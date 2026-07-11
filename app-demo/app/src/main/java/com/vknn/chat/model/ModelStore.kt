package com.vknn.chat.model

import android.app.Application
import android.net.ConnectivityManager
import android.os.StatFs
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap

sealed interface ModelState {
    data object Missing : ModelState
    data class Downloading(val bytesDone: Long, val bytesTotal: Long) : ModelState
    data class Paused(val bytesDone: Long, val bytesTotal: Long) : ModelState
    data class Verifying(val bytesDone: Long, val bytesTotal: Long) : ModelState
    data object Ready : ModelState
    data class Failed(val reason: String) : ModelState
}

// Owns every catalogue model on disk: resumable HTTP-Range downloads from HuggingFace, sha256
// verification, pause/delete, and a per-model state flow the mode screens key off. Lives on the
// Application so downloads keep running across activity recreation; partial files (.part) survive
// process death and resume from their byte offset.
class ModelStore(private val app: Application, private val catalog: StateFlow<List<ModelSpec>>) {

    // Model files live in getExternalFilesDir(null): app-scoped external storage, so no runtime
    // permission on any supported API level, removed by the OS on uninstall (no orphaned multi-GB
    // blobs — neither candidate location survives reinstall, which is right for cached downloads),
    // and attributed to the app in Settings -> Storage where the user can inspect and clear it.
    // Unlike filesDir it sits on the shared-storage volume, so on devices where that volume is a
    // larger partition (or an SD card) the multi-GB models do not compete with the internal
    // app-data budget. Falls back to filesDir only when external storage is unavailable. The first
    // release stored the Qwen model in filesDir; a complete file there is still honored (legacy
    // lookup below) and delete() clears both locations.
    private val root: File = app.getExternalFilesDir(null) ?: app.filesDir
    private val legacyRoot: File = app.filesDir

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val jobs = HashMap<String, Job>()                 // touched from the main thread only
    private val totals = ConcurrentHashMap<String, Long>()    // exact lengths learned this session

    private val _states = MutableStateFlow(catalog.value.associate { it.id to scanState(it) })
    val states: StateFlow<Map<String, ModelState>> = _states.asStateFlow()

    init {
        // Track the effective catalogue: a remote-manifest update adds newly-appeared models to the
        // state map at their on-disk state and drops entries that vanished, keeping the state of models
        // already tracked. The flow replays its current value, so this also seeds correctly on start.
        scope.launch {
            catalog.collect { list ->
                _states.update { cur -> list.associate { spec -> spec.id to (cur[spec.id] ?: scanState(spec)) } }
            }
        }
    }

    // Last failed-load reason per model (user-facing copy), shown on the Library card until the
    // model loads successfully or is deleted. Set by the mode view models.
    private val _loadErrors = MutableStateFlow<Map<String, String>>(emptyMap())
    val loadErrors: StateFlow<Map<String, String>> = _loadErrors.asStateFlow()

    fun reportLoadError(spec: ModelSpec, message: String) {
        _loadErrors.update { it + (spec.id to message) }
    }

    fun clearLoadError(spec: ModelSpec) {
        _loadErrors.update { it - spec.id }
    }

    fun state(spec: ModelSpec): ModelState = _states.value[spec.id] ?: ModelState.Missing
    fun isReady(spec: ModelSpec): Boolean = state(spec) is ModelState.Ready

    /** Where the model actually is: the primary root, else the legacy filesDir location. */
    fun file(spec: ModelSpec): File {
        val primary = File(root, spec.localFileName)
        if (primary.exists()) return primary
        val legacy = File(legacyRoot, spec.localFileName)
        return if (legacyComplete(spec, legacy)) legacy else primary
    }

    fun freeBytes(): Long = StatFs(root.absolutePath).availableBytes

    fun isMetered(): Boolean =
        app.getSystemService(ConnectivityManager::class.java)?.isActiveNetworkMetered == true

    /** Starts a download, or resumes one from its .part offset. No-op when already running or done. */
    fun start(spec: ModelSpec) {
        when (state(spec)) {
            is ModelState.Downloading, is ModelState.Verifying, is ModelState.Ready -> return
            else -> {}
        }
        val prev = jobs[spec.id] // a canceled pause/delete may still be draining its .part writes
        setState(spec, ModelState.Downloading(partFile(spec).length(), knownTotal(spec)))
        jobs[spec.id] = scope.launch {
            prev?.join()
            runDownload(spec)
        }
    }

    /** Cancels the transfer, keeping the .part for a later resume. */
    fun pause(spec: ModelSpec) {
        jobs[spec.id]?.cancel()
    }

    /** Removes the model (and any partial download) from every location to free its space. */
    fun delete(spec: ModelSpec) {
        val prev = jobs[spec.id]
        prev?.cancel()
        jobs[spec.id] = scope.launch {
            prev?.join()
            partFile(spec).delete()
            File(root, spec.localFileName).delete()
            File(legacyRoot, spec.localFileName).delete()
            for (aux in spec.auxFiles) auxFile(spec, aux).delete()
            clearLoadError(spec)
            setState(spec, ModelState.Missing)
        }
    }

    /** A companion repo file on disk (tokenizer etc.), stored under a model-scoped name. */
    fun auxFile(spec: ModelSpec, name: String): File = File(root, spec.auxLocalName(name))

    /**
     * Ad-hoc .vxm files in the model directories that are not catalogue downloads — pre-seeded
     * builds (e.g. pushed via run-as into the legacy files/ dir). Partial downloads (.part) never
     * match; a name present in both directories resolves to the primary root's copy.
     */
    fun adHocModelFiles(): List<File> {
        val catalogueNames = catalog.value.map { it.localFileName }.toSet()
        return listOf(root, legacyRoot).distinct()
            .flatMap { dir ->
                dir.listFiles { f: File -> f.isFile && f.name.endsWith(".vxm") && f.name !in catalogueNames }
                    ?.toList() ?: emptyList()
            }
            .distinctBy { it.name }
    }

    // A file under its final name passed verification (the rename happens after the checksum check);
    // a legacy-location file is trusted at its exact catalogue length, matching the check the first
    // release applied before renaming.
    private fun scanState(spec: ModelSpec): ModelState {
        if (file(spec).exists()) {
            if (spec.auxFiles.any { !auxFile(spec, it).exists() }) {
                return ModelState.Failed("Companion files missing — tap Retry")
            }
            return ModelState.Ready
        }
        val part = partFile(spec)
        val done = part.length()
        if (done > 0) return ModelState.Paused(done, maxOf(spec.approxBytes, done))
        return ModelState.Missing
    }

    private fun legacyComplete(spec: ModelSpec, f: File): Boolean =
        legacyRoot != root && f.exists() && f.length() == spec.approxBytes

    private fun partFile(spec: ModelSpec) = File(root, spec.localFileName + ".part")

    private fun knownTotal(spec: ModelSpec): Long = totals[spec.id] ?: spec.approxBytes

    private fun setState(spec: ModelSpec, st: ModelState) {
        _states.update { it + (spec.id to st) }
    }

    private suspend fun runDownload(spec: ModelSpec) {
        val part = partFile(spec)
        var total = knownTotal(spec)
        try {
            root.mkdirs()
            if (file(spec).exists()) { // model already on disk: only companion files are missing
                fetchAuxFiles(spec)
                setState(spec, ModelState.Ready)
                return
            }
            setState(spec, ModelState.Downloading(part.length(), total))
            // The API gives the authoritative length + sha256 (and reflects re-uploads); the pinned
            // catalogue values only stand in when the API call fails.
            val info = HfApi.fetch(spec)
            if (info != null) {
                total = info.size
                totals[spec.id] = info.size
            }
            var done = part.length()
            if (info != null && done > info.size) { // stale .part from a replaced upload
                part.delete()
                done = 0
            }

            val free = freeBytes()
            val need = total - done + STORAGE_MARGIN_BYTES
            if (free < need) {
                setState(
                    spec,
                    ModelState.Failed(
                        "Not enough storage: needs ${formatBytes(need)} free, ${formatBytes(free)} available"
                    ),
                )
                return
            }

            setState(spec, ModelState.Downloading(done, total))
            if (info == null || done < info.size) {
                val conn = openFollowingRedirects(spec.url, done)
                try {
                    when (conn.responseCode) {
                        HttpURLConnection.HTTP_PARTIAL ->
                            if (conn.contentLengthLong > 0) total = done + conn.contentLengthLong
                        HttpURLConnection.HTTP_OK -> { // server ignored the Range: restart from zero
                            part.delete()
                            done = 0
                            if (conn.contentLengthLong > 0) total = conn.contentLengthLong
                        }
                        416 -> {} // requested range starts at EOF: the .part is already complete
                        else -> throw IOException("HTTP ${conn.responseCode}")
                    }
                    if (conn.responseCode != 416) {
                        totals[spec.id] = total
                        conn.inputStream.use { input ->
                            FileOutputStream(part, done > 0).use { out ->
                                val buf = ByteArray(1 shl 20)
                                var sinceUpdate = 0L
                                while (true) {
                                    // Pause/delete cancel here; a stalled socket read can defer the
                                    // cancellation by up to the read timeout.
                                    currentCoroutineContext().ensureActive()
                                    val n = input.read(buf)
                                    if (n < 0) break
                                    out.write(buf, 0, n)
                                    done += n
                                    sinceUpdate += n
                                    if (sinceUpdate >= PROGRESS_STRIDE_BYTES) {
                                        sinceUpdate = 0
                                        setState(spec, ModelState.Downloading(done, maxOf(total, done)))
                                    }
                                }
                            }
                        }
                    }
                } finally {
                    conn.disconnect()
                }
            }

            val size = part.length()
            if (info != null && size != info.size) {
                throw IOException("download incomplete: $size of ${info.size} bytes")
            }
            val expectedSha = info?.sha256 ?: spec.sha256
            if (expectedSha != null) {
                setState(spec, ModelState.Verifying(0, size))
                val actual = sha256Of(part) { hashed -> setState(spec, ModelState.Verifying(hashed, size)) }
                if (!actual.equals(expectedSha, ignoreCase = true)) {
                    part.delete()
                    setState(spec, ModelState.Failed("Checksum mismatch — the corrupted download was deleted"))
                    return
                }
            }
            val out = File(root, spec.localFileName)
            if (!part.renameTo(out)) {
                part.copyTo(out, overwrite = true)
                part.delete()
            }
            fetchAuxFiles(spec)
            setState(spec, ModelState.Ready)
        } catch (e: CancellationException) {
            setState(spec, ModelState.Paused(part.length(), maxOf(total, part.length())))
            throw e
        } catch (e: Exception) {
            setState(spec, ModelState.Failed(e.message ?: e.javaClass.simpleName))
        }
    }

    // Small companion files (tokenizer vocab/merges) download whole — no resume or checksum; a
    // failure throws and surfaces as the model's Failed state, and Retry re-fetches only what is
    // still missing.
    private suspend fun fetchAuxFiles(spec: ModelSpec) {
        for (name in spec.auxFiles) {
            val dst = auxFile(spec, name)
            if (dst.exists()) continue
            currentCoroutineContext().ensureActive()
            val conn = openFollowingRedirects(spec.auxUrl(name), 0)
            try {
                if (conn.responseCode != HttpURLConnection.HTTP_OK) {
                    throw IOException("$name: HTTP ${conn.responseCode}")
                }
                val tmp = File(dst.path + ".tmp")
                conn.inputStream.use { input -> tmp.outputStream().use { input.copyTo(it) } }
                if (!tmp.renameTo(dst)) {
                    tmp.copyTo(dst, overwrite = true)
                    tmp.delete()
                }
            } finally {
                conn.disconnect()
            }
        }
    }

    // Follows redirects manually so the Range header is guaranteed on every hop (the HF resolve
    // endpoint 302s to a CDN host) and relative Location values resolve against the current URL.
    private fun openFollowingRedirects(url0: String, offset: Long): HttpURLConnection {
        var url = URL(url0)
        var redirects = 0
        while (true) {
            val conn = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 20_000
                readTimeout = 60_000
                instanceFollowRedirects = false
                setRequestProperty("User-Agent", HF_USER_AGENT)
                if (offset > 0) setRequestProperty("Range", "bytes=$offset-")
                connect()
            }
            if (conn.responseCode in 300..399 && redirects < 8) {
                val loc = conn.getHeaderField("Location") ?: return conn
                conn.disconnect()
                url = URL(url, loc)
                redirects++
                continue
            }
            return conn
        }
    }

    private suspend fun sha256Of(f: File, onProgress: (Long) -> Unit): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { input ->
            val buf = ByteArray(1 shl 20)
            var done = 0L
            var sinceUpdate = 0L
            while (true) {
                currentCoroutineContext().ensureActive()
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
                done += n
                sinceUpdate += n
                if (sinceUpdate >= PROGRESS_STRIDE_BYTES * 16) {
                    sinceUpdate = 0
                    onProgress(done)
                }
            }
        }
        // Manual hex: String.format("%x") localizes digits under some locales, which would false-fail
        // the comparison against the ASCII digest the HF API reports.
        val digest = md.digest()
        val hex = StringBuilder(digest.size * 2)
        for (b in digest) {
            hex.append(HEX_DIGITS[(b.toInt() ushr 4) and 0xF])
            hex.append(HEX_DIGITS[b.toInt() and 0xF])
        }
        return hex.toString()
    }

    private companion object {
        const val STORAGE_MARGIN_BYTES = 512L shl 20 // headroom kept free for the rest of the system
        const val PROGRESS_STRIDE_BYTES = 4L shl 20
        val HEX_DIGITS = "0123456789abcdef".toCharArray()
    }
}
