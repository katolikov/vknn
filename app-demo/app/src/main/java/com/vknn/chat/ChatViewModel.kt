package com.vknn.chat

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

enum class Role { USER, ASSISTANT }
data class Msg(val role: Role, val text: String)
data class Metrics(val ttftMs: Long = 0, val tokPerSec: Double = 0.0, val prefillMs: Long = 0, val tokens: Int = 0)
enum class Phase { NEED_DOWNLOAD, DOWNLOADING, DOWNLOADED, LOADING, READY }

data class UiState(
    val phase: Phase = Phase.NEED_DOWNLOAD,
    val downloadPct: Int = 0,
    val downloadMB: Long = 0,
    val totalMB: Long = 0,
    val messages: List<Msg> = emptyList(),
    val metrics: Metrics = Metrics(),
    val temperature: Float = 0f,
    val generating: Boolean = false,
    val contextUsed: Int = 0,
    val contextMax: Int = 0,
    val status: String? = null,
)

// Drives the on-device decoder: download/load the model, then encode -> prefill -> stream decode a reply
// while tracking latency metrics and the KV-cache context usage. All native/tokenizer work runs off the
// main thread; UI state is a StateFlow the Compose layer collects.
class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val _ui = MutableStateFlow(UiState())
    val ui: StateFlow<UiState> = _ui.asStateFlow()

    private val filesDir: File = app.filesDir
    private var tokenizer: Tokenizer? = null
    private var handle: Long = 0L
    private var position = 0
    private var contextMax = 0

    private val eos = 151643      // <|endoftext|>
    private val maxNew = 220
    private val topK = 40
    private val topP = 0.95f

    init {
        if (ModelDownloader.isPresent(filesDir)) _ui.value = _ui.value.copy(phase = Phase.DOWNLOADED)
    }

    fun setTemperature(t: Float) {
        _ui.value = _ui.value.copy(temperature = t)
    }

    fun download() {
        if (_ui.value.phase == Phase.DOWNLOADING) return
        _ui.value = _ui.value.copy(phase = Phase.DOWNLOADING, status = null, downloadPct = 0)
        viewModelScope.launch {
            try {
                ModelDownloader.download(filesDir) { pct, done, total ->
                    _ui.value = _ui.value.copy(downloadPct = pct, downloadMB = done shr 20, totalMB = total shr 20)
                }
                _ui.value = _ui.value.copy(phase = Phase.DOWNLOADED)
            } catch (e: Exception) {
                _ui.value = _ui.value.copy(phase = Phase.NEED_DOWNLOAD, status = "Download failed: ${e.message}")
            }
        }
    }

    fun loadModel() {
        if (_ui.value.phase != Phase.DOWNLOADED) return
        _ui.value = _ui.value.copy(phase = Phase.LOADING, status = null)
        viewModelScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    val ctx = getApplication<Application>()
                    tokenizer = Tokenizer(
                        ctx.assets.open("vocab.json").bufferedReader().use { it.readText() },
                        ctx.assets.open("merges.txt").bufferedReader().use { it.readText() },
                    )
                    val vxm = ModelDownloader.modelFile(filesDir).absolutePath
                    val cache = File(filesDir, "qwen.cache").absolutePath
                    val h = NativeLib.nativeInit(vxm, cache, "low")
                    if (h == 0L) throw RuntimeException("model load failed (is the device Vulkan-capable?)")
                    handle = h
                    contextMax = NativeLib.nativeInfo(h)[2] // {L,kvHeads,C,headDim,vocab}
                    NativeLib.nativeReset(h, 1234)
                    position = 0
                }
                _ui.value = _ui.value.copy(phase = Phase.READY, contextMax = contextMax, contextUsed = 0)
            } catch (e: Exception) {
                _ui.value = _ui.value.copy(phase = Phase.DOWNLOADED, status = "Load failed: ${e.message}")
            }
        }
    }

    fun reset() {
        val h = handle
        if (h == 0L) return
        NativeLib.nativeReset(h, 1234)
        position = 0
        _ui.value = _ui.value.copy(messages = emptyList(), contextUsed = 0, metrics = Metrics(), status = null)
    }

    fun send(text: String) {
        val h = handle
        val tk = tokenizer ?: return
        if (h == 0L || _ui.value.generating || text.isBlank()) return
        val temp = _ui.value.temperature
        val kMsgs = _ui.value.messages + Msg(Role.USER, text) + Msg(Role.ASSISTANT, "")
        val asstIdx = kMsgs.size - 1
        _ui.value = _ui.value.copy(messages = kMsgs, generating = true, status = null)

        viewModelScope.launch {
            withContext(Dispatchers.Default) {
                val t0 = System.nanoTime()
                // Prefill: continue the running context (a leading newline separates turns after the first).
                val prompt = tk.encode(if (position == 0) text else "\n$text")
                var ok = true
                for (id in prompt) {
                    if (position >= contextMax - 1) { ok = false; break }
                    if (NativeLib.nativeStep(h, id) != 0) { ok = false; break }
                    position++
                }
                val prefillMs = (System.nanoTime() - t0) / 1_000_000
                if (!ok) {
                    setAssistant(asstIdx, "[context full — tap reset to start over]", Metrics(prefillMs = prefillMs))
                    return@withContext
                }
                val dec = Tokenizer.StreamDecoder(tk)
                val sb = StringBuilder()
                var next = NativeLib.nativeSample(h, temp, if (temp > 0) topK else 0, topP)
                val firstNs = System.nanoTime()
                val ttftMs = (firstNs - t0) / 1_000_000
                var gen = 0
                while (gen < maxNew && next != eos && position < contextMax - 1) {
                    val piece = dec.add(next)
                    gen++
                    if (piece.isNotEmpty()) sb.append(piece)
                    val secs = (System.nanoTime() - firstNs) / 1e9
                    val tps = if (secs > 0) gen / secs else 0.0
                    setAssistant(asstIdx, sb.toString(), Metrics(ttftMs, tps, prefillMs, gen))
                    _ui.value = _ui.value.copy(contextUsed = position)
                    if (NativeLib.nativeStep(h, next) != 0) break
                    position++
                    next = NativeLib.nativeSample(h, temp, if (temp > 0) topK else 0, topP)
                }
                if (sb.isEmpty()) setAssistant(asstIdx, "…", Metrics(ttftMs, 0.0, prefillMs, gen))
                val tps = if (gen > 0) gen / ((System.nanoTime() - firstNs) / 1e9) else 0.0
                android.util.Log.i("vknnchat-app", "reply: gen=$gen ttft=${ttftMs}ms prefill=${prefillMs}ms tps=${(tps * 10).toInt() / 10.0} text=${sb.toString().replace("\n", "\\n")}")
            }
            _ui.value = _ui.value.copy(generating = false, contextUsed = position)
        }
    }

    private fun setAssistant(idx: Int, text: String, metrics: Metrics) {
        val cur = _ui.value.messages
        if (idx !in cur.indices) return
        val next = cur.toMutableList()
        next[idx] = next[idx].copy(text = text)
        _ui.value = _ui.value.copy(messages = next, metrics = metrics)
    }

    override fun onCleared() {
        val h = handle
        handle = 0L
        if (h != 0L) NativeLib.nativeFree(h)
        super.onCleared()
    }
}
