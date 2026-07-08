package com.vknn.chat.vlm

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.vknn.chat.Metrics
import com.vknn.chat.NativeLib
import com.vknn.chat.Tokenizer
import com.vknn.chat.VknnApp
import com.vknn.chat.model.BackendPolicy
import com.vknn.chat.model.InferenceBackend
import com.vknn.chat.model.ModelCatalog
import com.vknn.chat.model.ModelState
import com.vknn.chat.model.friendlyLoadError
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

enum class VlmPhase { MISSING, DOWNLOADED, LOADING, CAMERA, ANSWERING }

data class VlmUiState(
    val phase: VlmPhase = VlmPhase.MISSING,
    val photo: Bitmap? = null,
    val answer: String = "",
    val metrics: Metrics = Metrics(),
    val generating: Boolean = false,
    val backend: String = "", // engine backend of the loaded session (empty until loaded)
    val status: String? = null,
)

// Drives the camera coach: photo -> vision encode -> prompt prefill (image rows spliced over the
// image tokens) -> streamed greedy decode, all through the VLM native bridge. Each capture is a
// fresh single-turn conversation (KV cache reset). The model is owned by the ModelStore; this view
// model observes its presence and loads it onto the GPU. Cancellation stops calling the native
// side between steps — the session itself is never torn down mid-run.
class VlmViewModel(app: Application) : AndroidViewModel(app) {
    private val _ui = MutableStateFlow(VlmUiState())
    val ui: StateFlow<VlmUiState> = _ui.asStateFlow()

    private val store = (app as VknnApp).models
    private val settings = (app as VknnApp).settings
    private val prompts = (app as VknnApp).prompts
    private var tokenizer: Tokenizer? = null
    private var handle: Long = 0L
    private var contextMax = 0
    private var prefillWindow = 0
    private var imageSide = 384
    private var pendingFree = false
    private var nativeBusy = false // decode work in flight on the Default dispatcher

    @Volatile
    private var cancelRequested = false

    private val maxNew = 128

    init {
        viewModelScope.launch {
            store.states
                .map { it[ModelCatalog.SMOLVLM2.id] is ModelState.Ready }
                .distinctUntilChanged()
                .collect { present ->
                    if (present) {
                        if (_ui.value.phase == VlmPhase.MISSING) _ui.value = _ui.value.copy(phase = VlmPhase.DOWNLOADED)
                    } else {
                        if (handle != 0L) pendingFree = true
                        cancelRequested = true
                        _ui.value = VlmUiState(phase = VlmPhase.MISSING)
                        releaseIfIdle()
                    }
                }
        }
    }

    /** The vision bucket's square input side (pixels); valid once the model is loaded. */
    fun imageSide(): Int = imageSide

    /**
     * Cost of one coach turn asking [question], against the loaded model's prefill window. Null until
     * the model is loaded: both the tokenizer and the window come from it.
     */
    fun measurePrompt(question: String): VlmPromptMeasurement? {
        val tk = tokenizer ?: return null
        if (prefillWindow <= 0) return null
        return VlmPromptBudget.measure(tk, question, prefillWindow)
    }

    /** Persist the coach question, and retire an over-budget refusal that the new question may resolve. */
    fun setQuestion(question: String) {
        prompts.setVlmQuestion(question)
        if (_ui.value.phase == VlmPhase.CAMERA && !_ui.value.generating) _ui.value = _ui.value.copy(status = null)
    }

    fun loadModel() {
        if (_ui.value.phase != VlmPhase.DOWNLOADED) return
        val chosenBackend = settings.current()
        if (chosenBackend == InferenceBackend.CPU) {
            val verdict = BackendPolicy.cpuVerdict(ModelCatalog.SMOLVLM2, getApplication<VknnApp>().totalRamBytes())
            if (verdict is BackendPolicy.CpuVerdict.Blocked) {
                _ui.value = _ui.value.copy(status = "CPU backend: ${verdict.reason}")
                return
            }
        }
        _ui.value = _ui.value.copy(phase = VlmPhase.LOADING, status = null)
        viewModelScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    val spec = ModelCatalog.SMOLVLM2
                    tokenizer = VlmTemplate.tokenizer(
                        store.auxFile(spec, "vocab.json").readText(),
                        store.auxFile(spec, "merges.txt").readText(),
                    )
                    val ctx = getApplication<Application>()
                    val cache = File(ctx.filesDir, "smolvlm2.cache").absolutePath
                    val h = NativeLib.nativeVlmInit(store.file(spec).absolutePath, cache, "low", chosenBackend.engineName)
                    if (h == 0L) throw RuntimeException("model load failed (is the device Vulkan-capable?)")
                    handle = h
                    val info = NativeLib.nativeVlmInfo(h) // {L,kvHeads,C,headDim,vocab,prefillS,imageRows,H,imageSide}
                    contextMax = info[2]
                    prefillWindow = info[5]
                    imageSide = info[8]
                    NativeLib.nativeVlmReset(h, 1234)
                }
                if (store.isReady(ModelCatalog.SMOLVLM2)) {
                    store.clearLoadError(ModelCatalog.SMOLVLM2)
                    _ui.value = _ui.value.copy(phase = VlmPhase.CAMERA, backend = chosenBackend.engineName)
                } else {
                    freeHandle()
                    _ui.value = _ui.value.copy(phase = VlmPhase.MISSING)
                }
            } catch (e: Exception) {
                val friendly = friendlyLoadError(ModelCatalog.SMOLVLM2, e.message)
                store.reportLoadError(ModelCatalog.SMOLVLM2, friendly)
                val back = if (store.isReady(ModelCatalog.SMOLVLM2)) VlmPhase.DOWNLOADED else VlmPhase.MISSING
                _ui.value = _ui.value.copy(phase = back, status = friendly)
            }
        }
    }

    /** One captured photo (already rotated upright): run the full see-and-answer turn. */
    fun onCapture(photo: Bitmap) {
        val h = handle
        val tk = tokenizer ?: return
        if (h == 0L || _ui.value.generating || _ui.value.phase != VlmPhase.CAMERA) return
        // Pinned for the whole turn, so editing the question mid-answer cannot change the prompt in flight.
        val question = prompts.vlmQuestion.value
        // The editor refuses to save an over-budget question; a prompt persisted against a wider window
        // (a differently compiled .vxm) can still arrive here, and the native prefill would only say
        // "prefill failed". Name the real cause instead.
        val measurement = measurePrompt(question)
        if (measurement != null && !measurement.fitsPrefillWindow) {
            _ui.value = _ui.value.copy(
                status = "Prompt is ${measurement.promptTokenCount} tokens, over this model's " +
                    "${measurement.prefillWindowTokens}-token prefill window. Edit it to fit.",
            )
            return
        }
        cancelRequested = false
        _ui.value = _ui.value.copy(phase = VlmPhase.ANSWERING, photo = photo, answer = "", metrics = Metrics(), generating = true, status = null)
        nativeBusy = true

        viewModelScope.launch {
            withContext(Dispatchers.Default) {
                val t0 = System.nanoTime()
                // Center-follow the HF processor with image splitting off: the full photo resized to
                // exactly side x side (aspect distortion intentional), RGB, x/127.5 - 1, fp32 CHW.
                val scaled = Bitmap.createScaledBitmap(photo, imageSide, imageSide, true)
                val argb = IntArray(imageSide * imageSide)
                scaled.getPixels(argb, 0, imageSide, 0, 0, imageSide, imageSide)
                val pixels = normalizeToChw(argb, imageSide)

                val emb = NativeLib.nativeVisionEncode(h, pixels)
                if (emb == null) {
                    fail("vision encode failed")
                    return@withContext
                }
                if (cancelRequested) {
                    finish()
                    return@withContext
                }
                NativeLib.nativeVlmReset(h, 1234)
                var position = 0
                val prompt = VlmTemplate.promptIds(tk, question)
                if (NativeLib.nativeVlmPrefill(h, prompt, emb, VlmTemplate.IMAGE, VlmTemplate.PAD) != 0) {
                    fail("prefill failed")
                    return@withContext
                }
                position += prompt.size
                val prefillMs = (System.nanoTime() - t0) / 1_000_000

                val dec = Tokenizer.StreamDecoder(tk)
                val sb = StringBuilder()
                var next = NativeLib.nativeVlmSample(h, 0f, 0, 1f) // greedy: a coach answer is deterministic
                val firstNs = System.nanoTime()
                val ttftMs = (firstNs - t0) / 1_000_000
                var gen = 0
                while (gen < maxNew && next != VlmTemplate.EOS && position < contextMax - 1 && !cancelRequested) {
                    val piece = dec.add(next)
                    gen++
                    if (piece.isNotEmpty()) sb.append(piece)
                    val secs = (System.nanoTime() - firstNs) / 1e9
                    val tps = if (secs > 0) gen / secs else 0.0
                    _ui.value = _ui.value.copy(answer = sb.toString(), metrics = Metrics(ttftMs, tps, prefillMs, gen))
                    if (NativeLib.nativeVlmStep(h, next) != 0) break
                    position++
                    next = NativeLib.nativeVlmSample(h, 0f, 0, 1f)
                }
                if (sb.isEmpty() && !cancelRequested) {
                    _ui.value = _ui.value.copy(answer = "…", metrics = Metrics(ttftMs, 0.0, prefillMs, gen))
                }
                finish()
            }
            nativeBusy = false
            releaseIfIdle()
        }
    }

    /** Stops the decode loop between native calls; the loaded session stays intact. */
    fun cancel() {
        cancelRequested = true
    }

    /** Back to the live camera for another shot. */
    fun retake() {
        if (_ui.value.generating) return
        if (_ui.value.phase == VlmPhase.ANSWERING) {
            _ui.value = _ui.value.copy(phase = VlmPhase.CAMERA, photo = null, answer = "", metrics = Metrics(), status = null)
        }
    }

    private fun fail(msg: String) {
        _ui.value = _ui.value.copy(generating = false, status = msg)
    }

    private fun finish() {
        _ui.value = _ui.value.copy(generating = false)
    }

    // A delete while a reply streams defers the native free until the decode coroutine settles;
    // pendingFree and nativeBusy only mutate on the main dispatcher, so the pair is race-free.
    private fun releaseIfIdle() {
        if (pendingFree && !nativeBusy) {
            pendingFree = false
            freeHandle()
        }
    }

    private fun freeHandle() {
        val h = handle
        handle = 0L
        // The window belongs to the released session; no prompt can be measured until one is loaded again.
        prefillWindow = 0
        if (h != 0L) NativeLib.nativeVlmFree(h)
    }

    override fun onCleared() {
        cancelRequested = true
        freeHandle()
        super.onCleared()
    }
}
