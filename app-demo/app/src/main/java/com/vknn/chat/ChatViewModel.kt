package com.vknn.chat

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.vknn.chat.model.BackendPolicy
import com.vknn.chat.model.InferenceBackend
import com.vknn.chat.model.ModelCatalog
import com.vknn.chat.model.ModelResidency
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

enum class Role { USER, ASSISTANT }
data class Msg(val role: Role, val text: String)
data class Metrics(val ttftMs: Long = 0, val tokPerSec: Double = 0.0, val prefillMs: Long = 0, val tokens: Int = 0)
enum class Phase { MISSING, DOWNLOADED, LOADING, READY }

data class UiState(
    val phase: Phase = Phase.MISSING,
    val messages: List<Msg> = emptyList(),
    val metrics: Metrics = Metrics(),
    val temperature: Float = 0f,
    val generating: Boolean = false,
    val contextUsed: Int = 0,
    val contextMax: Int = 0,
    val backend: String = "", // engine backend of the loaded session (empty until loaded)
    val status: String? = null,
)

// Drives the on-device decoder: encode -> prefill -> stream decode a reply while tracking latency
// metrics and the KV-cache context usage. The model itself is owned by the ModelStore (Library tab);
// this view model only observes its presence and loads it onto the GPU. All native/tokenizer work
// runs off the main thread; UI state is a StateFlow the Compose layer collects. The shared
// ModelResidency keeps at most one mode's model resident: every load acquires the slot (freeing any
// other mode's model first) and Unload/library-delete release it.
class ChatViewModel(app: Application) : AndroidViewModel(app), ModelResidency.Holder {
    private val _ui = MutableStateFlow(UiState())
    val ui: StateFlow<UiState> = _ui.asStateFlow()

    private val store = (app as VknnApp).models
    private val settings = (app as VknnApp).settings
    private val prompts = (app as VknnApp).prompts
    private val residency = (app as VknnApp).residency
    private var tokenizer: Tokenizer? = null
    private var handle: Long = 0L
    private var position = 0
    private var contextMax = 0
    private var vocabSize = 0

    // Set when a residency release waits on the streaming reply; the decode loop then stops early.
    @Volatile
    private var cancelRequested = false

    private val maxNew = 220
    private val topK = 40
    private val topP = 0.95f

    init {
        // Follow the library: the chat mode is usable only while the Qwen model is on device, and
        // reverts to MISSING (releasing the decoder) when the model is deleted from the Library.
        viewModelScope.launch {
            store.states
                .map { it[ModelCatalog.QWEN.id] is ModelState.Ready }
                .distinctUntilChanged()
                .collect { present ->
                    if (present) {
                        if (_ui.value.phase == Phase.MISSING) _ui.value = _ui.value.copy(phase = Phase.DOWNLOADED)
                    } else {
                        _ui.value = _ui.value.copy(
                            phase = Phase.MISSING,
                            messages = emptyList(),
                            contextUsed = 0,
                            metrics = Metrics(),
                            status = null,
                        )
                        // Frees the decoder once any streaming reply settles; no-op when not loaded.
                        viewModelScope.launch { residency.releaseResidentModel(this@ChatViewModel) }
                    }
                }
        }
    }

    fun setTemperature(t: Float) {
        _ui.value = _ui.value.copy(temperature = t)
    }

    fun loadModel() {
        if (_ui.value.phase != Phase.DOWNLOADED) return
        val chosenBackend = settings.current()
        if (chosenBackend == InferenceBackend.CPU) {
            val verdict = BackendPolicy.cpuVerdict(ModelCatalog.QWEN, getApplication<VknnApp>().totalRamBytes())
            if (verdict is BackendPolicy.CpuVerdict.Blocked) {
                _ui.value = _ui.value.copy(status = "CPU backend: ${verdict.reason}")
                return
            }
        }
        _ui.value = _ui.value.copy(phase = Phase.LOADING, status = null)
        viewModelScope.launch {
            // Frees any other mode's resident model before this load allocates, so at most one
            // heavy model lives in the process.
            residency.acquireResidency(this@ChatViewModel)
            cancelRequested = false
            try {
                withContext(Dispatchers.Default) {
                    val ctx = getApplication<Application>()
                    tokenizer = Tokenizer(
                        ctx.assets.open("vocab.json").bufferedReader().use { it.readText() },
                        ctx.assets.open("merges.txt").bufferedReader().use { it.readText() },
                    )
                    val vxm = store.file(ModelCatalog.QWEN).absolutePath
                    val cache = File(ctx.filesDir, "qwen.cache").absolutePath
                    val h = NativeLib.nativeInit(vxm, cache, "low", chosenBackend.engineName)
                    if (h == 0L) throw RuntimeException("model load failed (is the device Vulkan-capable?)")
                    handle = h
                    val info = NativeLib.nativeInfo(h) // {L,kvHeads,C,headDim,vocab}
                    contextMax = info[2]
                    vocabSize = info[4]
                    NativeLib.nativeReset(h, 1234)
                    position = 0
                }
                residency.nativeCallSettled(this@ChatViewModel) // a release requested mid-load lands here
                if (!residency.isResident(this@ChatViewModel)) return@launch // released: UI already reset
                if (store.isReady(ModelCatalog.QWEN)) {
                    store.clearLoadError(ModelCatalog.QWEN)
                    _ui.value = _ui.value.copy(phase = Phase.READY, contextMax = contextMax, contextUsed = 0, backend = chosenBackend.engineName)
                } else { // deleted from the Library while loading: free the fresh session, revert to MISSING
                    residency.releaseResidentModel(this@ChatViewModel)
                }
            } catch (e: Exception) {
                val abandonedHandle = handle
                handle = 0L
                if (abandonedHandle != 0L) withContext(Dispatchers.Default) { NativeLib.nativeFree(abandonedHandle) }
                residency.dropResidency(this@ChatViewModel)
                val friendly = friendlyLoadError(ModelCatalog.QWEN, e.message)
                store.reportLoadError(ModelCatalog.QWEN, friendly)
                val back = if (store.isReady(ModelCatalog.QWEN)) Phase.DOWNLOADED else Phase.MISSING
                _ui.value = _ui.value.copy(phase = back, status = friendly)
            }
        }
    }

    /** The Unload control: frees the resident decoder once any streaming reply settles and returns to DOWNLOADED. */
    fun unloadModel() {
        viewModelScope.launch { residency.releaseResidentModel(this@ChatViewModel) }
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
        // Pinned for the whole turn, so editing the template mid-reply cannot change the prompt in flight.
        val template = prompts.chatPromptTemplate.value
        val temp = _ui.value.temperature
        val kMsgs = _ui.value.messages + Msg(Role.USER, text) + Msg(Role.ASSISTANT, "")
        val asstIdx = kMsgs.size - 1
        _ui.value = _ui.value.copy(messages = kMsgs, generating = true, status = null)
        cancelRequested = false
        residency.nativeCallStarted(this)

        viewModelScope.launch {
            withContext(Dispatchers.Default) {
                val t0 = System.nanoTime()
                // A template renders one self-contained turn, so it starts from a cleared context: appending
                // a fresh system block onto a running ChatML stream would leave the previous assistant turn
                // unclosed. An untemplated message continues the running context instead.
                if (ChatPromptTemplate.isActive(template)) {
                    NativeLib.nativeReset(h, 1234)
                    position = 0
                }
                val prompt = ChatPromptTemplate.encodeTurn(tk, template, text, continuingContext = position > 0)
                // Control-token ids sit above the bundled vocab.json; a decoder whose embedding table stops
                // short of them would index out of range rather than fail.
                if (vocabSize > 0 && prompt.any { it >= vocabSize }) {
                    setAssistant(asstIdx, "[this model has no chat-template tokens — clear the prompt template]", Metrics())
                    return@withContext
                }
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
                // A templated turn ends on the template's own stop token; the base model's end-of-stream
                // never arrives inside a ChatML turn.
                val stopTokens = ChatPromptTemplate.stopTokensFor(template)
                val dec = Tokenizer.StreamDecoder(tk)
                val sb = StringBuilder()
                var next = NativeLib.nativeSample(h, temp, if (temp > 0) topK else 0, topP)
                val firstNs = System.nanoTime()
                val ttftMs = (firstNs - t0) / 1_000_000
                var gen = 0
                while (gen < maxNew && next !in stopTokens && position < contextMax - 1 && !cancelRequested) {
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
            residency.nativeCallSettled(this@ChatViewModel) // a release requested mid-reply lands here
            if (residency.isResident(this@ChatViewModel)) {
                _ui.value = _ui.value.copy(generating = false, contextUsed = position)
            }
        }
    }

    private fun setAssistant(idx: Int, text: String, metrics: Metrics) {
        val cur = _ui.value.messages
        if (idx !in cur.indices) return
        val next = cur.toMutableList()
        next[idx] = next[idx].copy(text = text)
        _ui.value = _ui.value.copy(messages = next, metrics = metrics)
    }

    // ModelResidency.Holder: the coordinator calls these on the main dispatcher when this mode's
    // model has to leave the GPU (Unload, library delete, or another mode acquiring the slot).

    /** Stops a streaming reply between native calls, so a pending release settles quickly. */
    override fun onReleaseRequested() {
        cancelRequested = true
    }

    override suspend fun freeResidentSession() {
        val releasedHandle = handle
        handle = 0L
        position = 0
        if (releasedHandle != 0L) withContext(Dispatchers.Default) { NativeLib.nativeFree(releasedHandle) }
    }

    override fun resetToUnloadedState() {
        contextMax = 0
        vocabSize = 0
        _ui.value = UiState(
            phase = if (store.isReady(ModelCatalog.QWEN)) Phase.DOWNLOADED else Phase.MISSING,
            temperature = _ui.value.temperature,
        )
    }

    override fun onCleared() {
        cancelRequested = true
        val abandonedHandle = handle
        handle = 0L
        if (abandonedHandle != 0L) NativeLib.nativeFree(abandonedHandle)
        residency.dropResidency(this)
        super.onCleared()
    }
}
