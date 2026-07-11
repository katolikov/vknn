package com.vknn.chat

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.vknn.chat.model.BackendPolicy
import com.vknn.chat.model.InferenceBackend
import com.vknn.chat.model.ModelCatalog
import com.vknn.chat.model.ModelResidency
import com.vknn.chat.model.ModelSelection
import com.vknn.chat.model.ModelSpec
import com.vknn.chat.model.friendlyLoadError
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.drop
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
// metrics and the KV-cache context usage. The model itself is owned by the ModelStore (Library tab)
// and the ModelSelection picks WHICH chat variant loads; this view model observes both and loads
// the chosen file onto the GPU. All native/tokenizer work runs off the main thread; UI state is a
// StateFlow the Compose layer collects. The shared ModelResidency keeps at most one mode's model
// resident: every load acquires the slot (freeing any other mode's model first) and
// Unload/library-delete/variant-switch release it.
class ChatViewModel(app: Application) : AndroidViewModel(app), ModelResidency.Holder {
    private val _ui = MutableStateFlow(UiState())
    val ui: StateFlow<UiState> = _ui.asStateFlow()

    private val store = (app as VknnApp).models
    private val settings = (app as VknnApp).settings
    private val prompts = (app as VknnApp).prompts
    private val residency = (app as VknnApp).residency
    private val selection = (app as VknnApp).modelSelection
    private var tokenizer: Tokenizer? = null
    // The loaded model's chat family, fixed at load: it picks the tokenizer regex + source, the prompt
    // template, the control-token ids spliced into it, and the stop tokens. Defaults to the ChatML
    // (Qwen) dialect until a model loads.
    private var dialect: ChatPromptTemplate.Dialect = ChatPromptTemplate.QWEN
    private var handle: Long = 0L
    private var position = 0
    private var contextMax = 0
    private var vocabSize = 0

    // The loaded session registered the engine-side greedy argmax: the decode logits row is never
    // downloaded, so temperature sampling needs a reload (the send path pins the turn to greedy).
    private var engineArgMax = false

    // Set when a residency release waits on the streaming reply; the decode loop then stops early.
    @Volatile
    private var cancelRequested = false

    private val maxNew = 220
    private val topK = 40
    private val topP = 0.95f

    private fun currentSpec(): ModelSpec = selection.specFor(selection.chatKey.value, ModelCatalog.QWEN)

    init {
        // Follow the library and the variant selection: the chat mode is usable only while the
        // CHOSEN model is on device, and reverts to MISSING (releasing the decoder) when that file
        // is deleted from the Library.
        viewModelScope.launch {
            combine(store.states, selection.chatKey) { _, key -> selection.isPresent(key) }
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
        // A variant switch: any resident chat session belongs to the previous choice — release it
        // (deferred until an in-flight reply settles), then reflect the new choice's disk state.
        viewModelScope.launch {
            selection.chatKey.drop(1).collect {
                residency.releaseResidentModel(this@ChatViewModel)
                if (_ui.value.phase != Phase.LOADING && handle == 0L) {
                    _ui.value = _ui.value.copy(
                        phase = if (selection.isPresent(selection.chatKey.value)) Phase.DOWNLOADED else Phase.MISSING,
                        status = null,
                    )
                }
            }
        }
    }

    fun setTemperature(t: Float) {
        _ui.value = _ui.value.copy(temperature = t)
    }

    /** The variant picker: persists the choice; the collector above swaps the session. */
    fun selectModel(key: String) {
        if (key == selection.chatKey.value) return
        selection.setChatKey(key)
    }

    fun loadModel() {
        if (_ui.value.phase != Phase.DOWNLOADED) return
        val spec = currentSpec()
        val selectedKey = selection.chatKey.value
        val chosenBackend = settings.current()
        if (chosenBackend == InferenceBackend.CPU) {
            val verdict = BackendPolicy.cpuVerdict(spec, getApplication<VknnApp>().totalRamBytes())
            if (verdict is BackendPolicy.CpuVerdict.Blocked) {
                _ui.value = _ui.value.copy(status = "CPU backend: ${verdict.reason}")
                return
            }
        }
        // Engine-side greedy argmax is a load-time, session-lifetime registration (the decode
        // logits row is then never downloaded), so it engages only when the slider sits at greedy.
        val greedyArgMax = _ui.value.temperature <= 0f
        _ui.value = _ui.value.copy(phase = Phase.LOADING, status = null)
        viewModelScope.launch {
            // Frees any other mode's resident model before this load allocates, so at most one
            // heavy model lives in the process.
            residency.acquireResidency(this@ChatViewModel)
            cancelRequested = false
            try {
                withContext(Dispatchers.Default) {
                    val ctx = getApplication<Application>()
                    val chatDialect = ChatPromptTemplate.dialect(spec.chatDialectId)
                    // Qwen ships its vocab/merges in the app assets; Llama downloads them beside the
                    // model, read from the model's companion files like the VLM tokenizer.
                    tokenizer = if (chatDialect.tokenizerFromAssets) {
                        Tokenizer(
                            ctx.assets.open("vocab.json").bufferedReader().use { it.readText() },
                            ctx.assets.open("merges.txt").bufferedReader().use { it.readText() },
                            chatDialect.pattern,
                        )
                    } else {
                        Tokenizer(
                            store.auxFile(spec, "vocab.json").readText(),
                            store.auxFile(spec, "merges.txt").readText(),
                            chatDialect.pattern,
                        )
                    }
                    dialect = chatDialect
                    val vxm = store.file(spec).absolutePath
                    val cache = File(ctx.filesDir, ModelSelection.cacheFileName(spec)).absolutePath
                    val h = NativeLib.nativeInit(vxm, cache, "low", chosenBackend.engineName, greedyArgMax)
                    if (h == 0L) throw RuntimeException("model load failed (is the device Vulkan-capable?)")
                    handle = h
                    val info = NativeLib.nativeInfo(h) // {L,kvHeads,C,headDim,vocab,prefillS,engineArgMax}
                    contextMax = info[2]
                    vocabSize = info[4]
                    engineArgMax = info[6] == 1
                    NativeLib.nativeReset(h, 1234)
                    position = 0
                }
                residency.nativeCallSettled(this@ChatViewModel) // a release requested mid-load lands here
                if (!residency.isResident(this@ChatViewModel)) return@launch // released: UI already reset
                if (selection.isPresent(selectedKey) && selection.chatKey.value == selectedKey) {
                    store.clearLoadError(spec)
                    _ui.value = _ui.value.copy(phase = Phase.READY, contextMax = contextMax, contextUsed = 0, backend = chosenBackend.engineName)
                } else { // deleted or re-selected while loading: free the fresh session, reflect the current choice
                    residency.releaseResidentModel(this@ChatViewModel)
                }
            } catch (e: Exception) {
                val abandonedHandle = handle
                handle = 0L
                if (abandonedHandle != 0L) withContext(NonCancellable + Dispatchers.Default) { NativeLib.nativeFree(abandonedHandle) }
                residency.dropResidency(this@ChatViewModel)
                val friendly = friendlyLoadError(spec, e.message)
                store.reportLoadError(spec, friendly)
                val back = if (selection.isPresent(selectedKey)) Phase.DOWNLOADED else Phase.MISSING
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
        // Pinned for the whole turn, so editing the template mid-reply cannot change the prompt in
        // flight. The loaded dialect's template — a Qwen ChatML string would tokenize to garbage under
        // Llama-3 ids, so the two are read together.
        val template = prompts.chatTemplate(dialect)
        // An engine-argmax session never downloads the decode logits row, so temperature sampling
        // has nothing to sample from: the turn pins to greedy and the status names the way out.
        val requestedTemp = _ui.value.temperature
        val temp = if (engineArgMax && requestedTemp > 0f) 0f else requestedTemp
        val kMsgs = _ui.value.messages + Msg(Role.USER, text) + Msg(Role.ASSISTANT, "")
        val asstIdx = kMsgs.size - 1
        val tempStatus = if (engineArgMax && requestedTemp > 0f) "Temperature applies after a reload — this session decodes greedy on the GPU." else null
        _ui.value = _ui.value.copy(messages = kMsgs, generating = true, status = tempStatus)
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
                val prompt = ChatPromptTemplate.encodeTurn(tk, template, text, continuingContext = position > 0, dialect)
                // Control-token ids sit above the bundled vocab.json; a decoder whose embedding table stops
                // short of them would index out of range rather than fail.
                if (vocabSize > 0 && prompt.any { it >= vocabSize }) {
                    setAssistant(asstIdx, "[this model has no chat-template tokens — clear the prompt template]", Metrics())
                    return@withContext
                }
                // The whole prompt in one native call: batched prefill-bucket forwards when the
                // model carries a prefill bucket, per-token decode steps otherwise.
                val prefillResult = NativeLib.nativePrefill(h, LongArray(prompt.size) { prompt[it].toLong() })
                if (prefillResult == 0) position += prompt.size
                val prefillMs = (System.nanoTime() - t0) / 1_000_000
                if (prefillResult != 0) {
                    val reason = if (prefillResult == -2) "[context full — tap reset to start over]" else "[prompt feed failed — try reloading the model]"
                    setAssistant(asstIdx, reason, Metrics(prefillMs = prefillMs))
                    return@withContext
                }
                // A templated turn ends on the template's own stop token; the base model's end-of-stream
                // never arrives inside a ChatML turn. A negative sample is a native-side failure
                // (no logits row / argmax readback), never a token id.
                val stopTokens = ChatPromptTemplate.stopTokensFor(template, dialect)
                val dec = Tokenizer.StreamDecoder(tk)
                val sb = StringBuilder()
                var next = NativeLib.nativeSample(h, temp, if (temp > 0) topK else 0, topP)
                val firstNs = System.nanoTime()
                val ttftMs = (firstNs - t0) / 1_000_000
                var gen = 0
                while (next >= 0 && gen < maxNew && next !in stopTokens && position < contextMax - 1 && !cancelRequested) {
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
        if (releasedHandle != 0L) withContext(NonCancellable + Dispatchers.Default) { NativeLib.nativeFree(releasedHandle) }
    }

    override fun resetToUnloadedState() {
        contextMax = 0
        vocabSize = 0
        engineArgMax = false
        dialect = ChatPromptTemplate.QWEN
        _ui.value = UiState(
            phase = if (selection.isPresent(selection.chatKey.value)) Phase.DOWNLOADED else Phase.MISSING,
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
