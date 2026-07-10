package com.vknn.chat.splat

import android.app.Application
import android.graphics.Bitmap
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.vknn.chat.NativeLib
import com.vknn.chat.VknnApp
import com.vknn.chat.model.BackendPolicy
import com.vknn.chat.model.InferenceBackend
import com.vknn.chat.model.ModelCatalog
import com.vknn.chat.model.ModelResidency
import com.vknn.chat.model.ModelState
import com.vknn.chat.model.friendlyLoadError
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

enum class SplatPhase { MISSING, DOWNLOADED, LOADING, CAPTURING, ENCODING, VIEWER }

data class SplatUiState(
    val phase: SplatPhase = SplatPhase.MISSING,
    val captureSide: Int = 224, // square frame side the encoder expects; set at load
    val capturing: Boolean = false, // the guided grab loop is armed (Start was tapped)
    val framesCaptured: Int = 0,
    val framesTotal: Int = 8,
    val render: Bitmap? = null,
    val rendering: Boolean = false,
    val renderMs: Long = 0,
    val gaussians: Int = 0,
    val backend: String = "", // engine backend of the loaded encoder (the rasterizer is always Vulkan)
    val status: String? = null,
)

// Drives the 3D-splat mode: a guided multi-frame capture feeds the YoNoSplat encoder once, then the
// orbit viewer re-renders the uploaded Gaussians from user-driven camera poses. The encoder session
// loads once and survives recaptures; only the pose changes per render. The model file is owned by
// the ModelStore (Library tab). The shared ModelResidency keeps at most one mode's model resident:
// every load acquires the slot (freeing any other mode's model first) and Unload/library-delete
// release it.
class SplatViewModel(app: Application) : AndroidViewModel(app), ModelResidency.Holder {
    private val _ui = MutableStateFlow(SplatUiState())
    val ui: StateFlow<SplatUiState> = _ui.asStateFlow()

    private val store = (app as VknnApp).models
    private val settings = (app as VknnApp).settings
    private val residency = (app as VknnApp).residency
    private var handle: Long = 0L
    private var views = 8
    private var encoderSide = 224 // square side of the encoder's input frames; set at load

    // Capture accumulators (main-thread only: the analyzer executor is the main executor).
    private var frameSlab = FloatArray(0)
    private var framesFilled = 0
    private var lastFrameMs = 0L
    private var focalXNormalized = 1f
    private var focalYNormalized = 1f

    // Orbit state (main-thread only); renders serialize through renderJobActive.
    private var basePose = FloatArray(16)
    private var pivotDepth = 1f
    private var latestCamera = OrbitCamera()
    private var renderJobActive = false

    init {
        viewModelScope.launch {
            store.states
                .map { it[ModelCatalog.DL3DV.id] is ModelState.Ready }
                .distinctUntilChanged()
                .collect { present ->
                    if (present) {
                        if (_ui.value.phase == SplatPhase.MISSING) _ui.value = _ui.value.copy(phase = SplatPhase.DOWNLOADED)
                    } else {
                        _ui.value = SplatUiState(phase = SplatPhase.MISSING)
                        // Frees the encoder once any encode/render settles; no-op when not loaded.
                        viewModelScope.launch { residency.releaseResidentModel(this@SplatViewModel) }
                    }
                }
        }
    }

    /** The square side (pixels) each captured frame must be scaled to; valid once loaded. */
    fun captureSide(): Int = encoderSide

    fun loadModel() {
        if (_ui.value.phase != SplatPhase.DOWNLOADED) return
        val chosenBackend = settings.current()
        if (chosenBackend == InferenceBackend.CPU) {
            val verdict = BackendPolicy.cpuVerdict(ModelCatalog.DL3DV, getApplication<VknnApp>().totalRamBytes())
            if (verdict is BackendPolicy.CpuVerdict.Blocked) {
                _ui.value = _ui.value.copy(status = "CPU backend: ${verdict.reason}")
                return
            }
        }
        _ui.value = _ui.value.copy(phase = SplatPhase.LOADING, status = null)
        viewModelScope.launch {
            // Frees any other mode's resident model before this load allocates, so at most one
            // heavy model lives in the process.
            residency.acquireResidency(this@SplatViewModel)
            try {
                withContext(Dispatchers.Default) {
                    val spec = ModelCatalog.DL3DV
                    val cache = File(getApplication<Application>().filesDir, "dl3dv.cache").absolutePath
                    val loaded = NativeLib.nativeSplatLoad(store.file(spec).absolutePath, cache, "low", chosenBackend.engineName, RENDER_SIDE)
                    if (loaded == 0L) throw RuntimeException("encoder load failed (is the device Vulkan-capable?)")
                    handle = loaded
                    val info = NativeLib.nativeSplatInfo(loaded) // {gaussians, views, height, width}
                    views = info[1]
                    encoderSide = info[3]
                }
                residency.nativeCallSettled(this@SplatViewModel) // a release requested mid-load lands here
                if (!residency.isResident(this@SplatViewModel)) return@launch // released: UI already reset
                if (store.isReady(ModelCatalog.DL3DV)) {
                    store.clearLoadError(ModelCatalog.DL3DV)
                    _ui.value = _ui.value.copy(captureSide = encoderSide, backend = chosenBackend.engineName)
                    startFreshCapture()
                } else { // deleted from the Library while loading: free the fresh session, revert to MISSING
                    residency.releaseResidentModel(this@SplatViewModel)
                }
            } catch (e: Exception) {
                val abandonedHandle = handle
                handle = 0L
                if (abandonedHandle != 0L) withContext(NonCancellable + Dispatchers.Default) { NativeLib.nativeSplatFree(abandonedHandle) }
                residency.dropResidency(this@SplatViewModel)
                val friendly = friendlyLoadError(ModelCatalog.DL3DV, e.message)
                store.reportLoadError(ModelCatalog.DL3DV, friendly)
                val back = if (store.isReady(ModelCatalog.DL3DV)) SplatPhase.DOWNLOADED else SplatPhase.MISSING
                _ui.value = _ui.value.copy(phase = back, status = friendly)
            }
        }
    }

    /** The Unload control: frees the resident encoder once any encode/render settles and returns to DOWNLOADED. */
    fun unloadModel() {
        viewModelScope.launch { residency.releaseResidentModel(this@SplatViewModel) }
    }

    /** Arms the guided grab loop; frames then accumulate via addFrame. */
    fun startCapture() {
        if (_ui.value.phase != SplatPhase.CAPTURING) return
        lastFrameMs = 0L
        _ui.value = _ui.value.copy(capturing = true, status = null)
    }

    /**
     * One upright square frame (already scaled to captureSide) from the analyzer stream, with the
     * normalized focal lengths of the cropped square. Grabs are paced FRAME_GAP_MS apart; the
     * views-th frame closes the capture and starts the encode.
     */
    fun addFrame(squareFrame: Bitmap, fxNormalized: Float, fyNormalized: Float) {
        val state = _ui.value
        if (state.phase != SplatPhase.CAPTURING || !state.capturing || framesFilled >= views) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastFrameMs < FRAME_GAP_MS) return

        val side = captureSide()
        if (frameSlab.isEmpty()) frameSlab = FloatArray(views * 3 * side * side)
        val argb = IntArray(side * side)
        squareFrame.getPixels(argb, 0, side, 0, 0, side, side)
        val candidate = rgbToChw01(argb, side)
        // Parallax gate: the encoder needs baseline between views — eight near-identical frames
        // reconstruct as low-opacity fog. A frame too similar to the last accepted one is skipped
        // (re-checked at the analyzer rate) until the user actually moves.
        if (framesFilled > 0 && frameDifference(candidate, frameSlab, (framesFilled - 1) * 3 * side * side) < MIN_FRAME_DIFFERENCE) {
            _ui.value = _ui.value.copy(status = "Keep moving \u2014 arc around your subject")
            return
        }
        lastFrameMs = now
        candidate.copyInto(frameSlab, framesFilled * 3 * side * side)
        focalXNormalized = fxNormalized
        focalYNormalized = fyNormalized
        framesFilled++
        _ui.value = _ui.value.copy(framesCaptured = framesFilled, framesTotal = views, status = null)
        if (framesFilled == views) encodeCapturedFrames()
    }

    /** Mean absolute difference between a candidate frame and the slab frame at [slabOffset], subsampled. */
    private fun frameDifference(candidate: FloatArray, slab: FloatArray, slabOffset: Int): Float {
        var sum = 0f
        var count = 0
        var i = 0
        while (i < candidate.size) {
            sum += kotlin.math.abs(candidate[i] - slab[slabOffset + i])
            count++
            i += 17 // prime stride: samples every channel/row phase without scanning all 150k floats
        }
        return if (count == 0) 0f else sum / count
    }

    fun recapture() {
        if (_ui.value.phase != SplatPhase.VIEWER || _ui.value.rendering) return
        startFreshCapture()
    }

    /** Orbit-camera update from the viewer gestures; renders serialize and coalesce to the latest. */
    fun setOrbit(camera: OrbitCamera) {
        latestCamera = SplatPose.clamped(camera)
        scheduleRender()
    }

    private fun startFreshCapture() {
        framesFilled = 0
        frameSlab = FloatArray(0)
        latestCamera = OrbitCamera()
        _ui.value = _ui.value.copy(
            phase = SplatPhase.CAPTURING,
            capturing = false,
            framesCaptured = 0,
            framesTotal = views,
            render = null,
            renderMs = 0,
            status = null,
        )
    }

    private fun encodeCapturedFrames() {
        val loaded = handle
        if (loaded == 0L) return
        _ui.value = _ui.value.copy(phase = SplatPhase.ENCODING, capturing = false)
        residency.nativeCallStarted(this)
        viewModelScope.launch {
            val startedMs = SystemClock.elapsedRealtime()
            val outcome = withContext(Dispatchers.Default) {
                // Normalized K per view: fx/fy from the camera characteristics, principal point at
                // the crop center (cx = cy = 0.5).
                val intrinsics = FloatArray(views * 9)
                for (view in 0 until views) {
                    val base = view * 9
                    intrinsics[base + 0] = focalXNormalized
                    intrinsics[base + 2] = 0.5f
                    intrinsics[base + 4] = focalYNormalized
                    intrinsics[base + 5] = 0.5f
                    intrinsics[base + 8] = 1f
                }
                if (NativeLib.nativeSplatEncode(loaded, frameSlab, intrinsics) != 0) return@withContext null
                val poses = NativeLib.nativeSplatPoses(loaded)
                poses.copyInto(basePose, 0, 0, 16) // view 0 anchors the orbit
                pivotDepth = NativeLib.nativeSplatPivotDepth(loaded)
                val info = NativeLib.nativeSplatInfo(loaded)
                val gaussians = info[0]
                val fogPermille = if (info.size > 4) info[4] else 0
                val pixels = NativeLib.nativeSplatRender(loaded, basePose) ?: return@withContext null
                Triple(gaussians, fogPermille, Bitmap.createBitmap(pixels, RENDER_SIDE, RENDER_SIDE, Bitmap.Config.ARGB_8888))
            }
            frameSlab = FloatArray(0)
            residency.nativeCallSettled(this@SplatViewModel) // a release requested mid-encode lands here
            if (!residency.isResident(this@SplatViewModel)) return@launch // released: UI already reset
            if (outcome == null) {
                _ui.value = _ui.value.copy(phase = SplatPhase.CAPTURING, capturing = false, framesCaptured = 0, status = "Encode failed — try again")
                framesFilled = 0
                return@launch
            }
            _ui.value = _ui.value.copy(
                phase = SplatPhase.VIEWER,
                render = outcome.third,
                gaussians = outcome.first,
                renderMs = SystemClock.elapsedRealtime() - startedMs,
                // Fog signature: a degenerate capture spreads mid-opacity gaussians everywhere
                // instead of concentrating opacity on surfaces — tell the user how to fix it.
                status = if (outcome.second > FOG_WARNING_PERMILLE)
                    "Low capture quality \u2014 arc slowly around a well-lit subject and recapture"
                else null,
            )
        }
    }

    private fun scheduleRender() {
        if (renderJobActive || handle == 0L || _ui.value.phase != SplatPhase.VIEWER) return
        renderJobActive = true
        residency.nativeCallStarted(this)
        val camera = latestCamera
        val loaded = handle
        _ui.value = _ui.value.copy(rendering = true)
        viewModelScope.launch {
            val startedMs = SystemClock.elapsedRealtime()
            val bitmap = withContext(Dispatchers.Default) {
                val pose = if (camera == OrbitCamera()) basePose
                else SplatPose.orbitPose(basePose, pivotDepth, camera)
                NativeLib.nativeSplatRender(loaded, pose)
                    ?.let { Bitmap.createBitmap(it, RENDER_SIDE, RENDER_SIDE, Bitmap.Config.ARGB_8888) }
            }
            renderJobActive = false
            residency.nativeCallSettled(this@SplatViewModel) // a release requested mid-render lands here
            if (!residency.isResident(this@SplatViewModel)) return@launch // released: UI already reset
            _ui.value = _ui.value.copy(
                render = bitmap ?: _ui.value.render,
                rendering = false,
                renderMs = SystemClock.elapsedRealtime() - startedMs,
                status = if (bitmap == null) "Render failed" else _ui.value.status,
            )
            if (camera != latestCamera) scheduleRender()
        }
    }

    // ModelResidency.Holder: the coordinator calls these on the main dispatcher when this mode's
    // model has to leave the GPU (Unload, library delete, or another mode acquiring the slot).
    // Encode and render are single native calls with nothing to stop early, so onReleaseRequested
    // keeps its no-op default.

    override suspend fun freeResidentSession() {
        val releasedHandle = handle
        handle = 0L
        if (releasedHandle != 0L) withContext(NonCancellable + Dispatchers.Default) { NativeLib.nativeSplatFree(releasedHandle) }
    }

    override fun resetToUnloadedState() {
        framesFilled = 0
        frameSlab = FloatArray(0)
        latestCamera = OrbitCamera()
        _ui.value = SplatUiState(
            phase = if (store.isReady(ModelCatalog.DL3DV)) SplatPhase.DOWNLOADED else SplatPhase.MISSING,
        )
    }

    override fun onCleared() {
        val abandonedHandle = handle
        handle = 0L
        if (abandonedHandle != 0L) NativeLib.nativeSplatFree(abandonedHandle)
        residency.dropResidency(this)
        super.onCleared()
    }

    private companion object {
        const val FRAME_GAP_MS = 900L // pacing between grabs while the user arcs the phone
        // Minimum subsampled mean-abs pixel difference vs the last accepted frame: static or
        // rotation-free captures sit well below this; a real arc step sits well above.
        const val MIN_FRAME_DIFFERENCE = 0.012f
        // Fraction (permille) of gaussians above opacity 0.1: a coherent scene concentrates
        // opacity in few splats (~40 permille); degenerate captures spread mid-opacity fog
        // everywhere (~300 permille).
        const val FOG_WARNING_PERMILLE = 150

        // Rasterizer output side; independent of the encoder input side (intrinsics are normalized).
        const val RENDER_SIDE = 512
    }
}
