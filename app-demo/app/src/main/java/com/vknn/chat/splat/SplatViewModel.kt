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
// the ModelStore (Library tab).
class SplatViewModel(app: Application) : AndroidViewModel(app) {
    private val _ui = MutableStateFlow(SplatUiState())
    val ui: StateFlow<SplatUiState> = _ui.asStateFlow()

    private val store = (app as VknnApp).models
    private val settings = (app as VknnApp).settings
    private var handle: Long = 0L
    private var views = 8
    private var encoderSide = 224 // square side of the encoder's input frames; set at load
    private var pendingFree = false

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
    private var nativeBusy = false // native call in flight on the Default dispatcher

    init {
        viewModelScope.launch {
            store.states
                .map { it[ModelCatalog.DL3DV.id] is ModelState.Ready }
                .distinctUntilChanged()
                .collect { present ->
                    if (present) {
                        if (_ui.value.phase == SplatPhase.MISSING) _ui.value = _ui.value.copy(phase = SplatPhase.DOWNLOADED)
                    } else {
                        if (handle != 0L) pendingFree = true
                        _ui.value = SplatUiState(phase = SplatPhase.MISSING)
                        releaseIfIdle()
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
                if (store.isReady(ModelCatalog.DL3DV)) {
                    store.clearLoadError(ModelCatalog.DL3DV)
                    _ui.value = _ui.value.copy(captureSide = encoderSide, backend = chosenBackend.engineName)
                    startFreshCapture()
                } else {
                    freeHandle()
                    _ui.value = _ui.value.copy(phase = SplatPhase.MISSING)
                }
            } catch (e: Exception) {
                val friendly = friendlyLoadError(ModelCatalog.DL3DV, e.message)
                store.reportLoadError(ModelCatalog.DL3DV, friendly)
                val back = if (store.isReady(ModelCatalog.DL3DV)) SplatPhase.DOWNLOADED else SplatPhase.MISSING
                _ui.value = _ui.value.copy(phase = back, status = friendly)
            }
        }
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
        lastFrameMs = now

        val side = captureSide()
        if (frameSlab.isEmpty()) frameSlab = FloatArray(views * 3 * side * side)
        val argb = IntArray(side * side)
        squareFrame.getPixels(argb, 0, side, 0, 0, side, side)
        rgbToChw01(argb, side).copyInto(frameSlab, framesFilled * 3 * side * side)
        focalXNormalized = fxNormalized
        focalYNormalized = fyNormalized
        framesFilled++
        _ui.value = _ui.value.copy(framesCaptured = framesFilled, framesTotal = views)
        if (framesFilled == views) encodeCapturedFrames()
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
        nativeBusy = true
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
                val gaussians = NativeLib.nativeSplatInfo(loaded)[0]
                val pixels = NativeLib.nativeSplatRender(loaded, basePose) ?: return@withContext null
                gaussians to Bitmap.createBitmap(pixels, RENDER_SIDE, RENDER_SIDE, Bitmap.Config.ARGB_8888)
            }
            frameSlab = FloatArray(0)
            nativeBusy = false
            if (pendingFree) { // the model left the library mid-encode: stay reverted
                releaseIfIdle()
                return@launch
            }
            if (outcome == null) {
                _ui.value = _ui.value.copy(phase = SplatPhase.CAPTURING, capturing = false, framesCaptured = 0, status = "Encode failed — try again")
                framesFilled = 0
                return@launch
            }
            _ui.value = _ui.value.copy(
                phase = SplatPhase.VIEWER,
                render = outcome.second,
                gaussians = outcome.first,
                renderMs = SystemClock.elapsedRealtime() - startedMs,
                status = null,
            )
        }
    }

    private fun scheduleRender() {
        if (renderJobActive || handle == 0L || _ui.value.phase != SplatPhase.VIEWER) return
        renderJobActive = true
        nativeBusy = true
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
            nativeBusy = false
            if (pendingFree) { // the model left the library mid-render: stay reverted
                releaseIfIdle()
                return@launch
            }
            _ui.value = _ui.value.copy(
                render = bitmap ?: _ui.value.render,
                rendering = false,
                renderMs = SystemClock.elapsedRealtime() - startedMs,
                status = if (bitmap == null) "Render failed" else _ui.value.status,
            )
            if (camera != latestCamera) scheduleRender()
        }
    }

    // A model delete while encoding/rendering defers the native free until the in-flight call
    // settles (all mutation happens on the main dispatcher, so the flags are race-free).
    private fun releaseIfIdle() {
        if (pendingFree && !nativeBusy) {
            pendingFree = false
            freeHandle()
        }
    }

    private fun freeHandle() {
        val loaded = handle
        handle = 0L
        if (loaded != 0L) NativeLib.nativeSplatFree(loaded)
    }

    override fun onCleared() {
        freeHandle()
        super.onCleared()
    }

    private companion object {
        const val FRAME_GAP_MS = 900L // pacing between grabs while the user arcs the phone

        // Square side (pixels) of the rasterizer output. Independent of the encoder input side:
        // the intrinsics are normalized, so the render sharpens without re-encoding.
        const val RENDER_SIDE = 512
    }
}
