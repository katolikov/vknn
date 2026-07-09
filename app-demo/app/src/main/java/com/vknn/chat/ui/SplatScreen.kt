package com.vknn.chat.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Eject
import androidx.compose.material.icons.filled.Remove
import androidx.compose.material.icons.filled.ViewInAr
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.vknn.chat.model.ModelCatalog
import com.vknn.chat.model.formatBytes
import com.vknn.chat.splat.OrbitCamera
import com.vknn.chat.splat.SplatPhase
import com.vknn.chat.splat.SplatPose
import com.vknn.chat.splat.SplatUiState
import kotlin.math.roundToInt

// The 3D-splat mode: a guided 8-frame capture (slowly arc the phone around the subject) feeds the
// YoNoSplat encoder once, then the orbit viewer re-renders the splat scene — one finger orbits the
// look-at pivot in both axes, and double-tap, the zoom control, or a pinch dollies. Usable only once
// the dl3dv encoder is on device.
@Composable
fun SplatScreen(
    ui: SplatUiState,
    onLoad: () -> Unit,
    onUnload: () -> Unit,
    onStartCapture: () -> Unit,
    onFrame: (Bitmap, Float, Float) -> Unit,
    onOrbit: (OrbitCamera) -> Unit,
    onRecapture: () -> Unit,
    onOpenLibrary: () -> Unit,
) {
    val encoderLoaded = ui.phase == SplatPhase.CAPTURING ||
        ui.phase == SplatPhase.ENCODING ||
        ui.phase == SplatPhase.VIEWER
    Column(Modifier.fillMaxSize().background(Bg)) {
        SplatTopBar(showUnload = encoderLoaded, onUnload = onUnload)
        when (ui.phase) {
            SplatPhase.MISSING -> SplatSetup(
                title = "Model not downloaded",
                body = "Get the ${ModelCatalog.DL3DV.displayName} (~${formatBytes(ModelCatalog.DL3DV.approxBytes)}) from the Model Library to capture splat scenes.",
                button = "Open Library",
                status = ui.status,
                onClick = onOpenLibrary,
            )
            SplatPhase.DOWNLOADED -> SplatSetup(
                title = "Encoder ready",
                body = "Load it onto the GPU to start capturing.",
                button = "Load encoder",
                status = ui.status,
                onClick = onLoad,
            )
            SplatPhase.LOADING -> CenteredSpinner("Loading on GPU…")
            SplatPhase.CAPTURING -> CapturePane(ui, onStartCapture, onFrame)
            SplatPhase.ENCODING -> CenteredSpinner("Building the splat scene on the GPU…")
            SplatPhase.VIEWER -> ViewerPane(ui, onOrbit, onRecapture)
        }
    }
}

@Composable
private fun SplatTopBar(showUnload: Boolean, onUnload: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier.size(30.dp).background(Surface, RoundedCornerShape(9.dp)),
            contentAlignment = Alignment.Center,
        ) { Icon(Icons.Filled.ViewInAr, null, tint = Accent, modifier = Modifier.size(18.dp)) }
        Spacer(Modifier.width(9.dp))
        Column(Modifier.weight(1f)) {
            Text("YoNoSplat", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Medium)
            Text("3D splat capture — on-device", color = TextSecondary, fontSize = 11.sp)
        }
        if (showUnload) {
            TopBarIconButton(Icons.Filled.Eject, "unload model", onUnload)
            Spacer(Modifier.width(8.dp))
        }
        Row(
            Modifier.background(AccentDim, RoundedCornerShape(999.dp)).padding(horizontal = 9.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(Icons.Filled.Bolt, null, tint = Accent, modifier = Modifier.size(13.dp))
            Spacer(Modifier.width(4.dp))
            Text("GPU", color = Accent, fontSize = 11.sp)
        }
    }
}

@Composable
private fun SplatSetup(title: String, body: String, button: String, status: String?, onClick: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(title, color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Medium)
        Spacer(Modifier.height(6.dp))
        Text(body, color = TextSecondary, fontSize = 13.sp, lineHeight = 19.sp, textAlign = TextAlign.Center)
        Spacer(Modifier.height(20.dp))
        PrimaryButton(button, onClick)
        status?.let {
            Spacer(Modifier.height(14.dp))
            Text(it, color = Err, fontSize = 12.sp)
        }
    }
}

@Composable
private fun CenteredSpinner(caption: String) {
    Column(
        Modifier.fillMaxSize().padding(28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        CircularProgressIndicator(color = Accent)
        Spacer(Modifier.height(16.dp))
        Text(caption, color = TextSecondary, fontSize = 13.sp)
    }
}

// Live preview + the paced 8-frame grab. ImageAnalysis emits RGBA frames; each accepted frame is
// rotated upright, center-cropped to a square, scaled to the encoder side, and handed to the view
// model together with the normalized focal lengths of the cropped square.
@androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
@Composable
private fun CapturePane(ui: SplatUiState, onStartCapture: () -> Unit, onFrame: (Bitmap, Float, Float) -> Unit) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    if (!granted) {
        SplatSetup(
            title = "Camera access needed",
            body = "The capture arcs the camera around your subject to build the 3D scene.",
            button = "Grant camera access",
            status = null,
        ) { permission.launch(Manifest.permission.CAMERA) }
        return
    }

    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(Unit) {
        onDispose { ProcessCameraProvider.getInstance(context).get().unbindAll() }
    }
    Box(Modifier.fillMaxSize()) {
        AndroidView(
            factory = { viewContext ->
                val previewView = PreviewView(viewContext)
                val providerFuture = ProcessCameraProvider.getInstance(viewContext)
                providerFuture.addListener({
                    val provider = providerFuture.get()
                    val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                    val analysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                    provider.unbindAll()
                    val camera = provider.bindToLifecycle(
                        lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis,
                    )
                    val characteristics = Camera2CameraInfo.from(camera.cameraInfo)
                    val focalMm = characteristics
                        .getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                        ?.firstOrNull()
                    val sensorSize = characteristics
                        .getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                    analysis.setAnalyzer(ContextCompat.getMainExecutor(viewContext)) { frame ->
                        frame.use { image ->
                            val square = frameToUprightSquare(image, ui.captureSide)
                            if (square != null) {
                                val (focalXNorm, focalYNorm) = normalizedFocals(
                                    focalMm, sensorSize?.width, sensorSize?.height,
                                    image.width, image.height, image.imageInfo.rotationDegrees,
                                )
                                onFrame(square, focalXNorm, focalYNorm)
                            }
                        }
                    }
                }, ContextCompat.getMainExecutor(viewContext))
                previewView
            },
            modifier = Modifier.fillMaxSize(),
        )
        Text(
            if (ui.capturing) "Slowly arc the phone around your subject"
            else "Frame your subject, then start — keep it centered while you arc",
            color = TextPrimary,
            fontSize = 13.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(14.dp)
                .background(Bg.copy(alpha = 0.6f), RoundedCornerShape(12.dp))
                .padding(horizontal = 12.dp, vertical = 8.dp),
        )
        ui.status?.let {
            Text(
                it,
                color = Err,
                fontSize = 12.sp,
                modifier = Modifier
                    .align(Alignment.Center)
                    .background(Surface, RoundedCornerShape(10.dp))
                    .padding(horizontal = 10.dp, vertical = 6.dp),
            )
        }
        Box(Modifier.align(Alignment.BottomCenter).padding(bottom = 26.dp), contentAlignment = Alignment.Center) {
            if (!ui.capturing) {
                PrimaryButton("Start capture", onStartCapture)
            } else {
                CircularProgressIndicator(
                    progress = { ui.framesCaptured / ui.framesTotal.toFloat() },
                    modifier = Modifier.size(68.dp),
                    color = Accent,
                    trackColor = Surface,
                    strokeWidth = 5.dp,
                )
                Text(
                    "${ui.framesCaptured}/${ui.framesTotal}",
                    color = TextPrimary,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier.background(Bg.copy(alpha = 0.4f), CircleShape).padding(6.dp),
                )
            }
        }
    }
}

// The rendered splat with its orbit gestures. One finger drives the whole orbit: horizontal drag =
// azimuth, vertical drag = elevation. Zoom is reachable one-handed through a double tap (dolly in)
// and the +/- control pinned to the render's bottom corner; a pinch still dollies both ways for
// two-handed use. Every gesture update reaches the view model immediately; its single-flight
// render loop coalesces the stream to back-to-back renders of the latest pose, so the orbit
// tracks the finger continuously at whatever rate the rasterizer sustains.
@Composable
private fun ViewerPane(ui: SplatUiState, onOrbit: (OrbitCamera) -> Unit, onRecapture: () -> Unit) {
    var camera by remember { mutableStateOf(OrbitCamera()) }
    val moveCamera: (OrbitCamera) -> Unit = { next ->
        camera = next
        onOrbit(next)
    }
    Column(Modifier.fillMaxSize().padding(horizontal = 14.dp)) {
        Box(
            Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .clip(RoundedCornerShape(16.dp))
                .background(Surface)
                // The tap detector consumes only the initial down, which the transform detector reads
                // with requireUnconsumed = false; a drag past touch slop then consumes the moves and
                // cancels the pending tap. The two coexist without stealing from each other.
                .pointerInput(Unit) {
                    detectTapGestures(onDoubleTap = { moveCamera(SplatPose.doubleTapZoomStep(camera)) })
                }
                .pointerInput(Unit) {
                    detectTransformGestures { _, dragDelta, pinchScale, _ ->
                        moveCamera(
                            SplatPose.applyPinch(
                                SplatPose.applyDrag(camera, dragDelta.x, dragDelta.y),
                                pinchScale,
                            ),
                        )
                    }
                },
            contentAlignment = Alignment.Center,
        ) {
            ui.render?.let {
                Image(
                    it.asImageBitmap(),
                    contentDescription = "rendered splat view",
                    contentScale = ContentScale.FillBounds,
                    modifier = Modifier.fillMaxSize(),
                )
            }
            if (ui.rendering) {
                CircularProgressIndicator(
                    color = Accent,
                    modifier = Modifier.size(30.dp).align(Alignment.TopEnd).padding(6.dp),
                    strokeWidth = 3.dp,
                )
            }
            ZoomControl(
                onZoomIn = { moveCamera(SplatPose.zoomInStep(camera)) },
                onZoomOut = { moveCamera(SplatPose.zoomOutStep(camera)) },
                modifier = Modifier.align(Alignment.BottomEnd).padding(10.dp),
            )
        }
        Spacer(Modifier.height(10.dp))
        Text(
            "${ui.gaussians} gaussians · ${ui.renderMs} ms · ${ui.backend.ifEmpty { "vulkan" }} · drag to orbit, double-tap or +/− to zoom",
            color = TextSecondary,
            fontSize = 11.sp,
        )
        Text(
            "azimuth ${SplatPose.azimuthDegrees(camera).roundToInt()}° · " +
                "elevation ${SplatPose.elevationDegrees(camera).roundToInt()}° · " +
                "dolly ${(camera.dolly * 100).roundToInt()}%",
            color = TextSecondary,
            fontSize = 11.sp,
        )
        ui.status?.let {
            Spacer(Modifier.height(6.dp))
            Text(it, color = Err, fontSize = 12.sp)
        }
        Spacer(Modifier.height(14.dp))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
            PillButton("Recapture", accent = true, onClick = onRecapture)
        }
    }
}

// The one-handed zoom affordance: two discrete dolly steps in a translucent pill over the render's
// bottom corner, where a thumb already rests. Each button consumes the taps that land on it, so a tap
// zooms instead of orbiting; it leaves motion unconsumed, so a drag that starts on the pill still
// orbits. Steps flow through the view model's render coalescing, so a burst of taps renders the
// latest dolly.
@Composable
private fun ZoomControl(onZoomIn: () -> Unit, onZoomOut: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier
            .background(Bg.copy(alpha = 0.55f), RoundedCornerShape(999.dp))
            .padding(vertical = 2.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        ZoomButton(Icons.Filled.Add, "zoom in", onZoomIn)
        Box(Modifier.width(16.dp).height(1.dp).background(Line))
        ZoomButton(Icons.Filled.Remove, "zoom out", onZoomOut)
    }
}

@Composable
private fun ZoomButton(icon: ImageVector, description: String, onClick: () -> Unit) {
    Box(
        Modifier.size(40.dp).clip(CircleShape).clickableNoRipple(onClick),
        contentAlignment = Alignment.Center,
    ) { Icon(icon, description, tint = TextPrimary, modifier = Modifier.size(18.dp)) }
}

// One analyzer frame (RGBA_8888) -> upright square bitmap at [side]: compact the row stride,
// rotate by the frame's rotation, center-crop the shorter side, scale.
private fun frameToUprightSquare(image: ImageProxy, side: Int): Bitmap? {
    val plane = image.planes[0]
    val rowStrideBytes = plane.rowStride
    val widthBytes = image.width * 4
    val buffer = plane.buffer
    val fullFrame = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
    if (rowStrideBytes == widthBytes) {
        fullFrame.copyPixelsFromBuffer(buffer)
    } else {
        val compact = ByteArray(widthBytes * image.height)
        val row = ByteArray(rowStrideBytes)
        for (y in 0 until image.height) {
            buffer.position(y * rowStrideBytes)
            val available = minOf(rowStrideBytes, buffer.remaining())
            buffer.get(row, 0, available)
            System.arraycopy(row, 0, compact, y * widthBytes, widthBytes)
        }
        fullFrame.copyPixelsFromBuffer(java.nio.ByteBuffer.wrap(compact))
    }
    val rotationDegrees = image.imageInfo.rotationDegrees
    val upright = if (rotationDegrees == 0) fullFrame else Bitmap.createBitmap(
        fullFrame, 0, 0, fullFrame.width, fullFrame.height,
        Matrix().apply { postRotate(rotationDegrees.toFloat()) }, true,
    )
    val squareSide = minOf(upright.width, upright.height)
    val cropped = Bitmap.createBitmap(
        upright,
        (upright.width - squareSide) / 2,
        (upright.height - squareSide) / 2,
        squareSide,
        squareSide,
    )
    return Bitmap.createScaledBitmap(cropped, side, side, true)
}

// Normalized focal lengths of the center-cropped square: fx_px = f_mm * frameWidth / sensorWidth_mm,
// divided by the square side; a 90/270-degree frame rotation swaps the axes. Falls back to 1.0
// (a plausible phone FOV) when the characteristics are unavailable.
internal fun normalizedFocals(
    focalMm: Float?,
    sensorWidthMm: Float?,
    sensorHeightMm: Float?,
    frameWidth: Int,
    frameHeight: Int,
    rotationDegrees: Int,
): Pair<Float, Float> {
    if (focalMm == null || sensorWidthMm == null || sensorHeightMm == null ||
        sensorWidthMm <= 0f || sensorHeightMm <= 0f || frameWidth <= 0 || frameHeight <= 0
    ) {
        return 1f to 1f
    }
    val squareSide = minOf(frameWidth, frameHeight).toFloat()
    var focalXNorm = focalMm * frameWidth / sensorWidthMm / squareSide
    var focalYNorm = focalMm * frameHeight / sensorHeightMm / squareSide
    if (rotationDegrees % 180 != 0) {
        val swap = focalXNorm
        focalXNorm = focalYNorm
        focalYNorm = swap
    }
    return focalXNorm to focalYNorm
}

