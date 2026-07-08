package com.vknn.chat.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Eject
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.vknn.chat.Metrics
import com.vknn.chat.model.ModelCatalog
import com.vknn.chat.model.formatBytes
import com.vknn.chat.vlm.PromptBudgetVerdict
import com.vknn.chat.vlm.VlmPhase
import com.vknn.chat.vlm.VlmPromptMeasurement
import com.vknn.chat.vlm.VlmTemplate
import com.vknn.chat.vlm.VlmUiState
import kotlin.math.roundToInt

// Card scrim and shape. The scrim is opaque enough that streamed text stays readable over a bright
// scene; the camera is never assumed to be dark.
private val SuggestionCardScrim = Bg.copy(alpha = 0.72f)
private val SuggestionCardShape = RoundedCornerShape(18.dp)

// The card never covers more than this fraction of the preview; longer answers scroll inside it.
private const val SuggestionCardMaxHeightFraction = 0.35f

// The VLM camera coach: the camera preview is the main view, full-bleed, and the model's streamed
// suggestion floats over the top of it in a compact card. Usable only once the SmolVLM2 model is on
// device (Library).
@Composable
fun VlmScreen(
    ui: VlmUiState,
    onLoad: () -> Unit,
    onUnload: () -> Unit,
    onCapture: (Bitmap) -> Unit,
    onCancel: () -> Unit,
    onRetake: () -> Unit,
    onOpenLibrary: () -> Unit,
    question: String,
    onQuestion: (String) -> Unit,
    measurePrompt: (String) -> VlmPromptMeasurement?,
) {
    // Leaving the screen (tab switch) stops any in-flight generation between native calls.
    DisposableEffect(Unit) {
        onDispose { onCancel() }
    }
    var editingQuestion by remember { mutableStateOf(false) }
    // The coach prompt is editable in every phase, matching the Chat editor. Before the model is on the
    // GPU there is no tokenizer or prefill window to measure against, so the editor simply shows no token
    // counter then (validateCoachQuestion returns an empty verdict for an unmeasurable draft) and the
    // saved text is measured on the next load.

    if (editingQuestion) {
        PromptEditorDialog(
            title = "Coach prompt",
            explanation = "What the model is asked about your photo. The whole prompt — the image rows and the " +
                "chat template included — has to fit the decoder's prefill window.",
            initialPrompt = question,
            defaultPrompt = VlmTemplate.QUESTION,
            fieldPlaceholder = "Ask something about the photo",
            validate = { draft -> validateCoachQuestion(draft, measurePrompt) },
            onSave = onQuestion,
            onDismiss = { editingQuestion = false },
        )
    }

    Column(Modifier.fillMaxSize().background(Bg)) {
        VlmTopBar(
            showEditPrompt = true,
            onEditPrompt = { editingQuestion = true },
            showUnload = ui.phase == VlmPhase.CAMERA || ui.phase == VlmPhase.ANSWERING,
            onUnload = onUnload,
        )
        when (ui.phase) {
            VlmPhase.MISSING -> VlmSetup(
                title = "Model not downloaded",
                body = "Get ${ModelCatalog.SMOLVLM2.displayName} (~${formatBytes(ModelCatalog.SMOLVLM2.approxBytes)}) from the Model Library to use the camera coach.",
                button = "Open Library",
                status = ui.status,
                onClick = onOpenLibrary,
            )
            VlmPhase.DOWNLOADED -> VlmSetup(
                title = "Model ready",
                body = "Load it onto the GPU to start the camera coach.",
                button = "Load on GPU",
                status = ui.status,
                onClick = onLoad,
            )
            VlmPhase.LOADING -> Column(
                Modifier.fillMaxSize().padding(28.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                CircularProgressIndicator(color = Accent)
                Spacer(Modifier.height(16.dp))
                Text("Loading on GPU…", color = TextSecondary, fontSize = 13.sp)
            }
            // Both live and answering phases render the same stage; only the layer under the
            // overlay differs (live preview vs the frozen frame that was sent to the model).
            VlmPhase.CAMERA, VlmPhase.ANSWERING -> CameraStage(ui, onCapture, onCancel, onRetake)
        }
    }
}

// A coach question the editor will accept: non-empty, and — once the model can measure it — within the
// decoder's prefill window. An unmeasurable draft (no model loaded) carries no counter and no refusal.
internal fun validateCoachQuestion(
    draft: String,
    measurePrompt: (String) -> VlmPromptMeasurement?,
): PromptValidation {
    if (draft.isBlank()) return PromptValidation(rejectionReason = "Ask the model something about the photo")
    val measurement = measurePrompt(draft) ?: return PromptValidation()
    return PromptValidation(
        counterText = measurement.counterText(),
        tone = when (measurement.verdict) {
            PromptBudgetVerdict.OVER_BUDGET -> PromptCounterTone.ERROR
            PromptBudgetVerdict.NEAR_LIMIT -> PromptCounterTone.WARNING
            PromptBudgetVerdict.WITHIN_BUDGET -> PromptCounterTone.NEUTRAL
        },
        rejectionReason = measurement.rejectionReason(),
    )
}

@Composable
private fun VlmTopBar(showEditPrompt: Boolean, onEditPrompt: () -> Unit, showUnload: Boolean, onUnload: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier.size(30.dp).background(Surface, RoundedCornerShape(9.dp)),
            contentAlignment = Alignment.Center,
        ) { Icon(Icons.Filled.CameraAlt, null, tint = Accent, modifier = Modifier.size(18.dp)) }
        Spacer(Modifier.width(9.dp))
        Column(Modifier.weight(1f)) {
            Text("SmolVLM2 2.2B", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Medium)
            Text("camera coach — on-device", color = TextSecondary, fontSize = 11.sp)
        }
        if (showUnload) {
            TopBarIconButton(Icons.Filled.Eject, "unload model", onUnload)
            Spacer(Modifier.width(8.dp))
        }
        if (showEditPrompt) {
            Box(
                Modifier.size(30.dp).background(Surface, RoundedCornerShape(9.dp)).clickableNoRipple(onEditPrompt),
                contentAlignment = Alignment.Center,
            ) { Icon(Icons.Filled.Edit, "edit coach prompt", tint = TextSecondary, modifier = Modifier.size(16.dp)) }
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
private fun VlmSetup(title: String, body: String, button: String, status: String?, onClick: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(title, color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Medium)
        Spacer(Modifier.height(6.dp))
        Text(body, color = TextSecondary, fontSize = 13.sp, lineHeight = 19.sp)
        Spacer(Modifier.height(20.dp))
        PrimaryButton(button, onClick)
        status?.let {
            Spacer(Modifier.height(14.dp))
            Text(it, color = Err, fontSize = 12.sp)
        }
    }
}

// The five presentations of the suggestion card, derived from the view-model state. A status with
// no generation in flight is a failure (load refusal relayed through LoadErrors, or a vision
// encode / prefill failure); everything else follows the capture phase. Internal, JVM-tested: the
// failure check outranks the phase check, so an error never renders as a hint.
internal enum class SuggestionCardState { HINT, THINKING, STREAMING, DONE, ERROR }

internal fun suggestionCardStateFor(ui: VlmUiState): SuggestionCardState = when {
    ui.status != null && !ui.generating -> SuggestionCardState.ERROR
    ui.phase == VlmPhase.CAMERA -> SuggestionCardState.HINT
    ui.generating && ui.answer.isEmpty() -> SuggestionCardState.THINKING
    ui.generating -> SuggestionCardState.STREAMING
    else -> SuggestionCardState.DONE
}

// The caption under a finished answer. Cancelling mid-decode still leaves a valid TTFT and rate.
// Each term is dropped when unmeasured, so an immediate cancel yields an empty caption.
internal fun suggestionMetricsCaption(metrics: Metrics, backend: String): String {
    val parts = buildList {
        if (metrics.ttftMs > 0) add("TTFT ${metrics.ttftMs} ms")
        if (metrics.tokPerSec > 0) add("${(metrics.tokPerSec * 10).roundToInt() / 10.0} tok/s")
        if (backend.isNotEmpty()) add(backend)
    }
    return parts.joinToString("  ·  ")
}

// The full-bleed camera stage: preview (or the frozen capture) at the bottom, contrast scrims over
// it, the suggestion card floating at the top, and the shutter/cancel/retake controls at the bottom.
@Composable
private fun CameraStage(
    ui: VlmUiState,
    onCapture: (Bitmap) -> Unit,
    onCancel: () -> Unit,
    onRetake: () -> Unit,
) {
    val context = LocalContext.current
    var cameraGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { cameraGranted = it }
    if (!cameraGranted) {
        Column(
            Modifier.fillMaxSize().padding(28.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("Camera access needed", color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Medium)
            Spacer(Modifier.height(6.dp))
            Text("The coach answers questions about what the camera sees.", color = TextSecondary, fontSize = 13.sp)
            Spacer(Modifier.height(20.dp))
            PrimaryButton("Grant camera access") { permissionLauncher.launch(Manifest.permission.CAMERA) }
        }
        return
    }

    // Bound by the preview layer once CameraX has a session; cleared when that layer leaves the
    // composition, so the shutter can never fire against an unbound use case.
    var boundImageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var shutterBusy by remember { mutableStateOf(false) }
    val capturedPhoto = ui.photo
    val cardState = suggestionCardStateFor(ui)

    val takeShot = {
        val imageCapture = boundImageCapture
        if (imageCapture != null && !shutterBusy) {
            shutterBusy = true
            imageCapture.takePicture(
                ContextCompat.getMainExecutor(context),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        val photo = image.use { decodeUpright(it) }
                        shutterBusy = false
                        if (photo != null) onCapture(photo)
                    }

                    override fun onError(error: ImageCaptureException) {
                        shutterBusy = false
                    }
                },
            )
        }
    }

    BoxWithConstraints(Modifier.fillMaxSize()) {
        val suggestionCardMaxHeight = maxHeight * SuggestionCardMaxHeightFraction

        if (capturedPhoto == null) {
            LiveCameraPreviewLayer(onImageCaptureBound = { boundImageCapture = it })
        } else {
            // Crop reproduces PreviewView's FILL_CENTER framing, so the frozen frame lands where the
            // live preview stood. This crop is presentation only — the encoder receives the whole photo.
            Image(
                capturedPhoto.asImageBitmap(),
                contentDescription = "captured photo",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop,
            )
        }

        StageScrims()

        SuggestionOverlayCard(
            cardState = cardState,
            answerText = ui.answer,
            hintText = "Point at your shot and tap the shutter",
            errorText = ui.status,
            metricsCaption = suggestionMetricsCaption(ui.metrics, ui.backend),
            maxCardHeight = suggestionCardMaxHeight,
            modifier = Modifier.align(Alignment.TopCenter).padding(12.dp),
        )

        Box(
            Modifier.align(Alignment.BottomCenter).padding(bottom = 26.dp),
            contentAlignment = Alignment.Center,
        ) {
            when {
                capturedPhoto == null -> ShutterButton(busy = shutterBusy, onClick = takeShot)
                ui.generating -> PillButton("Cancel", accent = false, onClick = onCancel)
                else -> PillButton("Retake", accent = true, onClick = onRetake)
            }
        }
    }
}

// Gradients that hold the card and the controls legible over a bright scene. They sit above the
// preview and below the chrome, and take no touch input.
@Composable
private fun BoxScope.StageScrims() {
    Box(
        Modifier
            .align(Alignment.TopCenter)
            .fillMaxWidth()
            .fillMaxHeight(0.45f)
            .background(Brush.verticalGradient(listOf(Bg.copy(alpha = 0.50f), Color.Transparent))),
    )
    Box(
        Modifier
            .align(Alignment.BottomCenter)
            .fillMaxWidth()
            .fillMaxHeight(0.22f)
            .background(Brush.verticalGradient(listOf(Color.Transparent, Bg.copy(alpha = 0.60f)))),
    )
}

// Live CameraX preview. The bound ImageCapture is handed up so the stage's shutter can drive it,
// and released when this layer leaves the composition (a capture, or leaving the screen).
@Composable
private fun LiveCameraPreviewLayer(onImageCaptureBound: (ImageCapture?) -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(Unit) {
        onDispose {
            ProcessCameraProvider.getInstance(context).get().unbindAll()
            onImageCaptureBound(null)
        }
    }
    AndroidView(
        factory = { viewContext ->
            val previewView = PreviewView(viewContext)
            val providerFuture = ProcessCameraProvider.getInstance(viewContext)
            providerFuture.addListener({
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                val imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()
                provider.unbindAll()
                provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
                onImageCaptureBound(imageCapture)
            }, ContextCompat.getMainExecutor(viewContext))
            previewView
        },
        modifier = Modifier.fillMaxSize(),
    )
}

@Composable
private fun ShutterButton(busy: Boolean, onClick: () -> Unit) {
    Box(
        Modifier
            .size(68.dp)
            .background(if (busy) Surface else OnAccent, CircleShape)
            .clickableNoRipple(onClick),
        contentAlignment = Alignment.Center,
    ) {
        Box(Modifier.size(56.dp).background(Bg.copy(alpha = 0.15f), CircleShape))
    }
}

// The compact overlay panel: an eyebrow that names the state, the body (hint, streamed answer, or
// error), and — once the turn finishes — the latency caption. The body scrolls inside the card once
// the answer outgrows the height cap; the card itself animates between sizes as the text arrives.
@Composable
private fun SuggestionOverlayCard(
    cardState: SuggestionCardState,
    answerText: String,
    hintText: String,
    errorText: String?,
    metricsCaption: String,
    maxCardHeight: Dp,
    modifier: Modifier = Modifier,
) {
    val bodyScrollState = rememberScrollState()
    // While tokens arrive the newest line stays in view; maxValue settles after each layout pass.
    if (cardState == SuggestionCardState.STREAMING) {
        LaunchedEffect(bodyScrollState) {
            snapshotFlow { bodyScrollState.maxValue }.collect { bodyScrollState.scrollTo(it) }
        }
    }
    // A finished answer is read from its first word, so the card rewinds once the stream ends: following
    // the newest token is only useful while tokens are still arriving.
    if (cardState == SuggestionCardState.DONE) {
        LaunchedEffect(cardState) { bodyScrollState.animateScrollTo(0) }
    }
    Column(
        modifier
            .fillMaxWidth()
            .heightIn(max = maxCardHeight)
            .animateContentSize()
            .clip(SuggestionCardShape)
            .background(SuggestionCardScrim)
            .border(1.dp, Line.copy(alpha = 0.6f), SuggestionCardShape)
            .padding(14.dp),
    ) {
        SuggestionCardEyebrow(cardState)

        val bodyText = when (cardState) {
            SuggestionCardState.HINT -> hintText
            SuggestionCardState.THINKING -> ""
            SuggestionCardState.STREAMING -> answerText
            // A cancel landing before the first token leaves nothing to show.
            SuggestionCardState.DONE -> answerText.ifEmpty { "…" }
            SuggestionCardState.ERROR -> errorText.orEmpty()
        }
        if (bodyText.isNotEmpty()) {
            Spacer(Modifier.height(8.dp))
            Column(Modifier.weight(1f, fill = false).verticalScroll(bodyScrollState)) {
                Text(
                    bodyText,
                    color = when (cardState) {
                        SuggestionCardState.ERROR -> Err
                        SuggestionCardState.HINT -> TextSecondary
                        else -> TextPrimary
                    },
                    fontSize = 14.sp,
                    lineHeight = 20.sp,
                )
            }
        }

        AnimatedVisibility(visible = cardState == SuggestionCardState.DONE && metricsCaption.isNotEmpty()) {
            Column {
                Spacer(Modifier.height(9.dp))
                Text(metricsCaption, color = TextSecondary, fontSize = 11.sp)
            }
        }
    }
}

@Composable
private fun SuggestionCardEyebrow(cardState: SuggestionCardState) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        when (cardState) {
            SuggestionCardState.THINKING, SuggestionCardState.STREAMING ->
                CircularProgressIndicator(modifier = Modifier.size(13.dp), color = Accent, strokeWidth = 1.5.dp)
            SuggestionCardState.HINT ->
                Icon(Icons.Filled.CameraAlt, null, tint = TextSecondary, modifier = Modifier.size(13.dp))
            SuggestionCardState.DONE ->
                Icon(Icons.Filled.Bolt, null, tint = Accent, modifier = Modifier.size(13.dp))
            SuggestionCardState.ERROR ->
                Icon(Icons.Filled.ErrorOutline, null, tint = Err, modifier = Modifier.size(13.dp))
        }
        Spacer(Modifier.width(7.dp))
        Text(
            when (cardState) {
                SuggestionCardState.HINT -> "Coach"
                SuggestionCardState.THINKING -> "Looking…"
                SuggestionCardState.STREAMING, SuggestionCardState.DONE -> "Suggestion"
                SuggestionCardState.ERROR -> "Couldn't run"
            },
            color = if (cardState == SuggestionCardState.ERROR) Err else TextSecondary,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
        )
    }
}

// JPEG ImageProxy -> upright software Bitmap (the capture rotation is baked in by rotating the decode).
private fun decodeUpright(image: ImageProxy): Bitmap? {
    val buffer = image.planes[0].buffer
    val jpegBytes = ByteArray(buffer.remaining())
    buffer.get(jpegBytes)
    val decoded = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size) ?: return null
    val rotationDegrees = image.imageInfo.rotationDegrees
    if (rotationDegrees == 0) return decoded
    val rotation = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
    return Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, rotation, true)
}
