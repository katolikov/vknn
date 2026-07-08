package com.vknn.chat

import com.vknn.chat.splat.OrbitCamera
import com.vknn.chat.splat.SplatPose
import com.vknn.chat.splat.rgbToChw01
import com.vknn.chat.ui.normalizedFocals
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

// Validates the orbit-viewer math (OpenCV-convention lookAt + Rodrigues orbit), the one-finger
// gesture-to-camera mappings and their clamps, the [0,1] CHW conversion the YoNoSplat encoder
// expects, and the normalized-focal-length derivation.
class SplatPoseTest {
    private val identityPose = floatArrayOf(
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        0f, 0f, 0f, 1f,
    )

    // A base pose with no axis parallel to a world axis, so it cannot hide a convention error.
    private val tiltedPose = SplatPose.lookAt(
        floatArrayOf(1.5f, -2f, 3f),
        floatArrayOf(-0.5f, 0.25f, -1f),
        floatArrayOf(0.2f, 0.94f, -0.1f),
    )

    @Test
    fun zeroOrbitReproducesTheBasePose() {
        val orbited = SplatPose.orbitPose(identityPose, pivotDepth = 2f, camera = OrbitCamera())
        assertArrayEquals(identityPose, orbited, 1e-5f)
    }

    @Test
    fun orbitPreservesPivotDistanceAndLooksAtThePivot() {
        val pivotDepth = 2f
        val orbited = SplatPose.orbitPose(identityPose, pivotDepth, OrbitCamera(azimuthRadians = 0.7f))
        val eye = SplatPose.position(orbited)
        val pivot = floatArrayOf(0f, 0f, pivotDepth)
        val distance = distanceBetween(eye, pivot)
        assertEquals(pivotDepth, distance, 1e-4f)
        // The camera forward axis points from the eye back at the pivot.
        val forward = SplatPose.axis(orbited, 2)
        for (component in 0..2) {
            assertEquals((pivot[component] - eye[component]) / distance, forward[component], 1e-4f)
        }
    }

    @Test
    fun quarterOrbitSwingsTheEyeOntoTheCamerasRightAxis() {
        val quarterTurn = OrbitCamera(azimuthRadians = (PI / 2).toFloat())
        val orbited = SplatPose.orbitPose(identityPose, pivotDepth = 1f, camera = quarterTurn)
        val eye = SplatPose.position(orbited)
        // Base eye (0,0,0) - pivot (0,0,1) rotated a quarter turn about world up (0,-1,0). A positive
        // azimuth swings the eye toward the base camera's own right axis (+x).
        assertEquals(1f, eye[0], 1e-4f)
        assertEquals(0f, eye[1], 1e-4f)
        assertEquals(1f, eye[2], 1e-4f)
    }

    @Test
    fun dollyScalesThePivotDistance() {
        val orbited = SplatPose.orbitPose(identityPose, pivotDepth = 2f, camera = OrbitCamera(dolly = 0.5f))
        assertEquals(1f, SplatPose.position(orbited)[2], 1e-4f) // pivot at z=2, half distance -> z=1
    }

    @Test
    fun lookAtStaysOrthonormal() {
        val pose = SplatPose.lookAt(
            floatArrayOf(3f, -1f, 2f),
            floatArrayOf(0f, 0.5f, -1f),
            floatArrayOf(0.1f, 0.9f, 0.2f),
        )
        assertRotationIsOrthonormalAndRightHanded(pose)
    }

    // --- one-finger drag: horizontal -> azimuth, vertical -> elevation --------------------------

    @Test
    fun horizontalDragYawsAndVerticalDragElevatesAtTheSameSensitivity() {
        val draggedRight = SplatPose.applyDrag(OrbitCamera(), dragDeltaX = 100f, dragDeltaY = 0f)
        assertEquals(0.6f, draggedRight.azimuthRadians, 1e-5f) // 100 px * 0.006 rad/px
        assertEquals(0f, draggedRight.elevationRadians, 0f)

        // Screen y grows downward, so dragging up carries a negative delta and raises the elevation.
        val draggedUp = SplatPose.applyDrag(OrbitCamera(), dragDeltaX = 0f, dragDeltaY = -100f)
        assertEquals(0.6f, draggedUp.elevationRadians, 1e-5f)
        assertEquals(0f, draggedUp.azimuthRadians, 0f)

        val draggedDown = SplatPose.applyDrag(OrbitCamera(), dragDeltaX = 0f, dragDeltaY = 100f)
        assertEquals(-0.6f, draggedDown.elevationRadians, 1e-5f)
    }

    @Test
    fun dragAccumulatesAcrossMotionEvents() {
        var camera = OrbitCamera()
        repeat(10) { camera = SplatPose.applyDrag(camera, dragDeltaX = 10f, dragDeltaY = -5f) }
        assertEquals(0.6f, camera.azimuthRadians, 1e-5f)
        assertEquals(0.3f, camera.elevationRadians, 1e-5f)
        assertEquals(1f, camera.dolly, 0f) // a drag never dollies
    }

    @Test
    fun dragLeavesTheDollyUntouchedAndPinchLeavesTheAnglesUntouched() {
        val orbited = SplatPose.applyDrag(OrbitCamera(), 40f, -40f)
        val pinched = SplatPose.applyPinch(orbited, pinchScale = 2f) // fingers spreading pull in
        assertEquals(0.5f, pinched.dolly, 1e-5f)
        assertEquals(orbited.azimuthRadians, pinched.azimuthRadians, 0f)
        assertEquals(orbited.elevationRadians, pinched.elevationRadians, 0f)
        // A pure pan reports pinchScale 1 every event and must not drift the dolly.
        assertEquals(0.5f, SplatPose.applyPinch(pinched, 1f).dolly, 0f)
        // A degenerate scale is ignored rather than dividing by zero.
        assertEquals(0.5f, SplatPose.applyPinch(pinched, 0f).dolly, 0f)
        assertEquals(SplatPose.MIN_DOLLY, SplatPose.applyPinch(pinched, 1000f).dolly, 0f)
    }

    // --- elevation clamping at both poles -------------------------------------------------------

    @Test
    fun elevationClampsShortOfBothPoles() {
        assertEquals(85f, Math.toDegrees(SplatPose.MAX_ELEVATION_RADIANS.toDouble()).toFloat(), 1e-3f)
        assertEquals(SplatPose.MAX_ELEVATION_RADIANS, SplatPose.clampElevation(10f), 0f)
        assertEquals(-SplatPose.MAX_ELEVATION_RADIANS, SplatPose.clampElevation(-10f), 0f)
        assertEquals(0.4f, SplatPose.clampElevation(0.4f), 0f) // inside the band, untouched
    }

    @Test
    fun draggingPastEitherPoleParksTheCameraAtTheClamp() {
        var draggedUp = OrbitCamera()
        repeat(20) { draggedUp = SplatPose.applyDrag(draggedUp, 0f, dragDeltaY = -400f) }
        assertEquals(85f, SplatPose.elevationDegrees(draggedUp), 1e-3f)

        var draggedDown = OrbitCamera()
        repeat(20) { draggedDown = SplatPose.applyDrag(draggedDown, 0f, dragDeltaY = 400f) }
        assertEquals(-85f, SplatPose.elevationDegrees(draggedDown), 1e-3f)

        // The pole is never crossed: at the clamp the camera still sits above (world y below the
        // pivot plane, since the identity camera's up is -y) and the pose stays well conditioned.
        val topPose = SplatPose.orbitPose(identityPose, pivotDepth = 2f, camera = draggedUp)
        val bottomPose = SplatPose.orbitPose(identityPose, pivotDepth = 2f, camera = draggedDown)
        assertTrue(SplatPose.position(topPose)[1] < 0f)
        assertTrue(SplatPose.position(bottomPose)[1] > 0f)
        assertRotationIsOrthonormalAndRightHanded(topPose)
        assertRotationIsOrthonormalAndRightHanded(bottomPose)
    }

    @Test
    fun clampedPullsAnOutOfBoundsCameraBackIntoTheOrbit() {
        val clamped = SplatPose.clamped(OrbitCamera(azimuthRadians = 99f, elevationRadians = 9f, dolly = 99f))
        assertEquals(99f, clamped.azimuthRadians, 0f) // azimuth accumulates without bound
        assertEquals(SplatPose.MAX_ELEVATION_RADIANS, clamped.elevationRadians, 0f)
        assertEquals(SplatPose.MAX_DOLLY, clamped.dolly, 0f)
    }

    // --- elevation moves the camera up and down the orbit sphere --------------------------------

    @Test
    fun elevationRaisesAndLowersTheCameraAlongTheSceneUpAxis() {
        val pivotDepth = 2f
        val pivot = floatArrayOf(0f, 0f, pivotDepth)
        val elevationRadians = (PI / 6).toFloat() // 30 degrees up

        // The identity camera looks down +z with its own y pointing down, so the scene up axis is -y:
        // a positive elevation must drive the eye's world y negative by pivotDepth * sin(elevation).
        val raised = SplatPose.position(
            SplatPose.orbitPose(identityPose, pivotDepth, OrbitCamera(elevationRadians = elevationRadians)),
        )
        assertEquals(0f, raised[0], 1e-4f)
        assertEquals(-pivotDepth * sin(elevationRadians), raised[1], 1e-4f)
        assertEquals(pivotDepth * (1f - cos(elevationRadians)), raised[2], 1e-4f)

        val lowered = SplatPose.position(
            SplatPose.orbitPose(identityPose, pivotDepth, OrbitCamera(elevationRadians = -elevationRadians)),
        )
        assertEquals(+pivotDepth * sin(elevationRadians), lowered[1], 1e-4f)
        assertEquals(-raised[1], lowered[1], 1e-4f) // symmetric about the orbit plane

        // Elevation is a rotation, so it never changes the pivot distance.
        assertEquals(pivotDepth, distanceBetween(raised, pivot), 1e-4f)
        assertEquals(pivotDepth, distanceBetween(lowered, pivot), 1e-4f)
    }

    @Test
    fun elevationClimbsTheSceneUpAxisOfATiltedBasePose() {
        val pivotDepth = 3f
        val baseEye = SplatPose.position(tiltedPose)
        val baseForward = SplatPose.axis(tiltedPose, 2)
        val sceneUp = SplatPose.axis(tiltedPose, 1).map { -it } // up opposes the camera's down axis
        val pivot = FloatArray(3) { baseEye[it] + baseForward[it] * pivotDepth }

        val raised = SplatPose.position(
            SplatPose.orbitPose(tiltedPose, pivotDepth, OrbitCamera(elevationRadians = 0.5f)),
        )
        val climb = (0..2).sumOf { ((raised[it] - baseEye[it]) * sceneUp[it]).toDouble() }.toFloat()
        assertEquals(pivotDepth * sin(0.5f), climb, 1e-3f)
        assertEquals(pivotDepth, distanceBetween(raised, pivot), 1e-3f)

        // The same climb holds after a yaw: elevation pitches about the camera's own horizontal axis.
        val yawedThenRaised = SplatPose.position(
            SplatPose.orbitPose(tiltedPose, pivotDepth, OrbitCamera(azimuthRadians = 1.2f, elevationRadians = 0.5f)),
        )
        val yawedClimb = (0..2).sumOf { ((yawedThenRaised[it] - baseEye[it]) * sceneUp[it]).toDouble() }.toFloat()
        assertEquals(pivotDepth * sin(0.5f), yawedClimb, 1e-3f)
        assertEquals(pivotDepth, distanceBetween(yawedThenRaised, pivot), 1e-3f)
    }

    // --- azimuth + elevation compose into a valid view matrix ------------------------------------

    @Test
    fun azimuthAndElevationComposeIntoAnOrthonormalRightHandedPose() {
        val azimuths = listOf(0f, 0.7f, (PI / 2).toFloat(), 2.9f, -1.3f, (2 * PI).toFloat())
        val elevations = listOf(0f, 0.5f, -0.5f, SplatPose.MAX_ELEVATION_RADIANS, -SplatPose.MAX_ELEVATION_RADIANS)
        val dollies = listOf(SplatPose.MIN_DOLLY, 1f, SplatPose.MAX_DOLLY)
        for (basePose in listOf(identityPose, tiltedPose)) {
            for (azimuthRadians in azimuths) {
                for (elevationRadians in elevations) {
                    for (dolly in dollies) {
                        val camera = OrbitCamera(azimuthRadians, elevationRadians, dolly)
                        val pose = SplatPose.orbitPose(basePose, pivotDepth = 2f, camera = camera)
                        assertRotationIsOrthonormalAndRightHanded(pose)
                    }
                }
            }
        }
    }

    @Test
    fun everyOrbitPoseKeepsLookingAtThePivot() {
        val pivotDepth = 2.5f
        val baseEye = SplatPose.position(tiltedPose)
        val baseForward = SplatPose.axis(tiltedPose, 2)
        val pivot = FloatArray(3) { baseEye[it] + baseForward[it] * pivotDepth }
        val camera = OrbitCamera(azimuthRadians = -2.2f, elevationRadians = 1.1f, dolly = 0.4f)
        val pose = SplatPose.orbitPose(tiltedPose, pivotDepth, camera)
        val eye = SplatPose.position(pose)
        val distance = distanceBetween(eye, pivot)
        assertEquals(pivotDepth * camera.dolly, distance, 1e-3f)
        val forward = SplatPose.axis(pose, 2)
        for (component in 0..2) {
            assertEquals((pivot[component] - eye[component]) / distance, forward[component], 1e-3f)
        }
    }

    // --- one-finger zoom: double tap and the +/- control -----------------------------------------

    @Test
    fun doubleTapDollyStepsStayWithinTheDollyBounds() {
        var camera = OrbitCamera()
        repeat(40) {
            camera = SplatPose.doubleTapZoomStep(camera)
            assertDollyInBounds(camera)
        }
        assertEquals(SplatPose.MIN_DOLLY, camera.dolly, 0f) // saturates in, never past
        // A double tap moves in, so a single step from rest is closer than the base distance.
        assertTrue(SplatPose.doubleTapZoomStep(OrbitCamera()).dolly < 1f)
        // Zooming is orthogonal to the orbit angles.
        val orbited = OrbitCamera(azimuthRadians = 1.1f, elevationRadians = -0.4f)
        val zoomed = SplatPose.doubleTapZoomStep(orbited)
        assertEquals(orbited.azimuthRadians, zoomed.azimuthRadians, 0f)
        assertEquals(orbited.elevationRadians, zoomed.elevationRadians, 0f)
    }

    @Test
    fun zoomControlStepsStayWithinTheDollyBoundsAtBothEnds() {
        var zoomingIn = OrbitCamera()
        repeat(40) {
            zoomingIn = SplatPose.zoomInStep(zoomingIn)
            assertDollyInBounds(zoomingIn)
        }
        assertEquals(SplatPose.MIN_DOLLY, zoomingIn.dolly, 0f)

        var zoomingOut = OrbitCamera()
        repeat(40) {
            zoomingOut = SplatPose.zoomOutStep(zoomingOut)
            assertDollyInBounds(zoomingOut)
        }
        assertEquals(SplatPose.MAX_DOLLY, zoomingOut.dolly, 0f)

        // Backing out of a saturated zoom-in is immediate: the clamp holds no hidden state.
        assertTrue(SplatPose.zoomOutStep(zoomingIn).dolly > SplatPose.MIN_DOLLY)
        assertTrue(SplatPose.zoomInStep(zoomingOut).dolly < SplatPose.MAX_DOLLY)
    }

    @Test
    fun zoomInAndZoomOutStepsRoundTrip() {
        val roundTripped = SplatPose.zoomOutStep(SplatPose.zoomInStep(OrbitCamera()))
        assertEquals(1f, roundTripped.dolly, 1e-5f)
        assertTrue(SplatPose.zoomInStep(OrbitCamera()).dolly < 1f) // `+` pulls the camera in
        assertTrue(SplatPose.zoomOutStep(OrbitCamera()).dolly > 1f) // `-` pushes it out
    }

    @Test
    fun theDollyDrivesThePivotDistanceAtEveryBound() {
        val pivotDepth = 2f
        for (dolly in listOf(SplatPose.MIN_DOLLY, 0.75f, 1f, SplatPose.MAX_DOLLY)) {
            val pose = SplatPose.orbitPose(identityPose, pivotDepth, OrbitCamera(dolly = dolly))
            val distance = distanceBetween(SplatPose.position(pose), floatArrayOf(0f, 0f, pivotDepth))
            assertEquals(pivotDepth * dolly, distance, 1e-4f)
        }
    }

    // --- caption readout --------------------------------------------------------------------------

    @Test
    fun captionAnglesReadTheLiveCameraState() {
        // Azimuth accumulates without bound but reads out wrapped into (-180, 180].
        assertEquals(0f, SplatPose.azimuthDegrees(OrbitCamera()), 1e-3f)
        assertEquals(90f, SplatPose.azimuthDegrees(OrbitCamera(azimuthRadians = (PI / 2).toFloat())), 1e-3f)
        assertEquals(45f, SplatPose.azimuthDegrees(OrbitCamera(azimuthRadians = (2 * PI + PI / 4).toFloat())), 1e-3f)
        assertEquals(-135f, SplatPose.azimuthDegrees(OrbitCamera(azimuthRadians = (5 * PI / 4).toFloat())), 1e-3f)
        assertEquals(135f, SplatPose.azimuthDegrees(OrbitCamera(azimuthRadians = (-5 * PI / 4).toFloat())), 1e-3f)

        // Elevation lives inside the clamp, so it reads out unwrapped.
        assertEquals(30f, SplatPose.elevationDegrees(OrbitCamera(elevationRadians = (PI / 6).toFloat())), 1e-3f)
        assertEquals(-30f, SplatPose.elevationDegrees(OrbitCamera(elevationRadians = (-PI / 6).toFloat())), 1e-3f)
    }

    // --- encoder input plumbing -------------------------------------------------------------------

    @Test
    fun convertsArgbToUnitRangeChw() {
        val argb = intArrayOf(0xFFFF0000.toInt(), 0xFF00FF00.toInt(), 0xFF0000FF.toInt(), 0xFF808080.toInt())
        val chw = rgbToChw01(argb, 2)
        assertEquals(12, chw.size)
        assertEquals(1f, chw[0], 1e-6f)          // R of red
        assertEquals(0f, chw[1], 1e-6f)          // R of green
        assertEquals(1f, chw[4 + 1], 1e-6f)      // G of green
        assertEquals(1f, chw[8 + 2], 1e-6f)      // B of blue
        assertEquals(128f / 255f, chw[3], 1e-6f) // R of gray
    }

    @Test
    fun derivesNormalizedFocalsWithRotationSwap() {
        // 5.4mm lens, 8x6mm sensor, 640x480 frame: fx = 5.4*640/8 = 432px over the 480 square,
        // fy = 5.4*480/6 = 432px over the same square; a 90-degree rotation swaps the (equal) axes.
        val (focalXNorm, focalYNorm) = normalizedFocals(5.4f, 8f, 6f, 640, 480, 90)
        assertEquals(0.9f, focalXNorm, 1e-4f)
        assertEquals(0.9f, focalYNorm, 1e-4f)
        // Missing characteristics fall back to 1.0.
        assertEquals(1f to 1f, normalizedFocals(null, 8f, 6f, 640, 480, 0))
    }

    // --- helpers ------------------------------------------------------------------------------------

    /** R^T R = I over the pose's rotation columns, plus det(R) = +1 (no reflection through a pole). */
    private fun assertRotationIsOrthonormalAndRightHanded(pose: FloatArray) {
        for (columnA in 0..2) {
            for (columnB in 0..2) {
                val dotProduct = dot(SplatPose.axis(pose, columnA), SplatPose.axis(pose, columnB))
                assertEquals(if (columnA == columnB) 1f else 0f, dotProduct, 1e-4f)
            }
        }
        val right = SplatPose.axis(pose, 0)
        val down = SplatPose.axis(pose, 1)
        val forward = SplatPose.axis(pose, 2)
        assertEquals(1f, dot(right, crossProduct(down, forward)), 1e-4f)
    }

    private fun assertDollyInBounds(camera: OrbitCamera) {
        assertTrue(camera.dolly >= SplatPose.MIN_DOLLY)
        assertTrue(camera.dolly <= SplatPose.MAX_DOLLY)
    }

    private fun dot(left: FloatArray, right: FloatArray): Float =
        left[0] * right[0] + left[1] * right[1] + left[2] * right[2]

    private fun crossProduct(left: FloatArray, right: FloatArray): FloatArray = floatArrayOf(
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )

    private fun distanceBetween(from: FloatArray, to: FloatArray): Float = sqrt(
        (from[0] - to[0]) * (from[0] - to[0]) +
            (from[1] - to[1]) * (from[1] - to[1]) +
            (from[2] - to[2]) * (from[2] - to[2]),
    )
}
