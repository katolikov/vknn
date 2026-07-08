package com.vknn.chat.splat

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Orbit-viewer camera state, held as offsets from the encoder's view-0 pose: [azimuthRadians] yaws
 * around the scene's up axis, [elevationRadians] pitches the camera up (positive) or down (negative)
 * the orbit sphere, and [dolly] scales the pivot distance (below 1 sits closer to the pivot).
 * Azimuth accumulates without bound; elevation and dolly are clamped by [SplatPose].
 */
data class OrbitCamera(
    val azimuthRadians: Float = 0f,
    val elevationRadians: Float = 0f,
    val dolly: Float = 1f,
)

// Orbit-camera math over the encoder's predicted poses. Matrices are row-major 4x4 camera-to-world
// in the OpenCV convention (camera x right, y down, z forward): worldPoint = R * cameraPoint + t
// with R at [row*4+col] for rows/cols 0..2 and t at indices 3, 7, 11. Pure functions, JVM-tested:
// every gesture-to-camera mapping the viewer performs lives here, so the UI holds no pose math.
object SplatPose {

    /**
     * Elevation stops short of the poles. At exactly +/-90 degrees the camera forward aligns with
     * the scene up axis and the roll reference in [lookAt] degenerates; 85 degrees leaves the
     * reference cross product at 8.7% of unit length, far above float noise, and still shows the
     * scene from very nearly overhead.
     */
    const val MAX_ELEVATION_DEGREES = 85f

    /** Dolly bounds: a third of the pivot distance in, three times it out. */
    const val MIN_DOLLY = 0.3f
    const val MAX_DOLLY = 3f

    val MAX_ELEVATION_RADIANS = (MAX_ELEVATION_DEGREES * PI / 180.0).toFloat()

    // One finger drags the camera across the orbit sphere and the eye follows the finger: drag right
    // and the camera swings right, drag up and it climbs, so the scene sweeps the opposite way. Both
    // axes share the sensitivity, ~0.34 degrees per pixel, so a 500 px thumb sweep is a ~172-degree
    // turn -- the whole useful orbit in one comfortable stroke.
    private const val ORBIT_RADIANS_PER_PIXEL = 0.006f

    // Discrete one-finger dolly steps. The double tap is coarse enough to read as a jump; the zoom
    // control is fine enough to trim. Both clamp into [MIN_DOLLY, MAX_DOLLY].
    private const val DOUBLE_TAP_DOLLY_FACTOR = 0.6f
    private const val ZOOM_CONTROL_DOLLY_FACTOR = 0.8f

    fun position(pose: FloatArray): FloatArray = floatArrayOf(pose[3], pose[7], pose[11])

    /** Column `axisIndex` of the rotation: the camera's x/y/z axis expressed in world space. */
    fun axis(pose: FloatArray, axisIndex: Int): FloatArray =
        floatArrayOf(pose[axisIndex], pose[4 + axisIndex], pose[8 + axisIndex])

    /**
     * One-finger drag: the horizontal delta yaws the camera, the vertical delta raises or lowers it.
     * Screen y grows downward, so dragging up ([dragDeltaY] negative) raises the elevation.
     */
    fun applyDrag(camera: OrbitCamera, dragDeltaX: Float, dragDeltaY: Float): OrbitCamera = camera.copy(
        azimuthRadians = camera.azimuthRadians + dragDeltaX * ORBIT_RADIANS_PER_PIXEL,
        elevationRadians = clampElevation(camera.elevationRadians - dragDeltaY * ORBIT_RADIANS_PER_PIXEL),
    )

    /** Pinch dolly: [pinchScale] above 1 (fingers spreading) pulls the camera toward the pivot. */
    fun applyPinch(camera: OrbitCamera, pinchScale: Float): OrbitCamera =
        if (pinchScale <= 0f) camera else camera.copy(dolly = clampDolly(camera.dolly / pinchScale))

    /** One step of the zoom control's `+`: a fifth of the pivot distance closer. */
    fun zoomInStep(camera: OrbitCamera): OrbitCamera =
        camera.copy(dolly = clampDolly(camera.dolly * ZOOM_CONTROL_DOLLY_FACTOR))

    /** One step of the zoom control's `-`: the inverse of [zoomInStep], so the pair round-trips. */
    fun zoomOutStep(camera: OrbitCamera): OrbitCamera =
        camera.copy(dolly = clampDolly(camera.dolly / ZOOM_CONTROL_DOLLY_FACTOR))

    /** Double tap: a coarse dolly in. Pinch, or the zoom control's `-`, backs out again. */
    fun doubleTapZoomStep(camera: OrbitCamera): OrbitCamera =
        camera.copy(dolly = clampDolly(camera.dolly * DOUBLE_TAP_DOLLY_FACTOR))

    fun clampElevation(elevationRadians: Float): Float =
        elevationRadians.coerceIn(-MAX_ELEVATION_RADIANS, MAX_ELEVATION_RADIANS)

    fun clampDolly(dolly: Float): Float = dolly.coerceIn(MIN_DOLLY, MAX_DOLLY)

    /** Pulls an externally supplied camera back inside the orbit bounds. */
    fun clamped(camera: OrbitCamera): OrbitCamera = camera.copy(
        elevationRadians = clampElevation(camera.elevationRadians),
        dolly = clampDolly(camera.dolly),
    )

    /** Azimuth wrapped into (-180, 180] for display; the stored angle accumulates without bound. */
    fun azimuthDegrees(camera: OrbitCamera): Float {
        val wrappedDegrees = Math.toDegrees(camera.azimuthRadians.toDouble()).mod(360.0)
        return (if (wrappedDegrees > 180.0) wrappedDegrees - 360.0 else wrappedDegrees).toFloat()
    }

    fun elevationDegrees(camera: OrbitCamera): Float =
        Math.toDegrees(camera.elevationRadians.toDouble()).toFloat()

    /**
     * Camera-to-world for one orbit step: from [basePose], place the pivot at [pivotDepth] along the
     * base forward, yaw the camera position about the scene up axis, pitch it up the orbit sphere by
     * the elevation, scale the pivot distance by the dolly, and look back at the pivot with the base
     * camera's down as the roll reference.
     */
    fun orbitPose(basePose: FloatArray, pivotDepth: Float, camera: OrbitCamera): FloatArray {
        val baseEye = position(basePose)
        val baseForward = axis(basePose, 2)
        val baseDown = axis(basePose, 1)
        val pivot = FloatArray(3) { baseEye[it] + baseForward[it] * pivotDepth }
        val worldUp = floatArrayOf(-baseDown[0], -baseDown[1], -baseDown[2])
        val baseOffset = floatArrayOf(baseEye[0] - pivot[0], baseEye[1] - pivot[1], baseEye[2] - pivot[2])
        val yawedOffset = rotateAboutAxis(baseOffset, worldUp, camera.azimuthRadians)
        // baseOffset opposes the base forward, hence is perpendicular to worldUp, and the yaw about
        // worldUp keeps it there. A positive rotation about (offset x worldUp) therefore tilts the
        // offset straight toward worldUp -- the camera climbs the orbit sphere. Taking the axis after
        // the yaw pitches about the camera's own horizontal axis at every azimuth.
        val elevationAxis = cross(yawedOffset, worldUp)
        val pitchedOffset = rotateAboutAxis(yawedOffset, elevationAxis, camera.elevationRadians)
        val eye = FloatArray(3) { pivot[it] + pitchedOffset[it] * camera.dolly }
        // baseDown as the roll reference keeps the horizon level: the camera's right axis stays
        // perpendicular to the scene up axis at every elevation, so the orbit never rolls.
        return lookAt(eye, pivot, baseDown)
    }

    /** Rodrigues rotation of [vector] by [angleRadians] about [axis] (normalized internally). */
    fun rotateAboutAxis(vector: FloatArray, axis: FloatArray, angleRadians: Float): FloatArray {
        val axisLength = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        if (axisLength == 0f) return vector.copyOf()
        val unit = FloatArray(3) { axis[it] / axisLength }
        val cosAngle = cos(angleRadians)
        val sinAngle = sin(angleRadians)
        val crossProduct = cross(unit, vector)
        val dotProduct = unit[0] * vector[0] + unit[1] * vector[1] + unit[2] * vector[2]
        return FloatArray(3) {
            vector[it] * cosAngle + crossProduct[it] * sinAngle + unit[it] * dotProduct * (1f - cosAngle)
        }
    }

    /**
     * Row-major c2w whose camera sits at [eye] looking at [target], with [downReference] steering
     * the roll (camera y aligns with it as closely as orthonormality allows).
     */
    fun lookAt(eye: FloatArray, target: FloatArray, downReference: FloatArray): FloatArray {
        val forward = normalize(floatArrayOf(target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
        val right = normalize(cross(downReference, forward))
        val down = cross(forward, right)
        return floatArrayOf(
            right[0], down[0], forward[0], eye[0],
            right[1], down[1], forward[1], eye[1],
            right[2], down[2], forward[2], eye[2],
            0f, 0f, 0f, 1f,
        )
    }

    private fun cross(a: FloatArray, b: FloatArray): FloatArray = floatArrayOf(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )

    private fun normalize(vector: FloatArray): FloatArray {
        val length = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
        return if (length == 0f) vector else FloatArray(3) { vector[it] / length }
    }
}
