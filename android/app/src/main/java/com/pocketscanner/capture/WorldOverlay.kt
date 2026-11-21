package com.pocketscanner.capture

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

internal data class Vec3(val x: Float, val y: Float, val z: Float) {
    operator fun plus(other: Vec3) = Vec3(x + other.x, y + other.y, z + other.z)
    operator fun minus(other: Vec3) = Vec3(x - other.x, y - other.y, z - other.z)
    operator fun times(scale: Float) = Vec3(x * scale, y * scale, z * scale)

    fun distanceTo(other: Vec3): Float {
        val d = this - other
        return sqrt(d.x * d.x + d.y * d.y + d.z * d.z)
    }
}

internal data class Marker(val origin: Vec3)

internal fun markerInFront(pose: FloatArray): Marker {
    val camera = Vec3(pose[3], pose[7], pose[11])
    val forward = Vec3(pose[2], pose[6], pose[10])
    return Marker(camera + forward * 0.7f)
}

internal fun cameraPosition(pose: FloatArray) = Vec3(pose[3], pose[7], pose[11])

internal data class Look(
    val yaw: Float,
    val pitch: Float,
    val roll: Float,
    val position: Vec3,
)

/** Upright view along the pose's look direction. Roll starts at zero. */
internal fun lookFromPose(pose: FloatArray): Look {
    val forwardY = pose[6].coerceIn(-1f, 1f)
    val pitch = asin(forwardY).coerceIn(-PITCH_LIMIT, PITCH_LIMIT)
    val yaw = atan2(pose[2], pose[10])
    return Look(yaw, pitch, 0f, cameraPosition(pose))
}

/** Camera-to-world pose. Yaw around world up, then pitch, then roll. Pitch stays short of upside down. */
internal fun poseOf(look: Look): FloatArray {
    val rotation = mul3(rotationY(look.yaw), mul3(rotationX(-look.pitch), rotationZ(look.roll)))
    val pose = FloatArray(16)
    for (row in 0 until 3) {
        for (col in 0 until 3) pose[row * 4 + col] = rotation[row * 3 + col]
    }
    pose[3] = look.position.x
    pose[7] = look.position.y
    pose[11] = look.position.z
    pose[15] = 1f
    return pose
}

internal fun fly(look: Look, forward: Float, strafe: Float): Look {
    val pose = poseOf(look)
    val right = Vec3(pose[0], pose[4], pose[8])
    val ahead = Vec3(pose[2], pose[6], pose[10])
    return look.copy(position = look.position + ahead * forward + right * strafe)
}

internal fun lookAround(look: Look, yawDelta: Float, pitchDelta: Float): Look {
    return look.copy(
        yaw = look.yaw + yawDelta,
        pitch = (look.pitch + pitchDelta).coerceIn(-PITCH_LIMIT, PITCH_LIMIT),
    )
}

internal fun rollBy(look: Look, delta: Float) = look.copy(roll = look.roll + delta)

private const val PITCH_LIMIT = 1.2f

private fun rotationX(angle: Float): FloatArray {
    val c = cos(angle)
    val s = sin(angle)
    return floatArrayOf(
        1f, 0f, 0f,
        0f, c, -s,
        0f, s, c,
    )
}

private fun rotationY(angle: Float): FloatArray {
    val c = cos(angle)
    val s = sin(angle)
    return floatArrayOf(
        c, 0f, s,
        0f, 1f, 0f,
        -s, 0f, c,
    )
}

private fun rotationZ(angle: Float): FloatArray {
    val c = cos(angle)
    val s = sin(angle)
    return floatArrayOf(
        c, -s, 0f,
        s, c, 0f,
        0f, 0f, 1f,
    )
}

private fun mul3(left: FloatArray, right: FloatArray): FloatArray {
    val out = FloatArray(9)
    for (row in 0 until 3) {
        for (col in 0 until 3) {
            var sum = 0f
            for (k in 0 until 3) sum += left[row * 3 + k] * right[k * 3 + col]
            out[row * 3 + col] = sum
        }
    }
    return out
}

@Composable
internal fun WorldOverlay(
    pose: FloatArray?,
    fx: Float,
    fy: Float,
    cx: Float,
    cy: Float,
    imageWidth: Int,
    imageHeight: Int,
    marker: Marker?,
    trail: List<Vec3>,
    modifier: Modifier = Modifier,
) {
    Canvas(modifier) {
        val map = 96.dp.toPx()
        val pad = 12.dp.toPx()
        if (trail.size >= 2) {
            drawPathMap(trail, Offset(size.width - map - pad, pad), map)
        }
        if (pose == null || marker == null || fx <= 0f || imageWidth <= 0 || imageHeight <= 0) return@Canvas
        val camera = invertRigid(pose)
        val origin = marker.origin
        val arm = 0.2f
        drawAxis(camera, origin, origin + Vec3(arm, 0f, 0f), Color(0xFFE53935), fx, fy, cx, cy, imageWidth, imageHeight)
        drawAxis(camera, origin, origin + Vec3(0f, arm, 0f), Color(0xFF43A047), fx, fy, cx, cy, imageWidth, imageHeight)
        drawAxis(camera, origin, origin + Vec3(0f, 0f, arm), Color(0xFF1E88E5), fx, fy, cx, cy, imageWidth, imageHeight)
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawAxis(
    camera: FloatArray,
    from: Vec3,
    to: Vec3,
    color: Color,
    fx: Float,
    fy: Float,
    cx: Float,
    cy: Float,
    imageWidth: Int,
    imageHeight: Int,
) {
    val a = project(camera, from, fx, fy, cx, cy) ?: return
    val b = project(camera, to, fx, fy, cx, cy) ?: return
    val start = imageToView(a, size.width, size.height, imageWidth.toFloat(), imageHeight.toFloat())
    val end = imageToView(b, size.width, size.height, imageWidth.toFloat(), imageHeight.toFloat())
    drawLine(color, start, end, strokeWidth = 4.dp.toPx())
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawPathMap(
    trail: List<Vec3>,
    topLeft: Offset,
    side: Float,
) {
    drawRoundRect(
        color = Color.Black.copy(alpha = 0.55f),
        topLeft = topLeft,
        size = Size(side, side),
        cornerRadius = androidx.compose.ui.geometry.CornerRadius(12.dp.toPx()),
    )
    var minX = trail[0].x
    var maxX = trail[0].x
    var minZ = trail[0].z
    var maxZ = trail[0].z
    for (point in trail) {
        if (point.x < minX) minX = point.x
        if (point.x > maxX) maxX = point.x
        if (point.z < minZ) minZ = point.z
        if (point.z > maxZ) maxZ = point.z
    }
    val span = maxOf(maxX - minX, maxZ - minZ, 0.4f)
    val inset = 10.dp.toPx()
    fun map(point: Vec3): Offset {
        val nx = (point.x - minX) / span
        val nz = (point.z - minZ) / span
        return Offset(
            topLeft.x + inset + nx * (side - inset * 2f),
            topLeft.y + side - inset - nz * (side - inset * 2f),
        )
    }
    for (index in 1 until trail.size) {
        drawLine(Color.White.copy(alpha = 0.85f), map(trail[index - 1]), map(trail[index]), strokeWidth = 2.dp.toPx())
    }
    val head = map(trail.last())
    drawCircle(Color(0xFF8AB4F8), radius = 4.dp.toPx(), center = head)
    drawCircle(Color.White, radius = 4.dp.toPx(), center = head, style = Stroke(width = 1.dp.toPx()))
}

private fun invertRigid(worldFromCamera: FloatArray): FloatArray {
    val out = FloatArray(16)
    for (row in 0..2) {
        for (col in 0..2) {
            out[row * 4 + col] = worldFromCamera[col * 4 + row]
        }
    }
    val tx = worldFromCamera[3]
    val ty = worldFromCamera[7]
    val tz = worldFromCamera[11]
    out[3] = -(out[0] * tx + out[1] * ty + out[2] * tz)
    out[7] = -(out[4] * tx + out[5] * ty + out[6] * tz)
    out[11] = -(out[8] * tx + out[9] * ty + out[10] * tz)
    out[15] = 1f
    return out
}

private fun project(cameraFromWorld: FloatArray, point: Vec3, fx: Float, fy: Float, cx: Float, cy: Float): Offset? {
    val x = cameraFromWorld[0] * point.x + cameraFromWorld[1] * point.y + cameraFromWorld[2] * point.z + cameraFromWorld[3]
    val y = cameraFromWorld[4] * point.x + cameraFromWorld[5] * point.y + cameraFromWorld[6] * point.z + cameraFromWorld[7]
    val z = cameraFromWorld[8] * point.x + cameraFromWorld[9] * point.y + cameraFromWorld[10] * point.z + cameraFromWorld[11]
    if (z <= 0.05f) return null
    return Offset(fx * x / z + cx, fy * y / z + cy)
}

private fun imageToView(pixel: Offset, viewWidth: Float, viewHeight: Float, imageWidth: Float, imageHeight: Float): Offset {
    val viewAspect = viewWidth / viewHeight
    val frameAspect = imageWidth / imageHeight
    val contentWidth: Float
    val contentHeight: Float
    val offsetX: Float
    val offsetY: Float
    if (viewAspect > frameAspect) {
        contentHeight = viewHeight
        contentWidth = viewHeight * frameAspect
        offsetX = (viewWidth - contentWidth) / 2f
        offsetY = 0f
    } else {
        contentWidth = viewWidth
        contentHeight = viewWidth / frameAspect
        offsetX = 0f
        offsetY = (viewHeight - contentHeight) / 2f
    }
    return Offset(
        offsetX + pixel.x / imageWidth * contentWidth,
        offsetY + pixel.y / imageHeight * contentHeight,
    )
}
