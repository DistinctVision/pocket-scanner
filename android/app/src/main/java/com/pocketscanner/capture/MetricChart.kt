package com.pocketscanner.capture

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.unit.dp
data class ChartSeries(
    val color: Color,
    val dashed: Boolean,
    val points: List<Pair<Float, Float>>,
)

data class PlotLine(
    val color: Color,
    val dashed: Boolean,
    val l1: List<Pair<Float, Float>>,
    val ssim: List<Pair<Float, Float>>,
    val psnr: List<Pair<Float, Float>>,
)

@Composable
internal fun MetricChart(
    title: String,
    series: List<ChartSeries>,
    modifier: Modifier = Modifier,
    onFullscreen: (() -> Unit)? = null,
) {
    var scaleX by remember(title) { mutableFloatStateOf(1f) }
    var scaleY by remember(title) { mutableFloatStateOf(1f) }
    var panX by remember(title) { mutableFloatStateOf(0f) }
    var panY by remember(title) { mutableFloatStateOf(0f) }
    val scheme = MaterialTheme.colorScheme
    Box(modifier) {
        Text(
            title,
            color = scheme.onSurface,
            style = MaterialTheme.typography.labelLarge,
            modifier = Modifier.padding(start = 8.dp, top = 4.dp),
        )
        if (onFullscreen != null) {
            TextButton(onClick = onFullscreen, modifier = Modifier.align(Alignment.TopEnd)) { Text("Full") }
        }
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = 28.dp)
                .pointerInput(title) {
                    awaitEachGesture {
                        val down = awaitFirstDown()
                        val left = down.position.x < 48.dp.toPx()
                        val bottom = down.position.y > size.height - 48.dp.toPx()
                        var zoomed = false
                        do {
                            val event = awaitPointerEvent()
                            val zoom = event.calculateZoom()
                            if (zoom != 1f) {
                                zoomed = true
                                scaleX = (scaleX * zoom).coerceIn(0.2f, 40f)
                                scaleY = (scaleY * zoom).coerceIn(0.2f, 40f)
                            }
                            val pan = event.calculatePan()
                            if (!zoomed && event.changes.any { it.pressed }) {
                                when {
                                    left -> scaleY = (scaleY * (1f + pan.y / 400f)).coerceIn(0.2f, 40f)
                                    bottom -> scaleX = (scaleX * (1f - pan.x / 400f)).coerceIn(0.2f, 40f)
                                    else -> {
                                        panX += pan.x
                                        panY += pan.y
                                    }
                                }
                            }
                            event.changes.forEach { if (it.positionChanged()) it.consume() }
                        } while (event.changes.any { it.pressed })
                    }
                },
        ) {
            val points = series.flatMap { it.points }
            if (points.isEmpty()) return@Canvas
            val minX = points.minOf { it.first }
            val maxX = points.maxOf { it.first }.let { if (it == minX) it + 1f else it }
            val minY = points.minOf { it.second }
            val maxY = points.maxOf { it.second }.let { if (it == minY) it + 1f else it }
            val leftPad = 8f
            val plotW = size.width - leftPad
            val plotH = size.height - 8f
            fun map(x: Float, y: Float): Offset {
                val nx = (x - minX) / (maxX - minX)
                val ny = (y - minY) / (maxY - minY)
                val px = leftPad + (nx * plotW * scaleX) + panX
                val py = plotH - (ny * plotH * scaleY) + panY
                return Offset(px, py)
            }
            drawRect(scheme.surface.copy(alpha = 0.35f))
            val dash = PathEffect.dashPathEffect(floatArrayOf(12f, 8f))
            for (item in series) {
                if (item.points.size < 2) {
                    item.points.forEach { point ->
                        drawCircle(item.color, 4f, map(point.first, point.second))
                    }
                    continue
                }
                val path = Path()
                item.points.forEachIndexed { index, point ->
                    val at = map(point.first, point.second)
                    if (index == 0) path.moveTo(at.x, at.y) else path.lineTo(at.x, at.y)
                }
                drawPath(
                    path,
                    item.color,
                    style = Stroke(width = 3f, pathEffect = if (item.dashed) dash else null),
                )
            }
        }
    }
}

@Composable
internal fun MetricColumn(
    lines: List<PlotLine>,
    height: androidx.compose.ui.unit.Dp,
    fullscreen: String?,
    onFullscreen: (String) -> Unit,
) {
    listOf("L1", "SSIM", "PSNR").forEach { name ->
        if (fullscreen != null && fullscreen != name) return@forEach
        val picked = lines.map { line ->
            val values = when (name) {
                "SSIM" -> line.ssim
                "PSNR" -> line.psnr
                else -> line.l1
            }
            ChartSeries(line.color, line.dashed, values)
        }
        MetricChart(
            title = name,
            series = picked,
            modifier = if (fullscreen == name) Modifier.fillMaxSize() else Modifier.fillMaxWidth().height(height),
            onFullscreen = { onFullscreen(name) },
        )
    }
}
