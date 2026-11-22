package com.pocketscanner.capture

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.nestedscroll.NestedScrollConnection
import androidx.compose.ui.input.nestedscroll.NestedScrollSource
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.unit.dp
import com.patrykandpatrick.vico.compose.common.component.rememberTextComponent
import com.patrykandpatrick.vico.compose.cartesian.CartesianChartHost
import com.patrykandpatrick.vico.compose.cartesian.axis.rememberBottom
import com.patrykandpatrick.vico.compose.cartesian.axis.rememberStart
import com.patrykandpatrick.vico.compose.cartesian.layer.rememberLine
import com.patrykandpatrick.vico.compose.cartesian.layer.rememberLineCartesianLayer
import com.patrykandpatrick.vico.compose.cartesian.rememberCartesianChart
import com.patrykandpatrick.vico.compose.cartesian.rememberVicoZoomState
import com.patrykandpatrick.vico.core.cartesian.axis.HorizontalAxis
import com.patrykandpatrick.vico.core.cartesian.axis.VerticalAxis
import com.patrykandpatrick.vico.core.cartesian.data.CartesianChartModelProducer
import com.patrykandpatrick.vico.core.cartesian.data.lineSeries
import com.patrykandpatrick.vico.core.cartesian.layer.LineCartesianLayer
import com.patrykandpatrick.vico.core.common.Fill

data class PlotLine(
    val label: String,
    val color: Color,
    val l1: List<Pair<Float, Float>>,
    val ssim: List<Pair<Float, Float>>,
    val psnr: List<Pair<Float, Float>>,
)

private val holdChartScroll = object : NestedScrollConnection {
    override fun onPostScroll(consumed: Offset, available: Offset, source: NestedScrollSource): Offset = available
}

@Composable
internal fun MetricColumn(lines: List<PlotLine>, height: androidx.compose.ui.unit.Dp) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
        if (lines.isNotEmpty()) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                lines.forEach { line ->
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        Box(modifier = Modifier.size(10.dp).background(line.color))
                        Text(line.label, color = MaterialTheme.colorScheme.onSurface, style = MaterialTheme.typography.labelSmall)
                    }
                }
            }
        }
        MetricChart("L1", lines.map { it.color to it.l1 }, Modifier.fillMaxWidth().height(height))
        MetricChart("SSIM", lines.map { it.color to it.ssim }, Modifier.fillMaxWidth().height(height))
        MetricChart("PSNR", lines.map { it.color to it.psnr }, Modifier.fillMaxWidth().height(height))
    }
}

@Composable
private fun MetricChart(title: String, series: List<Pair<Color, List<Pair<Float, Float>>>>, modifier: Modifier) {
    val drawn = series.filter { it.second.size >= 2 }
    if (drawn.isEmpty()) {
        Text("No $title yet", color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = modifier.padding(8.dp))
        return
    }
    val producer = remember(title) { CartesianChartModelProducer() }
    LaunchedEffect(title, drawn) {
        producer.runTransaction {
            lineSeries {
                drawn.forEach { (_, points) ->
                    series(points.map { it.first }, points.map { it.second })
                }
            }
        }
    }
    val lines = drawn.map { (color, _) ->
        LineCartesianLayer.rememberLine(fill = LineCartesianLayer.LineFill.single(Fill(color.toArgb())))
    }
    val axisTitle = rememberTextComponent(color = MaterialTheme.colorScheme.onSurface)
    CartesianChartHost(
        chart = rememberCartesianChart(
            rememberLineCartesianLayer(lineProvider = LineCartesianLayer.LineProvider.series(lines)),
            startAxis = VerticalAxis.rememberStart(titleComponent = axisTitle, title = title),
            bottomAxis = HorizontalAxis.rememberBottom(titleComponent = axisTitle, title = "Epoch"),
        ),
        modelProducer = producer,
        modifier = modifier.clipToBounds().nestedScroll(holdChartScroll),
        zoomState = rememberVicoZoomState(zoomEnabled = true),
    )
}

private fun Color.toArgb(): Int {
    return (alpha * 255).toInt().shl(24) or (red * 255).toInt().shl(16) or (green * 255).toInt().shl(8) or (blue * 255).toInt()
}
