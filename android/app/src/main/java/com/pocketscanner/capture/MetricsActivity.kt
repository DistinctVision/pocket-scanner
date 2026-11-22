package com.pocketscanner.capture

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.addCallback
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Checkbox
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import java.io.File

class MetricsActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        onBackPressedDispatcher.addCallback(this) { finish() }
        setContent {
            MaterialTheme(colorScheme = Scheme) {
                MetricsScreen(onClose = { finish() })
            }
        }
    }
}

@Composable
private fun MetricsScreen(onClose: () -> Unit) {
    val context = LocalContext.current
    val runs = remember { SplatRuntime.runs(context) }
    var selected by remember { mutableStateOf(setOf<String>()) }
    var fullscreen by remember { mutableStateOf<String?>(null) }
    val live = SplatRuntime.ui
    val colors = listOf(
        Color(0xFF8AB4F8),
        Color(0xFFF6C945),
        Color(0xFF81C995),
        Color(0xFFF28B82),
        Color(0xFFC58AF9),
    )
    val lines = ArrayList<PlotLine>()
    if (live.running || live.train.isNotEmpty()) {
        lines.add(plotLine(colors[0], false, live.train))
        if (live.validation.isNotEmpty()) lines.add(plotLine(colors[0], true, live.validation))
    }
    var colorIndex = 1
    selected.forEach { path ->
        val pair = SplatRuntime.readMetrics(File(path))
        val color = colors[colorIndex % colors.size]
        colorIndex++
        if (pair.first.isNotEmpty()) lines.add(plotLine(color, false, pair.first))
        if (pair.second.isNotEmpty()) lines.add(plotLine(color, true, pair.second))
    }
    val scheme = MaterialTheme.colorScheme
    Column(
        modifier = Modifier
            .fillMaxSize()
            .belowSystemBars()
            .padding(12.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(
                if (live.running) "Epoch ${live.epoch} / ${live.epochs}" else "Metrics",
                color = scheme.onSurface,
                style = MaterialTheme.typography.titleMedium,
            )
            TextButton(onClick = { if (fullscreen != null) fullscreen = null else onClose() }) {
                Text(if (fullscreen != null) "Back" else "Close")
            }
        }
        if (live.running) {
            LinearProgressIndicator(progress = { live.fraction.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
        }
        if (fullscreen == null) {
            runs.forEach { dir ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(
                        checked = dir.absolutePath in selected,
                        onCheckedChange = { checked ->
                            selected = if (checked) selected + dir.absolutePath else selected - dir.absolutePath
                        },
                    )
                    Text(dir.name, color = scheme.onSurface)
                }
            }
        }
        MetricColumn(
            lines = lines,
            height = 180.dp,
            fullscreen = fullscreen,
            onFullscreen = { name -> fullscreen = if (fullscreen == name) null else name },
        )
        live.image?.let { image ->
            Image(bitmap = image.asImageBitmap(), contentDescription = "Validation", modifier = Modifier.fillMaxWidth())
        }
        if (live.running) {
            live.log.takeLast(6).forEach { line ->
                Text(line, color = scheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

private fun plotLine(color: Color, dashed: Boolean, metrics: List<TrainMetric>): PlotLine {
    return PlotLine(
        color,
        dashed,
        metrics.map { it.epoch.toFloat() to it.l1 },
        metrics.map { it.epoch.toFloat() to it.ssim },
        metrics.map { it.epoch.toFloat() to it.psnr },
    )
}
