package com.pocketscanner.capture

import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.VideoView
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import org.json.JSONObject
import java.io.File
import java.util.Locale

class MetricsActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
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
    var revision by remember { mutableIntStateOf(0) }
    val runs = remember(revision) { SplatRuntime.runs(context) }
    var selected by remember { mutableStateOf<File?>(null) }
    var pendingDelete by remember { mutableStateOf<File?>(null) }
    val live = SplatRuntime.ui
    BackHandler {
        if (selected != null) selected = null else onClose()
    }
    pendingDelete?.let { dir ->
        AlertDialog(
            onDismissRequest = { pendingDelete = null },
            title = { Text("Delete ${dir.name}?") },
            text = { Text("The metrics, validation pictures, and video in this run are removed.") },
            confirmButton = {
                TextButton(onClick = {
                    dir.deleteRecursively()
                    if (selected == dir) selected = null
                    pendingDelete = null
                    revision++
                }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { pendingDelete = null }) { Text("Cancel") } },
        )
    }
    val open = selected
    if (open != null && open.isDirectory) {
        RunDetail(open, live, onBack = { selected = null })
    } else {
        RunList(runs, live, onOpen = { selected = it }, onDelete = { pendingDelete = it }, onClose = onClose)
    }
}

@Composable
private fun RunList(
    runs: List<File>,
    live: TrainUi,
    onOpen: (File) -> Unit,
    onDelete: (File) -> Unit,
    onClose: () -> Unit,
) {
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
            Text("Runs", color = scheme.onSurface, style = MaterialTheme.typography.titleMedium)
            TextButton(onClick = onClose) { Text("Close") }
        }
        if (live.running) {
            Text(live.status, color = scheme.onSurface)
            LinearProgressIndicator(progress = { live.fraction.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
        }
        if (runs.isEmpty()) {
            Text("No runs yet", color = scheme.onSurfaceVariant)
        }
        runs.forEach { dir ->
            Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                Text(dir.name, color = scheme.onSurface, style = MaterialTheme.typography.titleSmall)
                Text(runLine(dir), color = scheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
                if (live.running && live.runDir == dir.absolutePath) {
                    Text(live.status, color = scheme.onSurface, style = MaterialTheme.typography.bodySmall)
                }
                Row {
                    TextButton(onClick = { onOpen(dir) }) { Text("Open") }
                    TextButton(onClick = { onDelete(dir) }) { Text("Delete") }
                }
            }
        }
    }
}

@Composable
private fun RunDetail(dir: File, live: TrainUi, onBack: () -> Unit) {
    val scheme = MaterialTheme.colorScheme
    val pair = remember(dir, live.epoch, live.fraction) { SplatRuntime.readMetrics(dir) }
    val lines = buildList {
        if (pair.first.isNotEmpty()) add(series("Train", Color(0xFF8AB4F8), pair.first))
        if (pair.second.isNotEmpty()) add(series("Val", Color(0xFFF28B82), pair.second))
    }
    val pictures = remember(dir, live.epoch) {
        File(dir, "val").listFiles { file -> file.isFile && file.name.endsWith(".png") }?.sortedBy { it.name } ?: emptyList()
    }
    val video = File(dir, "video.mp4")
    Column(
        modifier = Modifier
            .fillMaxSize()
            .belowSystemBars()
            .padding(12.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(dir.name, color = scheme.onSurface, style = MaterialTheme.typography.titleMedium)
            TextButton(onClick = onBack) { Text("Back") }
        }
        Text(runLine(dir), color = scheme.onSurfaceVariant)
        if (live.running && live.runDir == dir.absolutePath) {
            Text(live.status, color = scheme.onSurface)
            LinearProgressIndicator(progress = { live.fraction.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
        }
        MetricColumn(lines = lines, height = 180.dp)
        Text("Validation renders", color = scheme.onSurface, style = MaterialTheme.typography.titleSmall)
        if (pictures.isEmpty()) {
            Text("No validation pictures yet", color = scheme.onSurfaceVariant)
        } else {
            Row(modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                pictures.forEach { file ->
                    val bitmap = remember(file.absolutePath, file.lastModified()) { BitmapFactory.decodeFile(file.absolutePath) }
                    if (bitmap != null) {
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = file.nameWithoutExtension,
                            modifier = Modifier.width(160.dp).height(120.dp),
                            contentScale = ContentScale.Fit,
                        )
                    }
                }
            }
        }
        Text("Video", color = scheme.onSurface, style = MaterialTheme.typography.titleSmall)
        if (!video.isFile || video.length() < 32) {
            Text(
                "No video yet. It is written after a validation pass, once the scan has at least two frames.",
                color = scheme.onSurfaceVariant,
            )
        } else {
            AndroidView(
                modifier = Modifier.fillMaxWidth().height(220.dp),
                factory = { viewContext ->
                    VideoView(viewContext).apply {
                        setVideoURI(Uri.fromFile(video))
                        setOnPreparedListener { player ->
                            player.isLooping = true
                            start()
                        }
                    }
                },
                onRelease = { it.stopPlayback() },
            )
        }
    }
}

private fun series(label: String, color: Color, metrics: List<TrainMetric>): PlotLine {
    return PlotLine(
        label,
        color,
        metrics.map { it.epoch.toFloat() to it.l1 },
        metrics.map { it.epoch.toFloat() to it.ssim },
        metrics.map { it.epoch.toFloat() to it.psnr },
    )
}

private fun runLine(dir: File): String {
    val header = runCatching { JSONObject(File(dir, "run.json").readText()) }.getOrNull()
    val planned = header?.optInt("epochs", 0) ?: 0
    val validation = header?.optBoolean("validation", false) ?: false
    val (train, held) = SplatRuntime.readMetrics(dir)
    val done = train.maxOfOrNull { it.epoch } ?: 0
    val trainPsnr = train.lastOrNull()?.let { "train %.2f".format(Locale.US, it.psnr) } ?: "train —"
    val valPsnr = if (!validation) "validation off" else held.lastOrNull()?.let { "val %.2f".format(Locale.US, it.psnr) } ?: "val —"
    val epochs = if (planned > 0) "$done / $planned epochs" else "$done epochs"
    return "$epochs · $trainPsnr · $valPsnr"
}
