package com.pocketscanner.capture

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.addCallback
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import java.io.File

class TrainActivity : ComponentActivity() {
    private var retained = false
    private val notifications = registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        onBackPressedDispatcher.addCallback(this) {
            if (!SplatRuntime.ui.running) leave()
        }
        CaptureRuntime.retain(this)
        retained = true
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        SplatRuntime.start(this)
        setContent {
            val ui = SplatRuntime.ui
            LaunchedEffect(ui.finished) {
                if (ui.finished) leave()
            }
            MaterialTheme(colorScheme = Scheme) {
                TrainScreen(
                    ui = ui,
                    onStop = { SplatRuntime.requestStop() },
                    onBack = { leave() },
                    onMetrics = { startActivity(Intent(this, MetricsActivity::class.java)) },
                )
            }
        }
    }

    private fun leave() {
        startActivity(
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP),
        )
        finish()
    }

    override fun onDestroy() {
        if (retained) {
            retained = false
            CaptureRuntime.release()
        }
        super.onDestroy()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TrainScreen(ui: TrainUi, onStop: () -> Unit, onBack: () -> Unit, onMetrics: () -> Unit) {
    val scheme = MaterialTheme.colorScheme
    var compare by remember { mutableStateOf<File?>(null) }
    var open by remember { mutableStateOf(false) }
    val context = LocalContext.current
    val others = remember(ui.runDir) {
        SplatRuntime.runs(context).filter { it.absolutePath != ui.runDir }
    }
    val compared = compare?.let { SplatRuntime.readMetrics(it) }
    val lines = buildList {
        add(plot("Train", Color(0xFF8AB4F8), ui.train))
        if (ui.validation.isNotEmpty()) add(plot("Val", Color(0xFFF28B82), ui.validation))
        if (compared != null && compared.first.isNotEmpty()) add(plot("Compare train", Color(0xFFF6C945), compared.first))
        if (compared != null && compared.second.isNotEmpty()) add(plot("Compare val", Color(0xFF81C995), compared.second))
    }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .belowSystemBars()
            .padding(12.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(ui.status, color = scheme.onSurface, style = MaterialTheme.typography.titleMedium)
        LinearProgressIndicator(progress = { ui.fraction.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
        ExposedDropdownMenuBox(expanded = open, onExpandedChange = { open = it }) {
            OutlinedTextField(
                value = compare?.name ?: "Compare with a finished run",
                onValueChange = {},
                readOnly = true,
                modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable).fillMaxWidth(),
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = open) },
            )
            ExposedDropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                DropdownMenuItem(
                    text = { Text("None") },
                    onClick = {
                        compare = null
                        open = false
                    },
                )
                others.forEach { dir ->
                    DropdownMenuItem(
                        text = { Text(dir.name) },
                        onClick = {
                            compare = dir
                            open = false
                        },
                    )
                }
            }
        }
        MetricColumn(lines = lines, height = 160.dp)
        ui.image?.let { image ->
            Image(bitmap = image.asImageBitmap(), contentDescription = "Validation", modifier = Modifier.fillMaxWidth())
        }
        ui.log.forEach { line ->
            Text(line, color = scheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            if (ui.running) {
                Button(onClick = onStop, modifier = Modifier.weight(1f)) { Text("Stop") }
            } else {
                Button(onClick = onBack, modifier = Modifier.weight(1f)) { Text("Back") }
            }
            TextButton(onClick = onMetrics, modifier = Modifier.weight(1f)) { Text("Metrics") }
        }
    }
}

private fun plot(label: String, color: Color, metrics: List<TrainMetric>): PlotLine {
    return PlotLine(
        label,
        color,
        metrics.map { it.epoch.toFloat() to it.l1 },
        metrics.map { it.epoch.toFloat() to it.ssim },
        metrics.map { it.epoch.toFloat() to it.psnr },
        metrics.map { it.epoch.toFloat() to it.gaussians.toFloat() },
    )
}
