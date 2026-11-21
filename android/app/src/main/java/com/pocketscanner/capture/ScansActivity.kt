package com.pocketscanner.capture

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.addCallback
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.IconButton
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ScansActivity : ComponentActivity() {
    private var retained = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        onBackPressedDispatcher.addCallback(this) { finishScans() }
        val session = CaptureRuntime.retain(this)
        retained = true
        val fromRefine = intent.getBooleanExtra(EXTRA_FROM_REFINE, false)
        setContent {
            val ui = CaptureRuntime.ui
            MaterialTheme(colorScheme = Scheme) {
                ScansScreen(
                    scanning = ui.phase == "scanning",
                    onClose = { finishScans() },
                    onOpen = { uri, done ->
                        session.loadPackage(uri) { loaded ->
                            if (loaded) {
                                if (!fromRefine) openRefine()
                                finishScans()
                            } else {
                                done()
                            }
                        }
                    },
                )
            }
        }
    }

    override fun onDestroy() {
        if (retained) {
            retained = false
            CaptureRuntime.release()
        }
        super.onDestroy()
    }

    companion object {
        const val EXTRA_FROM_REFINE = "fromRefine"
    }
}

@Composable
private fun ScansScreen(
    scanning: Boolean,
    onClose: () -> Unit,
    onOpen: (Uri, () -> Unit) -> Unit,
) {
    val context = LocalContext.current
    var scans by remember { mutableStateOf(ScanLibrary.list(context)) }
    var selected by remember { mutableStateOf<String?>(null) }
    var menu by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    var pending by remember { mutableStateOf<Uri?>(null) }
    fun begin(uri: Uri) {
        busy = true
        onOpen(uri) { busy = false }
    }
    fun openSelected() {
        val uri = selected ?: return
        if (busy) return
        if (scanning) pending = Uri.parse(uri) else begin(Uri.parse(uri))
    }
    fun reload(keep: String?) {
        scans = ScanLibrary.list(context)
        selected = keep?.takeIf { id -> scans.any { it.uri == id } }
    }
    BackHandler(enabled = busy) {}
    Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 12.dp, vertical = 28.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box {
                    OverflowButton(onClick = { menu = true }, enabled = !busy)
                    DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        DropdownMenuItem(
                            text = { Text("Remove") },
                            enabled = selected != null,
                            onClick = {
                                menu = false
                                val uri = selected ?: return@DropdownMenuItem
                                ScanLibrary.forget(context, Uri.parse(uri))
                                reload(null)
                            },
                        )
                        DropdownMenuItem(
                            text = { Text("Remove all") },
                            onClick = {
                                menu = false
                                ScanLibrary.forgetAll(context)
                                reload(null)
                            },
                        )
                    }
                }
                Text(
                    "Saved scans",
                    color = Color(0xFFE8EAED),
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.weight(1f),
                )
                TextButton(onClick = onClose, enabled = !busy) { Text("Close") }
            }
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .clearSelectionOnEmptyTap(enabled = !busy) { selected = null },
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (scans.isEmpty()) {
                        Text(
                            "No saved scans",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 16.dp),
                        )
                    }
                    scans.forEach { scan ->
                        val on = scan.uri == selected
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(if (on) Color(0xFF1B3A5F) else Color(0xFF2C2C31))
                                .clickable(enabled = !busy) { selected = scan.uri }
                                .padding(horizontal = 16.dp, vertical = 16.dp),
                        ) {
                            Text(
                                scan.name,
                                color = if (on) Color(0xFFD3E4FF) else Color(0xFFE8EAED),
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                            Text(
                                scanLine(scan),
                                style = MaterialTheme.typography.bodySmall,
                                color = if (on) Color(0xFFB7C9E6) else Color(0xFFBDC1C6),
                            )
                        }
                    }
                }
            }
            Button(
                onClick = { openSelected() },
                enabled = selected != null && !busy,
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Open") }
        }
        if (busy) {
            Box(
                Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.72f))
                    .clickable(enabled = true, onClick = {}),
                contentAlignment = Alignment.Center,
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                ) {
                    CircularProgressIndicator()
                    Text("Loading")
                }
            }
        }
    }
    if (pending != null) {
        AlertDialog(
            onDismissRequest = { pending = null },
            title = { Text("Replace this scan?") },
            text = { Text("Loading a package stops the current scan.") },
            confirmButton = {
                TextButton(onClick = {
                    val uri = pending
                    pending = null
                    if (uri != null) begin(uri)
                }) { Text("Load") }
            },
            dismissButton = {
                TextButton(onClick = { pending = null }) { Text("Cancel") }
            },
        )
    }
}

@Composable
private fun OverflowButton(onClick: () -> Unit, enabled: Boolean) {
    IconButton(onClick = onClick, enabled = enabled) {
        Canvas(Modifier.size(22.dp)) {
            val color = Color.White.copy(alpha = if (enabled) 1f else 0.4f)
            val radius = 1.7.dp.toPx()
            val center = Offset(size.width / 2f, size.height / 2f)
            val gap = size.height / 4f
            drawCircle(color, radius, center.copy(y = center.y - gap))
            drawCircle(color, radius, center)
            drawCircle(color, radius, center.copy(y = center.y + gap))
        }
    }
}

private fun Modifier.clearSelectionOnEmptyTap(enabled: Boolean, onClear: () -> Unit): Modifier {
    return pointerInput(enabled) {
        awaitEachGesture {
            val down = awaitFirstDown(pass = PointerEventPass.Final, requireUnconsumed = false)
            if (down.isConsumed) return@awaitEachGesture
            val pointer = down.id
            val origin = down.position
            var moved = false
            while (true) {
                val event = awaitPointerEvent(PointerEventPass.Final)
                val change = event.changes.firstOrNull { it.id == pointer } ?: return@awaitEachGesture
                if (change.isConsumed) return@awaitEachGesture
                if ((change.position - origin).getDistance() > viewConfiguration.touchSlop) moved = true
                if (!change.pressed) {
                    if (!moved && enabled) onClear()
                    return@awaitEachGesture
                }
            }
        }
    }
}

private fun scanLine(scan: SavedScan): String {
    val whenText = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date(scan.savedAt))
    val frames = if (scan.frames == 1) "1 frame" else "${scan.frames} frames"
    return "$frames · $whenText"
}
