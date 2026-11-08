package com.pocketscanner.capture

import android.Manifest
import android.app.GameManager
import android.app.GameState
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat

class MainActivity : ComponentActivity() {
    private var ui by mutableStateOf(CaptureUi())
    private var session by mutableStateOf<RealSenseSession?>(null)
    private var preview: VkPreview? = null

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) startSession()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestGameMode()
        setContent {
            MaterialTheme(colorScheme = Scheme) {
                CaptureScaffold(
                    ui = ui,
                    onPreview = { view ->
                        preview = view
                        session?.attachPreview(view)
                    },
                    onShowsColor = { session?.setShowsColor(it) },
                    onSelect = { session?.selectProfile(it) },
                    onSnapshot = { session?.requestSnapshot() },
                )
            }
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startSession()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onResume() {
        super.onResume()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestGameMode()
    }

    private fun requestGameMode() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val manager = getSystemService(GameManager::class.java) ?: return
        manager.setGameState(GameState(false, GameState.MODE_GAMEPLAY_UNINTERRUPTIBLE))
    }

    private fun startSession() {
        if (session != null) return
        val created = RealSenseSession(applicationContext) { ui = it }
        session = created
        preview?.let { created.attachPreview(it) }
        created.start()
    }

    override fun onDestroy() {
        session?.stop()
        session = null
        super.onDestroy()
    }
}

private val Scheme = darkColorScheme(
    primary = Color(0xFF8AB4F8),
    onPrimary = Color(0xFF062E6F),
    primaryContainer = Color(0xFF1B3A5F),
    onPrimaryContainer = Color(0xFFD3E4FF),
    secondary = Color(0xFF4DD0E1),
    background = Color(0xFF000000),
    surface = Color(0xFF121212),
    surfaceContainer = Color(0xFF1E1E1E),
    onSurface = Color(0xFFE6E1E5),
    onSurfaceVariant = Color(0xFFCAC4D0),
    outline = Color(0xFF938F99),
)

private enum class PreviewMode { Color, Depth }

@Composable
private fun CaptureScaffold(
    ui: CaptureUi,
    onPreview: (VkPreview?) -> Unit,
    onShowsColor: (Boolean) -> Unit,
    onSelect: (String) -> Unit,
    onSnapshot: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    var mode by remember { mutableStateOf(PreviewMode.Color) }
    var marker by remember { mutableStateOf<Marker?>(null) }
    val trail = remember { androidx.compose.runtime.mutableStateListOf<Vec3>() }
    val pose = ui.track?.pose
    SideEffect {
        val current = pose ?: return@SideEffect
        val camera = cameraPosition(current)
        val last = trail.lastOrNull()
        if (last == null || last.distanceTo(camera) > 0.05f) {
            trail.add(camera)
            if (trail.size > 400) trail.removeAt(0)
        }
    }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(scheme.background)
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Pocket Scanner", color = scheme.onSurface, style = MaterialTheme.typography.titleLarge)
        Text(ui.headline, color = scheme.onSurface, style = MaterialTheme.typography.titleMedium)
        if (ui.deviceLine.isNotEmpty()) {
            Text(ui.deviceLine, color = scheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
        }
        Text(captureLine(ui), color = scheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
        ui.track?.let { sample ->
            Text(trackLine(sample), color = scheme.primary, style = MaterialTheme.typography.bodyMedium)
        }
        if (ui.saveLine.isNotEmpty()) {
            Text(ui.saveLine, color = scheme.onSurface, style = MaterialTheme.typography.bodyMedium)
        }
        if (ui.profiles.isNotEmpty()) {
            ProfileMenu(
                profiles = ui.profiles,
                selected = ui.selectedProfile.ifEmpty { ui.profiles.first() },
                onSelect = onSelect,
            )
        }
        PreviewSwitch(
            mode = mode,
            onMode = {
                mode = it
                onShowsColor(it == PreviewMode.Color)
            },
        )
        PreviewCard(modifier = Modifier.weight(1f)) {
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { ctx -> VkPreview(ctx).also(onPreview) },
                onRelease = { onPreview(null) },
            )
            if (mode == PreviewMode.Color) {
                val sample = ui.track
                WorldOverlay(
                    pose = sample?.pose,
                    fx = sample?.fx ?: 0f,
                    fy = sample?.fy ?: 0f,
                    cx = sample?.cx ?: 0f,
                    cy = sample?.cy ?: 0f,
                    imageWidth = sample?.width ?: 0,
                    imageHeight = sample?.height ?: 0,
                    marker = marker,
                    trail = trail,
                    modifier = Modifier.fillMaxSize(),
                )
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilledTonalButton(
                onClick = {
                    val current = ui.track?.pose ?: return@FilledTonalButton
                    marker = markerInFront(current)
                },
                enabled = ui.track?.pose != null,
                modifier = Modifier.weight(1f),
            ) {
                Text(if (marker == null) "Place marker" else "Move marker")
            }
            Button(onClick = onSnapshot, modifier = Modifier.weight(1f)) {
                Text("Save frame")
            }
        }
    }
}

private fun captureLine(ui: CaptureUi): String = buildString {
    if (ui.streamLine.isNotEmpty()) append(ui.streamLine)
    if (isNotEmpty()) append("   ")
    append("capture ${"%.1f".format(ui.fps)} fps")
    if (ui.detachCount > 0) append("   drops ${ui.detachCount}")
}

private fun trackLine(sample: TrackSample): String {
    val name = when (sample.state) {
        "tracking" -> "Tracking"
        "lost" -> "Lost"
        "initializing" -> "Initializing"
        "no-pose" -> "No pose"
        "off" -> "Tracking off"
        else -> sample.state
    }
    if (sample.fps <= 0f) return name
    return "$name · ${"%.1f".format(sample.fps)} fps · ${sample.frameMs} ms"
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PreviewSwitch(mode: PreviewMode, onMode: (PreviewMode) -> Unit) {
    SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
        SegmentedButton(
            selected = mode == PreviewMode.Color,
            onClick = { onMode(PreviewMode.Color) },
            shape = SegmentedButtonDefaults.itemShape(index = 0, count = 2),
        ) {
            Text("Color")
        }
        SegmentedButton(
            selected = mode == PreviewMode.Depth,
            onClick = { onMode(PreviewMode.Depth) },
            shape = SegmentedButtonDefaults.itemShape(index = 1, count = 2),
        ) {
            Text("Depth")
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ProfileMenu(
    profiles: List<String>,
    selected: String,
    onSelect: (String) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it },
        modifier = Modifier.fillMaxWidth(),
    ) {
        OutlinedTextField(
            value = selected,
            onValueChange = {},
            readOnly = true,
            singleLine = true,
            label = { Text("Profile") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .menuAnchor(MenuAnchorType.PrimaryNotEditable)
                .fillMaxWidth(),
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
        ) {
            profiles.forEach { label ->
                DropdownMenuItem(
                    text = { Text(label) },
                    onClick = {
                        expanded = false
                        if (label != selected) onSelect(label)
                    },
                )
            }
        }
    }
}

@Composable
private fun PreviewCard(
    modifier: Modifier,
    content: @Composable () -> Unit,
) {
    Box(
        modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(MaterialTheme.colorScheme.surfaceContainer),
    ) {
        content()
    }
}
