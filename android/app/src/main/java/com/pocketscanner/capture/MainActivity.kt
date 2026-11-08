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
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
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
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
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
                    onPreviewKind = { session?.setPreviewKind(it) },
                    onView = { pose, fx, fy, cx, cy, width, height ->
                        session?.setView(pose, fx, fy, cx, cy, width, height)
                    },
                    onSelect = { session?.selectProfile(it) },
                    onApply = { session?.applyScan(it) },
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

private enum class PreviewMode { Color, Depth, Mesh }

@Composable
private fun CaptureScaffold(
    ui: CaptureUi,
    onPreview: (VkPreview?) -> Unit,
    onPreviewKind: (Int) -> Unit,
    onView: (FloatArray, Float, Float, Float, Float, Int, Int) -> Unit,
    onSelect: (String) -> Unit,
    onApply: (ScanSettings) -> Unit,
    onSnapshot: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    var mode by remember { mutableStateOf(PreviewMode.Color) }
    var marker by remember { mutableStateOf<Marker?>(null) }
    var settingsOpen by remember { mutableStateOf(false) }
    var applied by remember { mutableStateOf(ScanSettings()) }
    var resetEpoch by remember { mutableIntStateOf(0) }
    val trail = remember { androidx.compose.runtime.mutableStateListOf<Vec3>() }
    LaunchedEffect(resetEpoch) {
        marker = null
        trail.clear()
    }
    SideEffect {
        val sample = ui.track
        val current = sample?.pose
        if (current != null) {
            onView(current, sample.fx, sample.fy, sample.cx, sample.cy, sample.width, sample.height)
            val camera = cameraPosition(current)
            val last = trail.lastOrNull()
            if (last == null || last.distanceTo(camera) > 0.05f) {
                trail.add(camera)
                if (trail.size > 400) trail.removeAt(0)
            }
        }
    }
    Box(Modifier.fillMaxSize()) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(scheme.background)
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text("Pocket Scanner", color = scheme.onSurface, style = MaterialTheme.typography.titleLarge)
            Row {
                TextButton(onClick = {
                    resetEpoch++
                    onApply(applied)
                }) { Text("Reset") }
                TextButton(onClick = { settingsOpen = true }) { Text("Settings") }
            }
        }
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
        PreviewSwitch(
            mode = mode,
            onMode = {
                mode = it
                onPreviewKind(it.ordinal)
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
        if (settingsOpen) {
            SettingsPanel(
                ui = ui,
                applied = applied,
                onSelect = onSelect,
                onApply = {
                    applied = it
                    resetEpoch++
                    onApply(it)
                    settingsOpen = false
                },
                onReset = {
                    resetEpoch++
                    onApply(applied)
                    settingsOpen = false
                },
                onClose = { settingsOpen = false },
            )
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
            shape = SegmentedButtonDefaults.itemShape(index = 0, count = 3),
        ) {
            Text("Color")
        }
        SegmentedButton(
            selected = mode == PreviewMode.Depth,
            onClick = { onMode(PreviewMode.Depth) },
            shape = SegmentedButtonDefaults.itemShape(index = 1, count = 3),
        ) {
            Text("Depth")
        }
        SegmentedButton(
            selected = mode == PreviewMode.Mesh,
            onClick = { onMode(PreviewMode.Mesh) },
            shape = SegmentedButtonDefaults.itemShape(index = 2, count = 3),
        ) {
            Text("Mesh")
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
private fun SettingsPanel(
    ui: CaptureUi,
    applied: ScanSettings,
    onSelect: (String) -> Unit,
    onApply: (ScanSettings) -> Unit,
    onReset: () -> Unit,
    onClose: () -> Unit,
) {
    var voxel by remember { mutableIntStateOf(applied.voxelCm) }
    var integrate by remember { mutableIntStateOf(applied.integrateWidth) }
    var minSize by remember { mutableIntStateOf(applied.minSize) }
    var levels by remember { mutableIntStateOf(applied.pyramidLevels) }
    var depth by remember { mutableIntStateOf(applied.depthMeters.toInt()) }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background.copy(alpha = 0.96f))
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(horizontal = 16.dp, vertical = 12.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Settings", color = MaterialTheme.colorScheme.onSurface, style = MaterialTheme.typography.titleLarge)
            TextButton(onClick = onClose) { Text("Close") }
        }
        Text(
            "Apply and Reset both restart tracking and clear the mesh. The camera stays on.",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            style = MaterialTheme.typography.bodySmall,
        )
        if (ui.profiles.isNotEmpty()) {
            ProfileMenu(
                profiles = ui.profiles,
                selected = ui.selectedProfile.ifEmpty { ui.profiles.first() },
                onSelect = onSelect,
            )
        }
        ChoiceRow("Voxel", listOf(2, 4, 8), voxel, { "$it cm" }) { voxel = it }
        ChoiceRow("Integrate", listOf(80, 160, 320), integrate, { "$it×${it * 3 / 4}" }) { integrate = it }
        ChoiceRow("Features", listOf(300, 500, 800), minSize, { "$it" }) { minSize = it }
        ChoiceRow("Pyramid", listOf(4, 6, 8), levels, { "$it" }) { levels = it }
        ChoiceRow("Depth", listOf(3, 6, 10), depth, { "$it m" }) { depth = it }
        Spacer(Modifier.height(4.dp))
        Button(
            onClick = {
                onApply(
                    ScanSettings(
                        voxelCm = voxel,
                        integrateWidth = integrate,
                        minSize = minSize,
                        pyramidLevels = levels,
                        depthMeters = depth.toFloat(),
                    ),
                )
            },
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Apply") }
        FilledTonalButton(onClick = onReset, modifier = Modifier.fillMaxWidth()) { Text("Reset") }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun <T> ChoiceRow(label: String, values: List<T>, selected: T, title: (T) -> String, onPick: (T) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.labelLarge)
        SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
            values.forEachIndexed { index, value ->
                SegmentedButton(
                    selected = value == selected,
                    onClick = { onPick(value) },
                    shape = SegmentedButtonDefaults.itemShape(index = index, count = values.size),
                ) { Text(title(value)) }
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
