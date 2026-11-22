package com.pocketscanner.capture

import android.Manifest
import android.app.GameManager
import android.app.GameState
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.VectorConverter
import androidx.compose.animation.core.spring
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.ui.Alignment
import androidx.compose.ui.geometry.Offset
import androidx.compose.foundation.layout.padding
import androidx.compose.ui.draw.alpha
import androidx.compose.foundation.layout.displayCutout
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.union
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.rememberDrawerState
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import kotlinx.coroutines.delay
import androidx.compose.runtime.MutableIntState
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import kotlinx.coroutines.launch
import kotlin.math.PI
import kotlin.math.atan2

class MainActivity : ComponentActivity() {
    private var session: RealSenseSession? = null
    private var preview: VkPreview? = null
    private var retained = false

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
            val ui = CaptureRuntime.ui
            MaterialTheme(colorScheme = Scheme) {
                ScanScreen(
                    ui = ui,
                    onPreview = { view ->
                        preview = view
                        view.onAttached = { attached ->
                            session?.attachPreview(attached)
                        }
                    },
                    onPreviewGone = { view -> session?.detachPreview(view) },
                    onPreviewKind = { session?.setPreviewKind(it) },
                    onView = { pose, fx, fy, cx, cy, width, height ->
                        session?.setView(pose, fx, fy, cx, cy, width, height)
                    },
                    onApply = { session?.applyScan(it) },
                    onBegin = { session?.beginScan(it) },
                    onStop = {
                        if (session?.stopAndRefine() == true) openRefine()
                    },
                    onSettings = { openSettings() },
                    chunkCount = { preview?.chunkCount() ?: 0 },
                    chunkPending = { preview?.chunkPending() ?: 0 },
                    chunkCap = { preview?.chunkCap() ?: 0 },
                    onRecord = { session?.setRecording(it) },
                    recordedCount = { session?.recordedCount() ?: 0 },
                    recording = session?.recordingEnabled() == true,
                    onScans = { openScans(fromRefine = false) },
                    onRuns = { startActivity(android.content.Intent(this, MetricsActivity::class.java)) },
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
        preview?.reclaim()
    }

    private fun requestGameMode() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val manager = getSystemService(GameManager::class.java) ?: return
        manager.setGameState(GameState(false, GameState.MODE_GAMEPLAY_UNINTERRUPTIBLE))
    }

    private fun startSession() {
        if (retained) return
        retained = true
        session = CaptureRuntime.retain(this)
        preview?.let { session?.attachPreview(it) }
    }

    override fun onDestroy() {
        if (retained) {
            retained = false
            CaptureRuntime.release()
            session = null
        }
        super.onDestroy()
    }
}

internal val Scheme = darkColorScheme(
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

private class LookHolder {
    var look = Look(0f, 0f, 0f, Vec3(0f, 0f, 0f))
    var anchor: FloatArray? = null
    var holdAnchor = false
}

private class LookGestures(private val holder: LookHolder, private val tick: MutableIntState) {
    fun flyBy(forward: Float, strafe: Float) {
        holder.holdAnchor = false
        holder.look = fly(holder.look, forward, strafe)
        tick.intValue++
    }

    fun lookBy(dx: Float, dy: Float) {
        holder.holdAnchor = false
        holder.look = lookAround(holder.look, dx * 0.007f, dy * 0.007f)
        tick.intValue++
    }

    fun twist(delta: Float) {
        holder.holdAnchor = false
        holder.look = rollBy(holder.look, -delta)
        tick.intValue++
    }

    fun snap(pose: FloatArray) {
        holder.anchor = pose.copyOf()
        holder.look = lookFromPose(pose)
        holder.holdAnchor = true
        tick.intValue++
    }

    fun viewPose(): FloatArray {
        val saved = holder.anchor
        return if (holder.holdAnchor && saved != null) saved else poseOf(holder.look)
    }
}

@Composable
private fun ScanScreen(
    ui: CaptureUi,
    onPreview: (VkPreview) -> Unit,
    onPreviewGone: (VkPreview) -> Unit,
    onPreviewKind: (Int) -> Unit,
    onView: (FloatArray, Float, Float, Float, Float, Int, Int) -> Unit,
    onApply: (ScanSettings) -> Unit,
    onBegin: (Boolean) -> Unit,
    onStop: () -> Unit,
    onSettings: () -> Unit,
    chunkCount: () -> Int,
    chunkPending: () -> Int,
    chunkCap: () -> Int,
    onRecord: (Boolean) -> Unit,
    recordedCount: () -> Int,
    recording: Boolean,
    onScans: () -> Unit,
    onRuns: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    var mode by remember { mutableStateOf(PreviewMode.Color) }
    var marker by remember { mutableStateOf<Marker?>(null) }
    var applied by remember { mutableStateOf(ScanSettings()) }
    var resetEpoch by remember { mutableIntStateOf(0) }
    var chunks by remember { mutableIntStateOf(0) }
    var pendingChunks by remember { mutableIntStateOf(0) }
    var chunkLimit by remember { mutableIntStateOf(0) }
    var record by remember { mutableStateOf(recording) }
    var recorded by remember { mutableIntStateOf(0) }
    var confirmNew by remember { mutableStateOf(false) }
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    val lifecycle = androidx.lifecycle.compose.LocalLifecycleOwner.current.lifecycle
    DisposableEffect(lifecycle, mode) {
        val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) onPreviewKind(mode.ordinal)
        }
        lifecycle.addObserver(observer)
        onDispose { lifecycle.removeObserver(observer) }
    }
    LaunchedEffect(Unit) {
        while (true) {
            chunks = chunkCount()
            pendingChunks = chunkPending()
            chunkLimit = chunkCap()
            recorded = recordedCount()
            delay(250)
        }
    }
    val trail = remember { androidx.compose.runtime.mutableStateListOf<Vec3>() }
    LaunchedEffect(resetEpoch) {
        marker = null
        trail.clear()
    }
    SideEffect {
        val sample = ui.track
        val current = sample?.pose
        if (ui.phase == "scanning" && mode != PreviewMode.Color && current != null) {
            onView(current, sample.fx, sample.fy, sample.cx, sample.cy, sample.width, sample.height)
        }
        if (current != null && ui.phase == "scanning") {
            val camera = cameraPosition(current)
            val last = trail.lastOrNull()
            if (last == null || last.distanceTo(camera) > 0.05f) {
                trail.add(camera)
                if (trail.size > 400) trail.removeAt(0)
            }
        }
    }
    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet {
                Column(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 28.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    Text(
                        "Scan",
                        style = MaterialTheme.typography.titleLarge,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                    )
                    NavigationDrawerItem(
                        label = { Text("Settings") },
                        selected = false,
                        onClick = {
                            scope.launch { drawerState.close() }
                            onSettings()
                        },
                    )
                    NavigationDrawerItem(
                        label = { Text("Scans") },
                        selected = false,
                        onClick = {
                            scope.launch { drawerState.close() }
                            onScans()
                        },
                    )
                    NavigationDrawerItem(
                        label = { Text("Runs") },
                        selected = false,
                        onClick = {
                            scope.launch { drawerState.close() }
                            onRuns()
                        },
                    )
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(if (record) "Recording $recorded" else "Record")
                        Switch(
                            checked = record,
                            onCheckedChange = {
                                record = it
                                onRecord(it)
                            },
                        )
                    }
                }
            }
        },
    ) {
    val menuOpen = drawerState.currentValue == DrawerValue.Open || drawerState.targetValue == DrawerValue.Open
    BackHandler(enabled = menuOpen) { scope.launch { drawerState.close() } }
    Box(Modifier.fillMaxSize().background(scheme.background)) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx -> VkPreview(ctx).also(onPreview) },
            onRelease = onPreviewGone,
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
        ScanChrome(
            ui = ui,
            mode = mode,
            chunks = chunks,
            pendingChunks = pendingChunks,
            chunkLimit = chunkLimit,
            onMode = {
                mode = it
                onPreviewKind(it.ordinal)
            },
            onReset = {
                resetEpoch++
                onApply(applied)
            },
            onMenu = { scope.launch { drawerState.open() } },
            onPrimary = {
                if (ui.phase == "scanning") onStop()
                else if (ui.framesOnDisk) confirmNew = true
                else onBegin(false)
            },
        )
        if (confirmNew) {
            AlertDialog(
                onDismissRequest = { confirmNew = false },
                title = { Text("Start a new scan?") },
                text = { Text("This clears the current scan.") },
                confirmButton = {
                    TextButton(onClick = {
                        confirmNew = false
                        onBegin(true)
                    }) { Text("Start") }
                },
                dismissButton = {
                    TextButton(onClick = { confirmNew = false }) { Text("Cancel") }
                },
            )
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

@Composable
internal fun MeshWorkBadge(chunks: Int, pending: Int) {
    if (chunks <= 0 || pending <= 0) return
    val done = ((chunks - pending) * 100 / chunks).coerceIn(0, 100)
    val scheme = MaterialTheme.colorScheme
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(14.dp))
            .background(scheme.surface.copy(alpha = 0.45f))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(14.dp),
            strokeWidth = 2.dp,
            color = scheme.primary,
        )
        Text("Mesh $done%", color = scheme.onSurface, style = MaterialTheme.typography.bodyMedium)
    }
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

@Composable
private fun ScanChrome(
    ui: CaptureUi,
    mode: PreviewMode,
    chunks: Int,
    pendingChunks: Int,
    chunkLimit: Int,
    onMode: (PreviewMode) -> Unit,
    onReset: () -> Unit,
    onMenu: () -> Unit,
    onPrimary: () -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    val status = buildString {
        append(ui.headline)
        if (chunkLimit > 0) append("  ·  $chunks/$chunkLimit")
        if (pendingChunks > 0) append("  ·  $pendingChunks waiting")
        ui.track?.let { append("  ·  ${trackLine(it)}") }
    }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.SpaceBetween,
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(18.dp))
                    .background(scheme.surface.copy(alpha = 0.45f))
                    .padding(start = 4.dp, end = 4.dp, top = 2.dp, bottom = 2.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                MenuButton(onMenu)
                Text(
                    status,
                    color = scheme.onSurface,
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f),
                )
                TextButton(onClick = onReset) { Text("Reset") }
            }
            MeshWorkBadge(chunks, pendingChunks)
        }
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            PreviewSwitch(mode = mode, onMode = onMode)
            Button(onClick = onPrimary, modifier = Modifier.fillMaxWidth()) {
                Text(if (ui.phase == "scanning") "Stop" else "Start scanning")
            }
        }
    }
}

@Composable
internal fun Modifier.belowSystemBars(): Modifier {
    return windowInsetsPadding(
        WindowInsets.safeDrawing
            .union(WindowInsets.statusBars)
            .union(WindowInsets.navigationBars)
            .union(WindowInsets.displayCutout),
    )
}

@Composable
internal fun RefineScreen(
    ui: CaptureUi,
    onPreview: (VkPreview) -> Unit,
    onPreviewGone: (VkPreview) -> Unit,
    onView: (FloatArray, Float, Float, Float, Float, Int, Int) -> Unit,
    onSettings: () -> Unit,
    onBack: () -> Unit,
    onSplat: () -> Unit,
    onSave: () -> Unit,
    onScans: () -> Unit,
    firstPose: () -> FloatArray?,
    lens: () -> ViewLens?,
    chunkCount: () -> Int,
    chunkPending: () -> Int,
    onMeshIdle: () -> Boolean,
) {
    val scheme = MaterialTheme.colorScheme
    val refining = ui.phase == "refining"
    val review = ui.phase == "review"
    val lookHolder = remember { LookHolder() }
    val lookTick = remember { mutableIntStateOf(0) }
    val gestures = remember { LookGestures(lookHolder, lookTick) }
    var meshChunks by remember { mutableIntStateOf(0) }
    var meshPending by remember { mutableIntStateOf(0) }
    LaunchedEffect(ui.phase, meshPending, meshChunks) {
        if (ui.phase != "review" || meshPending > 0 || meshChunks <= 0) return@LaunchedEffect
        delay(1500)
        if (meshPending > 0 || meshChunks <= 0) return@LaunchedEffect
        while (true) {
            val saved = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) { onMeshIdle() }
            if (saved) break
            delay(2000)
        }
    }
    LaunchedEffect(Unit) {
        while (true) {
            meshChunks = chunkCount()
            meshPending = chunkPending()
            delay(250)
        }
    }
    LaunchedEffect(ui.phase) {
        if (ui.phase != "review") return@LaunchedEffect
        repeat(8) {
            val pose = firstPose()
            if (pose != null) {
                gestures.snap(pose)
                return@LaunchedEffect
            }
            delay(200)
        }
    }
    val poseRevision = lookTick.intValue
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    SideEffect {
        val eye = lens() ?: return@SideEffect
        if (review && eye.fx > 0f && poseRevision != Int.MIN_VALUE) {
            onView(gestures.viewPose(), eye.fx, eye.fy, eye.cx, eye.cy, eye.width, eye.height)
        }
    }
    val menuOpen = drawerState.currentValue == DrawerValue.Open || drawerState.targetValue == DrawerValue.Open
    ModalNavigationDrawer(
        drawerState = drawerState,
        gesturesEnabled = menuOpen,
        drawerContent = {
            ModalDrawerSheet {
                Column(
                    modifier = Modifier.belowSystemBars().padding(horizontal = 12.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    Text(
                        "Pose refinement",
                        style = MaterialTheme.typography.titleLarge,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                    )
                    NavigationDrawerItem(
                        label = { Text("Settings") },
                        selected = false,
                        onClick = {
                            if (refining || ui.saving) return@NavigationDrawerItem
                            scope.launch { drawerState.close() }
                            onSettings()
                        },
                    )
                    NavigationDrawerItem(
                        label = { Text("Scans") },
                        selected = false,
                        onClick = {
                            if (ui.saving) return@NavigationDrawerItem
                            scope.launch { drawerState.close() }
                            onScans()
                        },
                    )
                }
            }
        },
    ) {
    BackHandler(enabled = menuOpen && !ui.saving) { scope.launch { drawerState.close() } }
    BackHandler(enabled = ui.saving) {}
    Box(Modifier.fillMaxSize().background(scheme.background)) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx -> VkPreview(ctx).also(onPreview) },
            onRelease = onPreviewGone,
        )
        if (review) {
            LookPad(
                onLook = { dx, dy -> gestures.lookBy(dx, dy) },
                onRoll = { delta -> gestures.twist(delta) },
            )
        }
        Column(
            modifier = Modifier
                .fillMaxSize()
                .belowSystemBars()
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.SpaceBetween,
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(18.dp))
                        .background(scheme.surface.copy(alpha = 0.45f))
                        .padding(start = 4.dp, end = 14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    MenuButton(onClick = { scope.launch { drawerState.open() } }, enabled = !refining)
                    Text(
                        "Pose refinement",
                        color = scheme.onSurface,
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.weight(1f),
                    )
                }
                MeshWorkBadge(meshChunks, meshPending)
            }
            if (!refining) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    if (ui.refineDetail.isNotEmpty() || ui.saveLine.isNotEmpty()) {
                        Text(
                            listOf(ui.refineDetail, ui.saveLine).filter { it.isNotEmpty() }.joinToString("  ·  "),
                            color = scheme.onSurface,
                            style = MaterialTheme.typography.bodySmall,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                    Trackball(enabled = true) { dx, dy -> gestures.flyBy(-dy * 0.008f, dx * 0.008f) }
                    FilledTonalButton(
                        onClick = { firstPose()?.let { gestures.snap(it) } },
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text("First frame") }
                    Button(
                        onClick = onSave,
                        enabled = ui.framesOnDisk,
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text("Save") }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        FilledTonalButton(onClick = onBack, modifier = Modifier.weight(1f)) { Text("Back") }
                        Button(
                            onClick = onSplat,
                            enabled = ui.framesOnDisk && meshPending == 0,
                            modifier = Modifier.weight(1f),
                        ) { Text("Start splatting") }
                    }
                }
            } else {
                Spacer(Modifier.height(1.dp))
            }
        }
        if (refining) {
            Column(
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(horizontal = 36.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                if (ui.refineDetail.isNotEmpty()) {
                    Text(ui.refineDetail, color = scheme.onSurface, style = MaterialTheme.typography.titleMedium)
                }
                LinearProgressIndicator(
                    progress = { ui.refineProgress.coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth(),
                )
                if (ui.saveLine.isNotEmpty()) {
                    Text(ui.saveLine, color = scheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
                }
            }
        }
        if (ui.saving) {
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
                    Text("Saving", color = Color.White)
                }
            }
        }
    }
    }
}

@Composable
private fun MenuButton(onClick: () -> Unit, enabled: Boolean = true) {
    IconButton(onClick = onClick, enabled = enabled) {
        Canvas(Modifier.size(22.dp)) {
            val color = Color.White.copy(alpha = if (enabled) 1f else 0.4f)
            val stroke = 2.dp.toPx()
            val left = 1.dp.toPx()
            val right = size.width - 1.dp.toPx()
            val y0 = size.height * 0.28f
            val y1 = size.height * 0.5f
            val y2 = size.height * 0.72f
            drawLine(color, Offset(left, y0), Offset(right, y0), strokeWidth = stroke)
            drawLine(color, Offset(left, y1), Offset(right, y1), strokeWidth = stroke)
            drawLine(color, Offset(left, y2), Offset(right, y2), strokeWidth = stroke)
        }
    }
}

@Composable
internal fun LookPad(onLook: (Float, Float) -> Unit, onRoll: (Float) -> Unit) {
    Box(
        Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false)
                    val previous = HashMap<Long, Offset>()
                    var twisting = false
                    while (true) {
                        val event = awaitPointerEvent()
                        val down = event.changes.filter { it.pressed }
                        if (down.isEmpty()) break
                        if (down.size >= 2) {
                            twisting = true
                            val a = down[0]
                            val b = down[1]
                            val beforeA = previous[a.id.value]
                            val beforeB = previous[b.id.value]
                            if (beforeA != null && beforeB != null) {
                                val was = atan2(beforeB.y - beforeA.y, beforeB.x - beforeA.x)
                                val now = atan2(b.position.y - a.position.y, b.position.x - a.position.x)
                                var delta = now - was
                                val turn = (2.0 * PI).toFloat()
                                if (delta > PI.toFloat()) delta -= turn
                                if (delta < -PI.toFloat()) delta += turn
                                if (delta != 0f) onRoll(delta)
                            }
                        } else if (!twisting) {
                            val change = down[0]
                            val delta = change.position - change.previousPosition
                            if (delta.x != 0f || delta.y != 0f) onLook(delta.x, delta.y)
                        }
                        for (change in down) previous[change.id.value] = change.position
                        event.changes.forEach { it.consume() }
                    }
                }
            },
    )
}

@Composable
internal fun Trackball(enabled: Boolean, onDrag: (Float, Float) -> Unit) {
    val knob = remember { Animatable(Offset.Zero, Offset.VectorConverter) }
    val scope = rememberCoroutineScope()
    Canvas(
        Modifier
            .size(124.dp)
            .alpha(0.5f)
            .pointerInput(enabled) {
                if (!enabled) return@pointerInput
                val limit = minOf(size.width, size.height) / 2f - 22.dp.toPx()
                detectDragGestures(
                    onDrag = { change, drag ->
                        change.consume()
                        val next = clampOffset(knob.value + drag, limit)
                        scope.launch { knob.snapTo(next) }
                        onDrag(drag.x, drag.y)
                    },
                    onDragEnd = { scope.launch { knob.animateTo(Offset.Zero, spring(stiffness = Spring.StiffnessMediumLow)) } },
                    onDragCancel = { scope.launch { knob.animateTo(Offset.Zero, spring(stiffness = Spring.StiffnessMediumLow)) } },
                )
            },
    ) {
        val radius = size.minDimension / 2f
        drawLitBall(center, radius, bright = false)
        drawLitBall(center + knob.value, 16.dp.toPx(), bright = true)
    }
}

private fun DrawScope.drawLitBall(center: Offset, radius: Float, bright: Boolean) {
    val light = center + Offset(-radius * 0.36f, -radius * 0.42f)
    val body = if (bright) {
        listOf(Color(0xFFF4F8FF), Color(0xFF9EC4F8), Color(0xFF24558C))
    } else {
        listOf(Color(0xFFFFFFFF), Color(0xFFD8D8DE), Color(0xFF8E8E96))
    }
    drawCircle(
        brush = Brush.radialGradient(body, center = light, radius = radius * 1.7f),
        radius = radius,
        center = center,
    )
    drawCircle(
        brush = Brush.radialGradient(
            listOf(Color.White.copy(alpha = if (bright) 0.85f else 0.45f), Color.Transparent),
            center = light,
            radius = radius * 0.42f,
        ),
        radius = radius * 0.42f,
        center = light,
    )
    drawCircle(
        color = Color.White.copy(alpha = 0.22f),
        radius = radius - 1.dp.toPx(),
        center = center,
        style = Stroke(width = 1.4.dp.toPx()),
    )
}

private fun clampOffset(offset: Offset, limit: Float): Offset {
    val length = offset.getDistance()
    if (length <= limit || length == 0f) return offset
    return offset * (limit / length)
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
internal fun SettingsPanel(
    ui: CaptureUi,
    applied: ScanSettings,
    showFrontier: Boolean,
    onFrontier: (Boolean) -> Unit,
    onSelect: (String) -> Unit,
    onApply: (ScanSettings) -> Unit,
    onReset: () -> Unit,
    onRefine: (Boolean, Int) -> Unit,
    onSplatting: (Int, Boolean, Int, Int, Int, Int) -> Unit,
    onClose: () -> Unit,
) {
    var voxel by remember { mutableIntStateOf(applied.voxelCm) }
    var integrate by remember { mutableIntStateOf(applied.integrateWidth) }
    var minSize by remember { mutableIntStateOf(applied.minSize) }
    var levels by remember { mutableIntStateOf(applied.pyramidLevels) }
    var depth by remember { mutableIntStateOf(applied.depthMeters.toInt()) }
    var closeLoops by remember { mutableStateOf(applied.closeLoops) }
    var reviewVoxel by remember { mutableIntStateOf(applied.reviewVoxelCm) }
    var epochs by remember { mutableIntStateOf(applied.epochs) }
    var validation by remember { mutableStateOf(applied.validation) }
    var validationPercent by remember { mutableIntStateOf(applied.validationPercent) }
    var validationEvery by remember { mutableIntStateOf(applied.validationEvery) }
    var gaussBudget by remember { mutableIntStateOf(applied.gaussBudget) }
    var videoSeconds by remember { mutableIntStateOf(applied.videoSeconds) }
    val scheme = MaterialTheme.colorScheme
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(scheme.background)
            .windowInsetsPadding(WindowInsets.safeDrawing)
            .padding(horizontal = 16.dp, vertical = 12.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Settings", color = scheme.onSurface, style = MaterialTheme.typography.titleLarge)
            TextButton(onClick = onClose) { Text("Close") }
        }
        Text("Tracking", color = scheme.onSurface, style = MaterialTheme.typography.titleMedium)
        Text(
            "Apply and Reset both restart tracking and clear the mesh. The camera stays on.",
            color = scheme.onSurfaceVariant,
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
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text("Unscanned edge", style = MaterialTheme.typography.labelLarge)
                Text(
                    "Amber outline. Applies immediately and does not restart tracking.",
                    color = scheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            Switch(checked = showFrontier, onCheckedChange = onFrontier)
        }
        Button(
            onClick = {
                onApply(
                    ScanSettings(
                        voxelCm = voxel,
                        integrateWidth = integrate,
                        minSize = minSize,
                        pyramidLevels = levels,
                        depthMeters = depth.toFloat(),
                        reviewVoxelCm = reviewVoxel,
                        closeLoops = closeLoops,
                        epochs = epochs,
                        validation = validation,
                        validationPercent = validationPercent,
                        validationEvery = validationEvery,
                        gaussBudget = gaussBudget,
                        videoSeconds = videoSeconds,
                    ),
                )
            },
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Apply") }
        FilledTonalButton(onClick = onReset, modifier = Modifier.fillMaxWidth()) { Text("Reset") }
        Text("Pose refinement", color = scheme.onSurface, style = MaterialTheme.typography.titleMedium)
        Text(
            "Used after Stop. These apply immediately and do not restart tracking.",
            color = scheme.onSurfaceVariant,
            style = MaterialTheme.typography.bodySmall,
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text("Close loops", style = MaterialTheme.typography.labelLarge)
                Text(
                    "After Stop, search for a return to a place already scanned.",
                    color = scheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            Switch(
                checked = closeLoops,
                onCheckedChange = {
                    closeLoops = it
                    onRefine(it, reviewVoxel)
                },
            )
        }
        ChoiceRow("Review mesh", listOf(2, 4, 8), reviewVoxel, { "$it cm" }) {
            reviewVoxel = it
            onRefine(closeLoops, it)
        }
        Text(
            "Default is 8 cm. 2 cm can fill the chunk budget in a large room.",
            color = scheme.onSurfaceVariant,
            style = MaterialTheme.typography.bodySmall,
        )
        Text("Splatting", color = scheme.onSurface, style = MaterialTheme.typography.titleMedium)
        Text(
            "Used by the next training run. Changing these does not restart tracking.",
            color = scheme.onSurfaceVariant,
            style = MaterialTheme.typography.bodySmall,
        )
        LabeledSlider(
            title = "Epochs",
            value = "$epochs",
            detail = "Training stops after this many passes over the frames.",
            sliderValue = epochs.toFloat(),
            range = 1f..500f,
        ) {
            epochs = it.toInt().coerceIn(1, 500)
            onSplatting(epochs, validation, validationPercent, validationEvery, gaussBudget, videoSeconds)
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Validation", style = MaterialTheme.typography.labelLarge)
            Switch(
                checked = validation,
                onCheckedChange = {
                    validation = it
                    onSplatting(epochs, validation, validationPercent, validationEvery, gaussBudget, videoSeconds)
                },
            )
        }
        ChoiceRow("Validation share", listOf(5, 8, 10), validationPercent, { "$it%" }) {
            validationPercent = it
            onSplatting(epochs, validation, it, validationEvery, gaussBudget, videoSeconds)
        }
        LabeledSlider(
            title = "Validation interval",
            value = "$validationEvery epochs",
            detail = "Validation runs after this many training epochs.",
            sliderValue = validationEvery.toFloat(),
            range = 1f..20f,
        ) {
            validationEvery = it.toInt().coerceIn(1, 20)
            onSplatting(epochs, validation, validationPercent, validationEvery, gaussBudget, videoSeconds)
        }
        LabeledSlider(
            title = "Gaussians",
            value = String.format(java.util.Locale.US, "%.1f million", gaussBudget / 1_000_000f),
            detail = "Most measured mesh vertices kept for the run.",
            sliderValue = gaussBudget / 1_000_000f,
            range = 0.5f..5f,
        ) {
            gaussBudget = (it * 1_000_000f).toInt().coerceIn(500_000, 5_000_000)
            onSplatting(epochs, validation, validationPercent, validationEvery, gaussBudget, videoSeconds)
        }
        LabeledSlider(
            title = "Video",
            value = "$videoSeconds s",
            detail = "Length of the clip taken from the middle of the path.",
            sliderValue = videoSeconds.toFloat(),
            range = 1f..30f,
        ) {
            videoSeconds = it.toInt().coerceIn(1, 30)
            onSplatting(epochs, validation, validationPercent, validationEvery, gaussBudget, videoSeconds)
        }
    }
}

@Composable
private fun LabeledSlider(
    title: String,
    value: String,
    detail: String,
    sliderValue: Float,
    range: ClosedFloatingPointRange<Float>,
    onValueChange: (Float) -> Unit,
) {
    val scheme = MaterialTheme.colorScheme
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(title, color = scheme.onSurface, style = MaterialTheme.typography.labelLarge)
            Text(value, color = scheme.onSurface, style = MaterialTheme.typography.labelLarge)
        }
        Text(detail, color = scheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
        androidx.compose.material3.Slider(value = sliderValue, onValueChange = onValueChange, valueRange = range)
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

