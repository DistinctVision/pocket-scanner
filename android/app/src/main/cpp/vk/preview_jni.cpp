#include "vk_preview.h"

#include <android/native_window_jni.h>
#include <jni.h>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_pocketscanner_capture_VkPreview_nativeCreate(JNIEnv* env, jobject, jobject surface) {
    ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
    if (window == nullptr) return 0;
    auto* preview = new Preview();
    if (!preview->create(window)) {
        delete preview;
        return 0;
    }
    return reinterpret_cast<jlong>(preview);
}

JNIEXPORT jboolean JNICALL Java_com_pocketscanner_capture_VkPreview_nativeAdopt(
    JNIEnv* env, jobject, jlong handle, jobject surface) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr) return JNI_FALSE;
    ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
    if (window == nullptr) return JNI_FALSE;
    return preview->adopt(window) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeDetach(
    JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview != nullptr) preview->detach();
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeResize(
    JNIEnv*, jobject, jlong handle, jint width, jint height) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview != nullptr) preview->resize(width, height);
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeUploadYuyv(
    JNIEnv* env, jobject, jlong handle, jbyteArray data, jint width, jint height, jint stride) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr || data == nullptr) return;
    jbyte* bytes = env->GetByteArrayElements(data, nullptr);
    if (bytes == nullptr) return;
    preview->uploadYuyv(reinterpret_cast<const uint8_t*>(bytes), env->GetArrayLength(data), width, height, stride);
    env->ReleaseByteArrayElements(data, bytes, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeUploadRgb(
    JNIEnv* env, jobject, jlong handle, jbyteArray data, jint width, jint height, jint stride, jint format) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr || data == nullptr) return;
    jbyte* bytes = env->GetByteArrayElements(data, nullptr);
    if (bytes == nullptr) return;
    preview->uploadRgb(reinterpret_cast<const uint8_t*>(bytes), env->GetArrayLength(data), width, height, stride, format);
    env->ReleaseByteArrayElements(data, bytes, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSetMode(
    JNIEnv*, jobject, jlong handle, jint mode) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview != nullptr) preview->setMode(mode);
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSetFrontier(
    JNIEnv*, jobject, jlong handle, jboolean enabled) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview != nullptr) preview->setFrontier(enabled == JNI_TRUE);
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSetView(
    JNIEnv* env, jobject, jlong handle, jfloatArray pose, jfloat fx, jfloat fy, jfloat cx, jfloat cy,
    jint width, jint height) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr || pose == nullptr) return;
    jfloat values[16];
    if (env->GetArrayLength(pose) < 16) return;
    env->GetFloatArrayRegion(pose, 0, 16, values);
    preview->setView(values, fx, fy, cx, cy, width, height);
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeMeshConfigure(
    JNIEnv*, jobject, jlong handle, jfloat voxelSize, jfloat maxDepth) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview != nullptr) preview->meshConfigure(voxelSize, maxDepth);
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeMeshUpsert(
    JNIEnv* env, jobject, jlong handle, jint id, jfloatArray pose, jfloatArray depth, jbyteArray color,
    jint width, jint height, jfloat fx, jfloat fy, jfloat cx, jfloat cy, jboolean force) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr || pose == nullptr || depth == nullptr || color == nullptr) return;
    if (env->GetArrayLength(pose) < 16) return;
    const jsize pixels = width * height;
    if (pixels <= 0 || env->GetArrayLength(depth) < pixels || env->GetArrayLength(color) < pixels * 3) return;
    jfloat poseValues[16];
    env->GetFloatArrayRegion(pose, 0, 16, poseValues);
    jfloat* depthValues = env->GetFloatArrayElements(depth, nullptr);
    jbyte* colorValues = env->GetByteArrayElements(color, nullptr);
    if (depthValues != nullptr && colorValues != nullptr) {
        preview->meshUpsert(id, poseValues, depthValues, reinterpret_cast<const uint8_t*>(colorValues),
                            width, height, fx, fy, cx, cy, force == JNI_TRUE);
    }
    if (depthValues != nullptr) env->ReleaseFloatArrayElements(depth, depthValues, JNI_ABORT);
    if (colorValues != nullptr) env->ReleaseByteArrayElements(color, colorValues, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeMeshErase(
    JNIEnv*, jobject, jlong handle, jint id) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview != nullptr) preview->meshErase(id);
}

JNIEXPORT jint JNICALL Java_com_pocketscanner_capture_VkPreview_nativeChunkCount(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    return preview == nullptr ? 0 : preview->chunkCount();
}

JNIEXPORT jint JNICALL Java_com_pocketscanner_capture_VkPreview_nativeChunkPending(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    return preview == nullptr ? 0 : preview->chunkPending();
}

JNIEXPORT jint JNICALL Java_com_pocketscanner_capture_VkPreview_nativeChunkUndrawn(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    return preview == nullptr ? 0 : preview->chunkUndrawn();
}

JNIEXPORT jint JNICALL Java_com_pocketscanner_capture_VkPreview_nativeChunkCap(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    return preview == nullptr ? 0 : preview->chunkCap();
}

JNIEXPORT jboolean JNICALL Java_com_pocketscanner_capture_VkPreview_nativeRedraw(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr) return JNI_FALSE;
    return preview->redraw() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jfloatArray JNICALL Java_com_pocketscanner_capture_VkPreview_nativeCopySurface(
    JNIEnv* env, jobject, jlong handle, jfloatArray voxelOut) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr) return nullptr;
    std::vector<float> data;
    float voxel = 0.f;
    preview->copySurface(data, voxel);
    if (voxelOut != nullptr && env->GetArrayLength(voxelOut) > 0) env->SetFloatArrayRegion(voxelOut, 0, 1, &voxel);
    jfloatArray out = env->NewFloatArray(static_cast<jsize>(data.size()));
    if (out != nullptr && !data.empty()) env->SetFloatArrayRegion(out, 0, static_cast<jsize>(data.size()), data.data());
    return out;
}

JNIEXPORT jboolean JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSaveMesh(
    JNIEnv* env, jobject, jlong handle, jstring path) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr || path == nullptr) return JNI_FALSE;
    const char* chars = env->GetStringUTFChars(path, nullptr);
    bool ok = chars != nullptr && preview->saveMesh(chars);
    if (chars != nullptr) env->ReleaseStringUTFChars(path, chars);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_pocketscanner_capture_VkPreview_nativeLoadMesh(
    JNIEnv* env, jobject, jlong handle, jstring path, jfloat depthMeters) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr || path == nullptr) return JNI_FALSE;
    const char* chars = env->GetStringUTFChars(path, nullptr);
    bool ok = chars != nullptr && preview->loadMesh(chars, depthMeters);
    if (chars != nullptr) env->ReleaseStringUTFChars(path, chars);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeDropMesh(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview != nullptr) preview->dropMesh();
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeDropQueue(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview != nullptr) preview->dropQueue();
}

JNIEXPORT jboolean JNICALL Java_com_pocketscanner_capture_VkPreview_nativeCaptureSeed(
    JNIEnv*, jobject, jlong handle, jint cap, jboolean collectCoverage) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    return preview != nullptr && preview->captureSeed(cap, collectCoverage == JNI_TRUE) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_pocketscanner_capture_VkPreview_nativeCaptureSeedFromFile(
    JNIEnv* env, jobject, jlong handle, jstring path, jint cap, jboolean collectCoverage) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr || path == nullptr) return JNI_FALSE;
    const char* chars = env->GetStringUTFChars(path, nullptr);
    const bool ok = chars != nullptr && preview->captureSeedFromFile(chars, cap, collectCoverage == JNI_TRUE);
    if (chars != nullptr) env->ReleaseStringUTFChars(path, chars);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jfloatArray JNICALL Java_com_pocketscanner_capture_VkPreview_nativeTakeSeedCoverage(
    JNIEnv* env, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr) return nullptr;
    const auto samples = preview->takeSeedCoverage();
    if (samples.empty()) return nullptr;
    const auto size = static_cast<jsize>(samples.size());
    jfloatArray result = env->NewFloatArray(size);
    if (result != nullptr) env->SetFloatArrayRegion(result, 0, size, samples.data());
    return result;
}

JNIEXPORT jstring JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSplatError(JNIEnv* env, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    const char* text = preview == nullptr ? "" : preview->splatError();
    return env->NewStringUTF(text == nullptr ? "" : text);
}

JNIEXPORT jstring JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSplatHyper(JNIEnv* env, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    const char* text = preview == nullptr ? "{}" : preview->splatHyper();
    return env->NewStringUTF(text == nullptr ? "{}" : text);
}

JNIEXPORT jint JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSplatGaussianCount(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    return preview == nullptr ? 0 : preview->splatGaussianCount();
}

JNIEXPORT jfloat JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSeedVoxel(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    return preview == nullptr ? 0.f : preview->seedVoxel();
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSplatConfigure(
    JNIEnv*, jobject, jlong handle, jboolean longTraining, jint interval, jfloat threshold) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview != nullptr) preview->splatConfigure(longTraining == JNI_TRUE, interval, threshold);
}

JNIEXPORT jboolean JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSplatSetResolution(
    JNIEnv*, jobject, jlong handle, jint width, jint height) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    return preview != nullptr && preview->splatSetResolution(width, height) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSplatInitFromSeed(
    JNIEnv*, jobject, jlong handle, jint views, jint width, jint height, jint budget, jint meanSteps, jint gridSteps) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    return preview != nullptr && preview->splatInitFromSeed(views, width, height, budget, meanSteps, gridSteps) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSplatInit(
    JNIEnv* env, jobject, jlong handle, jfloatArray verts, jfloat voxel, jint views, jint width, jint height, jint budget, jint meanSteps, jint gridSteps) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr || verts == nullptr) return JNI_FALSE;
    jfloat* values = env->GetFloatArrayElements(verts, nullptr);
    int count = env->GetArrayLength(verts) / 9;
    bool ok = values != nullptr && preview->splatInit(values, count, voxel, views, width, height, budget, meanSteps, gridSteps);
    if (values != nullptr) env->ReleaseFloatArrayElements(verts, values, JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSplatStep(
    JNIEnv* env, jobject, jlong handle, jfloatArray rgb, jfloatArray depth, jfloatArray pose, jfloat fx, jfloat fy, jfloat cx, jfloat cy,
    jint view, jboolean updateGauss, jboolean updateGrid, jboolean metricsAfter, jfloatArray metrics) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr || rgb == nullptr || pose == nullptr || metrics == nullptr) return JNI_FALSE;
    const int pixels = preview->splatWidth() * preview->splatHeight();
    if (pixels <= 0 || env->GetArrayLength(rgb) != pixels * 3 || env->GetArrayLength(pose) < 16 ||
        (depth != nullptr && env->GetArrayLength(depth) != pixels) ||
        env->GetArrayLength(metrics) < SplatTrainer::kMetricCount) return JNI_FALSE;
    jfloat poseValues[16];
    env->GetFloatArrayRegion(pose, 0, 16, poseValues);
    jfloat* image = env->GetFloatArrayElements(rgb, nullptr);
    jfloat* depthValues = depth == nullptr ? nullptr : env->GetFloatArrayElements(depth, nullptr);
    float out[SplatTrainer::kMetricCount]{};
    bool ok = image != nullptr && (depth == nullptr || depthValues != nullptr) &&
              preview->splatStep(image, depthValues, poseValues, fx, fy, cx, cy, view, updateGauss == JNI_TRUE,
                                                      updateGrid == JNI_TRUE, metricsAfter == JNI_TRUE, out);
    if (image != nullptr) env->ReleaseFloatArrayElements(rgb, image, JNI_ABORT);
    if (depthValues != nullptr) env->ReleaseFloatArrayElements(depth, depthValues, JNI_ABORT);
    if (ok) env->SetFloatArrayRegion(metrics, 0, SplatTrainer::kMetricCount, out);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSplatRender(
    JNIEnv* env, jobject, jlong handle, jfloatArray pose, jfloat fx, jfloat fy, jfloat cx, jfloat cy, jboolean corrected) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr || pose == nullptr) return nullptr;
    int pixels = preview->splatWidth() * preview->splatHeight();
    if (pixels <= 0) return nullptr;
    jfloat poseValues[16];
    env->GetFloatArrayRegion(pose, 0, 16, poseValues);
    std::vector<uint8_t> rgb(static_cast<size_t>(pixels) * 3);
    bool ok = corrected == JNI_TRUE
        ? preview->splatRead(rgb.data())
        : preview->splatRender(poseValues, fx, fy, cx, cy, rgb.data());
    if (!ok) return nullptr;
    jbyteArray out = env->NewByteArray(static_cast<jsize>(rgb.size()));
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(rgb.size()), reinterpret_cast<const jbyte*>(rgb.data()));
    return out;
}

JNIEXPORT jboolean JNICALL Java_com_pocketscanner_capture_VkPreview_nativeSplatWritePly(
    JNIEnv* env, jobject, jlong handle, jstring path) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr || path == nullptr) return JNI_FALSE;
    const char* chars = env->GetStringUTFChars(path, nullptr);
    bool ok = chars != nullptr && preview->splatWritePly(chars);
    if (chars != nullptr) env->ReleaseStringUTFChars(path, chars);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr) return;
    preview->destroy();
    delete preview;
}

} // extern "C"
