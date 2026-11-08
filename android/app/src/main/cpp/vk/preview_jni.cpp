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
    jint width, jint height, jfloat fx, jfloat fy, jfloat cx, jfloat cy) {
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
                            width, height, fx, fy, cx, cy);
    }
    if (depthValues != nullptr) env->ReleaseFloatArrayElements(depth, depthValues, JNI_ABORT);
    if (colorValues != nullptr) env->ReleaseByteArrayElements(color, colorValues, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeMeshErase(
    JNIEnv*, jobject, jlong handle, jint id) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview != nullptr) preview->meshErase(id);
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeRedraw(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview != nullptr) preview->redraw();
}

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr) return;
    preview->destroy();
    delete preview;
}

} // extern "C"
