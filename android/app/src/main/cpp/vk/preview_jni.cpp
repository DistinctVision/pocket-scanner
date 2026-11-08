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

JNIEXPORT void JNICALL Java_com_pocketscanner_capture_VkPreview_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    auto* preview = reinterpret_cast<Preview*>(handle);
    if (preview == nullptr) return;
    preview->destroy();
    delete preview;
}

} // extern "C"
