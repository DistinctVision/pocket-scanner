#include <jni.h>

#include <cstdio>
#include <memory>
#include <string>

#include <android/log.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "stella_vslam/config.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/publish/frame_publisher.h"
#include "stella_vslam/publish/map_publisher.h"
#include "stella_vslam/system.h"

#include <algorithm>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "pocket-slam", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "pocket-slam", __VA_ARGS__)

namespace {

struct Tracker {
    std::shared_ptr<stella_vslam::system> slam;
};

std::string state_token(const std::shared_ptr<stella_vslam::system>& slam) {
    const std::string raw = slam->get_frame_publisher()->get_tracking_state();
    if (raw == "Tracking") return "tracking";
    if (raw == "Lost") return "lost";
    if (raw == "Initializing") return "initializing";
    return "no-pose";
}

void write_pose(JNIEnv* env, jfloatArray pose_out, const std::shared_ptr<stella_vslam::Mat44_t>& pose) {
    if (pose_out == nullptr || env->GetArrayLength(pose_out) < 17) return;
    jfloat packed[17] = {};
    if (pose) {
        packed[0] = 1.f;
        const stella_vslam::Mat44_t& m = *pose;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                packed[1 + row * 4 + col] = static_cast<float>(m(row, col));
            }
        }
    }
    env->SetFloatArrayRegion(pose_out, 0, 17, packed);
}

void throw_runtime(JNIEnv* env, const char* message) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), message);
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_pocketscanner_capture_StellaTracker_nativeStart(JNIEnv* env, jobject, jstring vocab, jstring config) {
    const char* vocab_chars = env->GetStringUTFChars(vocab, nullptr);
    const char* config_chars = env->GetStringUTFChars(config, nullptr);
    const std::string vocab_path = vocab_chars == nullptr ? "" : vocab_chars;
    const std::string config_path = config_chars == nullptr ? "" : config_chars;
    if (vocab_chars != nullptr) {
        env->ReleaseStringUTFChars(vocab, vocab_chars);
    }
    if (config_chars != nullptr) {
        env->ReleaseStringUTFChars(config, config_chars);
    }
    try {
        // Pyramid levels already run in parallel. A second OpenCV pool inside
        // each level only fights those threads.
        cv::setNumThreads(1);
        auto cfg = std::make_shared<stella_vslam::config>(config_path);
        auto tracker = std::make_unique<Tracker>();
        tracker->slam = std::make_shared<stella_vslam::system>(cfg, vocab_path);
        tracker->slam->startup();
        LOGI("stella started");
        return reinterpret_cast<jlong>(tracker.release());
    } catch (const std::exception& error) {
        LOGE("start: %s", error.what());
        throw_runtime(env, error.what());
        return 0;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pocketscanner_capture_StellaTracker_nativeFeed(
    JNIEnv* env, jobject, jlong handle, jbyteArray color, jint width, jint height, jint stride,
    jbyteArray depth, jint depth_w, jint depth_h, jint depth_stride, jdouble timestamp,
    jfloatArray pose_out) {
    auto* tracker = reinterpret_cast<Tracker*>(handle);
    if (tracker == nullptr || tracker->slam == nullptr) {
        throw_runtime(env, "tracker is not running");
        return nullptr;
    }
    write_pose(env, pose_out, nullptr);
    if (width != depth_w || height != depth_h || width <= 0 || height <= 0) {
        return env->NewStringUTF("color and depth sizes differ");
    }
    jbyte* color_ptr = env->GetByteArrayElements(color, nullptr);
    jbyte* depth_ptr = env->GetByteArrayElements(depth, nullptr);
    if (color_ptr == nullptr || depth_ptr == nullptr) {
        if (color_ptr != nullptr) {
            env->ReleaseByteArrayElements(color, color_ptr, JNI_ABORT);
        }
        if (depth_ptr != nullptr) {
            env->ReleaseByteArrayElements(depth, depth_ptr, JNI_ABORT);
        }
        throw_runtime(env, "missing frame buffer");
        return nullptr;
    }
    std::string line;
    try {
        cv::Mat yuyv(height, width, CV_8UC2, color_ptr, static_cast<size_t>(stride));
        cv::Mat gray;
        cv::cvtColor(yuyv, gray, cv::COLOR_YUV2GRAY_YUY2);
        cv::Mat depth_mat(depth_h, depth_w, CV_16UC1, depth_ptr, static_cast<size_t>(depth_stride));
        const auto pose = tracker->slam->feed_RGBD_frame(gray, depth_mat, timestamp);
        write_pose(env, pose_out, pose);
        line = state_token(tracker->slam);
    } catch (const std::exception& error) {
        LOGE("feed: %s", error.what());
        line = error.what();
    }
    env->ReleaseByteArrayElements(color, color_ptr, JNI_ABORT);
    env->ReleaseByteArrayElements(depth, depth_ptr, JNI_ABORT);
    return env->NewStringUTF(line.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_pocketscanner_capture_StellaTracker_nativeCopyKeyframes(
    JNIEnv* env, jobject, jlong handle, jintArray ids, jdoubleArray times, jfloatArray poses) {
    auto* tracker = reinterpret_cast<Tracker*>(handle);
    if (tracker == nullptr || tracker->slam == nullptr || ids == nullptr || times == nullptr || poses == nullptr) {
        return 0;
    }
    const jsize cap = std::min(env->GetArrayLength(ids), env->GetArrayLength(times));
    if (cap <= 0 || env->GetArrayLength(poses) < cap * 16) return 0;
    std::vector<std::shared_ptr<stella_vslam::data::keyframe>> keyframes;
    tracker->slam->get_map_publisher()->get_keyframes(keyframes);
    const int count = std::min(static_cast<int>(keyframes.size()), static_cast<int>(cap));
    std::vector<jint> idOut(count);
    std::vector<jdouble> timeOut(count);
    std::vector<jfloat> poseOut(static_cast<size_t>(count) * 16);
    for (int i = 0; i < count; ++i) {
        const auto& keyframe = keyframes[i];
        if (!keyframe || keyframe->will_be_erased()) {
            idOut[i] = -1;
            continue;
        }
        idOut[i] = static_cast<jint>(keyframe->id_);
        timeOut[i] = keyframe->timestamp_;
        const stella_vslam::Mat44_t pose = keyframe->get_pose_wc();
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                poseOut[static_cast<size_t>(i) * 16 + row * 4 + col] = static_cast<float>(pose(row, col));
            }
        }
    }
    env->SetIntArrayRegion(ids, 0, count, idOut.data());
    env->SetDoubleArrayRegion(times, 0, count, timeOut.data());
    env->SetFloatArrayRegion(poses, 0, count * 16, poseOut.data());
    return count;
}

extern "C" JNIEXPORT void JNICALL
Java_com_pocketscanner_capture_StellaTracker_nativeStop(JNIEnv*, jobject, jlong handle) {
    auto* tracker = reinterpret_cast<Tracker*>(handle);
    if (tracker == nullptr) {
        return;
    }
    try {
        if (tracker->slam) {
            tracker->slam->shutdown();
        }
    } catch (const std::exception& error) {
        LOGE("stop: %s", error.what());
    }
    delete tracker;
}
