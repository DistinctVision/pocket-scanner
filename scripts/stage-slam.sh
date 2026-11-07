#!/bin/sh
# Copy the arm64 SLAM runtime into the app and strip it. The vocabulary is an asset.
set -eu
ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
PREFIX="${STELLA_PREFIX:-/home/alex/dev/android-prefix/arm64}"
NDK="${ANDROID_NDK_HOME:-/home/alex/dev/android-sdk/ndk/28.2.13676358}"
STRIP="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"
FBOW="${FBOW_LIBRARY:-/tmp/stella-android/3rd/FBoW/libfbow.so}"
VOCAB_SRC="${ORB_VOCAB:-/tmp/orb_vocab.fbow}"
DEST="$ROOT/android/app/src/main/slam-libs/arm64-v8a"
JNI="$ROOT/android/app/src/main/jniLibs/arm64-v8a"
OPENCV="$PREFIX/sdk/native/libs/arm64-v8a"
mkdir -p "$DEST" "$JNI" "$ROOT/android/app/src/main/assets"

stage() {
    src="$1"
    name="$(basename "$src")"
    if [ ! -f "$src" ]; then
        echo "missing $src" >&2
        exit 1
    fi
    cp -f "$src" "$DEST/$name"
    "$STRIP" --strip-unneeded "$DEST/$name"
    echo "$name $(wc -c < "$DEST/$name") bytes"
}

stage "$PREFIX/lib/libstella_vslam.so"
if [ -f "$PREFIX/lib/libfbow.so" ]; then
    stage "$PREFIX/lib/libfbow.so"
else
    stage "$FBOW"
fi
stage "$PREFIX/lib/libyaml-cpp.so"
stage "$PREFIX/lib/libsqlite3.so"
stage "$PREFIX/lib/libcxsparse.so"
for name in \
    libg2o_core.so libg2o_stuff.so libg2o_types_sba.so libg2o_types_sim3.so \
    libg2o_types_slam3d.so libg2o_solver_dense.so libg2o_solver_eigen.so \
    libg2o_solver_csparse.so libg2o_csparse_extension.so
do
    stage "$PREFIX/lib/$name"
done
for name in \
    libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so \
    libopencv_videoio.so libopencv_features2d.so libopencv_flann.so \
    libopencv_calib3d.so libopencv_highgui.so libopencv_objdetect.so
do
    stage "$OPENCV/$name"
done
stage "$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/19/lib/linux/aarch64/libomp.so"

# pocket_slam links these three, so AGP packages them from CMake. The rest are
# loaded because libstella_vslam.so needs them.
rm -rf "$JNI"
mkdir -p "$JNI"
for lib in "$DEST"/*.so; do
    name="$(basename "$lib")"
    case "$name" in
        libstella_vslam.so|libopencv_imgproc.so|libopencv_core.so) ;;
        *) cp -f "$lib" "$JNI/$name" ;;
    esac
done

if [ ! -f "$VOCAB_SRC" ]; then
    echo "missing vocab $VOCAB_SRC" >&2
    exit 1
fi
cp -f "$VOCAB_SRC" "$ROOT/android/app/src/main/assets/orb_vocab.fbow"
echo "vocab $(wc -c < "$ROOT/android/app/src/main/assets/orb_vocab.fbow") bytes"
