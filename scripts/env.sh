# Source this before Android builds:
#   source /home/alex/dev/pocket-scanner/scripts/env.sh
export ANDROID_HOME=/home/alex/dev/android-sdk
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/28.2.13676358"
export JAVA_HOME=/home/alex/dev/jdk-21
export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/cmake/3.31.6/bin:/home/alex/dev/slang/bin:$JAVA_HOME/bin:$PATH"
