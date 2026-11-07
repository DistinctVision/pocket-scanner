pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "pocket-capture"

include(":app")
include(":librealsense")
project(":librealsense").projectDir =
    file("/home/alex/dev/librealsense/wrappers/android/librealsense")

gradle.beforeProject {
    extra["defaultCompileSdkVersion"] = 35
    extra["defaultTargetSdkVersion"] = 35
    extra["defaultMinSdkVersion"] = 24
    extra["appcompatVersion"] = "1.6.1"
    extra["constraintlayoutVersion"] = "2.1.4"
    extra["materialVersion"] = "1.9.0"
    extra["junitVersion"] = "4.13.2"
    extra["androidextjunitVersion"] = "1.1.5"
    extra["espressoCoreVersion"] = "3.5.1"
}
