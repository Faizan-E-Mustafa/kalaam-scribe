pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // sherpa-onnx publishes its Android AAR as a GitHub release asset (its
        // JitPack build merely re-hosts that file), so fetch the official binary
        // directly instead of depending on a build service. Keep its version in
        // lockstep with onnxruntime-android in libs.versions.toml: sherpa-onnx
        // 1.13.4 is built and linked against ORT 1.27.0.
        ivy {
            name = "sherpa-onnx-github-releases"
            url = uri("https://github.com/k2-fsa/sherpa-onnx/releases/download/")
            content {
                includeModule("com.github.k2-fsa.sherpa-onnx", "sherpa-onnx")
            }
            patternLayout {
                artifact("v[revision]/[module]-[revision].aar")
            }
            metadataSources {
                artifact()
            }
        }
    }
}

rootProject.name = "Kalaam Scribe"
include(":app")
