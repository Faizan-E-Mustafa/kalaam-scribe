plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "dev.femustafa.kalaamscribe"
    compileSdk = 35

    signingConfigs {
        create("release") {
            storeFile = file("../app-signing.p12")
            storePassword = System.getenv("APK_SIGNING_PASSWORD")
            keyPassword = System.getenv("APK_SIGNING_PASSWORD")
            keyAlias = "kalaam-scribe"
        }
    }

    defaultConfig {
        applicationId = "dev.femustafa.kalaamscribe"
        minSdk = 26
        targetSdk = 35
        versionCode = 2
        versionName = "1.0.1"

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        // Both sherpa-onnx and onnxruntime-android ship their own libonnxruntime.so.
        // Keep exactly one copy per ABI (both are the ORT 1.27.0 build: sherpa-onnx
        // 1.13.4 bundles and links against ORT 1.27.0). The two must stay version-locked
        // — Android's linker matches ELF symbols by exact version (sherpa's
        // libsherpa-onnx-jni.so and onnxruntime's libonnxruntime4j_jni.so both require
        // OrtGetApiBase@VERS_1.27.0), so a mismatched runtime fails to load at runtime.
        // See docs/adr/0005-shared-libonnxruntime-version-lockstep.md.
        jniLibs {
            pickFirsts += listOf(
                "**/libonnxruntime.so",
            )
        }
    }

    testOptions {
        // android.util.Log and similar Android-only APIs return defaults (0/null) on the
        // JVM rather than throwing. Lets unit tests exercise code paths that log without
        // requiring a Robolectric or androidTest harness.
        unitTests.isReturnDefaultValues = true
        // Robolectric compose UI tests need the merged manifest + resources (the
        // same surface used for the device build), so layout bugs on small phones
        // are caught on the JVM.
        unitTests.isIncludeAndroidResources = true
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.splashscreen)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons)

    // onnxruntime-android first: its libonnxruntime.so wins the pickFirst merge below
    // (see packaging.jniLibs.pickFirsts). Its copy is baked in as the one shared ORT
    // runtime; it also carries the ai.onnxruntime JNI glue (libonnxruntime4j_jni.so)
    // that DolphinAttnEngine uses. Keep sherpaOnnx and onnxruntime in lockstep (both
    // 1.27.0) — see libs.versions.toml.
    implementation(libs.onnxruntime.android)

    // @aar: the sherpa-onnx GitHub-release ivy repo (settings.gradle.kts) serves a
    // bare AAR with no POM, so the extension must be pinned or AGP treats it as a JAR.
    implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx:${libs.versions.sherpaOnnx.get()}@aar")

    implementation(libs.whisper.android)

    // Pure-Java MP3 decoder for the file-upload transcription path.
    implementation(libs.jlayer)

    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(platform(libs.androidx.compose.bom))
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
    // JVM UI tests: Robolectric renders Compose at configurable phone sizes.
    testImplementation(platform(libs.androidx.compose.bom))
    testImplementation("androidx.compose.ui:ui-test-junit4")
    testImplementation(libs.androidx.test.core)
    testImplementation(libs.robolectric)
}
