plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "dev.femustafa.voicedictation"
    compileSdk = 35

    defaultConfig {
        applicationId = "dev.femustafa.voicedictation"
        minSdk = 26
        targetSdk = 35
        versionCode = 2
        versionName = "1.1"

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
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
        // They are the same shared runtime loaded by name, so keep exactly one copy.
        // onnxruntime-android is declared first so its 1.24.3 build wins: it is the
        // newer runtime (a superset of the C API sherpa 1.13.5 needs) and provides the
        // ai.onnxruntime JNI symbols the DolphinAttnEngine compiles against.
        jniLibs {
            pickFirsts += listOf(
                "**/libonnxruntime.so",
            )
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
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
    // (see packaging.jniLibs.pickFirsts), keeping one shared ORT runtime that serves
    // both sherpa-onnx and the DolphinAttnEngine.
    implementation(libs.onnxruntime.android)

    implementation("com.github.k2-fsa.sherpa-onnx:sherpa-onnx:1.13.5")

    implementation(libs.whisper.android)

    debugImplementation(libs.androidx.ui.tooling)

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
}
