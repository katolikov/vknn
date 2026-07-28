// The engine's VERSION file at the repo root is the single source for the version: the native
// library compiles the same string in, so the app's label and the engine it loads cannot drift.
// versionCode stays here -- it is a Play-store ordinal, not a version number.
val vknnVersion = rootProject.file("../VERSION").readText().trim()

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.vknn.chat"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.vknn.chat"
        minSdk = 28
        targetSdk = 34
        versionCode = 15
        versionName = vknnVersion
        ndk {
            // Only ship the arm64 native lib (the only ABI vknn is built for here).
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
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
        // VERSION_NAME/VERSION_CODE are shown in the Library screen next to the engine version the
        // native library reports, so the app half of that pair has to be readable at runtime.
        buildConfig = true
    }
    // Store the prebuilt libvknnchat.so uncompressed and page-aligned (useLegacyPackaging = false) so the
    // loader mmaps it directly — required for 16-KB-page-size devices; the .so is already release-built.
    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.09.02")
    implementation(composeBom)
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.6")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.6")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.animation:animation")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")

    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}
