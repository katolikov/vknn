# Unified build entry point for VKNN on Windows (PowerShell mirror of build.sh).
#
#   .\build.ps1                 host build  - CPU backend + IR + ONNX import + tools + tests
#   .\build.ps1 --android       Android arm64-v8a build (Vulkan backend, NDK toolchain; needs ninja)
#   .\build.ps1 --clean         remove build-host + build-android and stop (clean only, no build)
#   .\build.ps1 --convert       build only the model compiler (vknn_compile) for the chosen target
#   .\build.ps1 --test          build and run the host unit tests only (fast; skips examples/tools)
#   .\build.ps1 --docs          build the static documentation site (open docs/site/index.html)
#
# --clean alone just cleans and exits. Combined with a build flag it cleans that target's dir first,
# then builds - e.g.  .\build.ps1 --android --clean   or   .\build.ps1 --clean --convert
# Override the NDK with $env:ANDROID_NDK, the API level with $env:ANDROID_API.
#
# Toolchain: any CMake generator works. When ninja is on PATH it is preferred (single-config, same
# build-dir layout as build.sh); otherwise the Visual Studio generator is used and binaries land in
# build-host\Release\. MinGW-w64 and MSVC are both supported host compilers. --leakcheck is
# POSIX-only (ASan/LeakSanitizer on Linux, `leaks` on macOS) and is refused here.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Fail([string]$Message) {
    [Console]::Error.WriteLine($Message)
    exit 1
}

function Remove-Tree([string[]]$Paths) {
    # rm -rf under `set -e` fails loudly on a locked file; mirror that. -ErrorAction Ignore would
    # silently keep a half-deleted tree (and its CMakeCache generator choice) alive.
    foreach ($p in $Paths) {
        if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        if (Test-Path $p) { Fail "build.ps1: could not remove $p (files in use?)" }
    }
}

$android = $false; $clean = $false; $convertOnly = $false; $docs = $false; $test = $false
foreach ($a in $args) {
    switch ($a) {
        "--android"   { $android = $true }
        "--clean"     { $clean = $true }
        "--convert"   { $convertOnly = $true }
        "--test"      { $test = $true }
        "--docs"      { $docs = $true }
        "--leakcheck" {
            Fail "build.ps1: --leakcheck is POSIX-only (Linux ASan/LSan/UBSan, macOS 'leaks'); run it under WSL or on a Linux/macOS host"
        }
        { $_ -in "-h", "--help" } {
            Get-Content $PSCommandPath | Select-Object -First 17 | ForEach-Object { $_ -replace '^# ?', '' }
            exit 0
        }
        default { Fail "build.ps1: unknown flag '$a' (try --help)" }
    }
}

function Invoke-Checked {
    # Run an external command and fail the script on a non-zero exit (PowerShell does not stop
    # for native commands on its own the way `set -e` does).
    param([string]$Exe, [string[]]$Arguments, [switch]$Quiet)
    if ($Quiet) { & $Exe @Arguments | Out-Null } else { & $Exe @Arguments }
    if ($LASTEXITCODE -ne 0) { Fail "build.ps1: $Exe failed (exit $LASTEXITCODE)" }
}

if ($clean -and -not ($android -or $convertOnly -or $docs -or $test)) {
    Write-Host ">> clean: removing build-host and build-android"
    Remove-Tree build-host, build-android
    exit 0
}

# Find a Python for the docs site (and anything else scripted): the py launcher, then python3/python.
function Find-Python {
    foreach ($candidate in @("py", "python3", "python")) {
        if (Get-Command $candidate -ErrorAction Ignore) { return $candidate }
    }
    Fail "build.ps1: python not found on PATH (needed for this step)"
}

# --docs: build the static documentation site into docs/site (entry point: docs/site/index.html).
# Doxygen is optional/secondary - if installed, also emit the C++ API reference, linked from the site.
if ($docs) {
    if (Get-Command doxygen -ErrorAction Ignore) {
        Write-Host ">> generating API reference -> docs/api/html (doxygen)"
        Invoke-Checked doxygen @("docs/Doxyfile") -Quiet
    } else {
        Write-Host ">> doxygen not found; skipping the optional API reference"
    }
    Write-Host ">> building documentation site -> docs/site"
    Invoke-Checked (Find-Python) @("$Root/scripts/gen_site.py")
    Write-Host ">> open docs/site/index.html"
    exit 0
}

$jobs = if ($env:NUMBER_OF_PROCESSORS) { $env:NUMBER_OF_PROCESSORS } else { [Environment]::ProcessorCount }
$haveNinja = [bool](Get-Command ninja -ErrorAction Ignore)

# Host configure arguments: ninja when available (single-config, artifacts in build-host\ exactly
# like build.sh); otherwise CMake's default generator (Visual Studio, multi-config). An already
# configured build dir keeps its generator - CMake refuses to switch one in place.
function HostConfigArgs([string]$BuildDir) {
    $cfg = @("-DCMAKE_BUILD_TYPE=Release")
    if ($haveNinja -and -not (Test-Path "$BuildDir/CMakeCache.txt")) { $cfg = @("-G", "Ninja") + $cfg }
    return $cfg
}

# Where a just-built host binary landed: build-host\ for single-config generators,
# build-host\Release\ for multi-config (Visual Studio). After a generator switch both layouts
# can hold a binary, so the NEWEST existing candidate is the one this build produced.
function HostBinary([string]$BuildDir, [string]$Name) {
    # @(...) keeps a single surviving candidate an ARRAY: a bare pipeline result unrolls one match
    # to a plain string, whose [0] is its first character.
    $found = @(@("$BuildDir/$Name.exe", "$BuildDir/Release/$Name.exe", "$BuildDir/$Name") |
            Where-Object { Test-Path $_ } |
            Sort-Object { (Get-Item $_).LastWriteTimeUtc } -Descending)
    if ($found.Count -gt 0) { return $found[0] }
    Fail "build.ps1: $Name not found under $BuildDir"
}

function Build([string]$BuildDir, [string]$Target) {
    # --config selects the configuration under multi-config generators (Visual Studio) and is
    # silently ignored by single-config ones, so it is always passed.
    $buildArgs = @("--build", $BuildDir, "-j", "$jobs", "--config", "Release")
    if ($Target) { $buildArgs += @("--target", $Target) }
    Invoke-Checked cmake $buildArgs
}

# --test: build and run just the host unit tests (the vknn_tests target, which pulls in the vknn
# library). Skips the examples/tools/convert targets, so it is the fast inner loop for test work.
if ($test) {
    $buildDir = "build-host"
    if ($clean) { Write-Host ">> clean: removing $buildDir"; Remove-Tree $buildDir }
    Write-Host ">> VKNN host unit tests (build + run)"
    Invoke-Checked cmake (@("-S", ".", "-B", $buildDir) + (HostConfigArgs $buildDir)) -Quiet
    Build $buildDir "vknn_tests"
    Write-Host ">> running vknn_tests"
    Invoke-Checked (HostBinary $buildDir "vknn_tests") @()
    exit 0
}

if ($android) {
    if (-not $env:ANDROID_NDK) { $env:ANDROID_NDK = "$env:LOCALAPPDATA\Android\Sdk\ndk\27.0.12077973" }
    if (-not (Test-Path $env:ANDROID_NDK)) { Fail "ERROR: NDK not found at $env:ANDROID_NDK (set `$env:ANDROID_NDK)" }
    if (-not $haveNinja) { Fail "ERROR: the Android build needs ninja on PATH (winget install Ninja-build.Ninja)" }
    $buildDir = "build-android"
    $api = if ($env:ANDROID_API) { $env:ANDROID_API } else { "33" }
    $config = @("-G", "Ninja",
        "-DCMAKE_TOOLCHAIN_FILE=$env:ANDROID_NDK\build\cmake\android.toolchain.cmake",
        "-DANDROID_ABI=arm64-v8a",
        "-DANDROID_PLATFORM=android-$api",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DVKNN_ENABLE_VULKAN=ON",
        "-DVKNN_ENABLE_NEON=ON")
    Write-Host ">> VKNN Android build (arm64-v8a, NDK $env:ANDROID_NDK)"
} else {
    $buildDir = "build-host"
    $config = HostConfigArgs $buildDir
    Write-Host ">> VKNN host build"
}

if ($clean) { Write-Host ">> clean: removing $buildDir"; Remove-Tree $buildDir }

# Vendored shader compiler: build glslang (third_party/glslang) for the host once. It is a build-time
# tool (GLSL -> SPIR-V on this machine, not the device), so it is always built natively even for the
# Android target; CMake then prefers it over a system glslc. Only the Vulkan (Android) build compiles
# shaders, so skip it for the CPU-only host build. If the submodule is absent, CMake falls back to glslc.
$glslangDir = "$Root/third_party/glslang"
$glslangBin = "$glslangDir/build-host/StandAlone/glslang.exe"
if ($android -and (Test-Path "$glslangDir/CMakeLists.txt") -and -not (Test-Path $glslangBin)) {
    Write-Host ">> building vendored shader compiler (glslang, host)"
    Invoke-Checked cmake @("-S", $glslangDir, "-B", "$glslangDir/build-host", "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_OPT=OFF", "-DGLSLANG_TESTS=OFF", "-DBUILD_EXTERNAL=OFF") -Quiet
    Invoke-Checked cmake @("--build", "$glslangDir/build-host", "-j", "$jobs", "--target", "glslang-standalone") -Quiet
}

Invoke-Checked cmake (@("-S", ".", "-B", $buildDir) + $config) -Quiet
if ($convertOnly) {
    Write-Host ">> building model compiler (vknn_compile)"
    Build $buildDir "vknn_compile"
} else {
    Build $buildDir ""
}
Write-Host ">> artifacts in $buildDir/"
