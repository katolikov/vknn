package com.vknn.chat.model

import java.util.Locale

// The downloadable .vxm catalogue, one or more variants per app mode. A ModelSpec is the single
// place a model's HuggingFace location, size, and pinned digest live — repointing a model to a new
// upload means editing the one entry below.
data class ModelSpec(
    val id: String,             // stable key: state-map entry and persisted variant choice, never shown to the user
    val displayName: String,
    val mode: String,           // the app mode this model powers (library card subtitle, variant-picker role)
    val variant: String,        // weight format shown in the variant picker: "fp16" / "int4"
    val description: String,
    val repoId: String,         // HuggingFace model repo
    val repoFile: String,       // path inside the repo, fetched via /resolve/main/<repoFile>
    val approxBytes: Long,      // display + storage-guard size until the HF API reports the exact length
    val sha256: String?,        // pinned digest, used when the HF API is unreachable (null = API only)
    val auxFiles: List<String> = emptyList(), // small companion repo files (tokenizer etc.), fetched with the model
) {
    val url: String get() = "https://huggingface.co/$repoId/resolve/main/$repoFile"
    val localFileName: String get() = repoFile.substringAfterLast('/')
    fun auxUrl(name: String): String = "https://huggingface.co/$repoId/resolve/main/$name"
    fun auxLocalName(name: String): String = "$id.${name.substringAfterLast('/')}"
}

object ModelCatalog {
    val QWEN = ModelSpec(
        id = "qwen",
        displayName = "Qwen2.5-Coder 0.5B Instruct",
        mode = "Chat",
        variant = "fp16",
        description = "Instruction-tuned coder LLM behind Chat: answers questions rather than continuing them. Decodes entirely on the GPU (Vulkan) with a 1024-token context.",
        repoId = "katolikov/qwen-vknn",
        repoFile = "qwen2.5-coder-0.5b-instruct-c1024.vxm",
        approxBytes = 1_261_062_785L,
        sha256 = "5bb34db8b39645d5b095ceeaa435af68af6674e5c404ce224cd8d06d86c4836d",
    )

    val QWEN_INT4_PREFILL = ModelSpec(
        id = "qwen_int4_prefill",
        displayName = "Qwen2.5-Coder 0.5B Instruct (int4 + fast prefill)",
        mode = "Chat",
        variant = "int4",
        description = "The same instruct coder with int4-quantized weights and a 256-token prefill bucket: a quarter of the download, faster decode, and whole-window prompt ingestion.",
        repoId = "katolikov/qwen-vknn",
        repoFile = "qwen2.5-coder-0.5b-instruct-c1024-prefill256-int4.vxm",
        approxBytes = 541_745_785L,
        sha256 = "d433fc40bab476f44b25943d1c667149cf610eba627abf98fc95d58ea837eb89",
    )

    val QWEN_FP16_PREFILL = ModelSpec(
        id = "qwen_fp16_prefill",
        displayName = "Qwen2.5-Coder 0.5B Instruct (fp16 + fast prefill)",
        mode = "Chat",
        variant = "fp16",
        description = "The same instruct coder at full fp16 weights with a 256-token prefill bucket: reference quality plus whole-window prompt ingestion.",
        repoId = "katolikov/qwen-vknn",
        repoFile = "qwen2.5-coder-0.5b-instruct-c1024-prefill256.vxm",
        approxBytes = 1_265_737_329L,
        sha256 = "d76e51ec1d84bf123b2bbf00de73558c22cd5df82a4e088b05ec5e7c187107b0",
    )

    val SMOLVLM2 = ModelSpec(
        id = "smolvlm2",
        displayName = "SmolVLM2 2.2B",
        mode = "VLM",
        variant = "fp16",
        description = "Vision-language model behind the camera coach: describes and reasons about what the camera sees, on the GPU.",
        repoId = "katolikov/smolvlm2-vknn",
        repoFile = "smolvlm2-2.2b-fp16.vxm",
        approxBytes = 4_494_762_454L,
        sha256 = "8253752554ba36629bea5c0c7868c35cfcd73609eb02090f52bcaceaaf83fe3d",
        auxFiles = listOf("vocab.json", "merges.txt"), // the SmolVLM2 tokenizer rides along with the model
    )

    val DL3DV = ModelSpec(
        id = "dl3dv",
        displayName = "YoNoSplat encoder",
        mode = "3D Splat",
        variant = "fp16",
        description = "Feed-forward 3D-Gaussian-splatting encoder (dl3dv): turns 8 camera views into a splat scene.",
        repoId = "katolikov/yonosplat-vknn",
        repoFile = "dl3dv/encoder8_fp16.vxm",
        approxBytes = 2_336_982_453L,
        sha256 = "406dfebef5f9135af2085ec586f10ff7efbe8eb76c919242f64c464b13144835",
    )

    val ALL = listOf(QWEN, QWEN_INT4_PREFILL, QWEN_FP16_PREFILL, SMOLVLM2, DL3DV)

    fun byId(id: String): ModelSpec? = ALL.firstOrNull { it.id == id }

    /** The catalogue variants powering one app mode, in catalogue order. */
    fun forMode(mode: String): List<ModelSpec> = ALL.filter { it.mode == mode }
}

// Decimal units, matching how the model cards quote sizes ("1.3 GB", "450 MB").
fun formatBytes(bytes: Long): String = when {
    bytes >= 1_000_000_000L -> String.format(Locale.US, "%.1f GB", bytes / 1e9)
    else -> "${bytes / 1_000_000} MB"
}
