package com.vknn.chat.model

import com.vknn.chat.ChatPromptTemplate
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
    val chatDialectId: String? = null, // Chat models: the ChatPromptTemplate.Dialect id; null is the ChatML (Qwen) default
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

    val LLAMA = ModelSpec(
        id = "llama",
        displayName = "Llama 3.2 1B Instruct (int4)",
        mode = "Chat",
        variant = "int4",
        description = "Meta's instruction-tuned Llama 3.2 1B with int4-quantized weights: decodes entirely on the GPU (Vulkan). Its byte-level tokenizer downloads alongside the model.",
        repoId = "katolikov/Llama-3.2-1B-vknn",
        repoFile = "llama-3.2-1b-instruct-int4.vxm",
        approxBytes = 1_205_062_047L,
        sha256 = "cc6f4eb18479647a8cea7678e538a33bc8e0a1b3772fcc02aaecc63a98c60309",
        auxFiles = listOf("vocab.json", "merges.txt"), // the Llama-3 tokenizer rides along with the model
        chatDialectId = ChatPromptTemplate.LLAMA3.id,
    )

    val LLAMA31_8B = ModelSpec(
        id = "llama31_8b",
        displayName = "Llama 3.1 8B Instruct (int4)",
        mode = "Chat",
        variant = "int4",
        description = "Meta's instruction-tuned Llama 3.1 8B with int4-quantized weights: decodes entirely on the GPU (Vulkan). The largest model here — ~5.3 GB, ~6.4 GB peak GPU. Shares the Llama-3 tokenizer, which downloads alongside the model.",
        repoId = "katolikov/Llama-3.1-8B-vknn",
        repoFile = "llama-3.1-8b-instruct-int4.vxm",
        approxBytes = 5_306_875_173L,
        sha256 = "7a8ba6e8ca050660b97bc27f56d7612c6997c41e4ad90184e186d1bed943a035",
        auxFiles = listOf("vocab.json", "merges.txt"), // the Llama-3 tokenizer rides along with the model
        chatDialectId = ChatPromptTemplate.LLAMA3.id,
    )

    val SMOLVLM2 = ModelSpec(
        id = "smolvlm2",
        displayName = "SmolVLM2-2.2B-vknn",
        mode = "VLM",
        variant = "fp16",
        description = "Vision-language model behind the camera coach: describes and reasons about what the camera sees, fully on the GPU.",
        repoId = "katolikov/SmolVLM2-2.2B-vknn",
        repoFile = "smolvlm2-2.2b-fp16.vxm",
        approxBytes = 4_495_215_775L,
        sha256 = "c977ffdbc83ef3f13b9a66782d0dc15b44ae75849d75e566d3db5db71299dcab",
        auxFiles = listOf("vocab.json", "merges.txt"), // the SmolVLM2 tokenizer rides along with the model
    )

    val SMOLVLM2_INT4 = ModelSpec(
        id = "smolvlm2_int4",
        displayName = "SmolVLM2-2.2B-vknn (int4)",
        mode = "VLM",
        variant = "int4",
        description = "Int4-weight SmolVLM2 (calibration-free -Os quantization): the camera coach at ~1/3 the size, same full-GPU pipeline.",
        repoId = "katolikov/SmolVLM2-2.2B-vknn",
        repoFile = "smolvlm2-2.2b-i4.vxm",
        approxBytes = 1_354_375_152L,
        sha256 = "d8086425dcdde1dacf3173b45d1c7d50e1d241fd878593fd385125bc38c0bbad",
        auxFiles = listOf("vocab.json", "merges.txt"), // shares the SmolVLM2 tokenizer
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

    // The models compiled into the app: the offline / first-launch seed. The effective catalogue at
    // runtime is this list merged with a remote manifest (CatalogRepository), so a new model ships as a
    // catalog.json edit rather than an APK rebuild.
    val BUILTIN = listOf(QWEN, QWEN_INT4_PREFILL, QWEN_FP16_PREFILL, LLAMA, LLAMA31_8B, SMOLVLM2, SMOLVLM2_INT4, DL3DV)
}

/** The entry with this stable id, or null. Operates on the effective catalogue a caller holds. */
fun List<ModelSpec>.byId(id: String): ModelSpec? = firstOrNull { it.id == id }

/** The catalogue variants powering one app mode, in catalogue order. */
fun List<ModelSpec>.forMode(mode: String): List<ModelSpec> = filter { it.mode == mode }

// Decimal units, matching how the model cards quote sizes ("1.3 GB", "450 MB").
fun formatBytes(bytes: Long): String = when {
    bytes >= 1_000_000_000L -> String.format(Locale.US, "%.1f GB", bytes / 1e9)
    else -> "${bytes / 1_000_000} MB"
}
