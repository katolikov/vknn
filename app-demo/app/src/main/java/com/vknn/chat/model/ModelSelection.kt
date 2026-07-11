package com.vknn.chat.model

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

// One selectable model file in a mode's variant picker: a catalogue entry (downloadable, possibly
// already on disk) or an ad-hoc .vxm found in the app's model directories (a pre-seeded build).
data class ModelChoice(
    val key: String,       // persisted selection key: a catalogue id, or "local:<fileName>"
    val displayName: String,
    val variant: String,   // "fp16" / "int4"; inferred from the file name for ad-hoc files
    val sizeBytes: Long,   // catalogue size, or the on-disk length for ad-hoc files
    val onDevice: Boolean, // the file is present and loadable right now
    val spec: ModelSpec?,  // the catalogue entry behind this choice; null for an ad-hoc file
)

// The per-mode model-variant choice (Chat, VLM), persisted in SharedPreferences — the same settings
// store as BackendSetting. Changing the choice takes effect immediately: the mode's view model
// releases any resident session and reflects the chosen file's on-disk state.
//
// A selection key is a catalogue id ("qwen", "qwen_int4_prefill", ...) or "local:<fileName>" for an
// ad-hoc .vxm sitting in the app's model directories (ModelStore scans both the primary external
// dir and the legacy files/ dir, so run-as pre-seeded files are selectable too). A persisted
// catalogue key whose entry no longer exists (or moved modes) falls back to the mode's default.
class ModelSelection(
    context: Context,
    private val store: ModelStore,
    private val catalog: StateFlow<List<ModelSpec>>,
) {
    private val preferences: SharedPreferences =
        context.getSharedPreferences("vknn_settings", Context.MODE_PRIVATE)

    private val _chatKey = MutableStateFlow(validKey(preferences.getString(KEY_CHAT_MODEL, null), ModelCatalog.QWEN, catalog.value))
    val chatKey: StateFlow<String> = _chatKey.asStateFlow()

    private val _vlmKey = MutableStateFlow(validKey(preferences.getString(KEY_VLM_MODEL, null), ModelCatalog.SMOLVLM2, catalog.value))
    val vlmKey: StateFlow<String> = _vlmKey.asStateFlow()

    fun setChatKey(key: String) = put(KEY_CHAT_MODEL, key, _chatKey)

    fun setVlmKey(key: String) = put(KEY_VLM_MODEL, key, _vlmKey)

    /** Every selectable .vxm for [mode]: the catalogue variants plus ad-hoc files on disk. */
    fun choicesFor(mode: String): List<ModelChoice> {
        val catalogue = catalog.value.forMode(mode).map { spec ->
            ModelChoice(
                key = spec.id,
                displayName = spec.displayName,
                variant = spec.variant,
                sizeBytes = spec.approxBytes,
                onDevice = store.isReady(spec),
                spec = spec,
            )
        }
        // Ad-hoc files are offered only in the mode their name identifies: a VLM tab must never list a
        // Qwen text decoder (it has no vision bucket and its load fails), and the Chat tab must never
        // list a SmolVLM file. A file whose family the name does not reveal falls to Chat — a generic
        // text decoder is the safe default there, whereas VLM needs a specific vision model.
        val adHoc = store.adHocModelFiles()
            .filter { file ->
                val inferred = modeFromFileName(file.name)
                inferred == mode || (inferred == null && mode == ModelCatalog.QWEN.mode)
            }
            .map { file ->
                ModelChoice(
                    key = LOCAL_PREFIX + file.name,
                    displayName = file.name,
                    variant = variantFromFileName(file.name),
                    sizeBytes = file.length(),
                    onDevice = true,
                    spec = null,
                )
            }
        return catalogue + adHoc
    }

    /**
     * The [ModelSpec] behind a selection key: the catalogue entry, or a synthesized spec wrapping
     * an ad-hoc file (its id is the selection key, its size the on-disk length — enough for the
     * CPU-admission check, load-error copy, and [ModelStore.file] resolution). [fallback] covers a
     * stale catalogue key.
     */
    fun specFor(key: String, fallback: ModelSpec): ModelSpec {
        if (key.startsWith(LOCAL_PREFIX)) {
            val fileName = key.removePrefix(LOCAL_PREFIX)
            val file = store.adHocModelFiles().firstOrNull { it.name == fileName }
            return ModelSpec(
                id = key,
                displayName = fileName,
                mode = fallback.mode,
                variant = variantFromFileName(fileName),
                description = "Local model file",
                repoId = "",
                repoFile = fileName,
                approxBytes = file?.length() ?: 0L,
                sha256 = null,
            )
        }
        return catalog.value.byId(key) ?: fallback
    }

    /** True when [key]'s file is on disk right now (catalogue state, or ad-hoc file presence). */
    fun isPresent(key: String): Boolean {
        if (key.startsWith(LOCAL_PREFIX)) {
            val fileName = key.removePrefix(LOCAL_PREFIX)
            return store.adHocModelFiles().any { it.name == fileName }
        }
        return catalog.value.byId(key)?.let(store::isReady) == true
    }

    private fun put(prefKey: String, value: String, target: MutableStateFlow<String>) {
        preferences.edit().putString(prefKey, value).apply()
        target.value = value
    }

    companion object {
        const val LOCAL_PREFIX = "local:"
        private const val KEY_CHAT_MODEL = "chat_model_key"
        private const val KEY_VLM_MODEL = "vlm_model_key"

        /** The weight-format label for an ad-hoc file, read from its name. */
        fun variantFromFileName(name: String): String = if (name.contains("int4", ignoreCase = true)) "int4" else "fp16"

        /**
         * The app mode an ad-hoc .vxm file's NAME identifies, or null when the name reveals no
         * family. Used to keep a file out of the wrong tab's picker (a Qwen text model must never
         * appear in the VLM tab). Matches the catalogue's own naming families.
         */
        fun modeFromFileName(name: String): String? {
            val lower = name.lowercase()
            return when {
                lower.contains("smolvlm") || lower.contains("vlm") || lower.contains("idefics") -> ModelCatalog.SMOLVLM2.mode
                // Splat before Chat: "encoder" contains the "coder" substring the chat match uses.
                lower.contains("splat") || lower.contains("encoder") || lower.contains("yonosplat") -> ModelCatalog.DL3DV.mode
                lower.contains("qwen") || lower.contains("coder") || lower.contains("instruct") ||
                    lower.contains("decode") || lower.contains("chat") -> ModelCatalog.QWEN.mode
                else -> null
            }
        }

        /**
         * A persisted selection key that is still meaningful for [fallback]'s mode: a catalogue key
         * must name an entry of the same mode; a local key must name a file whose family matches (an
         * unnamed-family local file is allowed only in the Chat mode, mirroring choicesFor). Anything
         * else — including a stale cross-mode selection from before this filtering existed — reverts
         * to the mode's default.
         */
        fun validKey(persisted: String?, fallback: ModelSpec, catalog: List<ModelSpec>): String = when {
            persisted == null -> fallback.id
            persisted.startsWith(LOCAL_PREFIX) -> {
                val inferred = modeFromFileName(persisted.removePrefix(LOCAL_PREFIX))
                if (inferred == fallback.mode || (inferred == null && fallback.mode == ModelCatalog.QWEN.mode)) persisted else fallback.id
            }
            catalog.byId(persisted)?.mode == fallback.mode -> persisted
            else -> fallback.id
        }

        /** The per-variant engine cache file name; distinct variants never share a cache. */
        fun cacheFileName(spec: ModelSpec): String =
            spec.id.map { ch -> if (ch.isLetterOrDigit() || ch == '.' || ch == '_' || ch == '-') ch else '_' }
                .joinToString("") + ".cache"
    }
}
