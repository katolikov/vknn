package com.vknn.chat

import com.vknn.chat.model.ModelCatalog
import com.vknn.chat.model.ModelSelection
import org.junit.Assert.assertEquals
import org.junit.Test

// Validates the pure selection-key rules the variant picker persists through: catalogue keys must
// stay within their mode, local keys pass through, and cache file names are variant-scoped and
// filesystem-safe. Runs on the JVM: ./gradlew :app:testDebugUnitTest.
class ModelSelectionTest {

    @Test
    fun persistedCatalogueKeyOfTheSameModeIsKept() {
        assertEquals(
            ModelCatalog.QWEN_INT4_PREFILL.id,
            ModelSelection.validKey(ModelCatalog.QWEN_INT4_PREFILL.id, ModelCatalog.QWEN),
        )
    }

    @Test
    fun missingOrForeignKeysFallBackToTheModeDefault() {
        assertEquals(ModelCatalog.QWEN.id, ModelSelection.validKey(null, ModelCatalog.QWEN))
        assertEquals(ModelCatalog.QWEN.id, ModelSelection.validKey("removed_model", ModelCatalog.QWEN))
        // A VLM catalogue id persisted under the chat mode never selects a VLM model for Chat.
        assertEquals(ModelCatalog.QWEN.id, ModelSelection.validKey(ModelCatalog.SMOLVLM2.id, ModelCatalog.QWEN))
    }

    @Test
    fun localKeysPassThroughValidation() {
        val key = ModelSelection.LOCAL_PREFIX + "fresh-build.vxm"
        assertEquals(key, ModelSelection.validKey(key, ModelCatalog.QWEN))
    }

    @Test
    fun variantReadsFromTheFileName() {
        assertEquals("int4", ModelSelection.variantFromFileName("qwen-c1024-prefill256-INT4.vxm"))
        assertEquals("fp16", ModelSelection.variantFromFileName("qwen-c1024-prefill256.vxm"))
    }

    @Test
    fun cacheFileNamesAreVariantScopedAndFilesystemSafe() {
        // The pre-variant release cached the chat model as qwen.cache; the id-derived name keeps it.
        assertEquals("qwen.cache", ModelSelection.cacheFileName(ModelCatalog.QWEN))
        assertEquals("qwen_int4_prefill.cache", ModelSelection.cacheFileName(ModelCatalog.QWEN_INT4_PREFILL))
        assertEquals("smolvlm2.cache", ModelSelection.cacheFileName(ModelCatalog.SMOLVLM2))
        val local = ModelSelection.LOCAL_PREFIX + "fresh build.vxm"
        val spec = ModelCatalog.QWEN.copy(id = local)
        assertEquals("local_fresh_build.vxm.cache", ModelSelection.cacheFileName(spec))
    }
}
