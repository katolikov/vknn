package com.vknn.chat

import com.vknn.chat.model.HfApi
import com.vknn.chat.model.ModelCatalog
import com.vknn.chat.model.formatBytes
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

// Validates the HF API JSON walk that yields a download's expected size + sha256, plus the catalogue
// invariants ModelStore relies on (unique ids and on-disk names, well-formed resolve URLs). Runs on
// the JVM: ./gradlew :app:testDebugUnitTest.
class ModelCatalogTest {
    // Shape of https://huggingface.co/api/models/<repo>?blobs=true: non-LFS files carry only a size,
    // LFS files add the lfs object whose sha256 the download verification uses.
    private val sample = """
        {"id":"katolikov/yonosplat-vknn","siblings":[
          {"rfilename":"README.md","blobId":"a","size":1867},
          {"rfilename":"dl3dv/encoder8_fp16.vxm","blobId":"b","size":2336982453,
           "lfs":{"sha256":"406dfebef5f9135af2085ec586f10ff7efbe8eb76c919242f64c464b13144835","size":2336982453,"pointerSize":135}}
        ]}
    """.trimIndent()

    @Test
    fun parsesLfsSizeAndSha() {
        val info = HfApi.parse(sample, "dl3dv/encoder8_fp16.vxm")!!
        assertEquals(2336982453L, info.size)
        assertEquals("406dfebef5f9135af2085ec586f10ff7efbe8eb76c919242f64c464b13144835", info.sha256)
    }

    @Test
    fun parsesNonLfsFileWithoutSha() {
        val info = HfApi.parse(sample, "README.md")!!
        assertEquals(1867L, info.size)
        assertNull(info.sha256)
    }

    @Test
    fun missingFileParsesToNull() {
        assertNull(HfApi.parse(sample, "not-there.vxm"))
        assertNull(HfApi.parse("""{"id":"x"}""", "README.md"))
    }

    @Test
    fun catalogueIdsAndLocalNamesAreUnique() {
        assertEquals(ModelCatalog.ALL.size, ModelCatalog.ALL.map { it.id }.toSet().size)
        assertEquals(ModelCatalog.ALL.size, ModelCatalog.ALL.map { it.localFileName }.toSet().size)
        for (spec in ModelCatalog.ALL) {
            assertTrue(spec.approxBytes > 0)
            assertTrue(!spec.localFileName.contains('/'))
            assertTrue(spec.variant == "fp16" || spec.variant == "int4")
        }
    }

    @Test
    fun chatModeCarriesThreeVariants() {
        val chat = ModelCatalog.forMode(ModelCatalog.QWEN.mode)
        assertEquals(listOf(ModelCatalog.QWEN, ModelCatalog.QWEN_INT4_PREFILL, ModelCatalog.QWEN_FP16_PREFILL), chat)
        assertEquals("int4", ModelCatalog.QWEN_INT4_PREFILL.variant)
        assertEquals("fp16", ModelCatalog.QWEN_FP16_PREFILL.variant)
        assertEquals(ModelCatalog.QWEN_INT4_PREFILL, ModelCatalog.byId("qwen_int4_prefill"))
        assertNull(ModelCatalog.byId("no-such-model"))
    }

    @Test
    fun resolveUrls() {
        assertEquals(
            "https://huggingface.co/katolikov/qwen-vknn/resolve/main/qwen2.5-coder-0.5b-instruct-c1024.vxm",
            ModelCatalog.QWEN.url,
        )
        assertEquals(
            "https://huggingface.co/katolikov/qwen-vknn/resolve/main/qwen2.5-coder-0.5b-instruct-c1024-prefill256-int4.vxm",
            ModelCatalog.QWEN_INT4_PREFILL.url,
        )
        assertEquals(
            "https://huggingface.co/katolikov/qwen-vknn/resolve/main/qwen2.5-coder-0.5b-instruct-c1024-prefill256.vxm",
            ModelCatalog.QWEN_FP16_PREFILL.url,
        )
        assertEquals(
            "https://huggingface.co/katolikov/yonosplat-vknn/resolve/main/dl3dv/encoder8_fp16.vxm",
            ModelCatalog.DL3DV.url,
        )
        assertEquals("encoder8_fp16.vxm", ModelCatalog.DL3DV.localFileName)
    }

    @Test
    fun formatsBytes() {
        assertEquals("1.3 GB", formatBytes(1_261_050_516L))
        assertEquals("4.5 GB", formatBytes(4_490_000_000L))
        assertEquals("450 MB", formatBytes(450_000_000L))
        assertEquals("0 MB", formatBytes(0L))
    }
}
