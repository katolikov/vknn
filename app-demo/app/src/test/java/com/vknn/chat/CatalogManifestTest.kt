package com.vknn.chat

import com.vknn.chat.model.CatalogManifest
import com.vknn.chat.model.ModelSpec
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

// Unit tests for the pure catalog manifest parser + merge. No Android / no network.
class CatalogManifestTest {
    private val known = setOf("qwen", "llama3")

    private fun entry(
        id: String = "m1",
        mode: String = "Chat",
        repoId: String = "o/r",
        repoFile: String = "m.vxm",
        sha256: String? = "abc",
        approxBytes: Long? = 123L,
        dialect: String? = "llama3",
    ): String {
        val parts = mutableListOf(
            """"id":"$id"""", """"displayName":"D"""", """"mode":"$mode"""",
            """"variant":"int4"""", """"description":"desc"""",
            """"repoId":"$repoId"""", """"repoFile":"$repoFile"""",
            """"auxFiles":["vocab.json","merges.txt"]""",
        )
        if (sha256 != null) parts.add(""""sha256":"$sha256"""")
        if (approxBytes != null) parts.add(""""approxBytes":$approxBytes""")
        if (dialect != null) parts.add(""""chatDialectId":"$dialect"""")
        return "{" + parts.joinToString(",") + "}"
    }

    private fun doc(vararg entries: String) = """{"version":1,"models":[${entries.joinToString(",")}]}"""

    private fun spec(id: String, repoId: String = "o/r") = ModelSpec(
        id = id, displayName = "D", mode = "Chat", variant = "int4", description = "d",
        repoId = repoId, repoFile = "$id.vxm", approxBytes = 1, sha256 = "x", chatDialectId = "llama3",
    )

    @Test fun parsesValidEntry() {
        val specs = CatalogManifest.parse(doc(entry()), known)
        assertEquals(1, specs.size)
        val s = specs[0]
        assertEquals("m1", s.id)
        assertEquals("Chat", s.mode)
        assertEquals("o/r", s.repoId)
        assertEquals("m.vxm", s.repoFile)
        assertEquals(123L, s.approxBytes)
        assertEquals("abc", s.sha256)
        assertEquals(listOf("vocab.json", "merges.txt"), s.auxFiles)
        assertEquals("llama3", s.chatDialectId)
    }

    @Test fun skipsEntryMissingRepoIdButKeepsSibling() {
        val bad = """{"id":"bad","displayName":"D","mode":"Chat","variant":"int4","description":"d","repoFile":"m.vxm","sha256":"abc","auxFiles":[]}"""
        val specs = CatalogManifest.parse(doc(bad, entry(id = "good")), known)
        assertEquals(listOf("good"), specs.map { it.id })
    }

    @Test fun skipsChatEntryWithUnknownDialect() {
        assertTrue(CatalogManifest.parse(doc(entry(dialect = "mistral")), known).isEmpty())
    }

    @Test fun keepsChatEntryWithKnownDialect() {
        assertEquals(1, CatalogManifest.parse(doc(entry(dialect = "qwen")), known).size)
    }

    @Test fun keepsEntryWithNullDialect() {
        val specs = CatalogManifest.parse(doc(entry(mode = "VLM", dialect = null)), known)
        assertEquals(1, specs.size)
        assertNull(specs[0].chatDialectId)
    }

    @Test fun keepsEntryWithApproxBytesButNoSha() {
        assertEquals(1, CatalogManifest.parse(doc(entry(sha256 = null)), known).size)
    }

    @Test fun skipsEntryWithNeitherShaNorApproxBytes() {
        assertTrue(CatalogManifest.parse(doc(entry(sha256 = null, approxBytes = null)), known).isEmpty())
    }

    @Test fun malformedTopLevelReturnsEmpty() {
        assertTrue(CatalogManifest.parse("not json at all", known).isEmpty())
        assertTrue(CatalogManifest.parse("""{"version":1}""", known).isEmpty())
    }

    @Test fun mergeUnionsByIdRemoteAppendedAfterBuiltin() {
        val merged = CatalogManifest.merge(listOf(spec("a"), spec("b")), listOf(spec("c")))
        assertEquals(listOf("a", "b", "c"), merged.map { it.id })
    }

    @Test fun mergeRemoteOverridesBuiltinById() {
        val merged = CatalogManifest.merge(listOf(spec("a", repoId = "old")), listOf(spec("a", repoId = "new")))
        assertEquals(1, merged.size)
        assertEquals("new", merged[0].repoId)
    }

    // The committed seed manifest (served via GitHub raw) must be well-formed: every entry parses (none
    // silently dropped as malformed / unknown-dialect), so the app never ships a catalog.json a build
    // would partly ignore. Content is intentionally NOT pinned to BUILTIN — the remote may diverge.
    @Test fun seedCatalogJsonIsWellFormed() {
        val file = findUp("catalog.json")
        assertNotNull("catalog.json not found relative to the test working dir", file)
        val text = file!!.readText()
        val rawCount = JSONObject(text).getJSONArray("models").length()
        val parsed = CatalogManifest.parse(text, known)
        assertTrue("catalog.json must list at least one model", parsed.isNotEmpty())
        assertEquals("every catalog.json entry must be well-formed (none skipped)", rawCount, parsed.size)
        assertEquals("ids must be unique", parsed.size, parsed.map { it.id }.toSet().size)
    }

    private fun findUp(name: String): File? {
        var dir: File? = File(".").absoluteFile
        repeat(8) {
            File(dir, name).takeIf { it.exists() }?.let { return it }
            dir = dir?.parentFile
        }
        return null
    }
}
