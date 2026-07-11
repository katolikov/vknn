package com.vknn.chat.model

import org.json.JSONObject

// Parses the remote `catalog.json` manifest into ModelSpecs and merges it over the built-in list.
// Pure functions, no I/O — CatalogRepository owns the fetch/cache and calls these.
//
// Manifest shape: { "version": 1, "models": [ { <ModelSpec fields> }, ... ] }. Parsing is defensive:
// one malformed entry never drops its siblings, and a document that is not a well-formed manifest
// yields an empty list so the caller keeps whatever it already had.
object CatalogManifest {

    // Decode a manifest document. `knownDialectIds` are the chat dialects the app can actually render;
    // a Chat entry naming an unknown dialect is skipped rather than offered as an un-chattable model.
    fun parse(json: String, knownDialectIds: Set<String>): List<ModelSpec> {
        val root = try {
            JSONObject(json)
        } catch (e: Exception) {
            return emptyList()
        }
        val models = root.optJSONArray("models") ?: return emptyList()
        val out = ArrayList<ModelSpec>(models.length())
        for (i in 0 until models.length()) {
            val entry = models.optJSONObject(i) ?: continue
            parseEntry(entry, knownDialectIds)?.let(out::add)
        }
        return out
    }

    // Union by id with the remote entry winning on collision: a manifest edit can both add a model and
    // repoint/update an existing one. Built-in order is preserved (an overridden id keeps its position);
    // ids only in the remote list are appended.
    fun merge(builtin: List<ModelSpec>, remote: List<ModelSpec>): List<ModelSpec> {
        val byId = LinkedHashMap<String, ModelSpec>(builtin.size + remote.size)
        builtin.forEach { byId[it.id] = it }
        remote.forEach { byId[it.id] = it }
        return byId.values.toList()
    }

    // One entry -> a ModelSpec, or null if it is unusable. Required: id, displayName, mode, repoId,
    // repoFile, and at least one of sha256 / approxBytes (the download integrity + sizing anchor).
    private fun parseEntry(o: JSONObject, knownDialectIds: Set<String>): ModelSpec? {
        val id = o.optString("id").ifBlank { return null }
        val displayName = o.optString("displayName").ifBlank { return null }
        val mode = o.optString("mode").ifBlank { return null }
        val repoId = o.optString("repoId").ifBlank { return null }
        val repoFile = o.optString("repoFile").ifBlank { return null }
        val sha256 = o.optString("sha256").ifBlank { null }
        if (sha256 == null && !o.has("approxBytes")) return null

        val chatDialectId = o.optString("chatDialectId").ifBlank { null }
        if (mode == "Chat" && chatDialectId != null && chatDialectId !in knownDialectIds) return null

        val auxArray = o.optJSONArray("auxFiles")
        val auxFiles = if (auxArray == null) {
            emptyList()
        } else {
            (0 until auxArray.length()).mapNotNull { auxArray.optString(it).ifBlank { null } }
        }
        return ModelSpec(
            id = id,
            displayName = displayName,
            mode = mode,
            variant = o.optString("variant"),
            description = o.optString("description"),
            repoId = repoId,
            repoFile = repoFile,
            approxBytes = o.optLong("approxBytes", 0L),
            sha256 = sha256,
            auxFiles = auxFiles,
            chatDialectId = chatDialectId,
        )
    }
}
