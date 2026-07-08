package com.vknn.chat.model

import kotlinx.coroutines.CompletableDeferred

/**
 * Keeps at most one heavy model resident across the app's modes (Chat, VLM, 3D Splat). Each mode's
 * native session owns a full Vulkan device plus gigabytes of weights, so two resident models exceed
 * device memory. Every load therefore goes through [acquireResidency], which frees whichever mode
 * currently holds the slot before the new load allocates anything.
 *
 * Protocol for a mode (a [Holder], implemented by the mode's view model):
 *  - [acquireResidency] before loading. On return the mode owns the slot and counts as busy with
 *    the load; it calls [nativeCallSettled] once the load attempt finishes, success or failure.
 *  - [nativeCallStarted] / [nativeCallSettled] bracket every later native call, so a release never
 *    frees a session mid-call. After [nativeCallSettled] returns, the mode may no longer own the
 *    slot ([isResident] tells): a release requested during the call has then already freed the
 *    session and reset the mode's UI, and the caller stops publishing results.
 *  - [releaseResidentModel] is the user-facing Unload and the library-delete path. No-op unless
 *    the mode owns the slot, so a double Unload is safe.
 *  - [dropResidency] when the mode disposed its session on its own (failed load, view-model
 *    teardown); it unblocks pending acquirers without running the release callbacks.
 *
 * Confinement: every method runs on the app's main dispatcher (the view models' home), so the
 * class needs no locks. [Holder.freeResidentSession] suspends precisely so the native free can
 * hop to a worker dispatcher.
 */
class ModelResidency {

    /** One mode's residency hooks. */
    interface Holder {
        /** A release is waiting on the holder's in-flight native call; streaming loops stop early. */
        fun onReleaseRequested() {}

        /** Frees the mode's native session (worker dispatcher inside). In-flight calls have settled. */
        suspend fun freeResidentSession()

        /** Returns the mode's UI to its pre-load state: model on disk, nothing resident. */
        fun resetToUnloadedState()
    }

    private enum class ReleasePhase { NONE, REQUESTED, RUNNING }

    private var residentHolder: Holder? = null
    private var residentBusy = false
    private var releasePhase = ReleasePhase.NONE
    private val releaseWaiters = mutableListOf<CompletableDeferred<Unit>>()

    /** True while [holder] owns the residency slot. */
    fun isResident(holder: Holder): Boolean = residentHolder === holder

    /**
     * Grants [holder] the residency slot. The current resident (if any) is freed first — deferred
     * until its in-flight native call settles — and this returns only once that session is gone.
     * On return the holder is resident and busy with its load; it must call [nativeCallSettled]
     * when the load attempt ends.
     */
    suspend fun acquireResidency(holder: Holder) {
        while (true) {
            val occupant = residentHolder ?: break
            if (occupant === holder) {
                residentBusy = true
                return
            }
            requestReleaseAndAwait()
        }
        residentHolder = holder
        residentBusy = true
    }

    /**
     * Frees [holder]'s resident model and resets its UI — the Unload action and the library-delete
     * path. Defers until any in-flight native call settles; returns once the session is freed.
     * No-op when [holder] does not own the slot (double Unload, unload of a never-loaded mode).
     */
    suspend fun releaseResidentModel(holder: Holder) {
        if (residentHolder !== holder) return
        requestReleaseAndAwait()
    }

    /** Marks the resident holder's native call in flight; releases defer to [nativeCallSettled]. */
    fun nativeCallStarted(holder: Holder) {
        if (residentHolder === holder) residentBusy = true
    }

    /**
     * Marks the holder's native call settled and performs a release requested while it ran. When
     * this returns with [isResident] false, the session is freed and the mode's UI reset.
     */
    suspend fun nativeCallSettled(holder: Holder) {
        if (residentHolder !== holder) return
        residentBusy = false
        if (releasePhase == ReleasePhase.REQUESTED) releaseResidentNow()
    }

    /**
     * Clears [holder]'s slot after it disposed its session on its own (failed load, teardown).
     * Pending acquirers unblock; the release callbacks do not run.
     */
    fun dropResidency(holder: Holder) {
        if (residentHolder !== holder) return
        residentHolder = null
        residentBusy = false
        if (releasePhase == ReleasePhase.RUNNING) return // the running release completes the waiters
        releasePhase = ReleasePhase.NONE
        completeReleaseWaiters()
    }

    // Joins (or starts) the release of the current resident and suspends until it completes. A
    // resident mid-native-call is only marked (and told via onReleaseRequested, so streaming loops
    // stop early); nativeCallSettled then performs the release.
    private suspend fun requestReleaseAndAwait() {
        val waiter = CompletableDeferred<Unit>()
        releaseWaiters += waiter
        if (releasePhase == ReleasePhase.NONE) {
            if (residentBusy) {
                releasePhase = ReleasePhase.REQUESTED
                residentHolder?.onReleaseRequested()
            } else {
                releaseResidentNow() // completes every waiter, including the one above
            }
        }
        waiter.await()
    }

    private suspend fun releaseResidentNow() {
        val holder = residentHolder
        if (holder == null) {
            releasePhase = ReleasePhase.NONE
            completeReleaseWaiters()
            return
        }
        releasePhase = ReleasePhase.RUNNING
        holder.freeResidentSession()
        holder.resetToUnloadedState()
        if (residentHolder === holder) { // dropResidency may have cleared it during the free
            residentHolder = null
            residentBusy = false
        }
        releasePhase = ReleasePhase.NONE
        completeReleaseWaiters()
    }

    private fun completeReleaseWaiters() {
        // Drain before completing: a resumed waiter can re-enter and enqueue a fresh waiter.
        val completed = releaseWaiters.toList()
        releaseWaiters.clear()
        completed.forEach { it.complete(Unit) }
    }
}
