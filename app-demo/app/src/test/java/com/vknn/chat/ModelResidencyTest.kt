package com.vknn.chat

import com.vknn.chat.model.ModelResidency
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.yield
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

// Exercises the single-residency state machine exactly as the view models drive it: acquire before
// load, nativeCallStarted/nativeCallSettled around native work, releaseResidentModel for Unload,
// dropResidency for a failed load. runBlocking gives the same single-threaded confinement the app's
// main dispatcher gives, so waiter/defer ordering is deterministic.
class ModelResidencyTest {

    private class RecordingHolder(
        private val name: String,
        private val events: MutableList<String>,
    ) : ModelResidency.Holder {
        var freeCount = 0
        var resetCount = 0

        override fun onReleaseRequested() {
            events += "$name:releaseRequested"
        }

        override suspend fun freeResidentSession() {
            freeCount++
            events += "$name:freed"
        }

        override fun resetToUnloadedState() {
            resetCount++
            events += "$name:reset"
        }
    }

    @Test
    fun loadingSecondModeFreesResidentModeFirst() = runBlocking {
        val residency = ModelResidency()
        val events = mutableListOf<String>()
        val chatMode = RecordingHolder("chat", events)
        val vlmMode = RecordingHolder("vlm", events)

        residency.acquireResidency(chatMode)
        residency.nativeCallSettled(chatMode) // chat's load finished
        residency.acquireResidency(vlmMode)
        events += "vlm:loadStarts"

        assertEquals(listOf("chat:freed", "chat:reset", "vlm:loadStarts"), events)
        assertFalse(residency.isResident(chatMode))
        assertTrue(residency.isResident(vlmMode))
    }

    @Test
    fun unloadFreesSessionAndResetsState() = runBlocking {
        val residency = ModelResidency()
        val events = mutableListOf<String>()
        val chatMode = RecordingHolder("chat", events)

        residency.acquireResidency(chatMode)
        residency.nativeCallSettled(chatMode)
        residency.releaseResidentModel(chatMode)

        assertEquals(listOf("chat:freed", "chat:reset"), events)
        assertFalse(residency.isResident(chatMode))
    }

    @Test
    fun releaseDefersWhileNativeCallInFlightAndCompletesAfterItSettles() = runBlocking {
        val residency = ModelResidency()
        val events = mutableListOf<String>()
        val chatMode = RecordingHolder("chat", events)

        residency.acquireResidency(chatMode)
        residency.nativeCallSettled(chatMode)
        residency.nativeCallStarted(chatMode) // a decode is in flight

        val unload = launch {
            residency.releaseResidentModel(chatMode)
            events += "unload:returned"
        }
        yield() // the unload runs up to its deferred wait

        assertEquals(listOf("chat:releaseRequested"), events)
        assertTrue(residency.isResident(chatMode))
        assertEquals(0, chatMode.freeCount)

        residency.nativeCallSettled(chatMode) // the decode settles; the release lands here
        unload.join()

        assertEquals(listOf("chat:releaseRequested", "chat:freed", "chat:reset", "unload:returned"), events)
        assertFalse(residency.isResident(chatMode))
    }

    @Test
    fun acquireDuringResidentLoadDefersUntilThatLoadSettles() = runBlocking {
        val residency = ModelResidency()
        val events = mutableListOf<String>()
        val chatMode = RecordingHolder("chat", events)
        val vlmMode = RecordingHolder("vlm", events)

        residency.acquireResidency(chatMode) // chat's load is in flight (busy until settled)
        val vlmLoad = launch {
            residency.acquireResidency(vlmMode)
            events += "vlm:loadStarts"
        }
        yield()

        assertEquals(listOf("chat:releaseRequested"), events)
        assertTrue(residency.isResident(chatMode))

        residency.nativeCallSettled(chatMode) // chat's load lands; the release frees it
        vlmLoad.join()

        assertEquals(listOf("chat:releaseRequested", "chat:freed", "chat:reset", "vlm:loadStarts"), events)
        assertTrue(residency.isResident(vlmMode))
    }

    @Test
    fun doubleUnloadIsNoOp() = runBlocking {
        val residency = ModelResidency()
        val events = mutableListOf<String>()
        val chatMode = RecordingHolder("chat", events)

        residency.acquireResidency(chatMode)
        residency.nativeCallSettled(chatMode)
        residency.releaseResidentModel(chatMode)
        residency.releaseResidentModel(chatMode)

        assertEquals(1, chatMode.freeCount)
        assertEquals(1, chatMode.resetCount)
    }

    @Test
    fun concurrentUnloadsFreeOnce() = runBlocking {
        val residency = ModelResidency()
        val events = mutableListOf<String>()
        val chatMode = RecordingHolder("chat", events)

        residency.acquireResidency(chatMode)
        residency.nativeCallSettled(chatMode)
        residency.nativeCallStarted(chatMode)
        val firstUnload = launch { residency.releaseResidentModel(chatMode) }
        val secondUnload = launch { residency.releaseResidentModel(chatMode) }
        yield()
        residency.nativeCallSettled(chatMode)
        firstUnload.join()
        secondUnload.join()

        assertEquals(1, chatMode.freeCount)
        assertEquals(1, chatMode.resetCount)
        assertFalse(residency.isResident(chatMode))
    }

    @Test
    fun unloadOfNeverLoadedModeIsNoOp() = runBlocking {
        val residency = ModelResidency()
        val events = mutableListOf<String>()
        val chatMode = RecordingHolder("chat", events)

        residency.releaseResidentModel(chatMode)

        assertEquals(emptyList<String>(), events)
    }

    @Test
    fun dropResidencyUnblocksAcquirersWithoutReleaseCallbacks() = runBlocking {
        val residency = ModelResidency()
        val events = mutableListOf<String>()
        val chatMode = RecordingHolder("chat", events)
        val vlmMode = RecordingHolder("vlm", events)

        residency.acquireResidency(chatMode) // chat's load is in flight
        val vlmLoad = launch {
            residency.acquireResidency(vlmMode)
            events += "vlm:loadStarts"
        }
        yield()

        residency.dropResidency(chatMode) // chat's load failed; chat disposed its own session
        vlmLoad.join()

        assertEquals(0, chatMode.freeCount)
        assertEquals(0, chatMode.resetCount)
        assertTrue(residency.isResident(vlmMode))
        assertEquals(listOf("chat:releaseRequested", "vlm:loadStarts"), events)
    }

    @Test
    fun settledCallFromNonResidentHolderIsIgnored() = runBlocking {
        val residency = ModelResidency()
        val events = mutableListOf<String>()
        val chatMode = RecordingHolder("chat", events)
        val vlmMode = RecordingHolder("vlm", events)

        residency.acquireResidency(chatMode)
        residency.nativeCallStarted(vlmMode) // stale bracket from a mode that lost the slot
        residency.nativeCallSettled(vlmMode)

        assertTrue(residency.isResident(chatMode))
        assertEquals(emptyList<String>(), events)
    }

    @Test
    fun lastRequestedLoadWinsWhenTwoModesQueueBehindAResident() = runBlocking {
        val residency = ModelResidency()
        val events = mutableListOf<String>()
        val chatMode = RecordingHolder("chat", events)
        val vlmMode = RecordingHolder("vlm", events)
        val splatMode = RecordingHolder("splat", events)

        residency.acquireResidency(chatMode)
        residency.nativeCallSettled(chatMode)
        residency.nativeCallStarted(chatMode)

        val vlmLoad = launch {
            residency.acquireResidency(vlmMode)
            events += "vlm:loadStarts"
        }
        val splatLoad = launch {
            residency.acquireResidency(splatMode)
            events += "splat:loadStarts"
        }
        yield()
        residency.nativeCallSettled(chatMode) // chat's decode settles; its release lets vlm in
        vlmLoad.join()

        // vlm holds the slot, busy with its load; splat's request waits on that load.
        assertTrue(residency.isResident(vlmMode))
        assertEquals(
            listOf(
                "chat:releaseRequested",
                "chat:freed", "chat:reset",
                "vlm:loadStarts",
                "vlm:releaseRequested",
            ),
            events,
        )

        residency.nativeCallSettled(vlmMode) // vlm's load lands; splat's release frees it
        splatLoad.join()

        assertEquals(listOf("vlm:freed", "vlm:reset", "splat:loadStarts"), events.drop(5))
        assertFalse(residency.isResident(vlmMode))
        assertTrue(residency.isResident(splatMode))
    }
}
