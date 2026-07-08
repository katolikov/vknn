package com.vknn.chat

import com.vknn.chat.ui.SuggestionCardState
import com.vknn.chat.ui.suggestionCardStateFor
import com.vknn.chat.ui.suggestionMetricsCaption
import com.vknn.chat.vlm.VlmPhase
import com.vknn.chat.vlm.VlmUiState
import org.junit.Assert.assertEquals
import org.junit.Test

// Validates how the VLM suggestion overlay picks its presentation from the view-model state, and the
// latency caption it renders once a turn finishes.
class VlmCardStateTest {
    @Test
    fun liveCameraShowsTheHint() {
        assertEquals(
            SuggestionCardState.HINT,
            suggestionCardStateFor(VlmUiState(phase = VlmPhase.CAMERA)),
        )
    }

    @Test
    fun answeringWithNoTokensYetIsThinking() {
        assertEquals(
            SuggestionCardState.THINKING,
            suggestionCardStateFor(VlmUiState(phase = VlmPhase.ANSWERING, answer = "", generating = true)),
        )
    }

    @Test
    fun answeringWithTokensIsStreaming() {
        assertEquals(
            SuggestionCardState.STREAMING,
            suggestionCardStateFor(VlmUiState(phase = VlmPhase.ANSWERING, answer = "Move closer", generating = true)),
        )
    }

    @Test
    fun finishedGenerationIsDone() {
        assertEquals(
            SuggestionCardState.DONE,
            suggestionCardStateFor(VlmUiState(phase = VlmPhase.ANSWERING, answer = "Move closer", generating = false)),
        )
    }

    // The failure check outranks the phase check: a status with no generation in flight is an error
    // in every phase, so a relayed load or encode failure never renders as a hint or a finished answer.
    @Test
    fun aStatusWithNoGenerationInFlightIsAlwaysAnError() {
        assertEquals(
            SuggestionCardState.ERROR,
            suggestionCardStateFor(VlmUiState(phase = VlmPhase.CAMERA, status = "vision encode failed")),
        )
        assertEquals(
            SuggestionCardState.ERROR,
            suggestionCardStateFor(
                VlmUiState(phase = VlmPhase.ANSWERING, answer = "Move closer", status = "prefill failed"),
            ),
        )
    }

    @Test
    fun captionJoinsTheMeasuredTermsOnly() {
        assertEquals(
            "TTFT 812 ms  ·  17.4 tok/s  ·  vulkan",
            suggestionMetricsCaption(Metrics(ttftMs = 812, tokPerSec = 17.44), "vulkan"),
        )
        assertEquals("vulkan", suggestionMetricsCaption(Metrics(), "vulkan"))
        assertEquals("", suggestionMetricsCaption(Metrics(), ""))
    }
}
