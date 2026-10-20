//
//  ChatLLMTests+Reasoning
//  ChatLLMTests
//
//  Split out of ChatLLMTests.swift; part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
import SwiftUI
import SwiftData
import WebKit
import UIKit
import FoundationModels
@testable import ChatLLM

extension ChatLLMTests {
    @Test func thoughtDurationFormatterUsesNaturalUnits() {
        #expect(ThoughtDurationFormatter.string(for: 8) == "8 seconds")
        #expect(ThoughtDurationFormatter.string(for: 60) == "1 minute")
        #expect(ThoughtDurationFormatter.string(for: 68) == "1 minute and 8 seconds")
        #expect(ThoughtDurationFormatter.string(for: 3_600) == "1 hour")
        #expect(ThoughtDurationFormatter.string(for: 3_608) == "1 hour and 8 seconds")
        #expect(ThoughtDurationFormatter.string(for: 3_960) == "1 hour and 6 minutes")
    }

    @Test func cancellingActiveReasoningCapturesElapsedThinkingTime() throws {
        let schema = Schema([Conversation.self, ChatLLM.Message.self, MessageAttachment.self])
        let container = try ModelContainer(
            for: schema,
            configurations: [ModelConfiguration(isStoredInMemoryOnly: true)]
        )
        let context = ModelContext(container)
        let conversation = Conversation(title: "Test")
        let message = Message(
            role: .assistant,
            text: "",
            order: 0,
            conversation: conversation,
            isFinal: false,
            reasoning: "Working through the request.",
            isReasoningMode: true
        )
        let startedAt = Date(timeIntervalSince1970: 1_000)
        let cancelledAt = startedAt.addingTimeInterval(21)
        message.beginGenerationCapture(backend: "Test", modelName: nil, startedAt: startedAt)
        conversation.messages = [message]

        let viewModel = makeIsolatedChatViewModel(
            generator: TestLLMGenerator(),
            context: context,
            conversation: conversation
        )
        viewModel.isGenerating = true
        viewModel.streamingMessageID = message.id

        viewModel.cancelGeneration(at: cancelledAt)

        #expect(message.reasoningCompletedAt == cancelledAt)
        #expect(message.reasoningDuration == 21)
        #expect(ThoughtDurationFormatter.string(for: message.reasoningDuration ?? 0) == "21 seconds")
    }

    @Test func reasoningTextSanitizerRemovesAllThinkingTagsAndExtraGaps() {
        let sanitized = ReasoningTextSanitizer.string(from: """
        First reasoning block.

        </think>

        <thinking>
        Second reasoning block.
        </thinking>
        """)

        #expect(sanitized == """
        First reasoning block.

        Second reasoning block.
        """)
        #expect(!sanitized.contains("</think>"))
        #expect(!sanitized.contains("<thinking>"))
    }

    @Test func thinkingShimmerPhaseDependsOnlyOnClockTime() {
        // A rebuilt indicator must resume the sweep where the old one was, so the
        // offset is a pure function of the date and wraps every 1.6s.
        let date = Date(timeIntervalSinceReferenceDate: 1_000.4)
        let offset = InlineThinkingView.shimmerOffset(at: date)
        #expect(offset == InlineThinkingView.shimmerOffset(at: date))
        #expect(abs(offset - InlineThinkingView.shimmerOffset(at: date.addingTimeInterval(1.6))) < 0.0001)
        #expect(offset >= -0.4 && offset < 1.4)
    }

    /// A date `phase` (0..<1) of the way through a shimmer sweep.
    private func shimmerDate(atPhase phase: Double) -> Date {
        Date(timeIntervalSinceReferenceDate: InlineThinkingView.shimmerPeriod * (1_000 + phase))
    }

    @Test func statusLabelSwapWaitsOnlyWhileTheGlowCrossesTheLabel() {
        // Band hasn't reached the label yet, or has already left it: swap right away.
        #expect(InlineThinkingView.remainingSweepDuration(at: shimmerDate(atPhase: 0.02)) == 0)
        #expect(InlineThinkingView.remainingSweepDuration(at: shimmerDate(atPhase: 0.95)) == 0)

        // Mid-sweep: wait until the band has left the trailing edge.
        let midSweep = shimmerDate(atPhase: 0.5)
        let remaining = InlineThinkingView.remainingSweepDuration(at: midSweep)
        #expect(remaining > 0 && remaining < InlineThinkingView.shimmerPeriod)
        let afterSweep = midSweep.addingTimeInterval(remaining + 0.001)
        #expect(InlineThinkingView.shimmerOffset(at: afterSweep) - 0.25 >= 1)
    }

    @Test func streamingActivityFollowsFirstTokenAndSearches() {
        #expect(StreamingActivity.current(hasReceivedOutput: false, isSearching: false) == .loading)
        #expect(StreamingActivity.current(hasReceivedOutput: true, isSearching: false) == .thinking)
        #expect(StreamingActivity.current(hasReceivedOutput: false, isSearching: true) == .searching)
        #expect(StreamingActivity.current(hasReceivedOutput: true, isSearching: true) == .searching)
    }

    @Test func streamingActivityKeepsLoadingUntilTheSweepFinishes() async throws {
        let viewModel = try makeChat([])
        #expect(viewModel.streamingActivity == .loading)

        // First token mid-sweep: "Loading…" stays up until the glow has passed.
        viewModel.requestStreamingActivity(.thinking, now: shimmerDate(atPhase: 0.5))
        #expect(viewModel.streamingActivity == .loading)

        // A search starting before the swap replaces the pending target.
        viewModel.requestStreamingActivity(.searching, now: shimmerDate(atPhase: 0.6))
        #expect(viewModel.streamingActivity == .loading)

        // Going back to the label already shown drops the pending swap.
        viewModel.requestStreamingActivity(.loading, now: shimmerDate(atPhase: 0.7))
        #expect(viewModel.pendingStreamingActivity == nil)

        // Between sweeps the swap is immediate.
        viewModel.requestStreamingActivity(.searching, now: shimmerDate(atPhase: 0.95))
        #expect(viewModel.streamingActivity == .searching)

        // Near the end of a sweep the swap lands shortly after.
        viewModel.requestStreamingActivity(.thinking, now: shimmerDate(atPhase: 0.85))
        #expect(viewModel.streamingActivity == .searching)
        try await Task.sleep(for: .milliseconds(400))
        #expect(viewModel.streamingActivity == .thinking)

        viewModel.resetStreamingActivity()
        #expect(viewModel.streamingActivity == .loading)
    }

    @Test func reasoningStatusShowsSearchesEvenAfterReasoningCompletes() {
        typealias Bubble = ReasoningMessageBubble
        #expect(Bubble.liveActivity(isStreaming: true, hasCompletedReasoning: false, activity: .loading) == .loading)
        #expect(Bubble.liveActivity(isStreaming: true, hasCompletedReasoning: false, activity: .thinking) == .thinking)
        #expect(Bubble.liveActivity(isStreaming: true, hasCompletedReasoning: true, activity: .searching) == .searching)
        // Finished thinking and not searching: fall back to "Thought for…".
        #expect(Bubble.liveActivity(isStreaming: true, hasCompletedReasoning: true, activity: .thinking) == nil)
        #expect(Bubble.liveActivity(isStreaming: false, hasCompletedReasoning: false, activity: .searching) == nil)
    }

    @Test func plainReplyShowsSearchingBelowItsTextOnlyWhileSearching() {
        let message = Message(role: .assistant, text: "Let me check.", order: 0, isFinal: false)
        #expect(MessageCellView.showsSearchingBelowText(message: message, isStreaming: true, activity: .searching))
        #expect(!MessageCellView.showsSearchingBelowText(message: message, isStreaming: true, activity: .thinking))
        #expect(!MessageCellView.showsSearchingBelowText(message: message, isStreaming: false, activity: .searching))

        let reasoning = Message(role: .assistant, text: "", order: 0, isFinal: false, isReasoningMode: true)
        #expect(!MessageCellView.showsSearchingBelowText(message: reasoning, isStreaming: true, activity: .searching))
    }

    @Test func thinkingShimmerStopsOnceReasoningCompletes() throws {
        let schema = Schema([Conversation.self, ChatLLM.Message.self, MessageAttachment.self])
        let container = try ModelContainer(
            for: schema,
            configurations: [ModelConfiguration(isStoredInMemoryOnly: true)]
        )
        let context = ModelContext(container)
        let conversation = Conversation(title: "Test")
        let message = Message(
            role: .assistant,
            text: "",
            order: 0,
            conversation: conversation,
            isFinal: false,
            reasoning: "Working through the request.",
            isReasoningMode: true
        )
        context.insert(conversation)

        // Still streaming reasoning: the label shimmers, even though it is already tappable.
        message.streamingReasoningPhase = .initialThinking
        #expect(ReasoningMessageBubble.thinkingIndicatorAnimates(for: message, hasReasoningContent: true))

        // Moving on to the answer, or finishing, stops the shimmer on "Thought for…".
        message.streamingReasoningPhase = .finalAnswer
        #expect(!ReasoningMessageBubble.thinkingIndicatorAnimates(for: message, hasReasoningContent: true))

        message.streamingReasoningPhase = nil
        message.isFinal = true
        #expect(!ReasoningMessageBubble.thinkingIndicatorAnimates(for: message, hasReasoningContent: true))
    }

    @Test func chatFollowsGrowingReplyOnlyWhenReaderIsAtBottom() {
        typealias State = ChatView.ScrollFollowState
        let atBottom = State(contentHeight: 1_000, visibleHeight: 600, distanceFromBottom: 0)
        let scrolledUp = State(contentHeight: 1_000, visibleHeight: 600, distanceFromBottom: 400)

        // The reply grew while the reader was at the bottom: follow it.
        #expect(State(contentHeight: 1_200, visibleHeight: 600, distanceFromBottom: 200).shouldFollow(from: atBottom))
        // The reply grew while the reader was reading earlier messages: stay put.
        #expect(!State(contentHeight: 1_200, visibleHeight: 600, distanceFromBottom: 600).shouldFollow(from: scrolledUp))
        // The reader scrolled up themselves (no size change): don't pull them back.
        #expect(!scrolledUp.shouldFollow(from: atBottom))
        // The keyboard shrank the viewport while at the bottom: keep the latest message visible.
        #expect(State(contentHeight: 1_000, visibleHeight: 300, distanceFromBottom: 300).shouldFollow(from: atBottom))
        // Even a small overflow (within the near-bottom threshold) is revealed.
        #expect(State(contentHeight: 1_000, visibleHeight: 560, distanceFromBottom: 40).shouldFollow(from: atBottom))
        // Content that still fits the viewport needs no scrolling.
        #expect(!State(contentHeight: 400, visibleHeight: 560, distanceFromBottom: -160).shouldFollow(from: atBottom))
    }
}
