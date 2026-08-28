//
//  ChatLLMTests+Reasoning
//  ChatLLMTests
//
//  Split out of ChatLLMTests.swift; part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
import MLXLMCommon
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

        let viewModel = ChatViewModel(
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
}
