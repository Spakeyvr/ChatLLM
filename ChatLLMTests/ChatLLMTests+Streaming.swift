//
//  ChatLLMTests+Streaming
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
    @Test func searchSessionParserKeepsBulletAnswerAfterThinkingCloses() {
        let parsed = ChatViewModel.parseReasoningResponseForSearchSessionText(
            """
            I should verify the release notes.
            </think>

            Based on the search, Swift 6.2 is the latest stable release.

            - Adds more concurrency fixes
            - Available now
            """
        )

        #expect(parsed.reasoning == "I should verify the release notes.")
        #expect(parsed.finalAnswer == """
        Based on the search, Swift 6.2 is the latest stable release.

        - Adds more concurrency fixes
        - Available now
        """)
    }

    @Test func streamingReasoningUpdateKeepsPostSearchReasoningOutOfAnswerBubble() throws {
        let originalBackend = ModelBackendBridge.shared.selectedBackend
        ModelBackendBridge.shared.selectedBackend = .mlx
        defer { ModelBackendBridge.shared.selectedBackend = originalBackend }

        let viewModel = try makeViewModel()
        let conversation = viewModel.conversation
        let message = ChatLLM.Message(
            role: .assistant,
            text: "",
            order: 1,
            conversation: conversation,
            isReasoningMode: true
        )
        message.searchInvocations = [
            SearchInvocation(
                query: "latest swift release",
                results: "Swift 6.2 Released"
            )
        ]
        message.streamingReasoningPhase = .postToolReasoning
        message.postToolReasoningStartCloseTagCount = 1

        viewModel.updateMessageWithReasoningContent(
            message,
            fullText: """
            The user wants a surprising fact, so I should search for something interesting.
            </think>

            The first results are broad lists. I need to inspect them and decide whether another search is useful.

            This is still reasoning even though it does not start with a canned phrase.
            """,
            finalize: false
        )

        #expect(message.reasoning == """
        The user wants a surprising fact, so I should search for something interesting.

        The first results are broad lists. I need to inspect them and decide whether another search is useful.

        This is still reasoning even though it does not start with a canned phrase.
        """)
        #expect(message.finalAnswer == nil)
        #expect(message.text.isEmpty)
        #expect(message.streamingReasoningPhase == .postToolReasoning)
        #expect(!message.reasoning!.contains("</think>"))
    }

    @Test func streamingReasoningUpdateShowsAnswerAfterPostSearchReasoningTurnsIntoAnswer() throws {
        let originalBackend = ModelBackendBridge.shared.selectedBackend
        ModelBackendBridge.shared.selectedBackend = .mlx
        defer { ModelBackendBridge.shared.selectedBackend = originalBackend }

        let viewModel = try makeViewModel()
        let conversation = viewModel.conversation
        let message = ChatLLM.Message(
            role: .assistant,
            text: "",
            order: 1,
            conversation: conversation,
            isReasoningMode: true
        )
        message.searchInvocations = [
            SearchInvocation(
                query: "latest swift release",
                results: "Swift 6.2 Released"
            )
        ]
        message.streamingReasoningPhase = .postToolReasoning
        message.postToolReasoningStartCloseTagCount = 1

        viewModel.updateMessageWithReasoningContent(
            message,
            fullText: """
            I should verify the release notes.
            </think>

            The search results need one more pass before I can answer confidently.
            </think>

            Based on the search, Swift 6.2 is the latest stable release.
            """,
            finalize: false
        )

        #expect(message.reasoning == """
        I should verify the release notes.

        The search results need one more pass before I can answer confidently.
        """)
        #expect(message.finalAnswer == "Based on the search, Swift 6.2 is the latest stable release.")
        #expect(message.text == "Based on the search, Swift 6.2 is the latest stable release.")
        #expect(message.streamingReasoningPhase == .finalAnswer)
        #expect(!message.reasoning!.contains("</think>"))
    }
    @Test func streamingChunkMergePreservesSingleCharacterNumericChunks() {
        let merged = ChatViewModel.mergedStreamingChunk(
            currentText: "10",
            newText: "0"
        )

        #expect(merged == "100")
    }

    @Test func streamingChunkMergePreservesIntentionalPrefixOverlap() {
        let merged = ChatViewModel.mergedStreamingChunk(
            currentText: "Hello wor",
            newText: "world"
        )

        #expect(merged == "Hello worworld")
    }

    @Test func streamingChunkMergePreservesIntentionalRepeatedPhrase() {
        let repeated = "a recently repeated phrase"
        let current = String(repeating: "x", count: 20_000) + repeated
        let merged = ChatViewModel.mergedStreamingChunk(
            currentText: current,
            newText: repeated
        )

        #expect(merged == current + repeated)
    }

    @Test func foundationStreamingDeltaPreservesUnicodeAndYieldsOnlyNewText() {
        var yieldedUTF8Count = 0

        #expect(OnDeviceLLMGenerator.incrementalDelta(
            from: "Hello 👋",
            afterUTF8Count: &yieldedUTF8Count
        ) == "Hello 👋")
        #expect(OnDeviceLLMGenerator.incrementalDelta(
            from: "Hello 👋 world",
            afterUTF8Count: &yieldedUTF8Count
        ) == " world")
        #expect(OnDeviceLLMGenerator.incrementalDelta(
            from: "Hello 👋 world",
            afterUTF8Count: &yieldedUTF8Count
        ) == nil)
    }
}
