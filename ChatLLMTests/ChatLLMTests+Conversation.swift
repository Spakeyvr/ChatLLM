//
//  ChatLLMTests+Conversation
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
    @Test func conversationSearchableVisibleTextUsesDisplayedContent() {
        let conversation = Conversation(title: "Vision")
        let user = ChatLLM.Message(
            role: .user,
            text: """
            [HIDDEN_IMAGE_CONTEXT]ocr: secret token[/HIDDEN_IMAGE_CONTEXT]
            What does the sign say?
            """,
            order: 0,
            conversation: conversation,
            isFinal: true
        )
        let assistant = ChatLLM.Message(
            role: .assistant,
            text: "intermediate",
            order: 1,
            conversation: conversation,
            isFinal: true,
            reasoning: "internal reasoning",
            finalAnswer: "The sign says Welcome.",
            isReasoningMode: true
        )
        conversation.messages = [user, assistant]

        #expect(conversation.searchableVisibleText.contains("What does the sign say?"))
        #expect(conversation.searchableVisibleText.contains("The sign says Welcome."))
        #expect(!conversation.searchableVisibleText.contains("secret token"))
        #expect(!conversation.searchableVisibleText.contains("internal reasoning"))
    }

    @Test func messagesSortByOrderThenCreationDate() {
        let base = Date(timeIntervalSince1970: 1_000)
        let second = Message(role: .user, text: "second", createdAt: base.addingTimeInterval(10), order: 1)
        let first = Message(role: .user, text: "first", createdAt: base.addingTimeInterval(20), order: 0)
        let tieEarly = Message(role: .user, text: "tie-early", createdAt: base, order: 1)

        let sorted = [second, first, tieEarly].sortedByOrder

        #expect(sorted.map(\.text) == ["first", "tie-early", "second"])
    }

    // MARK: - JSON-backed message properties

    @Test func searchInvocationsRoundTripAndMirrorTheFirstQuery() {
        let message = Message(role: .assistant, text: "", order: 0)
        message.searchInvocations = [
            SearchInvocation(query: "weather today", results: "sunny"),
            SearchInvocation(query: "weather tomorrow", results: "rain")
        ]

        #expect(message.searchInvocations?.map(\.query) == ["weather today", "weather tomorrow"])
        // The legacy single-query field has to track the first invocation.
        #expect(message.searchQuery == "weather today")
        #expect(message.searchInvocationsJSON != nil)
    }

    @Test func clearingSearchInvocationsAlsoClearsTheLegacyQuery() {
        let message = Message(role: .assistant, text: "", order: 0)
        message.searchInvocations = [SearchInvocation(query: "q", results: "r")]

        message.searchInvocations = []
        #expect(message.searchInvocations == nil)
        #expect(message.searchQuery == nil)

        message.searchInvocations = [SearchInvocation(query: "q", results: "r")]
        message.searchInvocations = nil
        #expect(message.searchInvocations == nil)
        #expect(message.searchQuery == nil)
    }

    @Test func corruptSearchInvocationJSONDecodesToNilInsteadOfThrowing() {
        let message = Message(role: .assistant, text: "", order: 0)
        message.searchInvocationsJSON = "{not json"

        #expect(message.searchInvocations == nil)
    }

    @Test func reasoningStepsRoundTripThroughJSON() {
        let message = Message(role: .assistant, text: "", order: 0)
        let steps = [
            ReasoningStep(stepNumber: 1, title: "Read", content: "Read the question."),
            ReasoningStep(stepNumber: 2, title: nil, content: "Answer it.")
        ]
        message.reasoningSteps = steps

        #expect(message.reasoningSteps?.map(\.stepNumber) == [1, 2])
        #expect(message.reasoningSteps?.map(\.content) == ["Read the question.", "Answer it."])
        #expect(message.reasoningSteps?[1].title == nil)

        message.reasoningSteps = nil
        #expect(message.reasoningSteps == nil)
    }

    // MARK: - Conversation

    @Test func searchableVisibleTextSkipsSystemMessagesAndSources() {
        let conversation = Conversation(title: "Trip")
        conversation.messages = [
            Message(role: .system, text: "system prompt", order: 0, conversation: conversation),
            Message(role: .assistant, text: "Paris.\n\n<sources>1. https://a.example</sources>",
                    order: 2, conversation: conversation),
            Message(role: .user, text: "Capital of France?", order: 1, conversation: conversation),
            Message(role: .assistant, text: "   ", order: 3, conversation: conversation)
        ]

        #expect(conversation.searchableVisibleText == "Capital of France?\nParis.")
    }

    // MARK: - Settings
}
