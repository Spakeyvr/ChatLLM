//
//  ChatLLMTests+Conversation
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
    @Test func togglingReasoningInAnUnsentDraftDoesNotSaveIt() throws {
        let viewModel = try makeViewModel()
        #expect(viewModel.conversation.modelContext == nil)

        viewModel.setReasoningMode(true)
        viewModel.setSmartReasoningMode(true)

        // The draft stays out of the chat history but keeps the choice for its first send.
        #expect(viewModel.conversation.modelContext == nil)
        #expect(viewModel.conversation.smartReasoningMode)
        #expect(!viewModel.conversation.reasoningMode)
    }

    @Test func autoDeleteSkipsRecentChatsAndTheOpenChat() {
        let now = Date()
        let cutoff = now.addingTimeInterval(-30 * 86_400)
        let recent = Conversation(title: "Recent")
        recent.lastUpdated = now
        let old = Conversation(title: "Old")
        old.lastUpdated = cutoff.addingTimeInterval(-60)
        let oldButOpen = Conversation(title: "Old but open")
        oldButOpen.lastUpdated = cutoff.addingTimeInterval(-60)

        let deleted = ContentView.conversationsToAutoDelete(
            [recent, old, oldButOpen],
            olderThan: cutoff,
            keeping: oldButOpen.id
        )

        #expect(deleted.map(\.id) == [old.id])
    }

    @Test func attachmentCleanupKeepsFilesOfSavedChatsOnly() throws {
        let container = try ModelContainer(
            for: Schema([Conversation.self, ChatLLM.Message.self, MessageAttachment.self]),
            configurations: [ModelConfiguration(isStoredInMemoryOnly: true)]
        )
        let context = ModelContext(container)
        func chat(_ fileName: String) -> (Conversation, MessageAttachment) {
            let conversation = Conversation(title: fileName)
            context.insert(conversation)
            let message = ChatLLM.Message(role: .user, text: "Look", order: 0, conversation: conversation, isFinal: true)
            conversation.messages = [message]
            let attachment = MessageAttachment(type: .image, fileURL: URL(fileURLWithPath: "/tmp/\(fileName)"), fileName: fileName)
            message.attachments = [attachment]
            attachment.message = message
            return (conversation, attachment)
        }
        let (_, kept) = chat("kept.jpg")
        let (deleting, _) = chat("deleting.jpg")
        let orphan = MessageAttachment(type: .image, fileURL: URL(fileURLWithPath: "/tmp/orphan.jpg"), fileName: "orphan.jpg")
        context.insert(orphan)
        try context.save()

        let referenced = try MessageAttachment.referencedFileURLs(
            in: context,
            excludingConversationIDs: [deleting.id]
        )

        #expect(referenced == [kept.actualFileURL])
    }

    @Test func sidebarPreviewSkipsAReplyThatFailedBeforeWritingText() {
        let conversation = Conversation(title: "Failed reply")
        let user = ChatLLM.Message(role: .user, text: "Hello there", order: 0, conversation: conversation, isFinal: true)
        let failed = ChatLLM.Message(role: .assistant, text: "", order: 1, conversation: conversation, isFinal: true)
        failed.generationError = "The model isn't ready."
        conversation.messages = [user, failed]

        #expect(ConversationRow.previewText(for: conversation) == "Hello there")
    }

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
