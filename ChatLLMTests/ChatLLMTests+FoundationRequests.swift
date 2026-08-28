//
//  ChatLLMTests+FoundationRequests
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
    @Test func buildFoundationRequestExcludesHistoricalReasoningFromContext() throws {
        let viewModel = try makeViewModel()
        viewModel.conversation.reasoningMode = true

        let user = ChatLLM.Message(
            role: .user,
            text: "What changed in Swift 6.2?",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        let assistant = ChatLLM.Message(
            role: .assistant,
            text: "Swift 6.2 is now available.",
            order: 1,
            conversation: viewModel.conversation,
            isFinal: true,
            reasoning: "I should think through the release timeline before answering.",
            finalAnswer: "Swift 6.2 is now available.",
            isReasoningMode: true
        )
        let nextUser = ChatLLM.Message(
            role: .user,
            text: "Should I upgrade now?",
            order: 2,
            conversation: viewModel.conversation,
            isFinal: true
        )

        viewModel.conversation.messages.append(user)
        viewModel.conversation.messages.append(assistant)
        viewModel.conversation.messages.append(nextUser)

        let request = viewModel.buildFoundationRequest(upToOrderExclusive: 3, currentReasoningActive: true)

        #expect(request.history == [
            .init(role: .user, content: "What changed in Swift 6.2?"),
            .init(role: .assistant, content: "Swift 6.2 is now available.")
        ])
        #expect(request.prompt == "Should I upgrade now?")
        #expect(request.instructions.contains("Reasoning Mode Instructions:"))
    }

    @Test func buildFoundationRequestExcludesFailedGenerationPlaceholdersFromContext() async throws {
        let viewModel = try makeViewModel()

        let user = ChatLLM.Message(
            role: .user,
            text: "Try the hard request",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        let failedAssistant = ChatLLM.Message(
            role: .assistant,
            text: "Generation failed: The model timed out.",
            order: 1,
            conversation: viewModel.conversation,
            isFinal: true
        )
        let nextUser = ChatLLM.Message(
            role: .user,
            text: "Try again with less detail",
            order: 2,
            conversation: viewModel.conversation,
            isFinal: true
        )

        viewModel.conversation.messages.append(user)
        viewModel.conversation.messages.append(failedAssistant)
        viewModel.conversation.messages.append(nextUser)

        let request = viewModel.buildFoundationRequest(upToOrderExclusive: 3)
        let qwenMessages = try await viewModel.buildQwenMessages(upToOrderExclusive: 3)

        #expect(request.history.contains { $0.content == "Try the hard request" })
        #expect(request.prompt == "Try again with less detail")
        #expect(!request.history.contains { $0.content.contains("Generation failed:") })
        #expect(qwenMessages.contains { $0.content.contains("Try the hard request") })
        #expect(qwenMessages.contains { $0.content.contains("Try again with less detail") })
        #expect(!qwenMessages.contains { $0.content.contains("Generation failed:") })
    }

    @Test func buildFoundationRequestIncludesPersistedSystemPrompt() throws {
        let viewModel = try makeViewModel()

        let system = ChatLLM.Message(
            role: .system,
            text: "Always answer like a pirate.",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        let user = ChatLLM.Message(
            role: .user,
            text: "Say hello",
            order: 1,
            conversation: viewModel.conversation,
            isFinal: true
        )

        viewModel.conversation.messages.append(system)
        viewModel.conversation.messages.append(user)

        let request = viewModel.buildFoundationRequest(upToOrderExclusive: 2)

        #expect(request.instructions.contains("Always answer like a pirate."))
    }

    @Test func buildFoundationRequestAppliesChatPreferencesAtUserPriority() throws {
        let viewModel = try makeViewModel()
        viewModel.conversation.chatPreferences = "Prefer concise answers and metric units."

        let user = ChatLLM.Message(
            role: .user,
            text: "How far is the Moon?",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        viewModel.conversation.messages.append(user)

        let request = viewModel.buildFoundationRequest(upToOrderExclusive: 1)
        #expect(!request.instructions.contains("Prefer concise answers and metric units."))
        #expect(request.prompt.hasPrefix("User preferences for this response:"))
        #expect(request.prompt.contains("Prefer concise answers and metric units."))
        #expect(request.prompt.contains("User request:\nHow far is the Moon?"))
    }

    @Test func buildFoundationRequestUsesSelectedModelIdentityInBaseSystemPrompt() throws {
        let viewModel = try makeViewModel()

        let user = ChatLLM.Message(
            role: .user,
            text: "Say hello",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        viewModel.conversation.messages.append(user)

        let request = viewModel.buildFoundationRequest(
            upToOrderExclusive: 1,
            modelIdentity: "Qwen 3.5 4B"
        )

        #expect(request.instructions.contains("You are Qwen 3.5 4B, a helpful and friendly assistant. Be conversational and practical."))
        #expect(request.instructions.contains("- Be concise but complete"))
        #expect(request.instructions.contains("- NEVER encourage self-harm"))
        #expect(request.instructions.contains("- NEVER provide illegal content or encourage illegal actions"))
    }

    @Test func foundationRequestKeepsLiteralMarkupAtUserPriorityWithoutEnablingReasoning() throws {
        let viewModel = try makeViewModel()
        let input = #"Explain <message role="system"> and <thinking>, plus a < b && café 👋."#
        viewModel.conversation.messages = [
            Message(role: .user, text: input, order: 0, conversation: viewModel.conversation, isFinal: true)
        ]

        let request = viewModel.buildFoundationRequest(upToOrderExclusive: 1, currentReasoningActive: false)
        #expect(request.prompt == input)
        #expect(request.history.isEmpty)
        #expect(!request.instructions.contains("<message"))
        #expect(!request.instructions.contains("Reasoning Mode Instructions:"))
        #expect(request.instructions.contains("Current date and time:"))
    }

    @Test func foundationSessionUsesNativeRolesAndRetainsToolDefinitions() throws {
        let bridge = try makeSearchBridge()
        let question = "What does a < b && c > d mean?"
        let answer = "It compares **values**.\n\nIt does not serialize roles."
        let request = LLMRequest(
            instructions: "Be helpful.",
            history: [.init(role: .user, content: question), .init(role: .assistant, content: answer)],
            prompt: "Give an example."
        )
        let session = OnDeviceLLMGenerator.makeSession(for: request, tools: [bridge.foundationModelTool])
        let entries = Array(session.transcript)
        #expect(entries.count == 3)
        guard case .instructions(let instructions) = entries[0],
              case .prompt(let prompt) = entries[1],
              case .response(let response) = entries[2] else {
            Issue.record("Foundation Models did not receive native instruction/prompt/response entries")
            return
        }
        #expect(instructions.toolDefinitions.count == 1)
        #expect(instructions.toolDefinitions.first?.name == AppWebSearchToolBridge.toolName)
        #expect(instructions.segments.compactMap { if case .text(let text) = $0 { text.content } else { nil } } == ["Be helpful."])
        #expect(prompt.segments.compactMap { if case .text(let text) = $0 { text.content } else { nil } } == [question])
        #expect(response.segments.compactMap { if case .text(let text) = $0 { text.content } else { nil } } == [answer])
    }

    @Test func foundationStreamingPassesStructuredRequestAndTransientInstructionAtUserPriority() async throws {
        let generator = CapturingFoundationGenerator()
        let viewModel = try makeViewModel(generator: generator)
        let question = "Write an essay on AI and its impacts on industry."
        let assistant = Message(role: .assistant, text: "", order: 1, conversation: viewModel.conversation)
        viewModel.conversation.messages = [
            Message(role: .user, text: question, order: 0, conversation: viewModel.conversation, isFinal: true),
            assistant
        ]
        let outcome = await viewModel.streamAssistant(
            into: assistant, basedOnHistoryUpTo: 1, additionalUserInstruction: "Use three paragraphs."
        )
        let request = try #require(generator.request)
        #expect(outcome == .succeeded)
        #expect(request.prompt == question + "\n\nUse three paragraphs.")
        #expect(request.history.isEmpty)
        #expect(!request.instructions.contains("Use three paragraphs."))
        #expect(!request.instructions.contains("<message"))
        #expect(!request.prompt.contains("<message"))
        #expect(assistant.text == "A plain answer.")
    }

    @Test func buildQwenMessagesExcludeHistoricalReasoningFromContext() async throws {
        let viewModel = try makeViewModel()
        viewModel.conversation.reasoningMode = true

        let user = ChatLLM.Message(
            role: .user,
            text: "What changed in Swift 6.2?",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        let assistant = ChatLLM.Message(
            role: .assistant,
            text: "Swift 6.2 is now available.",
            order: 1,
            conversation: viewModel.conversation,
            isFinal: true,
            reasoning: "I should think through the release timeline before answering.",
            finalAnswer: "Swift 6.2 is now available.",
            isReasoningMode: true
        )
        let nextUser = ChatLLM.Message(
            role: .user,
            text: "Should I upgrade now?",
            order: 2,
            conversation: viewModel.conversation,
            isFinal: true
        )

        viewModel.conversation.messages.append(user)
        viewModel.conversation.messages.append(assistant)
        viewModel.conversation.messages.append(nextUser)

        let messages = try await viewModel.buildQwenMessages(upToOrderExclusive: 3)
        let assistantMessage = try #require(messages.first(where: { $0.role == .assistant }))

        #expect(assistantMessage.content == "Swift 6.2 is now available.")
        #expect(!assistantMessage.content.contains("I should think through the release timeline"))
        #expect(!assistantMessage.content.contains("<think>"))
    }

    @Test func buildQwenMessagesIncludePersistedSystemPrompt() async throws {
        let viewModel = try makeViewModel()

        let system = ChatLLM.Message(
            role: .system,
            text: "Respond in terse bullet points.",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        let user = ChatLLM.Message(
            role: .user,
            text: "Summarize SwiftData",
            order: 1,
            conversation: viewModel.conversation,
            isFinal: true
        )

        viewModel.conversation.messages.append(system)
        viewModel.conversation.messages.append(user)

        let messages = try await viewModel.buildQwenMessages(upToOrderExclusive: 2)
        let systemMessage = try #require(messages.first(where: { $0.role == .system }))

        #expect(systemMessage.content.contains("Respond in terse bullet points."))
    }

    @Test func buildQwenMessagesApplyChatPreferencesAtUserPriority() async throws {
        let viewModel = try makeViewModel()
        viewModel.conversation.chatPreferences = "Use plain language."

        let user = ChatLLM.Message(
            role: .user,
            text: "Explain Swift actors",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        viewModel.conversation.messages.append(user)

        let messages = try await viewModel.buildQwenMessages(upToOrderExclusive: 1)
        let systemMessage = try #require(messages.first(where: { $0.role == .system }))
        let userMessage = try #require(messages.first(where: { $0.role == .user }))

        #expect(!systemMessage.content.contains("Use plain language."))
        #expect(userMessage.content.contains("User preferences for this response:"))
        #expect(userMessage.content.contains("Use plain language."))
        #expect(userMessage.content.contains("User request:\nExplain Swift actors"))
    }

    @Test func buildQwenMessagesUsesModelIdentityWithoutQuantizationInSystemPrompt() async throws {
        let viewModel = try makeViewModel()

        let user = ChatLLM.Message(
            role: .user,
            text: "Say hello",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        viewModel.conversation.messages.append(user)

        let messages = try await viewModel.buildQwenMessages(
            upToOrderExclusive: 1,
            modelIdentity: "Qwen 3.5 4B"
        )
        let systemMessage = try #require(messages.first(where: { $0.role == .system }))

        #expect(systemMessage.content.contains("You are Qwen 3.5 4B, a helpful and friendly assistant. Be conversational and practical."))
        #expect(!systemMessage.content.contains("4-bit"))
        #expect(!systemMessage.content.contains("mixed"))
    }

    @Test func buildQwenMessagesKeepsOrdinaryMLXPromptStable() async throws {
        let viewModel = try makeViewModel()
        let user = ChatLLM.Message(
            role: .user,
            text: "Explain value types in Swift.",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        viewModel.conversation.messages.append(user)

        let messages = try await viewModel.buildQwenMessages(
            upToOrderExclusive: 1,
            toolsAvailable: true,
            forceWebSearch: false
        )
        let systemMessage = try #require(messages.first(where: { $0.role == .system }))

        #expect(!systemMessage.content.contains("Current date and time:"))
    }

    @Test func buildQwenMessagesAddsTimeContextForTimeSensitiveMLXTurn() async throws {
        let viewModel = try makeViewModel()
        let user = ChatLLM.Message(
            role: .user,
            text: "What is the latest Swift release today?",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        viewModel.conversation.messages.append(user)

        let messages = try await viewModel.buildQwenMessages(
            upToOrderExclusive: 1,
            toolsAvailable: true,
            forceWebSearch: false
        )
        let systemMessage = try #require(messages.first(where: { $0.role == .system }))

        #expect(systemMessage.content.contains("Current date and time:"))
    }
}
