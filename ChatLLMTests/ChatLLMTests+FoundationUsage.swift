//
//  ChatLLMTests+FoundationUsage
//  ChatLLMTests
//
//  Part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
import FoundationModels
@testable import ChatLLM

/// Answers with fixed text and reports fixed usage, like the Apple model does
/// once a response completes.
private struct UsageReportingGenerator: LLMGenerator {
    let metrics: GenerationMetrics

    func streamResponse(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error> {
        try await streamResponse(to: request, tools: tools, onMetrics: { _ in })
    }

    func streamResponse(
        to request: LLMRequest,
        tools: [any FoundationModelTool],
        onMetrics: @escaping @Sendable (GenerationMetrics) -> Void
    ) async throws -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield("The sign says OPEN.")
            onMetrics(metrics)
            continuation.finish()
        }
    }
}

extension ChatLLMTests {
    @Test func foundationUsageMapsOntoGenerationMetrics() {
        let metrics = OnDeviceLLMGenerator.generationMetrics(
            inputTokens: 261,
            cachedInputTokens: 40,
            outputTokens: 12,
            timeToFirstToken: 0.5,
            totalTime: 2.0
        )

        #expect(metrics == GenerationMetrics(
            promptTokenCount: 261,
            cachedPromptTokenCount: 40,
            promptTime: 0.5,
            outputTokenCount: 12,
            decodeTime: 1.5,
            timeToFirstToken: 0.5
        ))
    }

    @Test func foundationUsageWithoutAnyOutputCountsTheWholeTimeAsPrompt() {
        let metrics = OnDeviceLLMGenerator.generationMetrics(
            inputTokens: 80,
            cachedInputTokens: 0,
            outputTokens: 0,
            timeToFirstToken: nil,
            totalTime: 1.2
        )

        #expect(metrics.promptTime == 1.2)
        #expect(metrics.decodeTime == 0)
        #expect(metrics.timeToFirstToken == nil)
    }

    @Test func foundationReplyRecordsTheUsageTheModelReported() async throws {
        let reported = GenerationMetrics(
            promptTokenCount: 261,
            cachedPromptTokenCount: 0,
            promptTime: 0.4,
            outputTokenCount: 6,
            decodeTime: 0.3,
            timeToFirstToken: 0.4
        )
        let viewModel = try makeViewModel(generator: UsageReportingGenerator(metrics: reported))

        await viewModel.send(userText: "What does the sign say?")

        let reply = try #require(viewModel.conversation.messages.first { $0.role == .assistant })
        #expect(reply.generationBackend == ModelBackendBridge.Backend.foundationModels.displayName)
        #expect(reply.generationPromptTokenCount == 261)
        #expect(reply.generationOutputTokenCount == 6)
        #expect(reply.generationTimeToFirstToken == 0.4)
    }

    @Test func foundationReplyFromAGeneratorWithoutUsageRecordsNoTokenCounts() async throws {
        let viewModel = try makeViewModel(generator: SequencedTestGenerator(streams: [
            AsyncThrowingStream { continuation in
                continuation.yield("Hello.")
                continuation.finish()
            }
        ]))

        await viewModel.send(userText: "Hi")

        let reply = try #require(viewModel.conversation.messages.first { $0.role == .assistant })
        #expect(reply.generationPromptTokenCount == nil)
    }

    @Test func regenerationThatWritesNothingKeepsTheWholePreviousReply() async throws {
        let viewModel = try makeViewModel(generator: SequencedTestGenerator(streams: [
            AsyncThrowingStream { continuation in
                continuation.yield("First answer.")
                continuation.finish()
            },
            AsyncThrowingStream { continuation in continuation.finish() }
        ]))
        await viewModel.send(userText: "Hi")
        let reply = try #require(viewModel.conversation.messages.first { $0.role == .assistant })
        reply.searchInvocations = [SearchInvocation(query: "earlier search", results: "[1] Source\nhttps://example.com\nSnippet.")]
        let startedAt = try #require(reply.generationStartedAt)
        let backend = reply.generationBackend

        await viewModel.regenerateAfterAssistant(messageID: reply.id)

        // The empty retry put back the old reply, its sources and timings included.
        #expect(reply.text == "First answer.")
        #expect(reply.generationError == nil)
        #expect(reply.searchInvocations?.first?.query == "earlier search")
        #expect(reply.generationStartedAt == startedAt)
        #expect(reply.generationBackend == backend)
    }

    @Test func regenerationThatFailsBeforeWritingKeepsThePreviousAnswer() async throws {
        let viewModel = try makeViewModel(generator: SequencedTestGenerator(streams: [
            AsyncThrowingStream { continuation in
                continuation.yield("First answer.")
                continuation.finish()
            },
            AsyncThrowingStream { continuation in
                continuation.finish(throwing: NSError(
                    domain: "Test",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "The model stopped."]
                ))
            }
        ]))
        await viewModel.send(userText: "Hi")
        let reply = try #require(viewModel.conversation.messages.first { $0.role == .assistant })

        await viewModel.regenerateAfterAssistant(messageID: reply.id)

        #expect(reply.text == "First answer.")
        #expect(reply.isFinal)
        let error = try #require(reply.generationError)
        #expect(error.hasSuffix("The model stopped."))
        #expect(error != "The model stopped.")
    }

    @Test func regenerationThatCannotStartKeepsThePreviousAnswer() async throws {
        let viewModel = try makeViewModel(generator: SequencedTestGenerator(streams: [
            AsyncThrowingStream { continuation in
                continuation.yield("First answer.")
                continuation.finish()
            }
        ]))
        await viewModel.send(userText: "Hi")
        let reply = try #require(viewModel.conversation.messages.first { $0.role == .assistant })
        // A forced search with no search tool fails before the model is asked.
        reply.requiresWebSearch = true
        reply.searchQuery = "Hi"

        await viewModel.regenerateAfterAssistant(messageID: reply.id)

        #expect(reply.text == "First answer.")
        #expect(reply.generationError != nil)
    }

    @Test func unavailableAppleIntelligenceExplainsWhatToDo() {
        #expect(OnDeviceLLMGenerator.unavailableMessage(for: .unavailable(.deviceNotEligible)).contains("Manage Models"))
        #expect(OnDeviceLLMGenerator.unavailableMessage(for: .unavailable(.appleIntelligenceNotEnabled)).contains("turned off"))
        #expect(OnDeviceLLMGenerator.unavailableMessage(for: .unavailable(.modelNotReady)).contains("getting ready"))
    }
}
