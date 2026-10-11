//
//  ChatLLMTests+ContextWindowErrors
//  ChatLLMTests
//
//  Part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
import FoundationModels
@testable import ChatLLM

/// Fails every request the way Foundation Models reports an overflow.
private struct ContextOverflowGenerator: LLMGenerator {
    func streamResponse(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: NSError(
                domain: "FoundationModels.LanguageModelSession.GenerationError",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Exceeded model context window size"]
            ))
        }
    }
}

extension ChatLLMTests {
    @Test func appleIntelligenceOverflowMessageNamesTheAppleModelAndItsFixes() throws {
        let message = try #require(
            ChatViewModel.ContextWindowError(tokenLimit: 4_096, backend: .appleIntelligence).errorDescription
        )

        #expect(message.contains("Apple Intelligence"))
        #expect(message.contains("Attach fewer images"))
        #expect(!message.contains("MLX"))
        #expect(!message.contains("Context Window setting"))
    }

    @Test func mlxOverflowMessageKeepsTheContextWindowSettingAdvice() throws {
        let message = try #require(
            ChatViewModel.ContextWindowError(tokenLimit: 2_048, backend: .mlx).errorDescription
        )

        #expect(message.contains("MLX"))
        #expect(message.contains("Lower the Context Window setting"))
    }

    @Test func appleIntelligenceReplyThatOverflowsShowsTheAppleMessage() async throws {
        let viewModel = try makeViewModel(generator: ContextOverflowGenerator())

        await viewModel.send(userText: "Describe these photos")

        let reply = try #require(viewModel.conversation.messages.first { $0.role == .assistant })
        let error = try #require(reply.generationError)
        #expect(error.contains("Apple Intelligence"))
        #expect(!error.contains("MLX"))
    }
}
