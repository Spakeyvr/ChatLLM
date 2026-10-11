//
//  ChatLLMTests+ImagePrompts
//  ChatLLMTests
//
//  Part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
import FoundationModels
import MLXLMCommon
import UIKit
@testable import ChatLLM

extension ChatLLMTests {
    /// How image turns were saved while the app still described images in text.
    private static let legacyImageTurnText = """
    --- Image Analysis Data ---
    [IMAGE ANALYSIS]

    TEXT CONTENT:
      "OPEN 24 HOURS"

    --- End Image Analysis ---

    User's question: What does the sign say?

    Please respond based on the image analysis data above.
    """

    @discardableResult
    private func appendImageTurn(
        to viewModel: ChatViewModel,
        text: String,
        order: Int,
        imageURL: URL
    ) -> (message: ChatLLM.Message, attachment: MessageAttachment) {
        let message = ChatLLM.Message(
            role: .user,
            text: text,
            order: order,
            conversation: viewModel.conversation,
            isFinal: true
        )
        let attachment = MessageAttachment(
            type: .image,
            fileURL: imageURL,
            fileName: imageURL.lastPathComponent
        )
        attachment.message = message
        message.attachments.append(attachment)
        viewModel.conversation.messages.append(message)
        return (message, attachment)
    }

    private func appendAssistantTurn(to viewModel: ChatViewModel, text: String, order: Int) {
        viewModel.conversation.messages.append(
            ChatLLM.Message(
                role: .assistant,
                text: text,
                order: order,
                conversation: viewModel.conversation,
                isFinal: true
            )
        )
    }

    private func saveTestImage() async throws -> URL {
        try await ImageStore.shared.save(image: makeTestImage(width: 64, height: 48), maxDimension: 64)
    }

    @Test func buildFoundationRequestSendsTheLatestImageWithThePlainQuestion() async throws {
        let imageURL = try await saveTestImage()
        defer { Task { try? await ImageStore.shared.delete(url: imageURL) } }
        let viewModel = try makeViewModel()
        let turn = appendImageTurn(
            to: viewModel,
            text: "What does the sign say?",
            order: 0,
            imageURL: imageURL
        )

        let request = viewModel.buildFoundationRequest(
            upToOrderExclusive: 1,
            includeImages: true
        )

        #expect(request.imageURLs == [turn.attachment.actualFileURL])
        #expect(request.prompt == "What does the sign say?")
    }

    @Test func buildFoundationRequestSendsOnlyTheQuestionWhenImagesAreNotIncluded() async throws {
        let imageURL = try await saveTestImage()
        defer { Task { try? await ImageStore.shared.delete(url: imageURL) } }
        let viewModel = try makeViewModel()
        appendImageTurn(to: viewModel, text: "What does the sign say?", order: 0, imageURL: imageURL)

        let request = viewModel.buildFoundationRequest(upToOrderExclusive: 1)

        #expect(request.imageURLs.isEmpty)
        #expect(request.prompt == "What does the sign say?")
    }

    @Test func buildFoundationRequestSendsTheThreeNewestImagesAcrossTurns() async throws {
        var urls: [URL] = []
        for _ in 0..<4 { urls.append(try await saveTestImage()) }
        defer { Task { for url in urls { try? await ImageStore.shared.delete(url: url) } } }
        let viewModel = try makeViewModel()
        let questions = ["What is this?", "And this?", "How about this?", "Last one?"]
        var turns: [(message: ChatLLM.Message, attachment: MessageAttachment)] = []
        for (index, question) in questions.enumerated() {
            turns.append(appendImageTurn(to: viewModel, text: question, order: index * 2, imageURL: urls[index]))
            if index < questions.count - 1 {
                appendAssistantTurn(to: viewModel, text: "Answer \(index + 1).", order: index * 2 + 1)
            }
        }

        let request = viewModel.buildFoundationRequest(upToOrderExclusive: 7, includeImages: true)

        let userHistory = request.history.filter { $0.role == .user }
        #expect(userHistory.map(\.content) == ["What is this?", "And this?", "How about this?"])
        #expect(userHistory.map(\.imageURLs) == [
            [],
            [turns[1].attachment.actualFileURL],
            [turns[2].attachment.actualFileURL]
        ])
        #expect(request.prompt == "Last one?")
        #expect(request.imageURLs == [turns[3].attachment.actualFileURL])
    }

    @Test func buildFoundationRequestKeepsTheNewestImagesOfATurnOverTheLimit() async throws {
        var urls: [URL] = []
        for _ in 0..<4 { urls.append(try await saveTestImage()) }
        defer { Task { for url in urls { try? await ImageStore.shared.delete(url: url) } } }
        let viewModel = try makeViewModel()
        let turn = appendImageTurn(to: viewModel, text: "Compare these.", order: 0, imageURL: urls[0])
        for url in urls.dropFirst() {
            let attachment = MessageAttachment(type: .image, fileURL: url, fileName: url.lastPathComponent)
            attachment.message = turn.message
            turn.message.attachments.append(attachment)
        }

        let request = viewModel.buildFoundationRequest(upToOrderExclusive: 1, includeImages: true)

        #expect(request.imageURLs == turn.message.attachments.dropFirst().map(\.actualFileURL))
    }

    @Test func foundationPromptBudgetLeavesRoomForTheReplyAndInstructions() {
        let instructions = String(repeating: "a", count: 400)

        let budget = ChatViewModel.foundationPromptBudgetTokens(contextSize: 4_096, instructions: instructions)

        // 4,096 − 1,024 reply − 100 instruction − 256 overhead.
        #expect(budget == 2_716)
    }

    @Test func foundationSessionLabelsImagesInConversationOrder() {
        let first = URL(fileURLWithPath: "/tmp/first.jpg")
        let second = URL(fileURLWithPath: "/tmp/second.jpg")
        let request = LLMRequest(
            history: [
                .init(role: .user, content: "What is this?", imageURLs: [first]),
                .init(role: .assistant, content: "A dog."),
                .init(role: .user, content: "And this?", imageURLs: [second]),
                .init(role: .assistant, content: "A cat.")
            ],
            prompt: "Which is bigger?"
        )

        let session = OnDeviceLLMGenerator.makeSession(for: request, tools: [])

        let labels: [String] = session.transcript.flatMap { entry -> [String] in
            guard case .prompt(let prompt) = entry else { return [] }
            return prompt.segments.compactMap { segment in
                guard case .attachment(let attachment) = segment else { return nil }
                return attachment.label
            }
        }
        #expect(labels == ["image-1", "image-2"])
    }

    @Test func buildFoundationRequestSkipsImageFilesThatNoLongerExist() throws {
        let viewModel = try makeViewModel()
        appendImageTurn(
            to: viewModel,
            text: "What does the sign say?",
            order: 0,
            imageURL: URL(fileURLWithPath: "/tmp/missing-\(UUID().uuidString).jpg")
        )

        let request = viewModel.buildFoundationRequest(
            upToOrderExclusive: 1,
            includeImages: true
        )

        #expect(request.imageURLs.isEmpty)
        #expect(request.prompt == "What does the sign say?")
    }

    @Test func buildFoundationRequestStripsLegacyInlineImageDescriptions() async throws {
        let imageURL = try await saveTestImage()
        defer { Task { try? await ImageStore.shared.delete(url: imageURL) } }
        let viewModel = try makeViewModel()
        appendImageTurn(to: viewModel, text: Self.legacyImageTurnText, order: 0, imageURL: imageURL)

        let nativeRequest = viewModel.buildFoundationRequest(
            upToOrderExclusive: 1,
            includeImages: true
        )
        let textRequest = viewModel.buildFoundationRequest(upToOrderExclusive: 1)

        #expect(nativeRequest.prompt == "What does the sign say?")
        #expect(textRequest.prompt == "What does the sign say?")
    }

    @Test func buildQwenMessagesStripsLegacyInlineImageDescriptions() async throws {
        let imageURL = try await saveTestImage()
        defer { Task { try? await ImageStore.shared.delete(url: imageURL) } }
        let viewModel = try makeViewModel()
        appendImageTurn(to: viewModel, text: Self.legacyImageTurnText, order: 0, imageURL: imageURL)

        let messages = try await viewModel.buildQwenMessages(
            upToOrderExclusive: 1,
            includeLatestUserImages: false
        )
        let userMessage = try #require(messages.last(where: { $0.role == .user }))

        #expect(userMessage.content == "What does the sign say?")
        #expect(userMessage.images.isEmpty)
    }

    @Test func mlxBackendWithoutALoadedModelCannotTakeImages() throws {
        let bridge = ModelBackendBridge(defaults: try makeScratchDefaults(#function))
        bridge.selectedBackend = .mlx

        #expect(!bridge.nativeImageInputAvailable)
    }
}
