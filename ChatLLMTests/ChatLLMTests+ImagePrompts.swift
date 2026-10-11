//
//  ChatLLMTests+ImagePrompts
//  ChatLLMTests
//
//  Part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
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
            includeLatestUserImages: true
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

    @Test func buildFoundationRequestOnlySendsTheLatestUserTurnsImages() async throws {
        let firstURL = try await saveTestImage()
        let secondURL = try await saveTestImage()
        defer {
            Task {
                try? await ImageStore.shared.delete(url: firstURL)
                try? await ImageStore.shared.delete(url: secondURL)
            }
        }
        let viewModel = try makeViewModel()
        appendImageTurn(to: viewModel, text: "What breed is this?", order: 0, imageURL: firstURL)
        appendAssistantTurn(to: viewModel, text: "It looks like a beagle.", order: 1)
        let latest = appendImageTurn(to: viewModel, text: "And this one?", order: 2, imageURL: secondURL)

        let request = viewModel.buildFoundationRequest(
            upToOrderExclusive: 3,
            includeLatestUserImages: true
        )

        #expect(request.imageURLs == [latest.attachment.actualFileURL])
        #expect(request.history == [
            .init(role: .user, content: "What breed is this?"),
            .init(role: .assistant, content: "It looks like a beagle.")
        ])
        #expect(request.prompt == "And this one?")
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
            includeLatestUserImages: true
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
            includeLatestUserImages: true
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
