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
    private static let signAnalysis = VisionAnalysisResult(
        textBlocks: [RecognizedText(text: "OPEN 24 HOURS", confidence: 0.95, boundingBox: .zero)]
    )

    @discardableResult
    private func appendImageTurn(
        to viewModel: ChatViewModel,
        text: String,
        order: Int,
        imageURL: URL,
        analysis: VisionAnalysisResult? = nil
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
        if let analysis {
            attachment.storeAnalysisResult(analysis)
        }
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

    @Test func buildFoundationRequestSendsTheLatestImageInsteadOfItsVisionSummary() async throws {
        let imageURL = try await saveTestImage()
        defer { Task { try? await ImageStore.shared.delete(url: imageURL) } }
        let viewModel = try makeViewModel()
        let turn = appendImageTurn(
            to: viewModel,
            text: "What does the sign say?",
            order: 0,
            imageURL: imageURL,
            analysis: Self.signAnalysis
        )

        let request = viewModel.buildFoundationRequest(
            upToOrderExclusive: 1,
            includeLatestUserImages: true
        )

        #expect(request.imageURLs == [turn.attachment.actualFileURL])
        #expect(request.prompt == "What does the sign say?")
        #expect(!request.instructions.contains("IMAGE ANALYSIS INSTRUCTIONS"))
    }

    @Test func buildFoundationRequestAddsTheStoredVisionSummaryWhenImagesAreNotSent() async throws {
        let imageURL = try await saveTestImage()
        defer { Task { try? await ImageStore.shared.delete(url: imageURL) } }
        let viewModel = try makeViewModel()
        let turn = appendImageTurn(
            to: viewModel,
            text: "What does the sign say?",
            order: 0,
            imageURL: imageURL,
            analysis: Self.signAnalysis
        )

        let request = viewModel.buildFoundationRequest(upToOrderExclusive: 1)

        #expect(request.imageURLs.isEmpty)
        #expect(request.prompt.contains("--- Image Analysis Data ---"))
        #expect(request.prompt.contains("OPEN 24 HOURS"))
        #expect(request.prompt.contains("User's question: What does the sign say?"))
        #expect(request.instructions.contains("IMAGE ANALYSIS INSTRUCTIONS"))
        #expect(turn.message.text == "What does the sign say?")
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

    @Test func buildFoundationRequestStripsLegacyInlineAnalysisWhenSendingTheImage() async throws {
        let imageURL = try await saveTestImage()
        defer { Task { try? await ImageStore.shared.delete(url: imageURL) } }
        let viewModel = try makeViewModel()
        let legacyText = ChatViewModel.visionEnhancedText(
            userPrompt: "What is this?",
            analysisResult: Self.signAnalysis
        )
        appendImageTurn(to: viewModel, text: legacyText, order: 0, imageURL: imageURL)

        let nativeRequest = viewModel.buildFoundationRequest(
            upToOrderExclusive: 1,
            includeLatestUserImages: true
        )
        let textRequest = viewModel.buildFoundationRequest(upToOrderExclusive: 1)

        #expect(nativeRequest.prompt == "What is this?")
        #expect(textRequest.prompt == legacyText)
    }

    @Test func buildFoundationRequestFallsBackToTheVisionSummaryWhenTheImageFileIsMissing() throws {
        let viewModel = try makeViewModel()
        appendImageTurn(
            to: viewModel,
            text: "What does the sign say?",
            order: 0,
            imageURL: URL(fileURLWithPath: "/tmp/missing-\(UUID().uuidString).jpg"),
            analysis: Self.signAnalysis
        )

        let request = viewModel.buildFoundationRequest(
            upToOrderExclusive: 1,
            includeLatestUserImages: true
        )

        #expect(request.imageURLs.isEmpty)
        #expect(request.prompt.contains("OPEN 24 HOURS"))
    }

    @Test func buildFoundationRequestLeavesOlderUnanalyzedImageTurnsAsPlainText() async throws {
        let imageURL = try await saveTestImage()
        defer { Task { try? await ImageStore.shared.delete(url: imageURL) } }
        let viewModel = try makeViewModel()
        appendImageTurn(to: viewModel, text: "Describe this photo", order: 0, imageURL: imageURL)
        appendAssistantTurn(to: viewModel, text: "A red bicycle against a wall.", order: 1)
        viewModel.conversation.messages.append(
            ChatLLM.Message(
                role: .user,
                text: "What color is it?",
                order: 2,
                conversation: viewModel.conversation,
                isFinal: true
            )
        )

        let request = viewModel.buildFoundationRequest(upToOrderExclusive: 3)

        #expect(request.history.first == .init(role: .user, content: "Describe this photo"))
        #expect(!request.instructions.contains("IMAGE ANALYSIS INSTRUCTIONS"))
    }

    @Test func buildQwenMessagesAddsTheStoredVisionSummaryWhenImagesAreNotSent() async throws {
        let imageURL = try await saveTestImage()
        defer { Task { try? await ImageStore.shared.delete(url: imageURL) } }
        let viewModel = try makeViewModel()
        appendImageTurn(
            to: viewModel,
            text: "What does the sign say?",
            order: 0,
            imageURL: imageURL,
            analysis: Self.signAnalysis
        )

        let messages = try await viewModel.buildQwenMessages(
            upToOrderExclusive: 1,
            includeLatestUserImages: false
        )
        let userMessage = try #require(messages.last(where: { $0.role == .user }))

        #expect(userMessage.content.contains("OPEN 24 HOURS"))
        #expect(userMessage.content.contains("User's question: What does the sign say?"))
        #expect(userMessage.images.isEmpty)
    }
}
