//
//  ChatViewModel+ImageAttachments.swift
//  ChatLLM
//
//  Images are sent only to models that read them directly. The composer
//  disables attachments for every other model.
//

import Foundation
import SwiftData
import UIKit
import OSLog

extension ChatViewModel {

    /// Send a user message with an image attachment.
    func sendWithImage(
        text: String,
        image: UIImage,
        regenerateTitle: Bool = false,
        editedUserMessageID: UUID? = nil
    ) async {
        logger.debug("sendWithImage called")

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let userPrompt = trimmed.isEmpty ? "What do you see in this image?" : trimmed

        guard !isRegenerating else { return }
        await runGeneration { generationID in
            await self.sendImageTurn(
                userPrompt: userPrompt,
                image: image,
                generationID: generationID,
                regenerateTitle: regenerateTitle,
                editedUserMessageID: editedUserMessageID
            )
        }
    }

    private func sendImageTurn(
        userPrompt: String,
        image: UIImage,
        generationID: UUID,
        regenerateTitle: Bool,
        editedUserMessageID: UUID?
    ) async {
        let shouldUseReasoning = await resolvedReasoningMode(
            for: userPrompt,
            logContext: "image query"
        )

        guard isGenerationActive(generationID) else { return }

        guard let turn = prepareUserMessageForSend(
            userPrompt,
            forceAutoNaming: regenerateTitle,
            replacingMessageID: editedUserMessageID
        ) else { return }

        let isAttached: Bool
        do {
            try await addImageAttachment(image, to: turn.message)
            isAttached = true
        } catch {
            logger.error("Failed to attach image: \((error as NSError).localizedDescription, privacy: .public)")
            isAttached = false
        }
        guard isGenerationActive(generationID) else { return }
        let assistantMsg = appendAssistantPlaceholder(isReasoningMode: shouldUseReasoning)

        if isAttached {
            guard isGenerationActive(generationID) else { return }
            _ = await streamAssistant(
                into: assistantMsg,
                basedOnHistoryUpTo: assistantMsg.order,
                additionalUserInstruction: nil,
                disableWebSearch: true,
                allowNativeImages: true,
                generationID: generationID
            )
        } else {
            // Answering without the image would read as if the model had seen it.
            assistantMsg.generationError = String(
                localized: "The image couldn't be saved, so it wasn't sent. Try attaching it again.",
                bundle: .appLocalized
            )
            assistantMsg.markAsComplete()
            conversation.lastUpdated = Date()
            immediateSave()
        }

        guard isGenerationActive(generationID) else { return }
        await finishAutoNamingIfNeeded(turn.needsAutoNaming, userText: userPrompt)
    }

    // MARK: - Private Helpers
    
    /// Add image attachment (basic version)
    @discardableResult
    func addImageAttachment(
        _ image: UIImage,
        to message: Message
    ) async throws -> MessageAttachment {
        // Save image to disk
        let fileURL = try await ImageStore.shared.save(image: image)
        
        // Create attachment
        let attachment = MessageAttachment(
            type: .image,
            fileURL: fileURL,
            fileName: fileURL.lastPathComponent
        )
        
        // Add to message
        message.attachments.append(attachment)
        attachment.message = message
        
        conversation.lastUpdated = Date()
        immediateSave()
        
        return attachment
    }
}
