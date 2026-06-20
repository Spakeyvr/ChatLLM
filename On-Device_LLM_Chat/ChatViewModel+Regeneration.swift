//
//  ChatViewModel+Regeneration.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import SwiftData
import os

extension ChatViewModel {

    // MARK: - Scheduled Regeneration (Menu-Safe)

    /// Schedule a regeneration that will execute even if the calling menu/task is cancelled.
    /// Uses DispatchQueue to bypass Swift Concurrency's cancellation mechanism.
    nonisolated func scheduleRegeneration(messageID: UUID, instruction: String?) {
        // DispatchQueue scheduling is immune to Task cancellation from menu dismissal.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            guard let self else {
                return
            }

            // CRITICAL FIX (Bug 5): Capture self strongly; nil check already done above.
            // Detached task has no parent and won't be cancelled by menu dismissal.
            let viewModel = self
            Task.detached { @MainActor in
                if let instruction = instruction {
                    await viewModel.regenerateReplacingAssistant(messageID: messageID, instruction: instruction)
                } else {
                    await viewModel.regenerateAfterAssistant(messageID: messageID)
                }
            }
        }
    }

    func regenerateAfterAssistant(messageID: UUID) async {
        await regenerateAfterAssistantInternal(messageID: messageID, skipLockCheck: false)
    }

    /// Internal regeneration method that can optionally skip the lock check
    /// - Parameters:
    ///   - messageID: The assistant message to regenerate
    ///   - skipLockCheck: If true, skips the isRegenerating lock (used when called from other locked methods)
    func regenerateAfterAssistantInternal(messageID: UUID, skipLockCheck: Bool) async {
        // CRITICAL: Prevent concurrent regenerations; caller may already hold the lock (skipLockCheck).
        if !skipLockCheck {
            guard !isRegenerating else {
                logger.warning("Regenerate failed: already regenerating (double-tap protection)")
                return
            }
            isRegenerating = true
        }
        defer {
            if !skipLockCheck {
                isRegenerating = false
            }
        }

        guard await waitForStreamToFinish() else { return }

        guard let index = conversation.messages.firstIndex(where: { $0.id == messageID }) else {
            logger.warning("Regenerate failed: message not found")
            return
        }
        // Force SwiftData fault resolution before async boundaries
        let messageRole = conversation.messages[index].role
        let messageOrder = conversation.messages[index].order

        guard messageRole == .assistant else {
            logger.warning("Regenerate failed: message is not an assistant message")
            return
        }

        guard await waitForGenerationToFinish() else {
            logger.error("Regenerate failed: generation still in progress")
            return
        }

        logger.debug("Starting regeneration for message order \(messageOrder, privacy: .public)")

        let messagesToDelete = conversation.messages
            .filter { $0.order > messageOrder }

        let idsToDelete = Set(messagesToDelete.map(\.id))
        conversation.messages.removeAll { idsToDelete.contains($0.id) }
        for msg in messagesToDelete {
            context.delete(msg)
        }

        renumberMessagesByOrder()

        guard let updatedIndex = conversation.messages.firstIndex(where: { $0.id == messageID }) else {
            logger.error("Target message disappeared after renumbering")
            return
        }
        let updatedOrder = conversation.messages[updatedIndex].order

        conversation.lastUpdated = Date()
        immediateSave()

        // CRITICAL FIX (Bug 2): Use updatedOrder (post-renumber) instead of stale messageOrder
        let precedingUserMessage = conversation.messages
            .filter { $0.role == .user && $0.order < updatedOrder }
            .sortedByOrder
            .last

        let shouldUseReasoning = await resolvedReasoningMode(
            for: precedingUserMessage, logContext: "regenerateAfterAssistant")

        guard !Task.isCancelled else {
            logger.warning("Regenerate cancelled after reasoning evaluation")
            return
        }

        // CRITICAL FIX: Look up by ID, not index; SwiftData can reorder the array between lookups.
        guard let targetMessage = conversation.messages.first(where: { $0.id == messageID }) else {
            logger.error("Target message disappeared before applying reasoning mode")
            return
        }

        targetMessage.isReasoningMode = shouldUseReasoning
        targetMessage.promptSnapshot = nil
        conversation.lastUpdated = Date()
        immediateSave()
        invalidateMLXConversationSession(reason: "regenerate_after_assistant")

        let msg = targetMessage
        // CRITICAL FIX: Pass msg.order so buildPrompt excludes this assistant message itself
        // (basedOnHistoryUpTo is exclusive), correctly including all preceding user messages.
        await streamAssistant(into: msg, basedOnHistoryUpTo: msg.order)
        logger.debug("Regeneration completed for message order \(messageOrder, privacy: .public)")
    }

    func regenerateReplacingAssistant(messageID: UUID, instruction: String) async {
        guard !isRegenerating else {
            logger.warning("Regenerate with instruction failed: already regenerating (double-tap protection)")
            return
        }
        isRegenerating = true
        defer { isRegenerating = false }

        guard await waitForStreamToFinish() else { return }

        guard let index = conversation.messages.firstIndex(where: { $0.id == messageID }) else {
            logger.warning("Regenerate with instruction failed: message not found")
            return
        }
        // Force SwiftData fault resolution before async boundaries
        let messageRole = conversation.messages[index].role

        guard messageRole == .assistant else {
            logger.warning("Regenerate with instruction failed: message is not an assistant message")
            return
        }

        guard await waitForGenerationToFinish() else {
            logger.error("Regenerate with instruction failed: generation still in progress")
            return
        }

        logger.debug("Starting regeneration with custom instruction")

        let trimmed = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let shouldUseReasoning = await resolvedReasoningMode(
            for: trimmed, logContext: "regenerateReplacingAssistant")

        guard !Task.isCancelled else {
            logger.warning("Regenerate with instruction cancelled after reasoning evaluation")
            return
        }

        guard let targetMessage = conversation.messages.first(where: { $0.id == messageID }) else {
            logger.error("Target message disappeared after reasoning evaluation")
            return
        }
        targetMessage.isReasoningMode = shouldUseReasoning
        targetMessage.promptSnapshot = nil
        conversation.lastUpdated = Date()
        immediateSave()
        invalidateMLXConversationSession(reason: "regenerate_with_instruction")

        let msg = targetMessage
        // Do NOT insert a new user message; the instruction is passed transiently only.
        await streamAssistant(into: msg, basedOnHistoryUpTo: msg.order, additionalUserInstruction: trimmed)
    }

    func editUserMessageAndRegenerate(from messageID: UUID, newText: String) async {
        guard !isRegenerating else {
            logger.warning("Edit and regenerate failed: already regenerating (double-tap protection)")
            return
        }
        isRegenerating = true
        defer { isRegenerating = false }

        guard await waitForStreamToFinish() else { return }

        guard let index = conversation.messages.firstIndex(where: { $0.id == messageID }) else {
            logger.warning("Edit and regenerate failed: message not found")
            return
        }
        // Force SwiftData fault resolution before async boundaries
        let messageRole = conversation.messages[index].role
        let messageOrder = conversation.messages[index].order

        guard messageRole == .user else {
            logger.warning("Edit and regenerate failed: message is not a user message")
            return
        }

        guard !Task.isCancelled else {
            logger.warning("Edit and regenerate failed: task was cancelled")
            return
        }

        conversation.messages[index].text = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        conversation.lastUpdated = Date()

        let assistantToRegenerate = conversation.messages
            .filter({ $0.role == .assistant && $0.order > messageOrder })
            .sorted(by: { $0.order < $1.order })
            .first

        let assistantID = assistantToRegenerate?.id

        let messagesToDelete = conversation.messages
            .filter { $0.order > messageOrder }

        let idsToDelete = Set(messagesToDelete.lazy.filter { $0.id != assistantID }.map(\.id))
        conversation.messages.removeAll { idsToDelete.contains($0.id) }
        for msg in messagesToDelete where idsToDelete.contains(msg.id) {
            context.delete(msg)
        }

        renumberMessagesByOrder()
        immediateSave()

        if let targetID = assistantID {
            // Look up by ID after renumbering; index is stale.
            guard let assistant = conversation.messages.first(where: { $0.id == targetID }) else {
                logger.error("Edit and regenerate failed: Assistant message disappeared after renumbering")
                return
            }
            assistant.promptSnapshot = nil

            // CRITICAL FIX (Bug 1): Inline instead of calling internal method to keep
            // isRegenerating true throughout the entire operation.
            let updatedOrder = assistant.order

            let precedingUserMessage = conversation.messages
                .filter { $0.role == .user && $0.order < updatedOrder }
                .sortedByOrder
                .last

            let shouldUseReasoning = await resolvedReasoningMode(
                for: precedingUserMessage, logContext: "editAndRegenerate")

            guard !Task.isCancelled else {
                logger.warning("Edit and regenerate cancelled after reasoning evaluation")
                return
            }

            assistant.isReasoningMode = shouldUseReasoning
            conversation.lastUpdated = Date()
            immediateSave()
            invalidateMLXConversationSession(reason: "edit_user_message_and_regenerate")

            await streamAssistant(into: assistant, basedOnHistoryUpTo: assistant.order)
        }
    }

    func deleteMessageAndMaybeTrim(_ message: Message) async {
        if isGenerating, let streamingID = streamingMessageID, streamingID == message.id {
            cancelGeneration()
            guard await waitForStreamToFinish() else { return }
        }

        context.delete(message)
        conversation.messages.removeAll { $0.id == message.id }
        renumberMessagesByOrder()
        conversation.lastUpdated = Date()
        immediateSave()
        invalidateMLXConversationSession(reason: "message_deleted")
    }

    // MARK: - Private Helpers

    /// Waits up to 500ms for an in-progress generation to finish.
    /// Returns true if generation stopped, false if still running.
    private func waitForGenerationToFinish() async -> Bool {
        guard isGenerating else { return true }
        for _ in 0..<10 {
            try? await Task.sleep(for: .milliseconds(50))
            if !isGenerating { return true }
        }
        return !isGenerating
    }
}
