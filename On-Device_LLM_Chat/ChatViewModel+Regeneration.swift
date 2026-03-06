//
//  ChatViewModel+Regeneration.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import SwiftData

extension ChatViewModel {

    // MARK: - Scheduled Regeneration (Menu-Safe)

    /// Schedule a regeneration that will execute even if the calling menu/task is cancelled
    /// This uses DispatchQueue to completely bypass Swift Concurrency's cancellation mechanism
    nonisolated func scheduleRegeneration(messageID: UUID, instruction: String?) {
        print("📅 Scheduling regeneration for message \(messageID)")

        // DispatchQueue scheduling is immune to Task cancellation from menu dismissal.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            guard let self else {
                // CRITICAL FIX (Bug 5): Log when regeneration fails due to deallocation
                print("⚠️ Scheduled regeneration cancelled: ChatViewModel was deallocated")
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
                print("⚠️ Regenerate failed: already regenerating (double-tap protection)")
                return
            }
            isRegenerating = true
        }
        defer {
            if !skipLockCheck {
                isRegenerating = false
            }
        }

        await waitForStreamToFinish()

        guard let index = conversation.messages.firstIndex(where: { $0.id == messageID }) else {
            print("⚠️ Regenerate failed: message not found")
            return
        }
        // Force SwiftData fault resolution before async boundaries
        let messageRole = conversation.messages[index].role
        let messageOrder = conversation.messages[index].order

        guard messageRole == .assistant else {
            print("⚠️ Regenerate failed: message is not an assistant message")
            return
        }

        // waitForStreamToFinish may return slightly before isGenerating clears; give it a moment.
        if isGenerating {
            print("⚠️ Regenerate failed: still generating (waiting additional time)")
            for _ in 0..<10 {
                try? await Task.sleep(for: .milliseconds(50))
                if !isGenerating { break }
            }
            guard !isGenerating else {
                print("❌ Regenerate failed: generation in progress")
                return
            }
        }

        print("✅ Starting regeneration for message order \(messageOrder)")
        print("📊 Current message count: \(conversation.messages.count)")

        let messagesToDelete = conversation.messages
            .filter { $0.order > messageOrder }

        print("🗑️ Deleting \(messagesToDelete.count) messages after order \(messageOrder)")

        for msg in messagesToDelete {
            conversation.messages.removeAll(where: { $0.id == msg.id })
            context.delete(msg)
        }

        renumberMessagesByOrder()
        print("📊 After renumbering: \(conversation.messages.count) messages")

        guard let updatedIndex = conversation.messages.firstIndex(where: { $0.id == messageID }) else {
            print("❌ Target message disappeared after renumbering!")
            return
        }
        let updatedOrder = conversation.messages[updatedIndex].order
        print("📍 Target message now at order \(updatedOrder) (was \(messageOrder))")

        conversation.lastUpdated = Date()
        immediateSave()

        // CRITICAL FIX (Bug 2): Use updatedOrder (post-renumber) instead of stale messageOrder
        let precedingUserMessage = conversation.messages
            .filter { $0.role == .user && $0.order < updatedOrder }
            .sortedByOrder
            .last

        let shouldUseReasoning: Bool
        if let userMsg = precedingUserMessage {
            do {
                shouldUseReasoning = try await shouldUseReasoningForPrompt(userMsg.text)
            } catch let reasoningError as ReasoningEvaluationError {
                print("Error determining reasoning mode in regenerateAfterAssistant, using fallback: \(reasoningError.localizedDescription)")
                shouldUseReasoning = reasoningError.fallbackResult
            } catch {
                print("Unexpected error determining reasoning mode in regenerateAfterAssistant, using conversation fallback: \(error)")
                shouldUseReasoning = conversation.reasoningMode || conversation.smartReasoningMode
            }
        } else {
            shouldUseReasoning = conversation.reasoningMode || conversation.smartReasoningMode
        }

        guard !Task.isCancelled else {
            print("⚠️ Regenerate cancelled after reasoning evaluation (unexpected in scheduled context)")
            return
        }

        // CRITICAL FIX: Look up by ID, not index; SwiftData can reorder the array between lookups.
        guard let targetMessage = conversation.messages.first(where: { $0.id == messageID }) else {
            print("❌ Target message disappeared before applying reasoning mode!")
            return
        }

        print("🔍 Looking up message by ID \(messageID)")
        print("🔍 Found message details:")
        print("   - ID: \(targetMessage.id)")
        print("   - Role: \(targetMessage.role)")
        print("   - Order: \(targetMessage.order)")
        print("   - Text preview: '\(String(targetMessage.text.prefix(50)))...'")

        targetMessage.isReasoningMode = shouldUseReasoning
        targetMessage.promptSnapshot = nil
        conversation.lastUpdated = Date()
        immediateSave()

        let msg = targetMessage
        // CRITICAL FIX: Pass msg.order so buildPrompt excludes this assistant message itself
        // (basedOnHistoryUpTo is exclusive), correctly including all preceding user messages.
        print("🚀 Calling streamAssistant with order \(msg.order) for message \(msg.id)")
        print("📊 Message details: role=\(msg.role), text='\(msg.text.prefix(50))...', isFinal=\(msg.isFinal)")
        print("📊 All message orders in conversation: \(conversation.messages.map { "(\($0.role):\($0.order))" }.joined(separator: ", "))")
        await streamAssistant(into: msg, basedOnHistoryUpTo: msg.order)
        print("✅ Regeneration completed for message order \(messageOrder)")
    }

    func regenerateReplacingAssistant(messageID: UUID, instruction: String) async {
        guard !isRegenerating else {
            print("⚠️ Regenerate with instruction failed: already regenerating (double-tap protection)")
            return
        }
        isRegenerating = true
        defer { isRegenerating = false }

        await waitForStreamToFinish()

        guard let index = conversation.messages.firstIndex(where: { $0.id == messageID }) else {
            print("⚠️ Regenerate with instruction failed: message not found")
            return
        }
        // Force SwiftData fault resolution before async boundaries
        let messageRole = conversation.messages[index].role

        guard messageRole == .assistant else {
            print("⚠️ Regenerate with instruction failed: message is not an assistant message")
            return
        }

        if isGenerating {
            print("⚠️ Regenerate with instruction failed: still generating (waiting additional time)")
            for _ in 0..<10 {
                try? await Task.sleep(for: .milliseconds(50))
                if !isGenerating { break }
            }
            guard !isGenerating else {
                print("❌ Regenerate with instruction failed: generation in progress")
                return
            }
        }

        print("✅ Starting regeneration with custom instruction")

        let trimmed = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let shouldUseReasoning: Bool
        do {
            shouldUseReasoning = try await shouldUseReasoningForPrompt(trimmed)
        } catch let reasoningError as ReasoningEvaluationError {
            print("Error determining reasoning mode in regenerateReplacingAssistant, using fallback: \(reasoningError.localizedDescription)")
            shouldUseReasoning = reasoningError.fallbackResult
        } catch {
            print("Unexpected error determining reasoning mode in regenerateReplacingAssistant, using conversation fallback: \(error)")
            shouldUseReasoning = conversation.reasoningMode || conversation.smartReasoningMode
        }

        guard !Task.isCancelled else {
            print("⚠️ Regenerate with instruction cancelled after reasoning evaluation (unexpected in scheduled context)")
            return
        }

        guard let targetMessage = conversation.messages.first(where: { $0.id == messageID }) else {
            print("❌ Target message disappeared after reasoning evaluation!")
            return
        }
        targetMessage.isReasoningMode = shouldUseReasoning
        targetMessage.promptSnapshot = nil
        conversation.lastUpdated = Date()
        immediateSave()

        let msg = targetMessage
        // Do NOT insert a new user message; the instruction is passed transiently only.
        await streamAssistant(into: msg, basedOnHistoryUpTo: msg.order, additionalUserInstruction: trimmed)
    }

    func editUserMessageAndRegenerate(from messageID: UUID, newText: String) async {
        guard !isRegenerating else {
            print("⚠️ Edit and regenerate failed: already regenerating (double-tap protection)")
            return
        }
        isRegenerating = true
        defer { isRegenerating = false }

        await waitForStreamToFinish()

        guard let index = conversation.messages.firstIndex(where: { $0.id == messageID }) else {
            print("⚠️ Edit and regenerate failed: message not found")
            return
        }
        // Force SwiftData fault resolution before async boundaries
        let messageRole = conversation.messages[index].role
        let messageOrder = conversation.messages[index].order

        guard messageRole == .user else {
            print("⚠️ Edit and regenerate failed: message is not a user message")
            return
        }

        guard !Task.isCancelled else {
            print("⚠️ Edit and regenerate failed: task was cancelled")
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

        for msg in messagesToDelete {
            if msg.id != assistantID {
                conversation.messages.removeAll(where: { $0.id == msg.id })
                context.delete(msg)
            }
        }

        renumberMessagesByOrder()
        immediateSave()

        if let targetID = assistantID {
            // Look up by ID after renumbering; index is stale.
            guard let assistant = conversation.messages.first(where: { $0.id == targetID }) else {
                print("❌ Edit and regenerate failed: Assistant message disappeared after renumbering")
                return
            }
            assistant.promptSnapshot = nil

            // CRITICAL FIX (Bug 1): Inline instead of calling internal method to keep
            // isRegenerating true throughout the entire operation.
            let updatedOrder = assistant.order
            print("✅ Starting regeneration (via edit) for message order \(updatedOrder)")

    
            let precedingUserMessage = conversation.messages
                .filter { $0.role == .user && $0.order < updatedOrder }
                .sortedByOrder
                .last

            let shouldUseReasoning: Bool
            if let userMsg = precedingUserMessage {
                do {
                    shouldUseReasoning = try await shouldUseReasoningForPrompt(userMsg.text)
                } catch let reasoningError as ReasoningEvaluationError {
                    print("Error determining reasoning mode in editAndRegenerate, using fallback: \(reasoningError.localizedDescription)")
                    shouldUseReasoning = reasoningError.fallbackResult
                } catch {
                    print("Unexpected error determining reasoning mode in editAndRegenerate, using conversation fallback: \(error)")
                    shouldUseReasoning = conversation.reasoningMode || conversation.smartReasoningMode
                }
            } else {
                shouldUseReasoning = conversation.reasoningMode || conversation.smartReasoningMode
            }

            guard !Task.isCancelled else {
                print("⚠️ Edit and regenerate cancelled after reasoning evaluation")
                return
            }

            assistant.isReasoningMode = shouldUseReasoning
            conversation.lastUpdated = Date()
            immediateSave()

            await streamAssistant(into: assistant, basedOnHistoryUpTo: assistant.order)
            print("✅ Edit and regeneration completed")
        }
    }

    func deleteMessageAndMaybeTrim(_ message: Message) async {
        if isGenerating, let streamingID = streamingMessageID, streamingID == message.id {
            cancelGeneration()
            await waitForStreamToFinish()
        }

        conversation.messages.removeAll { $0.id == message.id }
        renumberMessagesByOrder()
        conversation.lastUpdated = Date()
        immediateSave()
    }
}
