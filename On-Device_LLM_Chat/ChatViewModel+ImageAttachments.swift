//
//  ChatViewModel+ImageAttachments.swift
//  On-Device_LLM_Chat
//
//  Updated to use unified VisionAnalyzer with all Vision framework features
//  and smarter prompt generation (no prompt bloat)
//

import Foundation
import SwiftData
import UIKit

extension ChatViewModel {
    
    /// Send a user message with an image attachment
    /// - Parameters:
    ///   - text: The message text
    ///   - image: The image to attach
    ///   - detections: Pre-computed object detections (optional, legacy support)
    ///   - analysisResult: Full Vision analysis result (preferred)
    func sendWithImage(
        text: String,
        image: UIImage,
        detections: [DetectedObject]? = nil,
        analysisResult: VisionAnalysisResult? = nil
    ) async {
        print("🖼️ sendWithImage called")
        print("📝 Text: '\(text)'")
        
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Ensure any previous stream is fully finished
        await waitForStreamToFinish()
        guard !isGenerating, !Task.isCancelled else { return }
        
        Speaker.shared.stop()
        
        // Build comprehensive image context from analysis result
        var imageContext = ""
        var hasAnalysisData = false
        
        if let result = analysisResult {
            // Use the full detailed description for best model understanding
            let detailedDescription = result.generateDetailedDescription()
            if !detailedDescription.isEmpty && !result.isEmpty {
                imageContext = detailedDescription
                hasAnalysisData = true
                print("📋 Using full Vision analysis context (\(detailedDescription.count) chars)")
            } else {
                // Fallback: provide basic context even if no objects detected
                imageContext = """
                [IMAGE ANALYSIS]
                
                The Vision framework analyzed the image but found:
                - No specific objects detected
                - No text recognized (OCR found nothing)
                - No faces detected
                
                This could mean the image contains:
                - Stylized or artistic text that OCR couldn't read
                - Illustrations or artwork without clear objects
                - Content with low contrast or unusual fonts
                
                Please base your response on what the user is asking about.
                """
                print("📋 Using fallback image context (empty analysis)")
            }
        } else if let detections = detections, !detections.isEmpty {
            // Legacy support: simple object list
            let objectList = detections.prefix(10).map { $0.label }.joined(separator: ", ")
            imageContext = "[IMAGE ANALYSIS]\n\nDETECTED OBJECTS:\n  \(objectList)"
            hasAnalysisData = true
            print("📋 Using legacy detection context: \(objectList)")
        } else {
            // No analysis data at all - still inform model about the image
            imageContext = """
            [IMAGE ANALYSIS]
            
            An image was attached but the Vision framework couldn't process it.
            This might be due to the image format or content type.
            
            Please acknowledge that you can see an image was shared but cannot analyze its contents directly.
            """
            print("📋 Using no-analysis fallback context")
        }
        
        // Create a simple enhanced prompt without overwhelming the model
        // IMPORTANT: Keep prompt structure clear and direct for Foundation Models LLM
        let userPrompt = trimmed.isEmpty ? "What do you see in this image?" : trimmed
        let enhancedText: String
        
        // Simplified format: provide context first, then the question
        // This structure is more natural for the LLM to process
        enhancedText = """
        --- Image Analysis Data ---
        \(imageContext)
        --- End Image Analysis ---
        
        User's question: \(userPrompt)
        
        Please respond based on the image analysis data above.
        """
        
        print("📋 Enhanced prompt length: \(enhancedText.count) characters")
        print("📋 Has actual analysis data: \(hasAnalysisData)")
        
        // Determine if reasoning should be used
        let shouldUseReasoning: Bool
        do {
            shouldUseReasoning = try await shouldUseReasoningForPrompt(enhancedText)
        } catch let reasoningError as ReasoningEvaluationError {
            print("Error determining reasoning mode, using fallback: \(reasoningError.localizedDescription)")
            shouldUseReasoning = reasoningError.fallbackResult
        } catch {
            print("Unexpected error determining reasoning mode: \(error)")
            shouldUseReasoning = conversation.reasoningMode || conversation.smartReasoningMode
        }
        
        guard !Task.isCancelled else { return }
        
        let baseOrder = nextOrder
        
        // Create user message with ENHANCED text that includes image context
        // This ensures the model has full context for this AND future messages
        let userMsg = Message(
            role: .user,
            text: enhancedText,  // Include image context in the stored message
            order: baseOrder,
            conversation: conversation,
            isFinal: true
        )
        conversation.messages.append(userMsg)
        
        // Add image attachment
        do {
            let attachment = try await addImageAttachment(image, to: userMsg)
            
            // Store the full analysis result if available, otherwise store basic detections
            if let result = analysisResult {
                attachment.storeAnalysisResult(result)
            } else if let detections = detections {
                let legacyResult = VisionAnalysisResult(objects: detections)
                attachment.storeAnalysisResult(legacyResult)
            }
            
            print("✅ Image attached successfully")
        } catch {
            print("⚠️ Failed to attach image: \(error)")
        }
        
        // Check if auto-naming is needed
        let userMessagesCount = conversation.messages.lazy.filter { $0.role == .user }.count
        let isFirstUserMessage = userMessagesCount == 1
        let needsAutoNaming = conversation.title == String(localized: "New Chat") &&
                             !conversation.hasAutoGeneratedTitle &&
                             isFirstUserMessage
        
        // Create assistant message
        let assistantMsg = Message(
            role: .assistant,
            text: "",
            order: baseOrder + 1,
            conversation: conversation,
            isFinal: false,
            isReasoningMode: shouldUseReasoning
        )
        conversation.messages.append(assistantMsg)
        conversation.lastUpdated = Date()
        immediateSave()
        
        guard !Task.isCancelled else { return }
        
        // Stream assistant response
        // Image context is already stored in the user message, so no additional instruction needed
        await streamAssistant(
            into: assistantMsg,
            basedOnHistoryUpTo: assistantMsg.order,
            additionalUserInstruction: nil,
            disableWebSearch: true  // Disable web search for image analysis to avoid confusion
        )
        
        // Generate title if needed
        if needsAutoNaming && !Task.isCancelled {
            await generateChatTitle(fromUserMessage: userPrompt)
        }
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
    
    /// Get attachments for a message
    func getAttachments(for message: Message) -> [MessageAttachment] {
        message.attachments
    }
    
    /// Delete an attachment
    func deleteAttachment(_ attachment: MessageAttachment) async {
        do {
            try await ImageStore.shared.delete(url: attachment.fileURL)
        } catch {
            print("Failed to delete attachment file: \(error)")
        }
        
        if let message = attachment.message {
            message.attachments.removeAll { $0.id == attachment.id }
        }
        
        context.delete(attachment)
        conversation.lastUpdated = Date()
        immediateSave()
    }
}
