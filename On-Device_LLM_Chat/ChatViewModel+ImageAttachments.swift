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

    private var canUseNativeImageSupport: Bool {
        let bridge = ModelBackendBridge.shared
        guard bridge.selectedBackend == .mlx else { return false }
        guard let model = bridge.modelManager?.currentModel else { return false }
        return model.supportsNativeImages
    }

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

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let userPrompt = trimmed.isEmpty ? "What do you see in this image?" : trimmed
        
        // Ensure any previous stream is fully finished
        await waitForStreamToFinish()
        guard !isGenerating, !Task.isCancelled else { return }

        if canUseNativeImageSupport {
            await sendWithNativeImage(
                userPrompt: userPrompt,
                image: image,
                detections: detections,
                analysisResult: analysisResult
            )
        } else {
            await sendWithVisionFallback(
                userPrompt: userPrompt,
                image: image,
                detections: detections,
                analysisResult: analysisResult
            )
        }
    }

    private func sendWithNativeImage(
        userPrompt: String,
        image: UIImage,
        detections: [DetectedObject]?,
        analysisResult: VisionAnalysisResult?
    ) async {
        let shouldUseReasoning = await resolvedReasoningMode(
            for: userPrompt,
            logContext: "native image query"
        )

        guard !Task.isCancelled else { return }

        let turn = appendUserMessage(userPrompt)
        let userMsg = turn.message
        let canonicalImageURL = await attachImage(
            image,
            detections: detections,
            analysisResult: analysisResult,
            to: userMsg
        )
        let assistantMsg = appendAssistantPlaceholder(isReasoningMode: shouldUseReasoning)

        guard !Task.isCancelled else { return }

        let firstAttempt = await streamAssistant(
            into: assistantMsg,
            basedOnHistoryUpTo: assistantMsg.order,
            additionalUserInstruction: nil,
            disableWebSearch: true,
            allowNativeImages: true
        )

        if firstAttempt == .failedBeforeOutput && !assistantMsg.hasContent && !Task.isCancelled {
            print("⚠️ Native image generation failed before output; falling back to Vision analysis context")

            let cachedAttachmentAnalysis = userMsg.attachments
                .first(where: { $0.type == .image })?
                .getAnalysisResult()
            let resolvedAnalysis = await resolveVisionAnalysisResult(
                canonicalImageURL: canonicalImageURL,
                fallbackImage: image,
                detections: detections,
                analysisResult: analysisResult ?? cachedAttachmentAnalysis
            )
            let fallbackText = buildVisionEnhancedText(
                userPrompt: userPrompt,
                detections: detections,
                analysisResult: resolvedAnalysis
            )
            userMsg.text = fallbackText
            storeImageAnalysis(
                resolvedAnalysis,
                detections: detections,
                on: userMsg
            )

            conversation.lastUpdated = Date()
            immediateSave()

            _ = await streamAssistant(
                into: assistantMsg,
                basedOnHistoryUpTo: assistantMsg.order,
                additionalUserInstruction: nil,
                disableWebSearch: true,
                allowNativeImages: false
            )
        }

        await finishAutoNamingIfNeeded(turn.needsAutoNaming, userText: userPrompt)
    }

    private func sendWithVisionFallback(
        userPrompt: String,
        image: UIImage,
        detections: [DetectedObject]?,
        analysisResult: VisionAnalysisResult?
    ) async {
        let enhancedText = buildVisionEnhancedText(
            userPrompt: userPrompt,
            detections: detections,
            analysisResult: analysisResult
        )

        let shouldUseReasoning = await resolvedReasoningMode(
            for: enhancedText,
            logContext: "image query"
        )

        guard !Task.isCancelled else { return }

        let turn = appendUserMessage(enhancedText)
        let userMsg = turn.message
        _ = await attachImage(
            image,
            detections: detections,
            analysisResult: analysisResult,
            to: userMsg
        )
        let assistantMsg = appendAssistantPlaceholder(isReasoningMode: shouldUseReasoning)

        guard !Task.isCancelled else { return }

        _ = await streamAssistant(
            into: assistantMsg,
            basedOnHistoryUpTo: assistantMsg.order,
            additionalUserInstruction: nil,
            disableWebSearch: true,
            allowNativeImages: false
        )

        await finishAutoNamingIfNeeded(turn.needsAutoNaming, userText: userPrompt)
    }

    private func attachImage(
        _ image: UIImage,
        detections: [DetectedObject]?,
        analysisResult: VisionAnalysisResult?,
        to message: Message
    ) async -> URL? {
        do {
            let attachment = try await addImageAttachment(image, to: message)
            storeImageAnalysis(analysisResult, detections: detections, on: attachment)
            return attachment.actualFileURL
        } catch {
            print("⚠️ Failed to attach image: \(error)")
            return nil
        }
    }

    private func storeImageAnalysis(
        _ analysisResult: VisionAnalysisResult?,
        detections: [DetectedObject]?,
        on message: Message
    ) {
        guard let attachment = message.attachments.first(where: { $0.type == .image }) else {
            return
        }
        storeImageAnalysis(analysisResult, detections: detections, on: attachment)
    }

    private func storeImageAnalysis(
        _ analysisResult: VisionAnalysisResult?,
        detections: [DetectedObject]?,
        on attachment: MessageAttachment
    ) {
        if let analysisResult {
            attachment.storeAnalysisResult(analysisResult)
        } else if let detections, !detections.isEmpty {
            attachment.storeAnalysisResult(VisionAnalysisResult(objects: detections))
        }
    }

    private func resolveVisionAnalysisResult(
        canonicalImageURL: URL?,
        fallbackImage: UIImage,
        detections: [DetectedObject]?,
        analysisResult: VisionAnalysisResult?
    ) async -> VisionAnalysisResult? {
        if let analysisResult {
            return analysisResult
        }

        if let detections, !detections.isEmpty {
            return VisionAnalysisResult(objects: detections)
        }

        let analyzer = VisionAnalyzer()
        // Use a lighter analysis set in the fallback path — we're already under memory pressure
        // (native VLM generation failed). Body pose, hands, saliency, and face quality are
        // not needed for the fallback Vision context sent to the LLM.
        var options = AnalysisOptions.objectsAndText
        options.minimumTextConfidence = 0.3
        options.useAccurateOCR = true

        if let canonicalImageURL {
            print("📎 Vision fallback source: canonical attachment (\(canonicalImageURL.lastPathComponent))")
            if let canonicalImage = await loadImageFromDisk(canonicalImageURL) {
                return try? await analyzer.analyze(image: canonicalImage, options: options)
            }
            print("⚠️ Failed to load canonical attachment for fallback analysis, using in-memory image")
        } else {
            print("⚠️ No canonical attachment URL for fallback analysis, using in-memory image")
        }

        return try? await analyzer.analyze(image: fallbackImage, options: options)
    }

    private func loadImageFromDisk(_ url: URL) async -> UIImage? {
        await DiskBackedImageLoader.loadImage(at: url)
    }

    private func buildVisionEnhancedText(
        userPrompt: String,
        detections: [DetectedObject]?,
        analysisResult: VisionAnalysisResult?
    ) -> String {
        var imageContext = ""
        if let result = analysisResult {
            let detailedDescription = result.generateDetailedDescription()
            if !detailedDescription.isEmpty && !result.isEmpty {
                imageContext = detailedDescription
            } else {
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
            }
        } else if let detections, !detections.isEmpty {
            let objectList = detections.prefix(10).map { $0.label }.joined(separator: ", ")
            imageContext = "[IMAGE ANALYSIS]\n\nDETECTED OBJECTS:\n  \(objectList)"
        } else {
            imageContext = """
            [IMAGE ANALYSIS]

            An image was attached but the Vision framework couldn't process it.
            This might be due to the image format or content type.

            Please acknowledge that you can see an image was shared but cannot analyze its contents directly.
            """
        }

        return """
        --- Image Analysis Data ---
        \(imageContext)
        --- End Image Analysis ---

        User's question: \(userPrompt)

        Please respond based on the image analysis data above.
        """
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
        let canonicalURL = attachment.actualFileURL
        do {
            try await ImageStore.shared.delete(url: canonicalURL)
        } catch {
            print("Failed to delete attachment file: \(error)")
        }
        await ImageStore.shared.deleteInferenceVariant(for: canonicalURL)
        
        if let message = attachment.message {
            message.attachments.removeAll { $0.id == attachment.id }
        }
        
        context.delete(attachment)
        conversation.lastUpdated = Date()
        immediateSave()
        invalidateMLXConversationSession(reason: "attachment_deleted")
    }
}
