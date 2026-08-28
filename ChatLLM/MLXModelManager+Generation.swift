//
//  MLXModelManager+Generation
//  ChatLLM
//
//  Split out of MLXModelManager.swift as part of the Type+Concern organization.
//

import Foundation
import MLX
import MLXLMCommon
import OSLog

extension MLXModelManager {
    // MARK: - Generation

    nonisolated private static let maximumPromptTokenCountCacheEntries = 256
    nonisolated private static let rotorQuantKeyBits = 3
    nonisolated private static let rotorQuantValueBits = 2
    nonisolated private static let rotorQuantSeed: UInt64 = 42
    nonisolated private static let rotorQuantExactBufferSize = 128
    nonisolated private static let rotorQuantAttentionBlockTokens = 128
    nonisolated private static let rotorQuantMediaExactBufferSize = 32
    nonisolated private static let rotorQuantMediaAttentionBlockTokens = 64
    nonisolated private static let memoryConstrainedMaxKVSize = 4096
    nonisolated internal static func promptTokenCountFingerprint(
        role: Chat.Message.Role,
        content: String
    ) -> Int {
        var hasher = Hasher()
        hasher.combine(role.rawValue)
        hasher.combine(content)
        return hasher.finalize()
    }

    func tokenCountForLoadedModelText(
        role: Chat.Message.Role,
        content: String
    ) async -> Int? {
        guard let currentModel, let container else { return nil }
        let cacheKey = PromptTokenCountCacheKey(
            modelID: currentModel.id,
            fingerprint: Self.promptTokenCountFingerprint(role: role, content: content)
        )
        if let cachedCount = promptTokenCountCache[cacheKey] {
            if let existingIndex = promptTokenCountCacheOrder.firstIndex(of: cacheKey) {
                promptTokenCountCacheOrder.remove(at: existingIndex)
            }
            promptTokenCountCacheOrder.append(cacheKey)
            return cachedCount
        }
        let tokenCount = await container.encode(content).count
        promptTokenCountCache[cacheKey] = tokenCount
        promptTokenCountCacheOrder.append(cacheKey)
        if promptTokenCountCacheOrder.count > Self.maximumPromptTokenCountCacheEntries {
            let evictedKey = promptTokenCountCacheOrder.removeFirst()
            promptTokenCountCache.removeValue(forKey: evictedKey)
        }
        return tokenCount
    }

    nonisolated internal static func shouldPersistSessionAcrossTurns(
        enableThinking: Bool,
        hasTools: Bool,
        hasMedia: Bool
    ) -> Bool {
        _ = enableThinking
        _ = hasTools
        _ = hasMedia
        return true
    }
    struct PromptTokenCountCacheKey: Hashable {
        let modelID: String
        let fingerprint: Int
    }

    struct MLXGenerationResult: Sendable {
        let toolInvocationCount: Int
    }

    enum CacheCompressionMode: Equatable, Sendable {
        case none
        case rotorQuant(RotorQuantConfiguration)

        nonisolated var diagnosticLabel: String {
            switch self {
            case .none:
                return "none"
            case .rotorQuant(let configuration):
                return "rotorquant(\(configuration.variant.rawValue),k\(configuration.keyBits),v\(configuration.valueBits),exact\(configuration.exactBufferSize),block\(configuration.attentionBlockTokens))"
            }
        }

        nonisolated var generateParametersCompression: KVCacheCompressionMode? {
            switch self {
            case .none:
                return nil
            case .rotorQuant(let configuration):
                return .rotorQuant(configuration)
            }
        }
    }

    struct GenerationConfiguration: Equatable, Sendable {
        let maxTokens: Int?
        let cachePolicy: MLXCachePolicy
        let cacheCompression: CacheCompressionMode

        var maxKVSize: Int? {
            cachePolicy.maxKVSize
        }

    }
    var prefersBoundedKVCacheAcrossTurns: Bool {
        deviceSupportProfile.hasLowMemoryForPersistentKVCache
    }

    func configuredContextWindowLimit(for model: MLXModelInfo? = nil) -> Int {
        UserDefaults.standard.mlxContextWindowTokens(
            deviceMaximum: deviceSupportProfile.maxContextWindowTokens(for: model ?? currentModel)
        )
    }

    var shouldPreferRotorQuant: Bool {
        UserDefaults.standard.mlxEnableRotorQuant
    }

    func generateTextStream(
        conversationID: UUID,
        messages: [Chat.Message],
        enableThinking: Bool,
        memoryConstrained: Bool = false,
        tools: [MLXToolSpec] = [],
        toolDispatch: (@Sendable (MLXToolCall) async throws -> String)? = nil,
        onToken: @escaping @Sendable (String) -> Void,
        onToolCall: @escaping @Sendable (MLXToolCall) -> Void = { _ in }
    ) async throws -> MLXGenerationResult {
        guard let container, let currentModel else {
            throw GenerationError.modelNotLoaded
        }
        await cancelDeferredTuningBeforeGeneration(for: currentModel)
        guard self.container === container, self.currentModel?.id == currentModel.id else {
            throw GenerationError.modelNotLoaded
        }
        let includesMedia = messages.contains { !$0.images.isEmpty || !$0.videos.isEmpty }
        let configuredMaxOutputTokens = UserDefaults.standard.mlxMaxOutputTokensLimit
        let configuredContextWindow = configuredContextWindowLimit(for: currentModel)
        let rotorQuantRequested = UserDefaults.standard.mlxEnableRotorQuant
        let generationConfiguration = Self.generationConfiguration(
            isEnabled: rotorQuantRequested,
            preferRotorQuant: shouldPreferRotorQuant,
            hasTools: !tools.isEmpty,
            hasMedia: includesMedia,
            memoryConstrained: memoryConstrained,
            prefersBoundedCache: prefersBoundedKVCacheAcrossTurns,
            configuredMaxOutputTokens: configuredMaxOutputTokens,
            configuredContextWindow: configuredContextWindow
        )
        if deferredPrewarmModelID == currentModel.id && !memoryConstrained {
            await prewarmModelShaders(for: currentModel, reason: "first-generation")
        }
        logger.notice(
            "MLX generation start: model=\(currentModel.localDirName, privacy: .public) conversation=\(conversationID.uuidString, privacy: .public) messages=\(messages.count, privacy: .public) thinking=\(enableThinking, privacy: .public) memory_constrained=\(memoryConstrained, privacy: .public) tools=\(tools.count, privacy: .public) media=\(includesMedia, privacy: .public) cache_policy=\(generationConfiguration.cachePolicy.diagnosticLabel, privacy: .public) cache_compression=\(generationConfiguration.cacheCompression.diagnosticLabel, privacy: .public) max_kv=\(generationConfiguration.maxKVSize ?? -1, privacy: .public)"
        )
        MLXMemoryDiagnostics.log(logger, "generation.start")
        let toolCallFormat = await container.configuration.toolCallFormat
        let suppressWrappedXMLToolMarkup =
            !tools.isEmpty &&
            toolCallFormat == .xmlFunction &&
            usesWrappedXMLToolCalls(for: currentModel)
        if suppressWrappedXMLToolMarkup {
            logger.notice("MLX wrapped XML tool-call filter enabled: model=\(currentModel.localDirName, privacy: .public)")
        }
        prepareMemoryForGeneration()

        let tunedPrefillStepSize = await inferenceWorker.prefillStepSize(for: currentModel.id)
        let prefillStepSize = Self.adaptivePrefillStepSize(
            tuned: tunedPrefillStepSize,
            messages: messages
        )
        let params = Self.makeGenerateParameters(
            maxTokens: generationConfiguration.maxTokens,
            maxKVSize: generationConfiguration.maxKVSize,
            cacheCompression: generationConfiguration.cacheCompression,
            enableThinking: enableThinking,
            includesMedia: includesMedia,
            currentModelID: currentModel.id,
            prefillStepSize: prefillStepSize,
            repetitionPenalty: UserDefaults.standard.mlxRepetitionPenaltyValue
        )
        let additionalContext: [String: any Sendable]? = ["enable_thinking": enableThinking]
        let processing = UserInput.Processing()

        if !tools.isEmpty {
            try validateToolTemplateSupport(for: currentModel)
        }

        let sessionKey = MLXSessionKey(
            conversationID: conversationID,
            modelID: currentModel.id,
            enableThinking: enableThinking,
            toolsEnabled: !tools.isEmpty,
            includesMedia: includesMedia,
            instructionFingerprint: messages.first(where: { $0.role == .system })?.content.hashValue ?? 0
        )
        let request = MLXInferenceRequest(
            conversationID: conversationID,
            sessionKey: sessionKey,
            model: currentModel,
            messages: messages,
            enableThinking: enableThinking,
            additionalContext: additionalContext,
            processing: processing,
            tools: tools,
            params: params,
            suppressWrappedXMLToolMarkup: suppressWrappedXMLToolMarkup
        )

        do {
            let response = try await inferenceWorker.generate(
                request: request,
                toolDispatch: toolDispatch,
                onToken: onToken,
                onToolCall: onToolCall
            )
            latestPerformanceSample = response.performanceSample
            cleanupMemoryAfterGeneration()
            if !memoryConstrained {
                await startDeferredTuningIfNeeded(container: container, model: currentModel)
            }
            return MLXGenerationResult(toolInvocationCount: response.toolInvocationCount)
        } catch {
            logger.error("MLX generation failed: \(error.localizedDescription, privacy: .public)")
            cleanupMemoryAfterGenerationError()
            throw error
        }
    }

    // MARK: - Errors

    enum GenerationError: LocalizedError {
        case modelNotLoaded
        case invalidChatHistory
        case missingToolDispatch
        case unsupportedToolTemplate(String)

        var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                return "No MLX model is loaded. Select the MLX backend and wait for the model to finish loading."
            case .invalidChatHistory:
                return "The MLX chat history is empty, so generation cannot start."
            case .missingToolDispatch:
                return "The model emitted a tool call, but no MLX tool dispatcher was configured."
            case .unsupportedToolTemplate(let message):
                return message
            }
        }
    }
    nonisolated internal static func generationConfiguration(
        isEnabled: Bool,
        preferRotorQuant: Bool,
        hasTools: Bool,
        hasMedia: Bool,
        memoryConstrained: Bool,
        prefersBoundedCache: Bool,
        configuredMaxOutputTokens: Int?,
        configuredContextWindow: Int
    ) -> GenerationConfiguration {
        let maxTokens: Int? = if hasTools {
            configuredMaxOutputTokens.map { min($0, 4096) } ?? 4096
        } else if memoryConstrained {
            configuredMaxOutputTokens.map { min($0, 1024) }
        } else {
            configuredMaxOutputTokens
        }
        let cachePolicy = cachePolicy(
            isEnabled: isEnabled && preferRotorQuant,
            hasTools: hasTools,
            prefersBoundedCache: prefersBoundedCache,
            memoryConstrained: memoryConstrained
        )
        let clampedCachePolicy = clampedCachePolicy(
            cachePolicy,
            configuredContextWindow: configuredContextWindow
        )
        return GenerationConfiguration(
            maxTokens: maxTokens,
            cachePolicy: clampedCachePolicy,
            cacheCompression: cacheCompressionMode(
                cachePolicy: clampedCachePolicy,
                preferRotorQuant: preferRotorQuant,
                hasMedia: hasMedia
            )
        )
    }

    nonisolated internal static func cachePolicy(
        isEnabled: Bool,
        hasTools: Bool,
        prefersBoundedCache: Bool,
        memoryConstrained: Bool
    ) -> MLXCachePolicy {
        if memoryConstrained || prefersBoundedCache {
            return .boundedRotating(maxKVSize: memoryConstrainedMaxKVSize)
        }
        guard !hasTools else {
            return .persistentSimple
        }

        return isEnabled ? .persistentRotorQuant : .persistentSimple
    }

    nonisolated internal static func cachePolicy(for parameters: GenerateParameters) -> MLXCachePolicy {
        if let maxKVSize = parameters.maxKVSize {
            return .boundedRotating(maxKVSize: maxKVSize)
        }
        if parameters.resolvedCacheCompression != nil {
            return .persistentRotorQuant
        }
        return .persistentSimple
    }

    nonisolated private static func clampedCachePolicy(
        _ cachePolicy: MLXCachePolicy,
        configuredContextWindow: Int
    ) -> MLXCachePolicy {
        switch cachePolicy {
        case .boundedRotating(let maxKVSize):
            return .boundedRotating(maxKVSize: min(maxKVSize, configuredContextWindow))
        case .persistentSimple, .persistentRotorQuant:
            return cachePolicy
        }
    }

    nonisolated internal static func cacheCompressionMode(
        cachePolicy: MLXCachePolicy,
        preferRotorQuant: Bool,
        hasMedia: Bool = false
    ) -> CacheCompressionMode {
        guard cachePolicy.usesRotorQuantCompression else {
            return .none
        }

        if preferRotorQuant {
            let exactBufferSize = hasMedia ? rotorQuantMediaExactBufferSize : rotorQuantExactBufferSize
            let attentionBlockTokens =
                hasMedia ? rotorQuantMediaAttentionBlockTokens : rotorQuantAttentionBlockTokens
            return .rotorQuant(
                RotorQuantConfiguration(
                    keyBits: rotorQuantKeyBits,
                    valueBits: rotorQuantValueBits,
                    seed: rotorQuantSeed,
                    exactBufferSize: exactBufferSize,
                    attentionBlockTokens: attentionBlockTokens,
                    variant: .iso
                )
            )
        }

        return .none
    }

    nonisolated private static func makeGenerateParameters(
        maxTokens: Int?,
        maxKVSize: Int?,
        cacheCompression: CacheCompressionMode,
        enableThinking: Bool,
        includesMedia: Bool,
        currentModelID: String,
        prefillStepSize: Int,
        repetitionPenalty: Float?
    ) -> GenerateParameters {
        let isQwen35Model = currentModelID.hasPrefix("qwen3.5-")

        if enableThinking && isQwen35Model {
            let qwen35ThinkingTemperature: Float = includesMedia ? 0.6 : 1.0
            let qwen35ThinkingPresencePenalty: Float = includesMedia ? 0.0 : 1.5
            return GenerateParameters(
                maxTokens: maxTokens,
                maxKVSize: maxKVSize,
                cacheCompression: cacheCompression.generateParametersCompression,
                temperature: qwen35ThinkingTemperature,
                topP: 0.95,
                topK: 20,
                minP: 0.0,
                presencePenalty: qwen35ThinkingPresencePenalty,
                repetitionPenalty: 1.0,
                prefillStepSize: prefillStepSize
            )
        } else if enableThinking {
            return GenerateParameters(
                maxTokens: maxTokens,
                maxKVSize: maxKVSize,
                cacheCompression: cacheCompression.generateParametersCompression,
                temperature: 0.6,
                topP: 0.95,
                topK: nil,
                minP: 0.0,
                presencePenalty: nil,
                repetitionPenalty: repetitionPenalty,
                prefillStepSize: prefillStepSize
            )
        } else if isQwen35Model {
            let qwen35NonThinkingTemperature: Float = includesMedia ? 0.7 : 1.0
            let qwen35NonThinkingTopP: Float = includesMedia ? 0.8 : 1.0
            let qwen35NonThinkingPresencePenalty: Float = includesMedia ? 1.5 : 2.0
            return GenerateParameters(
                maxTokens: maxTokens,
                maxKVSize: maxKVSize,
                cacheCompression: cacheCompression.generateParametersCompression,
                temperature: qwen35NonThinkingTemperature,
                topP: qwen35NonThinkingTopP,
                topK: 20,
                minP: 0.0,
                presencePenalty: qwen35NonThinkingPresencePenalty,
                repetitionPenalty: 1.0,
                prefillStepSize: prefillStepSize
            )
        } else {
            return GenerateParameters(
                maxTokens: maxTokens,
                maxKVSize: maxKVSize,
                cacheCompression: cacheCompression.generateParametersCompression,
                temperature: 0.7,
                topP: 0.8,
                topK: nil,
                minP: 0.0,
                presencePenalty: nil,
                repetitionPenalty: repetitionPenalty,
                prefillStepSize: prefillStepSize
            )
        }
    }
}
