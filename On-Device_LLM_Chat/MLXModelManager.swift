//
//  MLXModelManager.swift
//  On-Device_LLM_Chat
//
//  Manages MLX-based language models downloaded to Documents/Models/.
//

import Foundation
import Combine
import MLX
import MLXLMCommon
import MLXVLM
import Tokenizers
import OSLog

// MARK: - MLXModelManager

@MainActor
final class MLXModelManager: ObservableObject {
    private let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM", category: "MLXModelManager")

    // MARK: - Model Info

    struct MLXModelInfo: Identifiable {
        enum LoadPolicy: Equatable {
            case standard
            case qwenMultimodal

            var packageDescription: String {
                switch self {
                case .standard:
                    return "Standard package"
                case .qwenMultimodal:
                    return "Full multimodal package"
                }
            }

            var architectureHint: String? {
                switch self {
                case .standard:
                    return nil
                case .qwenMultimodal:
                    return "Qwen3_5ForConditionalGeneration"
                }
            }

            var simulatorUnsupportedReason: String? {
                switch self {
                case .standard:
                    return nil
                case .qwenMultimodal:
                    return "Qwen multimodal MLX loading is unavailable on the simulator because the upstream MLX/Metal VLM initialization path crashes during load."
                }
            }

            var defersAutomaticPrewarm: Bool {
                switch self {
                case .standard:
                    return false
                case .qwenMultimodal:
                    return true
                }
            }
        }

        let id: String
        let name: String
        let localDirName: String
        let hfRepoId: String
        let parameters: String
        let downloadSizeLabel: String
        let loadPolicy: LoadPolicy
        let description: String
        let contextLength: Int
        var isAvailable: Bool
        let supportsReasoning: Bool
        let supportsNativeImages: Bool
        let requiredProcessorClass: String?

        init(
            id: String,
            name: String,
            localDirName: String,
            hfRepoId: String,
            parameters: String,
            downloadSizeLabel: String,
            loadPolicy: LoadPolicy = .standard,
            description: String,
            contextLength: Int,
            isAvailable: Bool,
            supportsReasoning: Bool,
            supportsNativeImages: Bool = false,
            requiredProcessorClass: String? = nil
        ) {
            self.id = id
            self.name = name
            self.localDirName = localDirName
            self.hfRepoId = hfRepoId
            self.parameters = parameters
            self.downloadSizeLabel = downloadSizeLabel
            self.loadPolicy = loadPolicy
            self.description = description
            self.contextLength = contextLength
            self.isAvailable = isAvailable
            self.supportsReasoning = supportsReasoning
            self.supportsNativeImages = supportsNativeImages
            self.requiredProcessorClass = requiredProcessorClass
        }

        var displayName: String { "\(name) (\(parameters))" }
    }

    // MARK: - Published Properties

    @Published var availableModels: [MLXModelInfo] = []
    @Published var currentModel: MLXModelInfo?
    @Published var isLoading: Bool = false
    @Published var loadError: String?
    @Published private(set) var pendingModelToLoad: MLXModelInfo?

    @Published var downloadProgress: Double = 0
    @Published var isDownloading: Bool = false
    @Published var downloadError: String?

    var supportsNativeThinking: Bool { true }

    // MARK: - Private

    private var container: ModelContainer?
    private var loadTask: Task<Void, Never>?
    private let downloader = ModelDownloader()
    private var downloaderTask: Task<Void, Never>?
    private var compatibilityErrors: [String: String] = [:]
    private var toolTemplateInspectionCache: [String: ToolTemplateInspection] = [:]
    private var packageMetadataCache: [String: ModelPackageMetadata] = [:]
    private var activeLoadID: UUID?
    private var deferredPrewarmModelID: String?
    private var prewarmInFlightModelID: String?

    private static let modelDefinitions: [MLXModelInfo] = [
        MLXModelInfo(
            id: "qwen3.5-4b-4bit",
            name: "Qwen 3.5",
            localDirName: "Qwen3.5-4B-MLX-4bit",
            hfRepoId: "mlx-community/Qwen3.5-4B-MLX-4bit",
            parameters: "4B (4-bit)",
            downloadSizeLabel: "3.03 GB",
            loadPolicy: .qwenMultimodal,
            description: "Qwen 3.5 4B multimodal model with native reasoning and image support.",
            contextLength: 262144,
            isAvailable: false,
            supportsReasoning: true,
            supportsNativeImages: true,
            requiredProcessorClass: "Qwen3VLProcessor"
        ),
        MLXModelInfo(
            id: "qwen3.5-2b-4bit",
            name: "Qwen 3.5",
            localDirName: "Qwen3.5-2B-MLX-4bit",
            hfRepoId: "mlx-community/Qwen3.5-2B-MLX-4bit",
            parameters: "2B (4-bit)",
            downloadSizeLabel: "1.75 GB",
            loadPolicy: .qwenMultimodal,
            description: "Qwen 3.5 2B multimodal model with native reasoning and image support.",
            contextLength: 262144,
            isAvailable: false,
            supportsReasoning: true,
            supportsNativeImages: true,
            requiredProcessorClass: "Qwen3VLProcessor"
        )
    ]

    private let documentsDirectory: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first ?? FileManager.default.temporaryDirectory

    private struct ModelInstallationStatus {
        let isInstalled: Bool
        let isCompatible: Bool
        let compatibilityError: String?
    }

    private struct ModelPackageMetadata {
        let architecture: String?
        let processorClass: String?
        let modelType: String?
    }

    struct MLXGenerationResult: Sendable {
        let toolInvocationCount: Int
    }

    private enum ToolTemplateSupport: Equatable {
        case supported
        case unsupported(String)
    }

    private struct ToolTemplateInspection {
        let support: ToolTemplateSupport
        let preferredToolCallFormat: ToolCallFormat?
        let usesWrappedXMLToolCalls: Bool
    }

    // MARK: - Model Directory (Documents)

    func modelDirectoryURL(for info: MLXModelInfo) -> URL? {
        let dir = documentsDirectory.appendingPathComponent("Models/\(info.localDirName)")
        let config = dir.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: config.path) ? dir : nil
    }

    private func installationStatus(for info: MLXModelInfo) -> ModelInstallationStatus {
        let dir = documentsDirectory.appendingPathComponent("Models/\(info.localDirName)")
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.appendingPathComponent("config.json").path) else {
            #if targetEnvironment(simulator)
            if let simulatorUnsupportedReason = info.loadPolicy.simulatorUnsupportedReason {
                return ModelInstallationStatus(
                    isInstalled: false,
                    isCompatible: false,
                    compatibilityError: simulatorUnsupportedReason
                )
            }
            #endif
            return ModelInstallationStatus(isInstalled: false, isCompatible: false, compatibilityError: nil)
        }
        let contents = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
        guard contents.contains(where: { $0.hasSuffix(".safetensors") }) else {
            return ModelInstallationStatus(isInstalled: false, isCompatible: false, compatibilityError: nil)
        }

        #if targetEnvironment(simulator)
        if let simulatorUnsupportedReason = info.loadPolicy.simulatorUnsupportedReason {
            return ModelInstallationStatus(
                isInstalled: true,
                isCompatible: false,
                compatibilityError: simulatorUnsupportedReason
            )
        }
        #endif

        guard info.supportsNativeImages else {
            return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
        }

        guard let expectedProcessor = info.requiredProcessorClass else {
            return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
        }

        let detectedProcessorClass = packageMetadata(for: info)?.processorClass

        guard let detectedProcessorClass else {
            return ModelInstallationStatus(
                isInstalled: true,
                isCompatible: false,
                compatibilityError:
                    "Installed model '\(info.localDirName)' is missing processor metadata for native image support. Re-download \(info.localDirName) to enable native images."
            )
        }

        guard detectedProcessorClass == expectedProcessor else {
            return ModelInstallationStatus(
                isInstalled: true,
                isCompatible: false,
                compatibilityError:
                    "Installed model '\(info.localDirName)' is incompatible with native image support (processor '\(detectedProcessorClass)'). Re-download \(info.localDirName) to get the correct multimodal files."
            )
        }

        return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
    }

    private func isModelAvailable(_ info: MLXModelInfo) -> Bool {
        let status = installationStatus(for: info)
        return status.isInstalled && status.isCompatible
    }

    private func compatibilityError(for info: MLXModelInfo) -> String? {
        compatibilityErrors[info.id]
    }

    func availabilityIssue(for info: MLXModelInfo) -> String? {
        compatibilityError(for: info)
    }

    func packageArchitecture(for info: MLXModelInfo) -> String? {
        packageMetadata(for: info)?.architecture
    }

    // MARK: - Init

    init() {
        availableModels = Self.modelDefinitions.map { definition in
            var model = definition
            let status = installationStatus(for: model)
            model.isAvailable = status.isInstalled && status.isCompatible
            if let error = status.compatibilityError {
                compatibilityErrors[model.id] = error
            }
            return model
        }
    }

    // MARK: - Loading

    func startLoading(modelID: String? = nil, source: String = "unknown") {
        let preferredModel = modelID.flatMap { id in
            availableModels.first(where: { $0.id == id })
        }
        guard let model = preferredModel ?? availableModels.first else {
            return
        }
        guard model.isAvailable else {
            loadError = compatibilityError(for: model)
            return
        }

        if currentModel?.id == model.id, container != nil, !isLoading {
            logger.notice("MLX load request ignored: model already loaded id=\(model.id, privacy: .public) source=\(source, privacy: .public)")
            return
        }

        if pendingModelToLoad?.id == model.id, isLoading {
            logger.notice("MLX load request ignored: model already loading id=\(model.id, privacy: .public) source=\(source, privacy: .public)")
            return
        }

        let loadID = UUID()
        logger.notice("MLX load requested: id=\(model.id, privacy: .public) source=\(source, privacy: .public)")

        if let architecture = packageArchitecture(for: model) {
            logger.notice(
                "MLX package metadata: id=\(model.id, privacy: .public) architecture=\(architecture, privacy: .public) package=\(model.loadPolicy.packageDescription, privacy: .public)"
            )
        }

        cancelCurrentLoad(reason: "superseded by \(model.id)")
        tearDownCurrentModel(reason: "preparing to load \(model.id)")
        activeLoadID = loadID
        load(model, loadID: loadID, source: source)
    }

    func model(withID id: String) -> MLXModelInfo? {
        availableModels.first(where: { $0.id == id })
    }

    func cancelCurrentLoad(reason: String = "cancelled") {
        loadTask?.cancel()
        loadTask = nil
        logger.notice("MLX load cancelled: reason=\(reason, privacy: .public)")
        if isLoading {
            isLoading = false
            pendingModelToLoad = nil
        }
        if activeLoadID != nil {
            activeLoadID = nil
        }
        tearDownCurrentModel(reason: "cancelCurrentLoad(\(reason))")
    }

    func cancelAndLoad(_ model: MLXModelInfo) {
        startLoading(modelID: model.id, source: "manager.cancelAndLoad")
    }

    private func load(_ model: MLXModelInfo, loadID: UUID, source: String) {
        guard model.isAvailable, let modelURL = modelDirectoryURL(for: model) else {
            if let compatibilityError = compatibilityError(for: model) {
                loadError = compatibilityError
            } else {
                loadError = "Model '\(model.localDirName)' not found in Documents/Models/. Use the Download button to fetch it."
            }
            return
        }
        pendingModelToLoad = model
        isLoading = true
        loadError = nil

        loadTask = Task { [weak self] in
            guard let self else { return }
            do {
                self.logger.notice(
                    "MLX container load start: id=\(model.id, privacy: .public) source=\(source, privacy: .public) path=\(modelURL.lastPathComponent, privacy: .public)"
                )
                let loaded = try await loadModelContainer(directory: modelURL)
                await self.applyPreferredToolCallFormatIfNeeded(to: loaded, for: model)
                guard !Task.isCancelled else { return }
                guard self.activeLoadID == loadID else {
                    self.logger.notice("MLX container load discarded: stale load id=\(model.id, privacy: .public)")
                    return
                }
                await MainActor.run {
                    self.container = loaded
                    self.currentModel = model
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                    Memory.cacheLimit = 4 * 1024 * 1024
                    self.loadTask = nil
                }
                print("✅ MLX model loaded: \(model.displayName)")
                self.logger.notice("MLX container load finished: id=\(model.id, privacy: .public)")
                await MainActor.run {
                    self.logToolTemplateSupport(for: model)
                }
                if model.loadPolicy.defersAutomaticPrewarm {
                    self.deferredPrewarmModelID = model.id
                    self.logger.notice("MLX prewarm deferred until first generation: id=\(model.id, privacy: .public)")
                } else {
                    await self.prewarmModelShaders(for: model, reason: "post-load")
                }
            } catch is CancellationError {
                guard self.activeLoadID == loadID else { return }
                await MainActor.run {
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                    self.loadTask = nil
                }
                self.logger.notice("MLX container load cancelled during execution: id=\(model.id, privacy: .public)")
            } catch {
                guard self.activeLoadID == loadID else { return }
                await MainActor.run {
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                    self.loadTask = nil
                    self.loadError = "Failed to load model: \(error.localizedDescription)"
                }
                self.tearDownCurrentModel(reason: "load failure for \(model.id)")
                print("❌ MLX model load error: \(error)")
                self.logger.error("MLX container load failed: id=\(model.id, privacy: .public) error=\(error.localizedDescription, privacy: .public)")
            }
        }
    }

    // MARK: - Download

    func startDownload(for model: MLXModelInfo) {
        guard !isDownloading else { return }
        isDownloading = true
        downloadProgress = 0
        downloadError = nil

        let targetDir = documentsDirectory.appendingPathComponent("Models/\(model.localDirName)")

        downloaderTask = Task { [weak self] in
            guard let self else { return }
            do {
                if FileManager.default.fileExists(atPath: targetDir.path) {
                    try FileManager.default.removeItem(at: targetDir)
                }
                try await self.downloader.download(
                    repoId: model.hfRepoId,
                    to: targetDir,
                    onProgress: { [weak self] progress in
                        Task { @MainActor [weak self] in
                            self?.downloadProgress = progress
                        }
                    }
                )
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self.downloadProgress = 1.0
                    self.isDownloading = false
                    self.refreshModelAvailability()
                }
                await MainActor.run {
                    self.startLoading(modelID: model.id, source: "download_complete")
                }
                print("✅ Model downloaded: \(model.displayName)")
            } catch is CancellationError {
                await MainActor.run {
                    self.isDownloading = false
                    self.downloadProgress = 0
                }
                self.cleanupPartialDownload(at: targetDir)
            } catch {
                await MainActor.run {
                    self.isDownloading = false
                    self.downloadError = error.localizedDescription
                }
                self.cleanupPartialDownload(at: targetDir)
                print("❌ Download error: \(error)")
            }
        }
    }

    func cancelDownload() {
        downloaderTask?.cancel()
        downloaderTask = nil
    }

    private func cleanupPartialDownload(at dir: URL) {
        let fm = FileManager.default
        let items = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        for item in items where item.pathExtension == "download" {
            try? fm.removeItem(at: item)
        }
        try? fm.removeItem(at: dir)
    }

    // MARK: - Shader Pre-warming

    /// Runs a minimal text forward pass immediately after model load to pre-compile all LM Metal
    /// shaders (SSM, full-attention, MoE layers). Without this the compiler spike hits on the
    /// first real inference, when the device is also holding a user image in memory.
    /// Skipped on the simulator — the Metal compute stack there can't handle a full LM forward
    /// pass and triggers EXC_BAD_ACCESS.
    private func prewarmModelShaders(for model: MLXModelInfo, reason: String) async {
        #if targetEnvironment(simulator)
        print("⚠️ Skipping LM shader pre-warm on simulator")
        return
        #else
        guard let container else { return }
        guard currentModel?.id == model.id else { return }
        if deferredPrewarmModelID != model.id && prewarmInFlightModelID == nil && reason != "post-load" {
            return
        }
        if prewarmInFlightModelID == model.id {
            return
        }
        prewarmInFlightModelID = model.id
        logger.notice("MLX prewarm start: id=\(model.id, privacy: .public) reason=\(reason, privacy: .public)")
        print("🔥 Pre-warming LM Metal shaders...")
        // Free every cached Metal buffer before shader compilation so the compilation
        // spike has the maximum possible headroom on top of the ~3 GB model weights.
        Memory.clearCache()
        Memory.cacheLimit = 0
        do {
            let warmupMsg = Chat.Message.user("Hi")
            let userInput = UserInput(chat: [warmupMsg], processing: UserInput.Processing())
            let input = try await container.prepare(input: userInput)
            let params = GenerateParameters(maxTokens: 1, temperature: 1.0, topP: 1.0)
            let stream = try await container.generate(input: input, parameters: params)
            for await _ in stream { }
            Memory.clearCache()
            Memory.cacheLimit = 4 * 1024 * 1024
            deferredPrewarmModelID = nil
            prewarmInFlightModelID = nil
            print("✅ LM Metal shaders pre-warmed")
            logger.notice("MLX prewarm finished: id=\(model.id, privacy: .public)")
        } catch {
            Memory.cacheLimit = 4 * 1024 * 1024
            prewarmInFlightModelID = nil
            print("⚠️ Shader pre-warm failed (non-fatal): \(error.localizedDescription)")
            logger.error("MLX prewarm failed: id=\(model.id, privacy: .public) error=\(error.localizedDescription, privacy: .public)")
        }
        #endif
    }

    // MARK: - Generation

    func generateTextStream(
        messages: [Chat.Message],
        enableThinking: Bool,
        memoryConstrained: Bool = false,
        tools: [MLXToolSpec] = [],
        toolDispatch: (@Sendable (MLXToolCall) async throws -> String)? = nil,
        onToken: @escaping (String) -> Void
    ) async throws -> MLXGenerationResult {
        guard let container, let currentModel else {
            throw GenerationError.modelNotLoaded
        }
        if deferredPrewarmModelID == currentModel.id && !memoryConstrained {
            await prewarmModelShaders(for: currentModel, reason: "first-generation")
        }
        logger.notice(
            "MLX generation start: model=\(currentModel.localDirName, privacy: .public) messages=\(messages.count, privacy: .public) thinking=\(enableThinking, privacy: .public) memory_constrained=\(memoryConstrained, privacy: .public) tools=\(tools.count, privacy: .public)"
        )
        let toolCallFormat = await container.configuration.toolCallFormat
        let wrappedXMLToolCallFilter: WrappedXMLToolCallStreamFilter? =
            !tools.isEmpty &&
            toolCallFormat == .xmlFunction &&
            usesWrappedXMLToolCalls(for: currentModel)
            ? WrappedXMLToolCallStreamFilter()
            : nil
        if wrappedXMLToolCallFilter != nil {
            logger.notice("MLX wrapped XML tool-call filter enabled: model=\(currentModel.localDirName, privacy: .public)")
        }
        // Always free the Metal buffer pool before inference — on a device where the model
        // alone consumes ~3 GB, every megabyte matters during the activation spike.
        Memory.clearCache()
        Memory.cacheLimit = 0
        _ = messages.contains { !$0.images.isEmpty || !$0.videos.isEmpty }
        defer {
            Memory.cacheLimit = 4 * 1024 * 1024
        }

        let maxTokens: Int? = if !tools.isEmpty {
            4096
        } else if memoryConstrained {
            1024
        } else {
            nil
        }
        let maxKVSize: Int? = if !tools.isEmpty {
            8192
        } else if memoryConstrained {
            4096
        } else {
            nil
        }
        let params = enableThinking
            ? GenerateParameters(maxTokens: maxTokens,
                                 maxKVSize: maxKVSize,
                                 temperature: 0.6, topP: 0.95)
            : GenerateParameters(maxTokens: maxTokens,
                                 maxKVSize: maxKVSize,
                                 temperature: 0.7, topP: 0.8)
        let additionalContext: [String: any Sendable]? = enableThinking ? nil : ["enable_thinking": false]
        let processing = UserInput.Processing()

        if tools.isEmpty {
            logger.notice("MLX generation using plain path (no tools)")
            let userInput = UserInput(chat: messages, processing: processing, additionalContext: additionalContext)
            let input = try await container.prepare(input: userInput)
            logger.notice("MLX plain path prepared input successfully")
            let stream = try await container.generate(input: input, parameters: params)
            for await generation in stream {
                if case .chunk(let text) = generation {
                    onToken(text)
                }
            }
            logger.notice("MLX plain path completed")
            return MLXGenerationResult(toolInvocationCount: 0)
        }

        try validateToolTemplateSupport(for: currentModel)
        guard let lastMessage = messages.last else {
            throw GenerationError.invalidChatHistory
        }
        logger.notice(
            "MLX tool path enabled: last_role=\(lastMessage.role.rawValue, privacy: .public) last_chars=\(lastMessage.content.count, privacy: .public)"
        )

        do {
            let preflightInput = UserInput(
                chat: messages,
                processing: processing,
                tools: tools,
                additionalContext: additionalContext
            )
            let prepared = try await container.prepare(input: preflightInput)
            logger.notice(
                "MLX tool preflight prepare succeeded: token_count=\(prepared.text.tokens.size, privacy: .public) has_image=\(prepared.image != nil, privacy: .public) has_video=\(prepared.video != nil, privacy: .public)"
            )
            Memory.clearCache()
        } catch {
            logger.error("MLX tool preflight prepare failed: \(error.localizedDescription, privacy: .public)")
            throw error
        }

        if wrappedXMLToolCallFilter != nil {
            let result = try await generateWrappedXMLToolTextStream(
                container: container,
                messages: messages,
                processing: processing,
                additionalContext: additionalContext,
                tools: tools,
                params: params,
                toolDispatch: toolDispatch,
                onToken: onToken
            )
            logger.notice("MLX tool path completed: tool_invocations=\(result.toolInvocationCount, privacy: .public)")
            return result
        }

        let invocationTracker = ToolInvocationTracker()
        let session = ChatSession(
            container,
            history: Array(messages.dropLast()),
            generateParameters: params,
            processing: processing,
            additionalContext: additionalContext,
            tools: tools,
            toolDispatch: { toolCall in
                await invocationTracker.increment()
                self.logger.notice("MLX dispatching tool call: name=\(toolCall.function.name, privacy: .public)")
                // Free activation buffers from the previous generate step before
                // running the tool — the model → tool → model cycle would otherwise
                // hold two sets of activations simultaneously.
                Memory.clearCache()
                guard let toolDispatch else {
                    throw GenerationError.missingToolDispatch
                }
                return try await toolDispatch(toolCall)
            }
        )
        let stream = session.streamResponse(
            to: lastMessage.content,
            role: lastMessage.role,
            images: lastMessage.images,
            videos: lastMessage.videos
        )
        logger.notice("MLX tool path stream opened")

        for try await generation in stream {
            onToken(generation)
        }

        let toolInvocationCount = await invocationTracker.value
        logger.notice("MLX tool path completed: tool_invocations=\(toolInvocationCount, privacy: .public)")
        return MLXGenerationResult(toolInvocationCount: toolInvocationCount)
    }

    // MARK: - Diagnostics / Management

    func refreshModelAvailability() {
        compatibilityErrors = [:]
        toolTemplateInspectionCache = [:]
        packageMetadataCache = [:]
        availableModels = availableModels.map { model in
            var updated = model
            let status = installationStatus(for: model)
            updated.isAvailable = status.isInstalled && status.isCompatible
            if let error = status.compatibilityError {
                compatibilityErrors[model.id] = error
            }
            return updated
        }
    }

    func deleteModel(_ model: MLXModelInfo) throws {
        if currentModel?.id == model.id {
            tearDownCurrentModel(reason: "delete \(model.id)")
        }
        let dir = documentsDirectory.appendingPathComponent("Models/\(model.localDirName)")
        do {
            try FileManager.default.removeItem(at: dir)
            refreshModelAvailability()
            print("🗑️ Deleted model '\(model.localDirName)' from Documents/Models/")
        } catch let error as NSError where error.code == NSFileNoSuchFileError {
            print("ℹ️ Model '\(model.localDirName)' not found in Documents/Models/ — nothing to delete.")
        }
    }

    func loadModel(_ model: MLXModelInfo) async {
        startLoading(modelID: model.id, source: "manager.loadModel")
    }

    func unloadAllModels() {
        cancelCurrentLoad(reason: "unloadAllModels")
        tearDownCurrentModel(reason: "unloadAllModels")
        print("🧹 Unloaded all MLX models")
    }

    func printModelLocations() {
        for model in availableModels {
            if let url = modelDirectoryURL(for: model) {
                print("📁 \(model.name): \(url.path) — available: \(model.isAvailable)")
            } else {
                print("📁 \(model.name): not found in Documents/Models/ — available: \(model.isAvailable)")
            }
        }
    }

    func printModelInputOutputInfo() {
        guard let current = currentModel else {
            print("⚠️ No model loaded")
            return
        }
        print("ℹ️ Current model: \(current.displayName)")
        print("   Context length: \(current.contextLength) tokens")
        print("   Reasoning: \(current.supportsReasoning)")
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

    private actor ToolInvocationTracker {
        private(set) var value = 0

        func increment() {
            value += 1
        }
    }

    private func generateWrappedXMLToolTextStream(
        container: ModelContainer,
        messages: [Chat.Message],
        processing: UserInput.Processing,
        additionalContext: [String: any Sendable]?,
        tools: [MLXToolSpec],
        params: GenerateParameters,
        toolDispatch: (@Sendable (MLXToolCall) async throws -> String)?,
        onToken: @escaping (String) -> Void
    ) async throws -> MLXGenerationResult {
        var rawMessages = Qwen3VLMessageGenerator().generate(messages: messages)
        let conversationImages = messages.flatMap(\.images)
        let conversationVideos = messages.flatMap(\.videos)
        var toolInvocationCount = 0

        while true {
            var userInput = UserInput(
                messages: rawMessages,
                images: conversationImages,
                videos: conversationVideos,
                tools: tools,
                additionalContext: additionalContext
            )
            userInput.processing = processing

            let input = try await container.prepare(input: userInput)
            let stream = try await container.generate(input: input, parameters: params)
            logger.notice("MLX wrapped XML tool path stream opened")

            let outputFilter = WrappedXMLToolCallStreamFilter()
            var assistantVisibleText = ""
            var emittedToolCall: MLXToolCall?

            for await generation in stream {
                switch generation {
                case .chunk(let text):
                    if let visibleChunk = await outputFilter.consume(text) {
                        assistantVisibleText += visibleChunk
                        onToken(visibleChunk)
                    }
                case .toolCall(let toolCall):
                    emittedToolCall = toolCall
                    await outputFilter.didDispatchToolCall()
                case .info:
                    break
                }
            }

            if let trailingText = await outputFilter.finish() {
                assistantVisibleText += trailingText
                onToken(trailingText)
            }

            guard let emittedToolCall else {
                return MLXGenerationResult(toolInvocationCount: toolInvocationCount)
            }

            toolInvocationCount += 1
            logger.notice("MLX dispatching tool call: name=\(emittedToolCall.function.name, privacy: .public)")
            Memory.clearCache()
            guard let toolDispatch else {
                throw GenerationError.missingToolDispatch
            }

            let toolResult = try await toolDispatch(emittedToolCall)
            rawMessages.append(Self.assistantToolCallMessage(
                content: assistantVisibleText,
                toolCall: emittedToolCall
            ))
            rawMessages.append(Self.toolResponseMessage(toolResult))
        }
    }

    private static func assistantToolCallMessage(
        content: String,
        toolCall: MLXToolCall
    ) -> MLXLMCommon.Message {
        var message: MLXLMCommon.Message = [
            "role": Chat.Message.Role.assistant.rawValue,
            "content": content
        ]
        message["tool_calls"] = [[
            "function": [
                "name": toolCall.function.name,
                "arguments": Self.normalizedToolArguments(toolCall.function.arguments)
            ]
        ]]
        return message
    }

    private static func toolResponseMessage(_ content: String) -> MLXLMCommon.Message {
        [
            "role": Chat.Message.Role.tool.rawValue,
            "content": content
        ]
    }

    internal static func normalizedToolArguments(
        _ arguments: [String: any Sendable]
    ) -> [String: any Sendable] {
        arguments.reduce(into: [String: any Sendable]()) { partialResult, entry in
            partialResult[entry.key] = normalizedToolArgumentValue(entry.value)
        }
    }

    private static func normalizedToolArgumentValue(_ value: any Sendable) -> any Sendable {
        switch value {
        case let jsonValue as JSONValue:
            return normalized(jsonValue)
        case let string as String:
            return string
        case let bool as Bool:
            return bool
        case let int as Int:
            return int
        case let double as Double:
            return double
        case let array as [JSONValue]:
            return array.map { normalized($0) }
        case let object as [String: JSONValue]:
            return object.mapValues { normalized($0) }
        default:
            return String(describing: value)
        }
    }

    private static func normalized(_ value: JSONValue) -> any Sendable {
        switch value {
        case .null:
            return "null"
        case .bool(let bool):
            return bool
        case .int(let int):
            return int
        case .double(let double):
            return double
        case .string(let string):
            return string
        case .array(let array):
            return array.map { normalized($0) }
        case .object(let object):
            return object.mapValues { normalized($0) }
        }
    }

    private func applyPreferredToolCallFormatIfNeeded(
        to loaded: ModelContainer,
        for model: MLXModelInfo
    ) async {
        guard let preferredToolCallFormat = preferredToolCallFormat(for: model) else {
            return
        }

        let currentToolCallFormat = await loaded.configuration.toolCallFormat
        guard currentToolCallFormat != preferredToolCallFormat else {
            logger.notice(
                "MLX tool-call format retained: id=\(model.id, privacy: .public) format=\(preferredToolCallFormat.rawValue, privacy: .public)"
            )
            return
        }

        await loaded.update { context in
            context.configuration.toolCallFormat = preferredToolCallFormat
        }
        logger.notice(
            "MLX tool-call format override applied: id=\(model.id, privacy: .public) format=\(preferredToolCallFormat.rawValue, privacy: .public)"
        )
    }

    private func validateToolTemplateSupport(for model: MLXModelInfo) throws {
        switch toolTemplateSupport(for: model) {
        case .supported:
            return
        case .unsupported(let message):
            throw GenerationError.unsupportedToolTemplate(message)
        }
    }

    private func logToolTemplateSupport(for model: MLXModelInfo) {
        let inspection = toolTemplateInspection(for: model)
        switch inspection.support {
        case .supported:
            print("✅ MLX tool template support confirmed for \(model.displayName)")
            if let preferredToolCallFormat = inspection.preferredToolCallFormat {
                logger.notice(
                    "MLX tool template inspection: id=\(model.id, privacy: .public) format=\(preferredToolCallFormat.rawValue, privacy: .public) wrapped_xml=\(inspection.usesWrappedXMLToolCalls, privacy: .public)"
                )
            }
        case .unsupported(let message):
            print("⚠️ MLX tool template support unavailable: \(message)")
        }
    }

    private func toolTemplateSupport(for model: MLXModelInfo) -> ToolTemplateSupport {
        toolTemplateInspection(for: model).support
    }

    private func preferredToolCallFormat(for model: MLXModelInfo) -> ToolCallFormat? {
        toolTemplateInspection(for: model).preferredToolCallFormat
    }

    private func usesWrappedXMLToolCalls(for model: MLXModelInfo) -> Bool {
        toolTemplateInspection(for: model).usesWrappedXMLToolCalls
    }

    private func toolTemplateInspection(for model: MLXModelInfo) -> ToolTemplateInspection {
        if let cached = toolTemplateInspectionCache[model.id] {
            return cached
        }

        guard let modelURL = modelDirectoryURL(for: model) else {
            let inspection = ToolTemplateInspection(
                support: .unsupported(
                    "Installed model '\(model.localDirName)' could not be inspected for tool support. Re-download the model package to restore the tokenizer chat template."
                ),
                preferredToolCallFormat: nil,
                usesWrappedXMLToolCalls: false
            )
            toolTemplateInspectionCache[model.id] = inspection
            return inspection
        }

        let candidateFiles = [
            "tokenizer_config.json",
            "tokenizer.json",
            "chat_template.jinja",
            "chat_template.json",
            "processor_config.json",
            "preprocessor_config.json"
        ]

        let combinedContents = candidateFiles.compactMap { fileName -> String? in
            let url = modelURL.appendingPathComponent(fileName)
            guard FileManager.default.fileExists(atPath: url.path),
                  let data = try? Data(contentsOf: url),
                  let text = String(data: data, encoding: .utf8) else {
                return nil
            }
            return text
        }.joined(separator: "\n")

        let lowered = combinedContents.lowercased()
        let hasToolsContext = lowered.contains("tools")
        let hasToolRole = lowered.contains("\"tool\"") ||
            lowered.contains("'tool'") ||
            lowered.contains("role == 'tool'") ||
            lowered.contains("role != 'tool'")
        let hasToolCalls = lowered.contains("tool_call") || lowered.contains("tool_calls")
        let preferredToolCallFormat = Self.inferToolCallFormat(
            packageContents: combinedContents,
            modelType: packageMetadata(for: model)?.modelType
        )
        let usesWrappedXMLToolCalls = Self.usesWrappedXMLToolCallTemplate(packageContents: combinedContents)

        let support: ToolTemplateSupport
        if hasToolsContext && (hasToolRole || hasToolCalls) {
            support = .supported
        } else {
            support = .unsupported(
                "Installed model '\(model.localDirName)' does not expose a tool-aware chat template. Re-download or update this Qwen 3.5 MLX model package to enable MLX tool calling."
            )
        }

        let inspection = ToolTemplateInspection(
            support: support,
            preferredToolCallFormat: preferredToolCallFormat,
            usesWrappedXMLToolCalls: usesWrappedXMLToolCalls
        )
        toolTemplateInspectionCache[model.id] = inspection
        return inspection
    }

    private func packageMetadata(for info: MLXModelInfo) -> ModelPackageMetadata? {
        if let cached = packageMetadataCache[info.id] {
            return cached
        }

        guard let modelURL = modelDirectoryURL(for: info) else {
            return nil
        }

        let configURL = modelURL.appendingPathComponent("config.json")
        let configJSON = (
            (try? Data(contentsOf: configURL))
                .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        ) ?? [:]
        let architecture = (configJSON["architectures"] as? [String])?.first
        let modelType = configJSON["model_type"] as? String

        let processorURLCandidates = [
            modelURL.appendingPathComponent("preprocessor_config.json"),
            modelURL.appendingPathComponent("processor_config.json")
        ]

        var processorClass: String?
        for url in processorURLCandidates {
            guard let data = try? Data(contentsOf: url),
                  let rawObject = try? JSONSerialization.jsonObject(with: data),
                  let json = rawObject as? [String: Any],
                  let detected = json["processor_class"] as? String else {
                continue
            }
            processorClass = detected
            break
        }

        let metadata = ModelPackageMetadata(
            architecture: architecture,
            processorClass: processorClass,
            modelType: modelType
        )
        packageMetadataCache[info.id] = metadata
        return metadata
    }

    internal static func inferToolCallFormat(
        packageContents: String,
        modelType: String?
    ) -> ToolCallFormat? {
        let lowered = packageContents.lowercased()

        if lowered.contains("<function=") && lowered.contains("<parameter=") {
            return .xmlFunction
        }

        let hasJSONToolCallExample =
            lowered.contains("<tool_call>") &&
            lowered.contains("\"name\"") &&
            lowered.contains("\"arguments\"")
        if hasJSONToolCallExample {
            return .json
        }

        if let modelType,
           modelType.lowercased().hasPrefix("qwen3_5"),
           lowered.contains("tool_call") &&
           lowered.contains("function") {
            return .xmlFunction
        }

        return nil
    }

    internal static func usesWrappedXMLToolCallTemplate(packageContents: String) -> Bool {
        let lowered = packageContents.lowercased()
        return lowered.contains("<tool_call>") &&
            lowered.contains("<function=") &&
            lowered.contains("<parameter=")
    }

    private func tearDownCurrentModel(reason: String) {
        if container != nil || currentModel != nil || deferredPrewarmModelID != nil || prewarmInFlightModelID != nil {
            logger.notice("MLX model teardown: reason=\(reason, privacy: .public)")
        }
        container = nil
        currentModel = nil
        deferredPrewarmModelID = nil
        prewarmInFlightModelID = nil
        Memory.clearCache()
        Memory.cacheLimit = 4 * 1024 * 1024
    }
}

actor WrappedXMLToolCallStreamFilter {
    private static let toolMarkupMarkers = [
        "<tool_call>",
        "<function=",
        "<parameter=",
        "</parameter>",
        "</function>",
        "</tool_call>"
    ]

    private var pendingText = ""
    private var suppressingToolMarkup = false

    func consume(_ chunk: String) -> String? {
        guard !chunk.isEmpty else { return nil }

        if suppressingToolMarkup {
            return nil
        }

        pendingText += chunk

        if let markerRange = Self.firstMarkerRange(in: pendingText) {
            let visiblePrefix = String(pendingText[..<markerRange.lowerBound])
            pendingText = ""
            suppressingToolMarkup = true
            return visiblePrefix.isEmpty ? nil : visiblePrefix
        }

        let safePrefixLength = Self.safePrefixLength(in: pendingText)
        guard safePrefixLength > 0 else {
            return nil
        }

        let safeEnd = pendingText.index(pendingText.startIndex, offsetBy: safePrefixLength)
        let safeChunk = String(pendingText[..<safeEnd])
        pendingText = String(pendingText[safeEnd...])
        return safeChunk.isEmpty ? nil : safeChunk
    }

    func didDispatchToolCall() {
        pendingText = ""
        suppressingToolMarkup = false
    }

    func finish() -> String? {
        defer {
            pendingText = ""
            suppressingToolMarkup = false
        }

        guard !suppressingToolMarkup, !pendingText.isEmpty else {
            return nil
        }

        return pendingText
    }

    private static func firstMarkerRange(in text: String) -> Range<String.Index>? {
        toolMarkupMarkers
            .compactMap { marker in text.range(of: marker) }
            .min { $0.lowerBound < $1.lowerBound }
    }

    private static func safePrefixLength(in text: String) -> Int {
        guard text.contains("<") else {
            return text.count
        }

        let suffixLength = longestMarkerPrefixSuffix(in: text)
        return text.count - suffixLength
    }

    private static func longestMarkerPrefixSuffix(in text: String) -> Int {
        let maxSuffixLength = min(
            text.count,
            toolMarkupMarkers.map(\.count).max() ?? 0
        )
        guard maxSuffixLength > 0 else {
            return 0
        }

        for suffixLength in stride(from: maxSuffixLength, through: 1, by: -1) {
            let suffixStart = text.index(text.endIndex, offsetBy: -suffixLength)
            let suffix = String(text[suffixStart...])
            if toolMarkupMarkers.contains(where: { $0.hasPrefix(suffix) }) {
                return suffixLength
            }
        }

        return 0
    }
}
