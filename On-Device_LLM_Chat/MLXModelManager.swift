//
//  MLXModelManager.swift
//  On-Device_LLM_Chat
//
//  Manages MLX-based language models downloaded to Documents/Models/.
//

import Foundation
import Combine
import UIKit
import MLX
import MLXLMCommon

// MARK: - MLXModelManager

@MainActor
final class MLXModelManager: ObservableObject {

    // MARK: - Model Info

    struct MLXModelInfo: Identifiable {
        let id: String
        let name: String
        let localDirName: String
        let hfRepoId: String
        let parameters: String
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

    private static let modelDefinition = MLXModelInfo(
        id: "qwen3.5-4b-4bit",
        name: "Qwen 3.5",
        localDirName: "Qwen3.5-4B-MLX-4bit",
        hfRepoId: "mlx-community/Qwen3.5-4B-MLX-4bit",
        parameters: "4B (4-bit)",
        description: "Qwen 3.5 4B multimodal model with native reasoning and image support.",
        contextLength: 262144,
        isAvailable: false,
        supportsReasoning: true,
        supportsNativeImages: true,
        requiredProcessorClass: "Qwen3VLProcessor"
    )

    private let documentsDirectory: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

    private struct ModelInstallationStatus {
        let isInstalled: Bool
        let isCompatible: Bool
        let compatibilityError: String?
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
            return ModelInstallationStatus(isInstalled: false, isCompatible: false, compatibilityError: nil)
        }
        let contents = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
        guard contents.contains(where: { $0.hasSuffix(".safetensors") }) else {
            return ModelInstallationStatus(isInstalled: false, isCompatible: false, compatibilityError: nil)
        }

        guard info.supportsNativeImages else {
            return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
        }

        guard let expectedProcessor = info.requiredProcessorClass else {
            return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
        }

        let processorURLCandidates = [
            dir.appendingPathComponent("preprocessor_config.json"),
            dir.appendingPathComponent("processor_config.json")
        ]

        var detectedProcessorClass: String?
        for url in processorURLCandidates {
            guard let data = try? Data(contentsOf: url),
                  let rawObject = try? JSONSerialization.jsonObject(with: data),
                  let json = rawObject as? [String: Any],
                  let processorClass = json["processor_class"] as? String else {
                continue
            }
            detectedProcessorClass = processorClass
            break
        }

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

    // MARK: - Init

    init() {
        var model = Self.modelDefinition
        let status = installationStatus(for: model)
        model.isAvailable = status.isInstalled && status.isCompatible
        if let error = status.compatibilityError {
            compatibilityErrors[model.id] = error
        }
        availableModels = [model]
    }

    // MARK: - Loading

    func startLoading() {
        guard let model = availableModels.first else {
            loadError = nil
            return
        }
        guard model.isAvailable else {
            loadError = compatibilityError(for: model)
            return
        }
        cancelCurrentLoad()
        load(model)
    }

    func cancelCurrentLoad() {
        loadTask?.cancel()
        loadTask = nil
        if isLoading {
            isLoading = false
            pendingModelToLoad = nil
        }
    }

    func cancelAndLoad(_ model: MLXModelInfo) {
        cancelCurrentLoad()
        load(model)
    }

    private func load(_ model: MLXModelInfo) {
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
                let loaded = try await loadModelContainer(directory: modelURL)
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self.container = loaded
                    self.currentModel = model
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                    Memory.cacheLimit = 4 * 1024 * 1024
                }
                print("✅ MLX model loaded: \(model.displayName)")
                // Pre-warm the vision encoder immediately after model load.
                // VLM encoder weights load lazily on first image query; doing it
                // now (low memory pressure) prevents a concurrent spike when the
                // user also has a large UIImage in memory.
                if model.supportsNativeImages {
                    await self.prewarmVisionEncoder()
                }
            } catch is CancellationError {
                await MainActor.run {
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                }
            } catch {
                await MainActor.run {
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                    self.loadError = "Failed to load model: \(error.localizedDescription)"
                }
                print("❌ MLX model load error: \(error)")
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
                    self.load(model)
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
    }

    // MARK: - Vision Encoder Pre-warming

    /// Runs the vision encoder on a tiny dummy image immediately after model load.
    /// This ensures encoder Metal weights are resident before the user sends a photo,
    /// eliminating the concurrent spike of (encoder load + UIImage decode) that causes OOM.
    private func prewarmVisionEncoder() async {
        guard let container else { return }
        print("🔥 Pre-warming VLM vision encoder...")
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("vlm_prewarm_\(UUID().uuidString).jpg")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        do {
            // 224×224 = 8×8 Qwen3 patches (28px each) — pre-compiles vision encoder
            // + prefill Metal shaders at model-load time so first user image query
            // hits cached shaders rather than compiling under memory pressure.
            let prewarmData: Data = autoreleasepool {
                let size = CGSize(width: 224, height: 224)
                let renderer = UIGraphicsImageRenderer(size: size)
                let img = renderer.image { ctx in
                    UIColor.gray.setFill()
                    ctx.fill(CGRect(origin: .zero, size: size))
                }
                return img.jpegData(compressionQuality: 0.8) ?? Data()
            }
            guard !prewarmData.isEmpty else { return }
            try prewarmData.write(to: tmpURL)

            let dummyMsg = Chat.Message.user("x", images: [.url(tmpURL)])
            let userInput = UserInput(chat: [dummyMsg], processing: UserInput.Processing())
            _ = try await container.prepare(input: userInput)
            Memory.clearCache()
            print("✅ VLM vision encoder pre-warmed")
        } catch {
            print("⚠️ VLM pre-warm failed (non-fatal): \(error.localizedDescription)")
        }
    }

    // MARK: - Generation

    func generateTextStream(
        messages: [Chat.Message],
        enableThinking: Bool,
        hasImage: Bool = false,
        onToken: @escaping (String) -> Void
    ) async throws {
        guard let container else {
            throw GenerationError.modelNotLoaded
        }
        Memory.clearCache()
        if hasImage {
            // Drop the Metal buffer pool to zero before vision encoding.
            // The vision encoder needs all the headroom it can get on top of the 2.5 GB model weights.
            // Restore the normal limit after generation so text-only calls benefit from the pool again.
            Memory.cacheLimit = 0
        }
        defer {
            if hasImage { Memory.cacheLimit = 4 * 1024 * 1024 }
        }
        let additionalContext: [String: any Sendable]? = enableThinking ? nil : ["enable_thinking": false]
        let processing = UserInput.Processing()
        let userInput = UserInput(chat: messages, processing: processing, additionalContext: additionalContext)
        let input = try await container.prepare(input: userInput)
        let params = enableThinking
            ? GenerateParameters(temperature: 0.6, topP: 0.95)
            : GenerateParameters(temperature: 0.7, topP: 0.8)
        let stream = try await container.generate(input: input, parameters: params)

        for await generation in stream {
            if case .chunk(let text) = generation {
                onToken(text)
            }
        }
    }

    // MARK: - Diagnostics / Management

    func refreshModelAvailability() {
        compatibilityErrors = [:]
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
            currentModel = nil
            container = nil
            Memory.clearCache()
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
        load(model)
    }

    func unloadAllModels() {
        cancelCurrentLoad()
        container = nil
        currentModel = nil
        Memory.clearCache()
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

        var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                return "No MLX model is loaded. Select the MLX backend and wait for the model to finish loading."
            }
        }
    }
}
