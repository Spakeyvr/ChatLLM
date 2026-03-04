//
//  MLXModelManager.swift
//  On-Device_LLM_Chat
//
//  Manages MLX-based language models downloaded to Documents/Models/.
//

import Foundation
import Combine
import MLX
import MLXLLM
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

    private static let modelDefinition = MLXModelInfo(
        id: "qwen3.5-4b-4bit",
        name: "Qwen 3.5",
        localDirName: "Qwen3.5-4B-MLX-4bit",
        hfRepoId: "Qwen/Qwen3-4B-MLX-4bit",
        parameters: "4B (4-bit)",
        description: "Alibaba's Qwen 3.5 4B model with extended 262K context and native reasoning support.",
        contextLength: 262144,
        isAvailable: false,
        supportsReasoning: true
    )

    private let documentsDirectory: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

    // MARK: - Model Directory (Documents)

    func modelDirectoryURL(for info: MLXModelInfo) -> URL? {
        let dir = documentsDirectory.appendingPathComponent("Models/\(info.localDirName)")
        let config = dir.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: config.path) ? dir : nil
    }

    private func isModelAvailable(_ info: MLXModelInfo) -> Bool {
        let dir = documentsDirectory.appendingPathComponent("Models/\(info.localDirName)")
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.appendingPathComponent("config.json").path) else { return false }
        let contents = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
        return contents.contains { $0.hasSuffix(".safetensors") }
    }

    // MARK: - Init

    init() {
        var model = Self.modelDefinition
        model.isAvailable = isModelAvailable(model)
        availableModels = [model]
    }

    // MARK: - Loading

    func startLoading() {
        guard let model = availableModels.first, model.isAvailable else {
            loadError = nil
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
            loadError = "Model '\(model.localDirName)' not found in Documents/Models/. Use the Download button to fetch it."
            return
        }
        pendingModelToLoad = model
        isLoading = true
        loadError = nil

        loadTask = Task { [weak self] in
            guard let self else { return }
            do {
                let config = ModelConfiguration(directory: modelURL)
                let loaded = try await LLMModelFactory.shared.loadContainer(configuration: config)
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self.container = loaded
                    self.currentModel = model
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                    Memory.cacheLimit = 20 * 1024 * 1024
                }
                print("✅ MLX model loaded: \(model.displayName)")
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

    // MARK: - Generation

    func generateTextStream(
        messages: [Chat.Message],
        enableThinking: Bool,
        onToken: @escaping (String) -> Void
    ) async throws {
        guard let container else {
            throw GenerationError.modelNotLoaded
        }
        defer { Memory.clearCache() }

        let additionalContext: [String: any Sendable]? = enableThinking ? nil : ["enable_thinking": false]
        let userInput = UserInput(chat: messages, additionalContext: additionalContext)
        let input = try await container.prepare(input: userInput)
        let params = enableThinking
            ? GenerateParameters(temperature: 0.6, topP: 0.95)
            : GenerateParameters(temperature: 0.7, topP: 0.8)
        let stream = try await container.generate(input: input, parameters: params)

        for await generation in stream {
            if case .chunk(let text) = generation {
                await MainActor.run { onToken(text) }
            }
        }
    }

    // MARK: - Diagnostics / Management

    func refreshModelAvailability() {
        availableModels = availableModels.map { model in
            var updated = model
            updated.isAvailable = isModelAvailable(model)
            return updated
        }
    }

    func deleteModel(_ model: MLXModelInfo) throws {
        if currentModel?.id == model.id {
            currentModel = nil
            container = nil
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
