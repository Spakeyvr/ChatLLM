//
//  MLXModelManager.swift
//  On-Device_LLM_Chat
//
//  Manages MLX-based language models bundled inside the app.
//

import Foundation
import Combine
import MLX
import MLXLLM
import MLXVLM
import MLXLMCommon

// MARK: - MLXModelManager

@MainActor
final class MLXModelManager: ObservableObject {

    // MARK: - Model Info

    struct MLXModelInfo: Identifiable {
        let id: String
        let name: String
        let localDirName: String
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

    var supportsNativeThinking: Bool { true }

    // MARK: - Private

    private var container: ModelContainer?
    private var loadTask: Task<Void, Never>?

    private static let modelDefinition = MLXModelInfo(
        id: "qwen3.5-4b-4bit",
        name: "Qwen 3.5",
        localDirName: "Qwen3.5-4B-MLX-4bit",
        parameters: "4B (4-bit)",
        description: "Alibaba's Qwen 3.5 4B model with extended 262K context and native reasoning support.",
        contextLength: 262144,
        isAvailable: false,
        supportsReasoning: true
    )

    private func modelDirectoryURL(for info: MLXModelInfo) -> URL? {
        // PBXFileSystemSynchronizedRootGroup copies the model folder's contents flat
        // into the app bundle root, so the model files live at Bundle.main.bundleURL directly.
        let bundleURL = Bundle.main.bundleURL
        if FileManager.default.fileExists(atPath: bundleURL.appendingPathComponent("config.json").path) {
            return bundleURL
        }
        return nil
    }

    private func isModelAvailable(_ info: MLXModelInfo) -> Bool {
        guard let url = modelDirectoryURL(for: info) else { return false }
        return FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path)
    }

    // MARK: - Init

    init() {
        var model = Self.modelDefinition
        model.isAvailable = isModelAvailable(model)
        availableModels = [model]
    }

    // MARK: - Loading

    /// Called when the MLX backend is selected to kick off model loading.
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
            loadError = "Model bundle '\(model.localDirName)' not found. Drag it into Xcode as a folder reference and rebuild."
            return
        }
        pendingModelToLoad = model
        isLoading = true
        loadError = nil

        loadTask = Task { [weak self] in
            guard let self else { return }
            do {
                let config = ModelConfiguration(directory: modelURL)
                let loaded = try await VLMModelFactory.shared.loadContainer(configuration: config) { progress in
                    Task { @MainActor in
                        _ = progress
                    }
                }
                guard !Task.isCancelled else { return }
                self.container = loaded
                self.currentModel = model
                self.isLoading = false
                self.pendingModelToLoad = nil
                // Cap the Metal buffer pool to avoid unbounded RAM growth on iOS.
                Memory.cacheLimit = 20 * 1024 * 1024
                print("✅ MLX model loaded: \(model.displayName)")
            } catch is CancellationError {
                self.isLoading = false
                self.pendingModelToLoad = nil
            } catch {
                self.isLoading = false
                self.pendingModelToLoad = nil
                self.loadError = "Failed to load model: \(error.localizedDescription)"
                print("❌ MLX model load error: \(error)")
            }
        }
    }

    // MARK: - Generation

    /// Streams generated tokens for the given conversation messages.
    /// The Qwen3VLProcessor applies the Jinja chat template via `applyChatTemplate`.
    /// `enable_thinking` is passed through `additionalContext` so the template emits
    /// the correct prefix: open `<think>` (on) or empty `<think></think>` (off).
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
        // Bundled models are read-only; unload from memory only.
        if currentModel?.id == model.id {
            currentModel = nil
            container = nil
        }
        print("ℹ️ Bundled model '\(model.localDirName)' unloaded from memory (remove from Xcode to free disk space).")
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
                print("📁 \(model.name): not found in bundle — available: \(model.isAvailable)")
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
