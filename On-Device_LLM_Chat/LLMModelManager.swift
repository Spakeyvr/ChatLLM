//
//  LLMModelManager.swift
//  On-Device_LLM_Chat
//
//  Created by Xcode Assistant
//  Manages multiple CoreML language models
//

import Foundation
import CoreML
import NaturalLanguage
import Combine

// Import ZIPFoundation if available
// Add via: File → Add Package Dependencies → https://github.com/weichsel/ZIPFoundation
#if canImport(ZIPFoundation)
import ZIPFoundation
#endif

/// Manages loading and inference for CoreML language models
@MainActor
final class LLMModelManager: ObservableObject {
    
    // MARK: - Published Properties
    
    @Published var availableModels: [LLMModelInfo] = []
    @Published var currentModel: LLMModelInfo?
    @Published var isLoading: Bool = false
    @Published var loadError: String?
    @Published var downloadProgress: [String: Double] = [:]  // Model ID -> Progress (0.0 to 1.0)
    @Published var isDownloading: [String: Bool] = [:]       // Model ID -> Is downloading
    @Published var downloadStatus: [String: DownloadStatus] = [:]  // Model ID -> Download status
    
    // MARK: - Private Properties
    
    private var loadedModels: [String: MLModel] = [:]
    private var modelAdapters: [String: LLMModelAdapter] = [:]
    private var modelConfigurations: [String: MLModelConfiguration] = [:]
    private var downloadTasks: [String: URLSessionDownloadTask] = [:]
    private var loadingTask: Task<Void, Never>?
    @Published private(set) var pendingModelToLoad: LLMModelInfo?
    
    // MARK: - Download Status
    
    enum DownloadStatus: Equatable {
        case idle
        case downloading(bytesDownloaded: Int64, totalBytes: Int64)
        case extracting
        case initializing
        case completed
        case failed(String)
        
        var message: String {
            switch self {
            case .idle:
                return ""
            case .downloading(let downloaded, let total):
                let formatter = ByteCountFormatter()
                formatter.allowedUnits = [.useGB, .useMB]
                formatter.countStyle = .file
                let downloadedStr = formatter.string(fromByteCount: downloaded)
                let totalStr = formatter.string(fromByteCount: total)
                return "Downloading \(downloadedStr) of \(totalStr)"
            case .extracting:
                return "Extracting model files..."
            case .initializing:
                return "Initializing model..."
            case .completed:
                return "Download complete!"
            case .failed(let error):
                return "Failed: \(error)"
            }
        }
        
        var progress: Double {
            switch self {
            case .idle:
                return 0.0
            case .downloading(let downloaded, let total):
                guard total > 0 else { return 0.0 }
                return Double(downloaded) / Double(total)
            case .extracting:
                return 0.9  // Show near completion
            case .initializing:
                return 0.95
            case .completed:
                return 1.0
            case .failed:
                return 0.0
            }
        }
    }
    
    // MARK: - Model Information
    
    struct LLMModelInfo: Identifiable, Codable, Hashable {
        let id: String
        let name: String
        let modelFileName: String
        let description: String
        let parameters: String      // e.g., "3B", "7B"
        let contextLength: Int      // Max tokens
        let isAvailable: Bool       // Whether the model file exists locally
        let downloadURL: String?    // Optional: Hugging Face download URL
        let fileSize: Int64?        // Optional: File size in bytes
        let supportsReasoning: Bool // Whether the model has native reasoning/thinking capabilities
        
        var displayName: String {
            "\(name) (\(parameters))"
        }
        
        var fileSizeFormatted: String? {
            guard let size = fileSize else { return nil }
            let formatter = ByteCountFormatter()
            formatter.allowedUnits = [.useGB, .useMB]
            formatter.countStyle = .file
            return formatter.string(fromByteCount: size)
        }
    }
    
    // MARK: - Initialization
    
    init() {
        setupAvailableModels()
        // Do NOT call loadDefaultModel() here.
        // LLMModelManager is also instantiated by ModelSelectionView, ModelPickerView,
        // and ChatInputView as @StateObject. Each of those triggered a concurrent 1 GB
        // model load, spiking RAM to 3-4 GB and crashing. Only the canonical instance
        // owned by ModelBackendBridge should load; it calls startLoading() explicitly.
    }

    /// Trigger model loading. Call this only on the canonical instance (ModelBackendBridge).
    func startLoading() {
        loadDefaultModel()
    }
    
    // MARK: - Setup
    
    private func setupAvailableModels() {
        // Define all models that your app supports
        var models: [LLMModelInfo] = []
        
        // Check for Qwen 3 1.7B Instruct - 4-bit quantized
        let qwen3_17B_4bit = LLMModelInfo(
            id: "qwen3-1.7b-4bit",
            name: "Qwen 3 Instruct",
            modelFileName: "Qwen3-1.7B-4bit",
            description: "Alibaba's instruction-tuned Qwen 3 model with 1.7 billion parameters (4-bit quantization)",
            parameters: "1.7B (4-bit)",
            contextLength: 32768,
            isAvailable: checkModelAvailability("Qwen3-1.7B-4bit"),
            downloadURL: nil,
            fileSize: 860_000_000,  // ~860MB (4-bit quantized)
            supportsReasoning: true
        )
        models.append(qwen3_17B_4bit)
        
        // You can add more models here in the future
        // For example, other Qwen variants or models from Hugging Face
        
        availableModels = models
    }
    
    private func loadDefaultModel() {
        guard let first = availableModels.first(where: { $0.isAvailable }) else { return }
        loadingTask = Task { await loadModel(first) }
    }
    
    // MARK: - Model Loading
    
    func loadModel(_ modelInfo: LLMModelInfo) async {
        guard modelInfo.isAvailable else {
            loadError = "Model \(modelInfo.name) is not available in the app bundle"
            return
        }
        
        // If already loaded, just switch to it
        if let _ = loadedModels[modelInfo.id] {
            currentModel = modelInfo
            return
        }

        // Prevent concurrent loads — each MLModel.load() allocates ~1 GB; two simultaneous
        // calls would push RAM to 2–3 GB and trigger jetsam even on 6 GB devices.
        guard !isLoading else {
            print("⚠️ loadModel: already loading, ignoring duplicate call")
            return
        }

        isLoading = true
        loadError = nil
        
        do {
            // Release any previously loaded models before allocating new weights
            loadedModels.removeAll()
            modelAdapters.removeAll()
            modelConfigurations.removeAll()

            var modelURL: URL?
            
            // Prefer .mlmodelc (pre-compiled) — avoids on-device compilation that peaks at 2-3× weight size
            if let url = Bundle.main.url(forResource: modelInfo.modelFileName, withExtension: "mlmodelc") {
                modelURL = url
            } else if let url = Bundle.main.url(forResource: modelInfo.modelFileName, withExtension: "mlpackage") {
                modelURL = url
            }
            
            // Check Documents/Models folder if not in bundle
            if modelURL == nil {
                let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                let modelsPath = documentsPath.appendingPathComponent("Models", isDirectory: true)
                
                // Prefer .mlmodelc (pre-compiled, avoids on-device compilation peak RAM)
                let mlmodelcPath = modelsPath.appendingPathComponent("\(modelInfo.modelFileName).mlmodelc")
                if FileManager.default.fileExists(atPath: mlmodelcPath.path) {
                    modelURL = mlmodelcPath
                } else {
                    let mlpackagePath = modelsPath.appendingPathComponent("\(modelInfo.modelFileName).mlpackage")
                    if FileManager.default.fileExists(atPath: mlpackagePath.path) {
                        modelURL = mlpackagePath
                    }
                }
            }
            
            guard let url = modelURL else {
                throw NSError(
                    domain: "LLMModelManager",
                    code: 404,
                    userInfo: [NSLocalizedDescriptionKey: "Model file not found: \(modelInfo.modelFileName)"]
                )
            }
            
            // Verify file exists and is reachable
            guard FileManager.default.fileExists(atPath: url.path) else {
                 throw NSError(
                     domain: "LLMModelManager",
                     code: 404,
                     userInfo: [NSLocalizedDescriptionKey: "Model URL points to non-existent file: \(url.path)"]
                 )
            }
            
            print("📂 Loading model from: \(url.path)")
            // Calculate total size (recursive for directories like .mlmodelc)
            let totalSize: Int64 = {
                var size: Int64 = 0
                if let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) {
                    for case let fileURL as URL in enumerator {
                        if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                            size += Int64(fileSize)
                        }
                    }
                }
                return size
            }()
            print("   Size: \(ByteCountFormatter.string(fromByteCount: totalSize, countStyle: .file))")

            // Use Metal GPU first (no ANE compilation spike → avoids OOM on load),
            // then fall back to CPU-only.
            let computeUnitOptions: [(MLComputeUnits, String)] = [
                (.cpuAndGPU, "CPU + GPU (Metal)"),
                (.cpuOnly,   "CPU Only"),
            ]
            
            var loadedModel: MLModel?
            var lastError: Error?
            var activeConfig: MLModelConfiguration?

            for (computeUnits, description) in computeUnitOptions {
                print("⏳ Attempting to load with compute units: \(description)...")
                
                let config = MLModelConfiguration()
                config.computeUnits = computeUnits
                config.allowLowPrecisionAccumulationOnGPU = true
                
                do {
                    loadedModel = try await MLModel.load(contentsOf: url, configuration: config)
                    activeConfig = config
                    print("✅ Successfully loaded with \(description)")
                    break // Success!
                } catch {
                    print("⚠️ Failed to load with \(description): \(error.localizedDescription)")
                    lastError = error
                    // Continue to next option
                }
            }
            
            // Discard the model if the task was cancelled while MLModel.load was running
            try Task.checkCancellation()

            guard let model = loadedModel, let finalConfig = activeConfig else {
                throw lastError ?? NSError(
                    domain: "LLMModelManager",
                    code: -14,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to load model with all compute unit options."]
                )
            }
            
            modelConfigurations[modelInfo.id] = finalConfig
            
            // Create appropriate adapter for the model
            let adapter = try createAdapter(for: model, modelInfo: modelInfo)

            // Load tokenizer if this is a QwenAdapter
            if let qwenAdapter = adapter as? QwenAdapter {
                await loadTokenizerForQwen(adapter: qwenAdapter, modelInfo: modelInfo)
            } else if let autoAdapter = adapter as? AutoDetectingAdapter,
                      autoAdapter.requiresAutoregressiveGeneration {
                // For auto-detecting adapters that use Qwen underneath
                // Try to load tokenizer by checking if underlying adapter is Qwen
                await loadTokenizerForAutoAdapter(adapter: autoAdapter, modelInfo: modelInfo)
            }

            // Cache the loaded model and adapter
            loadedModels[modelInfo.id] = model
            modelAdapters[modelInfo.id] = adapter
            currentModel = modelInfo

            print("✅ Model ready: \(modelInfo.displayName)")
            print("   Adapter type: \(type(of: adapter))")
            if adapter.supportsNativeThinking {
                print("   🧠 Native thinking: SUPPORTED")
            }
            
        } catch is CancellationError {
            isLoading = false
            if let next = pendingModelToLoad {
                pendingModelToLoad = nil
                loadingTask = Task { await loadModel(next) }
            }
            return
        } catch {
            loadError = "Failed to load model: \(error.localizedDescription)"
            print("❌ Model loading error: \(error)")
            if let nsError = error as NSError? {
               print("   Domain: \(nsError.domain)")
               print("   Code: \(nsError.code)")
               print("   UserInfo: \(nsError.userInfo)")
            }
        }

        isLoading = false
    }
    
    // MARK: - Model Inference

    /// Perform inference with the current model
    /// - Parameters:
    ///   - prompt: The input prompt text
    ///   - maxTokens: Maximum number of tokens to generate (default: 4096 for longer responses)
    ///   - temperature: Sampling temperature (0.0 to 1.0)
    ///   - enableThinking: Whether to enable native thinking mode (if supported)
    /// - Returns: Generated text response
    func generateText(
        prompt: String,
        maxTokens: Int = 1024,
        temperature: Double = 0.7,
        enableThinking: Bool = false
    ) async throws -> String {
        guard let currentModel = currentModel else {
            throw NSError(
                domain: "LLMModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "No model is currently loaded"]
            )
        }

        guard let model = loadedModels[currentModel.id] else {
            throw NSError(
                domain: "LLMModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model is not loaded"]
            )
        }

        guard let adapter = modelAdapters[currentModel.id] else {
            throw NSError(
                domain: "LLMModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model adapter not found"]
            )
        }

        print("🔄 Generating text with \(currentModel.displayName)")
        print("   Thinking mode: \(enableThinking ? "ENABLED" : "DISABLED")")

        // Check if adapter requires autoregressive generation
        if adapter.requiresAutoregressiveGeneration {
            return try await generateTextAutoregressive(
                model: model,
                adapter: adapter,
                prompt: prompt,
                maxTokens: maxTokens,
                temperature: temperature,
                enableThinking: enableThinking
            )
        }

        // Standard single-pass generation for text-to-text models
        let input = try adapter.prepareInput(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            enableThinking: enableThinking
        )

        // Perform prediction
        let output = try await model.prediction(from: input)

        // Extract and return the generated text using the adapter
        let generatedText = try adapter.extractOutput(from: output)

        print("✅ Generated \(generatedText.count) characters")

        return generatedText
    }

    /// Autoregressive text generation for token-based models (like Qwen3)
    /// Generates tokens one at a time until EOS or max tokens reached
    private func generateTextAutoregressive(
        model: MLModel,
        adapter: LLMModelAdapter,
        prompt: String,
        maxTokens: Int,
        temperature: Double,
        enableThinking: Bool
    ) async throws -> String {
        guard let tokenizer = adapter.tokenizer else {
            throw NSError(
                domain: "LLMModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Tokenizer required for autoregressive generation"]
            )
        }

        // Reset per-session diagnostics so shape + top-5 logs fire on every generation
        adapter.resetSessionDiagnostics()

        // prompt is already fully formatted by ChatViewModel; encode it directly
        var currentTokens = tokenizer.encode(prompt)
        var generatedTokens: [Int32] = []
        let eosTokenId = adapter.eosTokenId

        print("📝 Starting autoregressive generation...")
        print("   Initial tokens: \(currentTokens.count)")
        print("   Max new tokens: \(maxTokens)")
        print("   EOS token ID: \(eosTokenId)")

        // Generation loop
        for step in 0..<maxTokens {
            // Prepare input with current token sequence
            let input = try adapter.prepareTokenInput(tokenIds: currentTokens)

            // Get model prediction (logits)
            let output = try await model.prediction(from: input)

            // Sample next token from logits
            let nextToken = try adapter.sampleNextToken(from: output, temperature: temperature)

            // Check for EOS
            if nextToken == eosTokenId {
                print("🛑 EOS token reached at step \(step)")
                break
            }

            // Append token
            generatedTokens.append(nextToken)
            currentTokens.append(nextToken)

            // Progress logging every 50 tokens
            if step > 0 && step % 50 == 0 {
                print("   Generated \(step) tokens...")
            }
        }

        print("✅ Generated \(generatedTokens.count) tokens")

        // Decode generated tokens to text
        var generatedText = tokenizer.decode(generatedTokens)

        // Clean up the output (remove special tokens, etc.)
        if let qwenAdapter = adapter as? QwenAdapter {
            generatedText = qwenAdapter.cleanOutput(generatedText)
        }

        print("✅ Decoded to \(generatedText.count) characters")

        return generatedText
    }
    
    /// Stream text generation token by token
    /// - Parameters:
    ///   - prompt: The input prompt text
    ///   - maxTokens: Maximum number of tokens to generate (default: 4096 for longer responses)
    ///   - temperature: Sampling temperature
    ///   - enableThinking: Whether to enable native thinking mode (if supported)
    ///   - onToken: Callback for each generated token
    func generateTextStream(
        prompt: String,
        maxTokens: Int = 1024,
        temperature: Double = 0.7,
        enableThinking: Bool = false,
        onToken: @escaping (String) -> Void
    ) async throws {
        guard let currentModel = currentModel else {
            throw NSError(
                domain: "LLMModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "No model is currently loaded"]
            )
        }

        guard let model = loadedModels[currentModel.id] else {
            throw NSError(
                domain: "LLMModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model is not loaded"]
            )
        }

        guard let adapter = modelAdapters[currentModel.id] else {
            throw NSError(
                domain: "LLMModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model adapter not found"]
            )
        }

        // Check if adapter supports true streaming via autoregressive generation
        if adapter.requiresAutoregressiveGeneration {
            try await generateTextStreamAutoregressive(
                model: model,
                adapter: adapter,
                prompt: prompt,
                maxTokens: maxTokens,
                temperature: temperature,
                enableThinking: enableThinking,
                onToken: onToken
            )
            return
        }

        // Fallback: generate full response and simulate streaming
        let fullResponse = try await generateText(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            enableThinking: enableThinking
        )

        // Simulate streaming by breaking into words
        let words = fullResponse.split(separator: " ")
        for word in words {
            onToken(String(word) + " ")
            try? await Task.sleep(for: .milliseconds(50))
        }
    }

    /// True streaming generation for autoregressive models
    /// Calls onToken callback as each token is generated
    private func generateTextStreamAutoregressive(
        model: MLModel,
        adapter: LLMModelAdapter,
        prompt: String,
        maxTokens: Int,
        temperature: Double,
        enableThinking: Bool,
        onToken: @escaping (String) -> Void
    ) async throws {
        guard let tokenizer = adapter.tokenizer else {
            throw NSError(
                domain: "LLMModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Tokenizer required for streaming generation"]
            )
        }

        // Reset per-session diagnostics so shape + top-5 logs fire on every generation
        adapter.resetSessionDiagnostics()

        // prompt is already fully formatted by ChatViewModel (builtPrompt);
        // encode it directly rather than re-wrapping with formatChatPrompt.
        var currentTokens = tokenizer.encode(prompt)
        let eosTokenId = adapter.eosTokenId

        print("📝 Starting streaming autoregressive generation...")
        print("   Initial tokens: \(currentTokens.count)")
        // Diagnostic: show first 12 token IDs — expected to start with 151644 (<|im_start|>)
        let previewTokens = Array(currentTokens.prefix(12))
        let previewDecoded = tokenizer.decode(previewTokens)
        print("   First 12 token IDs: \(previewTokens)")
        print("   Decoded preview: \(previewDecoded.prefix(80))")

        // Buffer for accumulating partial tokens (for multi-byte characters)
        var tokenBuffer: [Int32] = []

        // Generation loop with streaming
        for step in 0..<maxTokens {
            guard !Task.isCancelled else { break }

            // Sliding window: cap context to last 1024 tokens to bound activation tensor size
            let windowSize = 1024
            let contextTokens = Array(currentTokens.suffix(windowSize))
            let input = try adapter.prepareTokenInput(tokenIds: contextTokens)

            // Get prediction
            let output = try await model.prediction(from: input)

            // Sample next token
            let nextToken = try adapter.sampleNextToken(from: output, temperature: temperature)

            // Check for EOS
            if nextToken == eosTokenId {
                print("🛑 EOS token reached at step \(step)")
                // Flush any remaining buffer
                if !tokenBuffer.isEmpty {
                    let remainingText = tokenizer.decode(tokenBuffer)
                    if !remainingText.isEmpty {
                        onToken(remainingText)
                    }
                }
                break
            }

            // Add to current sequence
            currentTokens.append(nextToken)
            tokenBuffer.append(nextToken)

            // Try to decode buffer and stream
            let decodedText = tokenizer.decode(tokenBuffer)
            if !decodedText.isEmpty {
                // Clean output if it's a Qwen adapter
                var cleanedText = decodedText
                if let qwenAdapter = adapter as? QwenAdapter {
                    cleanedText = qwenAdapter.cleanOutput(decodedText)
                }

                if !cleanedText.isEmpty {
                    onToken(cleanedText)
                    tokenBuffer.removeAll()
                }
            }

            // Progress logging
            if step > 0 && step % 100 == 0 {
                print("   Streamed \(step) tokens...")
            }
        }

        print("✅ Streaming generation complete")
    }
    
    // MARK: - Helper Methods
    
    /// Create the appropriate adapter for a model
    private func createAdapter(for model: MLModel, modelInfo: LLMModelInfo) throws -> LLMModelAdapter {
        // Always use auto-detection to inspect actual model input/output format
        // This ensures we use the right adapter based on what the model actually expects
        print("🔍 Auto-detecting adapter for \(modelInfo.id)")
        let adapter = try AutoDetectingAdapter(model: model)

        // Log what we detected
        if adapter.requiresAutoregressiveGeneration {
            print("   📝 Model requires autoregressive generation (token-based)")
        }
        if adapter.supportsNativeThinking {
            print("   🧠 Model supports native thinking mode")
        }

        return adapter
    }

    /// Load tokenizer for a QwenAdapter
    private func loadTokenizerForQwen(adapter: QwenAdapter, modelInfo: LLMModelInfo) async {
        // Try to find tokenizer.json in various locations
        let tokenizerPaths = getTokenizerSearchPaths(for: modelInfo)

        for path in tokenizerPaths {
            if FileManager.default.fileExists(atPath: path) {
                do {
                    try await adapter.loadTokenizer(from: path)
                    print("✅ Tokenizer loaded from: \(path)")
                    return
                } catch {
                    print("⚠️ Failed to load tokenizer from \(path): \(error)")
                }
            }
        }

        print("⚠️ No tokenizer.json found. Using fallback tokenization.")
        print("   Searched paths:")
        for path in tokenizerPaths {
            print("   - \(path)")
        }
    }

    /// Load tokenizer for an AutoDetectingAdapter that uses Qwen underneath
    private func loadTokenizerForAutoAdapter(adapter: AutoDetectingAdapter, modelInfo: LLMModelInfo) async {
        // Get the underlying tokenizer from the adapter
        guard let tokenizer = adapter.tokenizer else {
            print("⚠️ AutoDetectingAdapter has no tokenizer to load")
            return
        }

        // Try to find tokenizer.json in various locations
        let tokenizerPaths = getTokenizerSearchPaths(for: modelInfo)

        for path in tokenizerPaths {
            if FileManager.default.fileExists(atPath: path) {
                do {
                    try await tokenizer.loadFromFile(path)
                    print("✅ Tokenizer loaded from: \(path)")
                    return
                } catch {
                    print("⚠️ Failed to load tokenizer from \(path): \(error)")
                }
            }
        }

        print("⚠️ No tokenizer.json found. Using fallback tokenization.")
    }

    /// Get list of paths to search for tokenizer.json
    private func getTokenizerSearchPaths(for modelInfo: LLMModelInfo) -> [String] {
        var paths: [String] = []

        // 1. Check inside the mlpackage's Data folder (tokenizer bundled with model)
        if let bundleModelURL = Bundle.main.url(forResource: modelInfo.modelFileName, withExtension: "mlpackage") {
            paths.append(bundleModelURL.appendingPathComponent("Data/tokenizer.json").path)
            let bundleDir = bundleModelURL.deletingLastPathComponent()
            paths.append(bundleDir.appendingPathComponent("tokenizer.json").path)
        }
        if let bundleModelURL = Bundle.main.url(forResource: modelInfo.modelFileName, withExtension: "mlmodelc") {
            let bundleDir = bundleModelURL.deletingLastPathComponent()
            paths.append(bundleDir.appendingPathComponent("tokenizer.json").path)
        }

        // 2. Check in app bundle Resources
        if let bundlePath = Bundle.main.resourcePath {
            paths.append((bundlePath as NSString).appendingPathComponent("tokenizer.json"))
        }

        // 3. Check in Documents/Models folder
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let modelsPath = documentsPath.appendingPathComponent("Models", isDirectory: true)
        paths.append(modelsPath.appendingPathComponent("tokenizer.json").path)

        // 4. Check in On-Device_LLM_Chat folder (for development)
        if let bundlePath = Bundle.main.resourcePath {
            let projectRoot = (bundlePath as NSString).deletingLastPathComponent
            paths.append((projectRoot as NSString).appendingPathComponent("On-Device_LLM_Chat/tokenizer.json"))
        }

        return paths
    }
    
    // MARK: - Cancel-Aware Loading

    /// Switch to a different model even while one is loading.
    /// If a load is in progress, it is cancelled and the new model starts after cleanup.
    func cancelAndLoad(_ modelInfo: LLMModelInfo) {
        if isLoading {
            pendingModelToLoad = modelInfo
            loadingTask?.cancel()
        } else {
            loadingTask = Task { await loadModel(modelInfo) }
        }
    }

    /// Cancel any in-progress model load (e.g. when switching to FoundationModels).
    func cancelCurrentLoad() {
        pendingModelToLoad = nil
        loadingTask?.cancel()
    }

    // MARK: - Model Management

    func unloadModel(_ modelInfo: LLMModelInfo) {
        loadedModels.removeValue(forKey: modelInfo.id)
        modelAdapters.removeValue(forKey: modelInfo.id)
        modelConfigurations.removeValue(forKey: modelInfo.id)
        
        if currentModel?.id == modelInfo.id {
            currentModel = nil
        }
        
        print("🗑️ Unloaded model: \(modelInfo.displayName)")
    }
    
    func unloadAllModels() {
        loadedModels.removeAll()
        modelAdapters.removeAll()
        modelConfigurations.removeAll()
        currentModel = nil
        print("🗑️ Unloaded all models")
    }
    
    // MARK: - Model Information
    
    func getModelInfo(for modelId: String) -> LLMModelInfo? {
        return availableModels.first { $0.id == modelId }
    }
    
    /// Check if the current model supports native thinking mode
    var supportsNativeThinking: Bool {
        guard let currentModel = currentModel,
              let adapter = modelAdapters[currentModel.id] else {
            return false
        }
        return adapter.supportsNativeThinking
    }
    
    func refreshModelAvailability() {
        print("🔄 Refreshing model availability...")
        setupAvailableModels()
        print("✅ Model availability refreshed")
    }
    
    func printModelLocations() {
        print("\n📍 Model Locations:")
        print("===================")
        
        // Print bundle path
        if let bundlePath = Bundle.main.resourcePath {
            print("\n📦 App Bundle Path:")
            print("   \(bundlePath)")
        }
        
        // Print Documents/Models path
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let modelsPath = documentsPath.appendingPathComponent("Models", isDirectory: true)
        print("\n📁 Documents/Models Path:")
        print("   \(modelsPath.path)")
        
        // Print available models
        print("\n📋 Available Models:")
        for model in availableModels {
            print("\n   \(model.displayName)")
            print("   ID: \(model.id)")
            print("   Filename: \(model.modelFileName)")
            print("   Available: \(model.isAvailable ? "✅" : "❌")")
            
            // Try to find actual location
            if model.isAvailable {
                if let url = Bundle.main.url(forResource: model.modelFileName, withExtension: "mlmodelc") {
                    print("   Location: Bundle - \(url.path)")
                } else if let url = Bundle.main.url(forResource: model.modelFileName, withExtension: "mlpackage") {
                    print("   Location: Bundle - \(url.path)")
                } else {
                    let mlpackagePath = modelsPath.appendingPathComponent("\(model.modelFileName).mlpackage")
                    let mlmodelcPath = modelsPath.appendingPathComponent("\(model.modelFileName).mlmodelc")
                    
                    if FileManager.default.fileExists(atPath: mlpackagePath.path) {
                        print("   Location: Documents - \(mlpackagePath.path)")
                    } else if FileManager.default.fileExists(atPath: mlmodelcPath.path) {
                        print("   Location: Documents - \(mlmodelcPath.path)")
                    }
                }
            }
        }
        print("\n===================\n")
    }
    
    func printModelInputOutputInfo() {
        guard let currentModel = currentModel,
              let model = loadedModels[currentModel.id] else {
            print("No model loaded")
            return
        }
        
        let description = model.modelDescription
        
        print("📊 Model Information:")
        print("Name: \(currentModel.displayName)")
        print("\nInput Features:")
        for (name, desc) in description.inputDescriptionsByName {
            print("  - \(name): \(desc.type)")
        }
        print("\nOutput Features:")
        for (name, desc) in description.outputDescriptionsByName {
            print("  - \(name): \(desc.type)")
        }
    }
    
    // MARK: - Download Management
    
    /// Download a model from Hugging Face
    func downloadModel(_ modelInfo: LLMModelInfo) async throws {
        print("\n🔽 Starting download for: \(modelInfo.displayName)")
        
        guard let downloadURLString = modelInfo.downloadURL,
              let downloadURL = URL(string: downloadURLString) else {
            print("❌ No download URL available")
            throw NSError(
                domain: "LLMModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "No download URL available for this model"]
            )
        }
        
        print("📍 Download URL: \(downloadURLString)")
        
        // Check if already downloading
        if isDownloading[modelInfo.id] == true {
            print("⚠️ Download already in progress for \(modelInfo.displayName)")
            return
        }
        
        // Check disk space before starting
        print("💾 Checking disk space...")
        if !hasEnoughDiskSpace(for: modelInfo) {
            let requiredSpace = getRequiredDiskSpace(for: modelInfo) ?? "unknown"
            print("❌ Not enough disk space. Need: \(requiredSpace)")
            throw NSError(
                domain: "LLMModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Not enough disk space. Need approximately \(requiredSpace) free."]
            )
        }
        print("✅ Disk space check passed")
        
        // Set downloading state
        print("📊 Setting download state...")
        isDownloading[modelInfo.id] = true
        downloadProgress[modelInfo.id] = 0.0
        downloadStatus[modelInfo.id] = .downloading(bytesDownloaded: 0, totalBytes: modelInfo.fileSize ?? 0)
        
        do {
            print("🌐 Creating URLSession...")
            // Create a custom URLSession with delegate for progress tracking
            let config = URLSessionConfiguration.default
            config.timeoutIntervalForRequest = 60
            config.timeoutIntervalForResource = 3600 // 1 hour for large files
            
            let delegate = DownloadDelegate(modelInfo: modelInfo, manager: self)
            let session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
            
            print("📋 Creating download task...")
            // Create download task
            let downloadTask = session.downloadTask(with: downloadURL)
            
            // Store the task for cancellation support
            downloadTasks[modelInfo.id] = downloadTask
            
            print("▶️ Starting download task...")
            // Start download
            downloadTask.resume()
            
            print("⏳ Waiting for download to complete...")
            // Wait for download to complete using continuation
            let localURL = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<URL, Error>) in
                delegate.continuation = continuation
            }
            
            print("✅ Download continuation resolved!")
            print("📁 Downloaded file location: \(localURL.path)")
            
            // Validate the downloaded file
            guard FileManager.default.fileExists(atPath: localURL.path) else {
                print("❌ Downloaded file doesn't exist at: \(localURL.path)")
                throw NSError(
                    domain: "LLMModelManager",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Downloaded file not found"]
                )
            }
            
            // Update status to extracting
            downloadStatus[modelInfo.id] = .extracting
            downloadProgress[modelInfo.id] = 0.9
            
            // Get Documents directory
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let modelsPath = documentsPath.appendingPathComponent("Models", isDirectory: true)
            
            // Create Models directory if it doesn't exist
            try FileManager.default.createDirectory(at: modelsPath, withIntermediateDirectories: true)
            
            // Determine file extension from URL
            let fileExtension: String
            if downloadURLString.hasSuffix(".mlmodelc") || downloadURLString.hasSuffix(".mlmodelc.zip") {
                fileExtension = "mlmodelc"
            } else if downloadURLString.hasSuffix(".mlpackage") || downloadURLString.hasSuffix(".mlpackage.zip") {
                fileExtension = "mlpackage"
            } else if downloadURLString.hasSuffix(".zip") {
                fileExtension = "zip"
            } else {
                fileExtension = "mlpackage" // Default assumption for CoreML models
            }
            
            // Check if downloaded file is a zip
            let isZip = downloadURLString.hasSuffix(".zip")
            
            if isZip {
                // Unzip the file
                try await unzipModel(from: localURL, to: modelsPath, modelFileName: modelInfo.modelFileName)
            } else {
                // Move the file directly (works for both .mlmodelc and .mlpackage)
                let destinationURL = modelsPath.appendingPathComponent("\(modelInfo.modelFileName).\(fileExtension)")
                
                // Remove existing file if present
                try? FileManager.default.removeItem(at: destinationURL)
                
                try FileManager.default.moveItem(at: localURL, to: destinationURL)
                print("✅ Model file moved to: \(destinationURL.path)")
            }
            
            // Update status to initializing
            downloadStatus[modelInfo.id] = .initializing
            downloadProgress[modelInfo.id] = 0.95
            
            // Small delay to show initializing status
            try? await Task.sleep(for: .seconds(1))
            
            print("✅ Model downloaded successfully")
            
            // Update model availability
            if let index = availableModels.firstIndex(where: { $0.id == modelInfo.id }) {
                let updatedModel = availableModels[index]
                let newModel = LLMModelInfo(
                    id: updatedModel.id,
                    name: updatedModel.name,
                    modelFileName: updatedModel.modelFileName,
                    description: updatedModel.description,
                    parameters: updatedModel.parameters,
                    contextLength: updatedModel.contextLength,
                    isAvailable: true,
                    downloadURL: updatedModel.downloadURL,
                    fileSize: updatedModel.fileSize,
                    supportsReasoning: updatedModel.supportsReasoning
                )
                availableModels[index] = newModel
            }
            
            downloadStatus[modelInfo.id] = .completed
            downloadProgress[modelInfo.id] = 1.0
            isDownloading[modelInfo.id] = false
            
            // Clean up
            downloadTasks.removeValue(forKey: modelInfo.id)
            session.finishTasksAndInvalidate()
            
            // Clear completed status after a delay
            Task {
                try? await Task.sleep(for: .seconds(2))
                await MainActor.run {
                    downloadStatus[modelInfo.id] = .idle
                }
            }
            
        } catch {
            print("❌❌❌ Download ERROR caught ❌❌❌")
            print("Error type: \(type(of: error))")
            print("Error description: \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("Error domain: \(nsError.domain)")
                print("Error code: \(nsError.code)")
                print("Error userInfo: \(nsError.userInfo)")
            }
            
            isDownloading[modelInfo.id] = false
            downloadProgress[modelInfo.id] = 0.0
            downloadStatus[modelInfo.id] = .failed(error.localizedDescription)
            
            // Clean up
            downloadTasks.removeValue(forKey: modelInfo.id)
            
            // Clear failed status after a delay
            Task {
                try? await Task.sleep(for: .seconds(5))
                await MainActor.run {
                    downloadStatus[modelInfo.id] = .idle
                }
            }
            throw error
        }
    }
    
    /// Cancel download for a specific model
    func cancelDownload(for modelID: String) {
        guard let task = downloadTasks[modelID] else {
            print("⚠️ No download task found for model ID: \(modelID)")
            return
        }
        
        task.cancel()
        downloadTasks.removeValue(forKey: modelID)
        isDownloading[modelID] = false
        downloadProgress[modelID] = 0.0
        downloadStatus[modelID] = .idle
        
        print("🚫 Cancelled download for model ID: \(modelID)")
    }
    
    /// Check if there's enough disk space for a download
    func hasEnoughDiskSpace(for modelInfo: LLMModelInfo) -> Bool {
        guard let fileSize = modelInfo.fileSize else {
            // If we don't know the file size, assume we have space
            return true
        }
        
        do {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let attributes = try FileManager.default.attributesOfFileSystem(forPath: documentsPath.path)
            
            if let freeSpace = attributes[.systemFreeSize] as? Int64 {
                // Require 1.5x the file size to account for extraction
                let requiredSpace = Int64(Double(fileSize) * 1.5)
                return freeSpace > requiredSpace
            }
        } catch {
            print("❌ Failed to check disk space: \(error)")
        }
        
        // If we can't check, assume we have space
        return true
    }
    
    /// Get formatted string for required disk space
    func getRequiredDiskSpace(for modelInfo: LLMModelInfo) -> String? {
        guard let fileSize = modelInfo.fileSize else { return nil }
        
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        
        // Show 1.5x file size to account for extraction
        let requiredSpace = Int64(Double(fileSize) * 1.5)
        return formatter.string(fromByteCount: requiredSpace)
    }
    
    /// Delete a downloaded model
    func deleteModel(_ modelInfo: LLMModelInfo) throws {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let modelsPath = documentsPath.appendingPathComponent("Models", isDirectory: true)
        
        // Unload if currently loaded
        if currentModel?.id == modelInfo.id {
            unloadModel(modelInfo)
        }
        
        // Try both .mlpackage and .mlmodelc extensions
        let mlpackagePath = modelsPath.appendingPathComponent("\(modelInfo.modelFileName).mlpackage")
        let mlmodelcPath = modelsPath.appendingPathComponent("\(modelInfo.modelFileName).mlmodelc")
        
        var deleted = false
        
        // Delete .mlpackage if it exists
        if FileManager.default.fileExists(atPath: mlpackagePath.path) {
            try FileManager.default.removeItem(at: mlpackagePath)
            deleted = true
        }
        
        // Delete .mlmodelc if it exists
        if FileManager.default.fileExists(atPath: mlmodelcPath.path) {
            try FileManager.default.removeItem(at: mlmodelcPath)
            deleted = true
        }
        
        if !deleted {
            throw NSError(
                domain: "LLMModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model file not found"]
            )
        }
        
        // Update availability
        if let index = availableModels.firstIndex(where: { $0.id == modelInfo.id }) {
            let updatedModel = availableModels[index]
            let newModel = LLMModelInfo(
                id: updatedModel.id,
                name: updatedModel.name,
                modelFileName: updatedModel.modelFileName,
                description: updatedModel.description,
                parameters: updatedModel.parameters,
                contextLength: updatedModel.contextLength,
                isAvailable: false,
                downloadURL: updatedModel.downloadURL,
                fileSize: updatedModel.fileSize,
                supportsReasoning: updatedModel.supportsReasoning
            )
            availableModels[index] = newModel
        }
        
        print("🗑️ Deleted model: \(modelInfo.displayName)")
    }
    
    /// Unzip a model file using platform-specific methods
    private func unzipModel(from zipURL: URL, to destinationFolder: URL, modelFileName: String) async throws {
        print("📦 Unzipping model from: \(zipURL.path)")
        print("📂 Destination: \(destinationFolder.path)")
        
        #if canImport(ZIPFoundation)
        // Create temporary extraction directory
        let tempExtractPath = destinationFolder.appendingPathComponent("temp_extract")
        try? FileManager.default.removeItem(at: tempExtractPath)
        try FileManager.default.createDirectory(at: tempExtractPath, withIntermediateDirectories: true)
        
        // Extract the zip file using ZIPFoundation
        try FileManager.default.unzipItem(at: zipURL, to: tempExtractPath)
        
        // Find and move the extracted .mlpackage
        print("📦 Looking for extracted .mlpackage...")
        let extractedContents = try FileManager.default.contentsOfDirectory(
            at: tempExtractPath,
            includingPropertiesForKeys: nil
        )
        
        guard let mlpackage = extractedContents.first(where: { $0.pathExtension == "mlpackage" }) else {
            throw NSError(
                domain: "LLMModelManager",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Could not find .mlpackage in extracted files"]
            )
        }
        
        // Move to final destination
        let finalPath = destinationFolder.appendingPathComponent("\(modelFileName).mlpackage")
        try? FileManager.default.removeItem(at: finalPath)
        try FileManager.default.moveItem(at: mlpackage, to: finalPath)
        
        // Clean up temporary files
        try? FileManager.default.removeItem(at: tempExtractPath)
        try? FileManager.default.removeItem(at: zipURL)
        
        print("✅ Successfully extracted model to: \(finalPath.path)")
        #else
        // ZIPFoundation is required for ZIP extraction
        throw NSError(
            domain: "LLMModelManager",
            code: -1,
            userInfo: [
                NSLocalizedDescriptionKey: """
                ZIP extraction requires ZIPFoundation package.
                
                Add ZIPFoundation via:
                File → Add Package Dependencies
                URL: https://github.com/weichsel/ZIPFoundation
                
                Alternatively, download and extract the model manually, then add the .mlpackage to the app bundle.
                """
            ]
        )
        #endif
    }
    
    /// Get local storage path for models
    private func getModelsDirectory() -> URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsPath.appendingPathComponent("Models", isDirectory: true)
    }
}

// MARK: - Update checkModelAvailability to check Documents folder

extension LLMModelManager {
    private func checkModelAvailability(_ modelName: String) -> Bool {
        // Check app bundle first
        if Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") != nil {
            return true
        }
        if Bundle.main.url(forResource: modelName, withExtension: "mlpackage") != nil {
            return true
        }
        
        // Check Documents/Models folder for both extensions
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let modelsPath = documentsPath.appendingPathComponent("Models", isDirectory: true)
        
        let mlpackagePath = modelsPath.appendingPathComponent("\(modelName).mlpackage")
        let mlmodelcPath = modelsPath.appendingPathComponent("\(modelName).mlmodelc")
        
        if FileManager.default.fileExists(atPath: mlpackagePath.path) {
            return true
        }
        
        if FileManager.default.fileExists(atPath: mlmodelcPath.path) {
            return true
        }
        
        // Check resource path
        if let resourcePath = Bundle.main.resourcePath {
            let mlmodelcPath = (resourcePath as NSString).appendingPathComponent("\(modelName).mlmodelc")
            let mlpackagePath = (resourcePath as NSString).appendingPathComponent("\(modelName).mlpackage")
            if FileManager.default.fileExists(atPath: mlmodelcPath) ||
               FileManager.default.fileExists(atPath: mlpackagePath) {
                return true
            }
        }
        return false
    }
}

// MARK: - AppStorage Extension for Model Selection

extension UserDefaults {
    private enum Keys {
        static let selectedModelKey = "selectedLLMModel"
    }
    
    var selectedLLMModel: String? {
        get {
            string(forKey: Keys.selectedModelKey)
        }
        set {
            set(newValue, forKey: Keys.selectedModelKey)
        }
    }
}

// MARK: - Download Delegate

class DownloadDelegate: NSObject, URLSessionDownloadDelegate {
    let modelInfo: LLMModelManager.LLMModelInfo
    weak var manager: LLMModelManager?
    var continuation: CheckedContinuation<URL, Error>?
    
    init(modelInfo: LLMModelManager.LLMModelInfo, manager: LLMModelManager) {
        self.modelInfo = modelInfo
        self.manager = manager
    }
    
    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        Task { @MainActor in
            guard let manager = manager else { return }
            
            let progress = totalBytesExpectedToWrite > 0 
                ? Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
                : 0.0
            
            manager.downloadProgress[modelInfo.id] = progress
            manager.downloadStatus[modelInfo.id] = .downloading(
                bytesDownloaded: totalBytesWritten,
                totalBytes: totalBytesExpectedToWrite
            )
        }
    }
    
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        print("📥 Download finished, file at: \(location.path)")
        
        // Move the file to a temporary location that will persist after this method returns
        let tempDirectory = FileManager.default.temporaryDirectory
        let tempFile = tempDirectory.appendingPathComponent(UUID().uuidString)
        
        do {
            // Copy (don't move) because the system will delete the original location
            try FileManager.default.copyItem(at: location, to: tempFile)
            print("✅ Copied to temp location: \(tempFile.path)")
            continuation?.resume(returning: tempFile)
            continuation = nil  // Clear to prevent double-resume
        } catch {
            print("❌ Failed to copy download: \(error)")
            continuation?.resume(throwing: error)
            continuation = nil
        }
    }
    
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            print("❌ Download task failed: \(error.localizedDescription)")
            // Only resume if continuation hasn't been used yet
            if continuation != nil {
                continuation?.resume(throwing: error)
                continuation = nil
            }
        } else {
            print("✅ Download task completed successfully")
            // Don't resume here if successful - didFinishDownloadingTo handles it
        }
    }
}
