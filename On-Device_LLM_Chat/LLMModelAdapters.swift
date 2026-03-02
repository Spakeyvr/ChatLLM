//
//  LLMModelAdapters.swift
//  On-Device_LLM_Chat
//
//  Created by Xcode Assistant
//  Adapters for different CoreML model formats from Hugging Face
//

import Foundation
import CoreML
import Tokenizers

// MARK: - Qwen3 Tokenizer Wrapper

/// Wrapper around swift-transformers tokenizer for Qwen3 models
final class Qwen3Tokenizer {
    // Qwen3 special tokens
    static let padTokenId: Int32 = 151643
    static let eosTokenId: Int32 = 151645  // <|im_end|>
    static let imStartTokenId: Int32 = 151644  // <|im_start|>
    static let imEndTokenId: Int32 = 151645    // <|im_end|>

    private var tokenizer: Tokenizers.Tokenizer?
    private var isLoaded = false

    init() {}

    /// Load tokenizer from a tokenizer.json file
    func loadFromFile(_ path: String) async throws {
        let url = URL(fileURLWithPath: path)
        let folderURL = url.deletingLastPathComponent()
        tokenizer = try await AutoTokenizer.from(modelFolder: folderURL)
        isLoaded = true
        print("✅ Tokenizer loaded from: \(path)")
        runSelfTest()
    }

    /// Load tokenizer from a folder containing tokenizer.json
    func loadFromFolder(_ folderPath: String) async throws {
        let url = URL(fileURLWithPath: folderPath)
        tokenizer = try await AutoTokenizer.from(modelFolder: url)
        isLoaded = true
        print("✅ Tokenizer loaded from folder: \(folderPath)")
        runSelfTest()
    }

    /// Verify that special tokens are encoded to the correct single-token IDs
    private func runSelfTest() {
        let imStart = encode("<|im_start|>")
        let imEnd   = encode("<|im_end|>")
        let hello   = encode("Hello")
        print("🧪 Tokenizer self-test:")
        print("   <|im_start|> → \(imStart)  (expected: [151644])")
        print("   <|im_end|>   → \(imEnd)    (expected: [151645])")
        print("   'Hello'      → \(hello)")
        if imStart == [151644] && imEnd == [151645] {
            print("   ✅ Special tokens OK")
        } else {
            print("   ❌ Special tokens WRONG — tokenizer_config.json or tokenizer.json may be missing/corrupt")
        }
    }

    /// Check if tokenizer is loaded
    var loaded: Bool { isLoaded }

    /// Encode text to token IDs
    func encode(_ text: String) -> [Int32] {
        guard let tokenizer = tokenizer else {
            print("⚠️ Tokenizer not loaded, using fallback encoding")
            return fallbackEncode(text)
        }

        let tokens = tokenizer.encode(text: text)
        return tokens.map { Int32($0) }
    }

    /// Decode token IDs back to text
    func decode(_ tokenIds: [Int32]) -> String {
        guard let tokenizer = tokenizer else {
            print("⚠️ Tokenizer not loaded, using fallback decoding")
            return fallbackDecode(tokenIds)
        }

        let intTokens = tokenIds.map { Int($0) }
        return tokenizer.decode(tokens: intTokens)
    }

    /// Format a prompt with Qwen3's chat template
    func formatChatPrompt(_ userMessage: String, systemPrompt: String = "You are a helpful AI assistant.", enableThinking: Bool = false) -> [Int32] {
        let thinkTag = enableThinking ? "/think" : "/no_think"
        let promptText = "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\n\(userMessage) \(thinkTag)<|im_end|>\n<|im_start|>assistant\n"
        return encode(promptText)
    }

    // MARK: - Fallback Methods (when tokenizer.json not loaded)

    private func fallbackEncode(_ text: String) -> [Int32] {
        // Cannot produce valid Qwen3 tokens without the real tokenizer
        print("❌ Qwen3Tokenizer: Cannot encode without tokenizer.json loaded. Returning empty.")
        return []
    }

    private func fallbackDecode(_ tokenIds: [Int32]) -> String {
        // Cannot decode Qwen3 tokens without the real tokenizer
        print("❌ Qwen3Tokenizer: Cannot decode without tokenizer.json loaded. Returning empty.")
        return ""
    }
}

// MARK: - Protocol

/// Protocol for different model input/output formats
protocol LLMModelAdapter {
    /// Prepare input features for the specific model format
    /// - Parameters:
    ///   - prompt: The input prompt text
    ///   - maxTokens: Maximum number of tokens to generate
    ///   - temperature: Sampling temperature (0.0 to 1.0)
    ///   - enableThinking: Whether to enable native thinking mode (if supported)
    func prepareInput(prompt: String, maxTokens: Int, temperature: Double, enableThinking: Bool) throws -> MLFeatureProvider

    /// Extract output text from model prediction
    func extractOutput(from output: MLFeatureProvider) throws -> String

    /// Model-specific configuration
    var modelConfiguration: MLModelConfiguration { get }

    /// Whether this adapter supports native thinking mode
    var supportsNativeThinking: Bool { get }

    /// Whether this adapter requires autoregressive generation
    var requiresAutoregressiveGeneration: Bool { get }

    /// Get the tokenizer for this adapter (if any)
    var tokenizer: Qwen3Tokenizer? { get }

    /// Prepare token-based input for autoregressive generation
    func prepareTokenInput(tokenIds: [Int32]) throws -> MLFeatureProvider

    /// Sample next token from logits output
    func sampleNextToken(from output: MLFeatureProvider, temperature: Double) throws -> Int32

    /// Get the EOS token ID for stopping generation
    var eosTokenId: Int32 { get }

    /// Reset per-generation diagnostic counters (shape log, top-5 log)
    func resetSessionDiagnostics()
}

// MARK: - Protocol Default Implementations

extension LLMModelAdapter {
    var requiresAutoregressiveGeneration: Bool { false }
    var tokenizer: Qwen3Tokenizer? { nil }
    var eosTokenId: Int32 { 0 }

    func prepareTokenInput(tokenIds: [Int32]) throws -> MLFeatureProvider {
        throw LLMModelError.unsupportedModelFormat
    }

    func sampleNextToken(from output: MLFeatureProvider, temperature: Double) throws -> Int32 {
        throw LLMModelError.unsupportedModelFormat
    }

    func resetSessionDiagnostics() {}  // no-op for adapters that don't need it
}

// MARK: - Standard Text-to-Text Adapter

/// For models that accept and return plain text
struct TextToTextAdapter: LLMModelAdapter {

    /// Keys used by the model for input/output
    let inputTextKey: String
    let outputTextKey: String
    let maxTokensKey: String?
    let temperatureKey: String?

    init(
        inputTextKey: String = "text",
        outputTextKey: String = "generated_text",
        maxTokensKey: String? = "max_tokens",
        temperatureKey: String? = "temperature"
    ) {
        self.inputTextKey = inputTextKey
        self.outputTextKey = outputTextKey
        self.maxTokensKey = maxTokensKey
        self.temperatureKey = temperatureKey
    }

    func prepareInput(prompt: String, maxTokens: Int, temperature: Double, enableThinking: Bool) throws -> MLFeatureProvider {
        var inputDict: [String: Any] = [
            inputTextKey: prompt
        ]

        if let maxTokensKey = maxTokensKey {
            inputDict[maxTokensKey] = maxTokens
        }

        if let temperatureKey = temperatureKey {
            inputDict[temperatureKey] = temperature
        }

        // Note: enableThinking is ignored for generic text adapters
        // as they don't have native thinking support

        return try MLDictionaryFeatureProvider(dictionary: inputDict)
    }

    func extractOutput(from output: MLFeatureProvider) throws -> String {
        guard let text = output.featureValue(for: outputTextKey)?.stringValue else {
            throw LLMModelError.outputExtractionFailed(
                "Could not find text output for key '\(outputTextKey)'"
            )
        }
        return text
    }

    var modelConfiguration: MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        return config
    }

    var supportsNativeThinking: Bool {
        return false  // Generic text adapters don't have native thinking
    }
}

// MARK: - Token-based Adapter

/// For models that use tokenization (input/output as token IDs)
struct TokenBasedAdapter: LLMModelAdapter {
    
    /// Tokenizer for converting text to/from token IDs
    let tokenizer: BasicTokenizer
    
    let inputIdsKey: String
    let attentionMaskKey: String?
    let outputIdsKey: String
    
    init(
        tokenizer: BasicTokenizer,
        inputIdsKey: String = "input_ids",
        attentionMaskKey: String? = "attention_mask",
        outputIdsKey: String = "output_ids"
    ) {
        self.tokenizer = tokenizer
        self.inputIdsKey = inputIdsKey
        self.attentionMaskKey = attentionMaskKey
        self.outputIdsKey = outputIdsKey
    }
    
    func prepareInput(prompt: String, maxTokens: Int, temperature: Double, enableThinking: Bool) throws -> MLFeatureProvider {
        // Tokenize the prompt
        let tokens = tokenizer.encode(prompt)
        
        // Create MLMultiArray for input_ids
        let inputIds = try MLMultiArray(shape: [1, tokens.count] as [NSNumber], dataType: .int32)
        for (i, token) in tokens.enumerated() {
            inputIds[i] = NSNumber(value: token)
        }
        
        var inputDict: [String: Any] = [
            inputIdsKey: inputIds
        ]
        
        // Create attention mask if needed (all 1s for valid tokens)
        if let attentionMaskKey = attentionMaskKey {
            let attentionMask = try MLMultiArray(shape: [1, tokens.count] as [NSNumber], dataType: .int32)
            for i in 0..<tokens.count {
                attentionMask[i] = 1
            }
            inputDict[attentionMaskKey] = attentionMask
        }
        
        // Note: maxTokens, temperature, and enableThinking might need to be handled differently
        // depending on your specific model implementation
        
        return try MLDictionaryFeatureProvider(dictionary: inputDict)
    }
    
    func extractOutput(from output: MLFeatureProvider) throws -> String {
        guard let outputIds = output.featureValue(for: outputIdsKey)?.multiArrayValue else {
            throw LLMModelError.outputExtractionFailed(
                "Could not find output token IDs for key '\(outputIdsKey)'"
            )
        }
        
        // Convert token IDs back to text
        var tokenIds: [Int] = []
        for i in 0..<outputIds.count {
            let value = outputIds[i].intValue
            tokenIds.append(value)
        }
        
        return tokenizer.decode(tokenIds)
    }
    
    var modelConfiguration: MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        return config
    }
    
    var supportsNativeThinking: Bool {
        return false  // Token-based adapters don't have native thinking by default
    }
}

// MARK: - Qwen-specific Adapter (Token-based with Autoregressive Generation)

/// Specialized adapter for Qwen3 models converted to CoreML
/// Uses token-based input/output with autoregressive generation
/// Supports native thinking mode controlled by user's reasoning settings
final class QwenAdapter: LLMModelAdapter {

    // MARK: - Properties

    private let _tokenizer = Qwen3Tokenizer()
    private var cachedPromptTokens: [Int32] = []
    private var enableThinkingMode: Bool = false
    private var tokenizerLoaded = false

    // Qwen3 model input/output keys
    let inputIdsKey = "input_ids"
    let logitsKey: String

    private var lastActualLength: Int = 0
    private var hasLoggedShape = false
    private var diagnosticStepCount = 0

    init(logitsOutputKey: String = "logits") {
        self.logitsKey = logitsOutputKey
    }

    /// Load the tokenizer from a file path
    func loadTokenizer(from path: String) async throws {
        try await _tokenizer.loadFromFile(path)
        tokenizerLoaded = true
    }

    /// Load the tokenizer from a folder containing tokenizer.json
    func loadTokenizerFromFolder(_ folderPath: String) async throws {
        try await _tokenizer.loadFromFolder(folderPath)
        tokenizerLoaded = true
    }

    /// Check if tokenizer is loaded
    var isTokenizerLoaded: Bool {
        return _tokenizer.loaded
    }

    // MARK: - LLMModelAdapter Protocol

    var tokenizer: Qwen3Tokenizer? {
        return _tokenizer
    }

    var requiresAutoregressiveGeneration: Bool {
        return true  // Qwen3 CoreML models output logits, need generation loop
    }

    var supportsNativeThinking: Bool {
        return true  // Qwen3 supports thinking via prompt formatting
    }

    var eosTokenId: Int32 {
        return Qwen3Tokenizer.imEndTokenId  // 151645 - <|im_end|>
    }

    var modelConfiguration: MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU  // .all triggers ANE compilation which OOMs on load
        return config
    }

    /// Reset per-session state so diagnostics fire again on the next generation
    func resetSessionDiagnostics() {
        hasLoggedShape = false
        diagnosticStepCount = 0
    }

    /// Prepare initial input from text prompt (used to start generation)
    func prepareInput(prompt: String, maxTokens: Int, temperature: Double, enableThinking: Bool) throws -> MLFeatureProvider {
        self.enableThinkingMode = enableThinking

        if enableThinking {
            print("🧠 Qwen Native Thinking: ENABLED (user requested reasoning mode)")
        } else {
            print("💬 Qwen Native Thinking: DISABLED (normal conversation mode)")
        }

        // Tokenize the prompt with chat template
        let tokens = _tokenizer.formatChatPrompt(prompt, enableThinking: enableThinking)
        cachedPromptTokens = tokens

        print("📝 Tokenized prompt: \(tokens.count) tokens")

        return try prepareTokenInput(tokenIds: tokens)
    }

    /// Prepare token-based input for next-token prediction
    func prepareTokenInput(tokenIds: [Int32]) throws -> MLFeatureProvider {
        let fixedLength = 512

        // Truncate to last 512 tokens or right-pad to 512 with pad token
        let actualTokens: [Int32]
        if tokenIds.count > fixedLength {
            actualTokens = Array(tokenIds.suffix(fixedLength))
        } else {
            let padding = Array(repeating: Qwen3Tokenizer.padTokenId, count: fixedLength - tokenIds.count)
            actualTokens = tokenIds + padding
        }
        lastActualLength = min(tokenIds.count, fixedLength)

        // Create input_ids MLMultiArray [1, 512]
        let inputIds = try MLMultiArray(shape: [1, fixedLength] as [NSNumber], dataType: .int32)
        for (i, tokenId) in actualTokens.enumerated() {
            inputIds[i] = NSNumber(value: tokenId)
        }

        let inputDict: [String: Any] = [
            inputIdsKey: inputIds
        ]

        return try MLDictionaryFeatureProvider(dictionary: inputDict)
    }

    /// Extract text output - for autoregressive models, this decodes generated tokens
    func extractOutput(from output: MLFeatureProvider) throws -> String {
        // This method is called after generation is complete
        // The generated tokens should be stored externally
        throw LLMModelError.outputExtractionFailed(
            "QwenAdapter requires autoregressive generation. Use generateText() in LLMModelManager instead."
        )
    }

    /// Sample next token from logits using temperature sampling
    func sampleNextToken(from output: MLFeatureProvider, temperature: Double) throws -> Int32 {
        guard let logits = output.featureValue(for: logitsKey)?.multiArrayValue else {
            throw LLMModelError.outputExtractionFailed(
                "Could not find logits output for key '\(logitsKey)'"
            )
        }

        // Logits shape is typically [1, seq_length, vocab_size]
        // We want the last token's logits
        let shape = logits.shape.map { $0.intValue }

        guard shape.count >= 2 else {
            throw LLMModelError.outputExtractionFailed(
                "Unexpected logits shape: \(shape)"
            )
        }

        // Strides may be non-contiguous when ANE returns data (e.g. padded alignment)
        let strides = logits.strides.map { $0.intValue }

        // Log shape + strides once — critical for diagnosing ANE memory layout
        if !hasLoggedShape {
            print("🔬 Logits shape: \(shape), strides: \(strides), dataType: \(logits.dataType.rawValue)")
            hasLoggedShape = true
        }

        let vocabSize: Int
        let tokenPos: Int   // 0-based position of the last real token in the sequence

        if shape.count == 3 {
            // [batch, seqLen, vocab]
            vocabSize = shape[2]
            let seqLen = shape[1]
            tokenPos = seqLen <= 1 ? 0 : min(max(lastActualLength - 1, 0), seqLen - 1)
        } else if shape.count == 2 {
            // [seqLen, vocab] or [1, vocab]
            vocabSize = shape[1]
            tokenPos = shape[0] == 1 ? 0 : min(max(lastActualLength - 1, 0), shape[0] - 1)
        } else {
            throw LLMModelError.outputExtractionFailed("Unexpected logits shape: \(shape)")
        }

        // Compute the element-level base offset for [batch=0, tokenPos, :] using actual strides.
        // This is correct even when the ANE returns non-contiguous (e.g. padded/transposed) memory.
        let s0 = strides.count > 2 ? strides[0] : 0   // batch stride (unused, batch=0)
        let s1 = strides.count > 1 ? strides[strides.count - 2] : vocabSize  // seq stride
        let s2 = strides.last ?? 1                                             // vocab stride
        let baseOffset = 0 * s0 + tokenPos * s1   // element offset to [0, tokenPos, 0]

        // Extract logits using stride-aware indexing
        var lastLogits: [Float] = []
        lastLogits.reserveCapacity(vocabSize)

        if logits.dataType == .float16 {
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<vocabSize {
                lastLogits.append(Float(ptr[baseOffset + i * s2]))
            }
        } else {
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<vocabSize {
                lastLogits.append(ptr[baseOffset + i * s2])
            }
        }

        diagnosticStepCount += 1

        // Step 1: compare top-1 across three key positions to identify the correct read position
        if diagnosticStepCount == 1 {
            let lastSeqPos = (shape.count == 3 ? shape[1] : shape[0]) - 1
            let checkPositions: [(String, Int)] = [
                ("pos0",   0),
                ("posN-1", tokenPos),
                ("posEnd", lastSeqPos)
            ]
            for (label, pos) in checkPositions {
                let offset = pos * s1  // stride-aware, batch=0
                var top1Val: Float = -.infinity
                var top1Idx = 0
                if logits.dataType == .float16 {
                    let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
                    for i in 0..<vocabSize {
                        let v = Float(ptr[offset + i * s2])
                        if v > top1Val { top1Val = v; top1Idx = i }
                    }
                } else {
                    let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
                    for i in 0..<vocabSize {
                        let v = ptr[offset + i * s2]
                        if v > top1Val { top1Val = v; top1Idx = i }
                    }
                }
                let decoded = _tokenizer.decode([Int32(top1Idx)])
                print("📍 \(label) (seq=\(pos)) top1: [\(top1Idx)]\(String(format: "%.2f", top1Val)) '\(decoded)'")
            }
        }

        // Steps 1-5: full top-5 from the chosen position
        if diagnosticStepCount <= 5 {
            let top5 = lastLogits.enumerated()
                .sorted { $0.element > $1.element }
                .prefix(5)
            let top5Desc = top5.map { (idx, val) -> String in
                let text = _tokenizer.decode([Int32(idx)])
                return "[\(idx)]\(String(format: "%.1f", val)) '\(text)'"
            }.joined(separator: ", ")
            print("🎯 Step \(diagnosticStepCount) top-5: \(top5Desc)  (tokenPos=\(tokenPos), actualLen=\(lastActualLength))")
        }

        // Apply temperature and sample
        let sampledToken = sampleWithTemperature(logits: lastLogits, temperature: temperature)

        return Int32(sampledToken)
    }

    // MARK: - Sampling Helpers

    /// Sample from logits using temperature-based softmax
    private func sampleWithTemperature(logits: [Float], temperature: Double) -> Int {
        if temperature <= 0.01 {
            // Greedy sampling - return argmax
            var maxIdx = 0
            var maxVal = logits[0]
            for (i, val) in logits.enumerated() {
                if val > maxVal {
                    maxVal = val
                    maxIdx = i
                }
            }
            return maxIdx
        }

        // Apply temperature
        let temp = Float(temperature)
        var scaledLogits = logits.map { $0 / temp }

        // Compute softmax
        let maxLogit = scaledLogits.max() ?? 0
        var expSum: Float = 0
        for i in 0..<scaledLogits.count {
            scaledLogits[i] = exp(scaledLogits[i] - maxLogit)
            expSum += scaledLogits[i]
        }

        // Normalize to probabilities
        for i in 0..<scaledLogits.count {
            scaledLogits[i] /= expSum
        }

        // Sample from distribution
        let random = Float.random(in: 0..<1)
        var cumSum: Float = 0
        for (i, prob) in scaledLogits.enumerated() {
            cumSum += prob
            if random < cumSum {
                return i
            }
        }

        return scaledLogits.count - 1
    }

    /// Clean Qwen output by removing special tokens
    func cleanOutput(_ text: String) -> String {
        var cleaned = text

        // Only remove actual Qwen special token markers
        let specialTokens = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>"
        ]

        for token in specialTokens {
            cleaned = cleaned.replacingOccurrences(of: token, with: "")
        }

        // Remove role prefixes only when they appear at the start of a line
        // (these are template artifacts, not user content)
        cleaned = cleaned.replacingOccurrences(
            of: "(?m)^(system|assistant|user)\\n",
            with: "",
            options: .regularExpression
        )

        cleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned
    }
}

// MARK: - Auto-detecting Adapter

/// Automatically detects the best adapter based on model description
final class AutoDetectingAdapter: LLMModelAdapter {

    private let model: MLModel
    private let underlyingAdapter: LLMModelAdapter

    init(model: MLModel) throws {
        self.model = model

        // Analyze model inputs to determine the best adapter
        let inputNames = Set(model.modelDescription.inputDescriptionsByName.keys)
        let outputNames = Set(model.modelDescription.outputDescriptionsByName.keys)

        print("🔍 Auto-detecting adapter...")
        print("   Input features: \(inputNames)")
        print("   Output features: \(outputNames)")

        let logitOutputKey = outputNames.first {
            model.modelDescription.outputDescriptionsByName[$0]?.type == .multiArray
        }
        if inputNames.contains("input_ids"), let logitKey = logitOutputKey {
            // Token-based model with logits output (like Qwen3 CoreML)
            // This requires autoregressive generation
            print("   → Detected token-based model with logits output (\(logitKey))")
            print("   → Using QwenAdapter for autoregressive generation")
            underlyingAdapter = QwenAdapter(logitsOutputKey: logitKey)
        } else if inputNames.contains("input_ids") {
            // Token-based model with direct output
            let tokenizer = SimpleTokenizer()
            underlyingAdapter = TokenBasedAdapter(tokenizer: tokenizer)
        } else if inputNames.contains("prompt") {
            // Text-to-text model (Qwen or similar)
            underlyingAdapter = QwenAdapter()
        } else {
            // Default to text-to-text
            let possibleInputKeys = ["text", "prompt", "input", "input_text"]
            let possibleOutputKeys = ["generated_text", "output", "text", "output_text"]

            let inputKey = inputNames.first { possibleInputKeys.contains($0) } ?? "text"
            let outputKey = outputNames.first { possibleOutputKeys.contains($0) } ?? "generated_text"

            underlyingAdapter = TextToTextAdapter(
                inputTextKey: inputKey,
                outputTextKey: outputKey
            )
        }
    }

    func prepareInput(prompt: String, maxTokens: Int, temperature: Double, enableThinking: Bool) throws -> MLFeatureProvider {
        try underlyingAdapter.prepareInput(prompt: prompt, maxTokens: maxTokens, temperature: temperature, enableThinking: enableThinking)
    }

    func extractOutput(from output: MLFeatureProvider) throws -> String {
        try underlyingAdapter.extractOutput(from: output)
    }

    var modelConfiguration: MLModelConfiguration {
        underlyingAdapter.modelConfiguration
    }

    var supportsNativeThinking: Bool {
        underlyingAdapter.supportsNativeThinking
    }

    var requiresAutoregressiveGeneration: Bool {
        underlyingAdapter.requiresAutoregressiveGeneration
    }

    var tokenizer: Qwen3Tokenizer? {
        underlyingAdapter.tokenizer
    }

    var eosTokenId: Int32 {
        underlyingAdapter.eosTokenId
    }

    func prepareTokenInput(tokenIds: [Int32]) throws -> MLFeatureProvider {
        try underlyingAdapter.prepareTokenInput(tokenIds: tokenIds)
    }

    func sampleNextToken(from output: MLFeatureProvider, temperature: Double) throws -> Int32 {
        try underlyingAdapter.sampleNextToken(from: output, temperature: temperature)
    }

    func resetSessionDiagnostics() {
        underlyingAdapter.resetSessionDiagnostics()
    }
}

// MARK: - Simple Tokenizer (Placeholder)

/// A simple tokenizer placeholder for non-Qwen models
protocol BasicTokenizer {
    func encode(_ text: String) -> [Int]
    func decode(_ tokens: [Int]) -> String
}

struct SimpleTokenizer: BasicTokenizer {
    func encode(_ text: String) -> [Int] {
        // Placeholder: split by whitespace and convert to simple IDs
        // In a real implementation, use the model's actual tokenizer
        return text.split(separator: " ").enumerated().map { $0.offset }
    }
    
    func decode(_ tokens: [Int]) -> String {
        // Placeholder implementation
        return tokens.map { String($0) }.joined(separator: " ")
    }
}

// MARK: - Error Handling

enum LLMModelError: LocalizedError {
    case outputExtractionFailed(String)
    case inputPreparationFailed(String)
    case unsupportedModelFormat
    
    var errorDescription: String? {
        switch self {
        case .outputExtractionFailed(let message):
            return "Failed to extract output: \(message)"
        case .inputPreparationFailed(let message):
            return "Failed to prepare input: \(message)"
        case .unsupportedModelFormat:
            return "Unsupported model format"
        }
    }
}

// MARK: - Usage Examples

/*
 
 Example 1: Using Qwen adapter with user-controlled thinking
 
 ```swift
 // User has enabled reasoning mode in settings
 let userWantsReasoning = conversation.reasoningMode || conversation.smartReasoningMode
 
 let adapter = QwenAdapter()
 let input = try adapter.prepareInput(
     prompt: "What is Swift?",
     maxTokens: 512,
     temperature: 0.7,
     enableThinking: userWantsReasoning  // Controlled by user settings!
 )
 let output = try model.prediction(from: input)
 let text = try adapter.extractOutput(from: output)
 ```
 
 Example 2: Auto-detecting adapter with thinking support
 
 ```swift
 let model = try MLModel(contentsOf: modelURL)
 let adapter = try AutoDetectingAdapter(model: model)
 
 // Check if model supports native thinking
 if adapter.supportsNativeThinking {
     print("This model supports native thinking mode!")
 }
 
 // Use thinking based on user's reasoning mode settings
 let useThinking = conversation.reasoningMode || messageIsComplex
 let input = try adapter.prepareInput(
     prompt: "Tell me a story",
     maxTokens: 512,
     temperature: 0.7,
     enableThinking: useThinking
 )
 let output = try model.prediction(from: input)
 let text = try adapter.extractOutput(from: output)
 ```
 
 Example 3: Integration with ModelBackendBridge
 
 ```swift
 // Check if current model supports native thinking
 let bridge = ModelBackendBridge()
 if bridge.supportsNativeThinking {
     // Determine if thinking should be enabled based on user settings
     let shouldUseThinking = bridge.shouldEnableThinking(
         reasoningMode: conversation.reasoningMode,
         smartReasoningMode: conversation.smartReasoningMode,
         messageReasoningMode: message.isReasoningMode
     )
     
     // Pass to adapter
     let adapter = QwenAdapter()
     let input = try adapter.prepareInput(
         prompt: prompt,
         maxTokens: 512,
         temperature: 0.7,
         enableThinking: shouldUseThinking  // User-controlled!
     )
 }
 ```
 
 */
