import Foundation
import Hub
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

private struct BenchmarkOptions: Sendable {
    let modelID: String
    let pplTokens: Int
    let promptTokenTarget: Int
    let generatedTokens: Int
    let prefillStepSize: Int
    let rotorExactBufferSize: Int
    let rotorAttentionBlockTokens: Int
    let variant: RotorQuantVariant
    let scenarioKeys: [String]
    let text: String

    static func fromEnvironment() -> BenchmarkOptions {
        let environment = ProcessInfo.processInfo.environment
        let modelID = environment["ROTORQUANT_EVAL_MODEL"]
            ?? "mlx-community/Qwen3.5-4B-MLX-4bit"
        let pplTokens = Int(environment["ROTORQUANT_EVAL_PPL_TOKENS"] ?? "") ?? 128
        let promptTokenTarget = Int(environment["ROTORQUANT_EVAL_PROMPT_TOKENS"] ?? "") ?? 1024
        let generatedTokens = Int(environment["ROTORQUANT_EVAL_GENERATED_TOKENS"] ?? "") ?? 128
        let prefillStepSize = Int(environment["ROTORQUANT_EVAL_PREFILL_STEP"] ?? "") ?? 256
        let rotorExactBufferSize = Int(environment["ROTORQUANT_EVAL_EXACT_BUFFER"] ?? "") ?? 128
        let rotorAttentionBlockTokens = Int(environment["ROTORQUANT_EVAL_BLOCK_TOKENS"] ?? "") ?? 128
        let variant = RotorQuantVariant(rawValue: environment["ROTORQUANT_EVAL_VARIANT"] ?? "")
            ?? .iso
        let scenarioKeys = (environment["ROTORQUANT_EVAL_SCENARIOS"] ?? "")
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty }
        let text = environment["ROTORQUANT_EVAL_TEXT"]
            ?? """
            Local language models trade memory, latency, and output quality when the context window grows. \
            A cache compressor should preserve attention behavior, keep recurrent-state layers untouched, \
            and avoid special cases that fail when key and value head dimensions change.
            """
        return BenchmarkOptions(
            modelID: modelID,
            pplTokens: max(8, pplTokens),
            promptTokenTarget: max(32, promptTokenTarget),
            generatedTokens: max(1, generatedTokens),
            prefillStepSize: max(1, prefillStepSize),
            rotorExactBufferSize: max(0, rotorExactBufferSize),
            rotorAttentionBlockTokens: max(1, rotorAttentionBlockTokens),
            variant: variant,
            scenarioKeys: scenarioKeys,
            text: text
        )
    }

    var longPrompt: String {
        Array(repeating: text, count: max(1, promptTokenTarget / 24)).joined(separator: " ")
    }
}

private struct Scenario: Sendable {
    let key: String
    let name: String
    let compression: KVCacheCompressionMode?
}

private struct CacheSummary: Sendable {
    var simple = 0
    var quantized = 0
    var rotorQuant = 0
    var mamba = 0
    var other = 0

    var description: String {
        "simple=\(simple) quantized=\(quantized) rotor=\(rotorQuant) mamba=\(mamba) other=\(other)"
    }
}

private struct PerplexityResult: Sendable {
    let tokenCount: Int
    let loss: Double
    let perplexity: Double
    let elapsed: TimeInterval
    let cacheSummary: CacheSummary
}

private struct GenerationResult: Sendable {
    let promptTokens: Int
    let generatedTokens: Int
    let prefillTokensPerSecond: Double
    let decodeTokensPerSecond: Double
}

private struct MemoryResult: Sendable {
    let kvBytes: Int
    let peakActiveBytes: Int
    let workspaceBytes: Int
    let cacheSummary: CacheSummary
}

private struct ScenarioResult: Sendable {
    let scenario: Scenario
    let ppl: PerplexityResult
    let generation: GenerationResult
    let memory: MemoryResult
}

private func formatMiB(_ bytes: Int) -> String {
    String(format: "%.1f MiB", Double(bytes) / (1024.0 * 1024.0))
}

private func logLine(_ message: String = "") {
    FileHandle.standardOutput.write(Data((message + "\n").utf8))
}

private func stableNegativeLogLikelihood(logits: [Float], target: Int) -> Double {
    let maxLogit = logits.max() ?? 0
    var sum = 0.0
    for value in logits {
        sum += exp(Double(value - maxLogit))
    }
    return Double(maxLogit) + log(sum) - Double(logits[target])
}

private func summarize(_ cache: [KVCache]) -> CacheSummary {
    var summary = CacheSummary()
    for item in cache {
        switch item {
        case is RotorQuantKVCache:
            summary.rotorQuant += 1
        case is QuantizedKVCache:
            summary.quantized += 1
        case is MambaCache:
            summary.mamba += 1
        case is KVCacheSimple:
            summary.simple += 1
        default:
            summary.other += 1
        }
    }
    return summary
}

private func makeParameters(
    options: BenchmarkOptions,
    compression: KVCacheCompressionMode?
) -> GenerateParameters {
    GenerateParameters(
        maxTokens: options.generatedTokens,
        cacheCompression: compression,
        temperature: 0,
        prefillStepSize: options.prefillStepSize
    )
}

private func measurePerplexity(
    context: ModelContext,
    tokens: [Int],
    parameters: GenerateParameters
) throws -> PerplexityResult {
    var cache = context.model.newCache(parameters: parameters)
    var state: LMOutput.State?
    var negativeLogLikelihood = 0.0
    let start = Date()

    for index in 0 ..< (tokens.count - 1) {
        let input = MLXArray([tokens[index]]).reshaped([1, 1])
        let output = context.model(
            LMInput.Text(tokens: input),
            cache: cache.isEmpty ? nil : cache,
            state: state
        )
        state = output.state
        maybeApplyKVCacheCompression(cache: &cache, compression: parameters.resolvedCacheCompression)

        let logits = output.logits[0, -1, 0...].asType(.float32)
        eval(logits)
        let values = logits.asArray(Float.self)
        let target = tokens[index + 1]
        guard target >= 0 && target < values.count else {
            throw NSError(
                domain: "RotorQuantEval",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Target token \(target) is outside logits size \(values.count)"]
            )
        }
        negativeLogLikelihood += stableNegativeLogLikelihood(logits: values, target: target)
    }

    let predictions = max(1, tokens.count - 1)
    let loss = negativeLogLikelihood / Double(predictions)
    return PerplexityResult(
        tokenCount: predictions,
        loss: loss,
        perplexity: exp(loss),
        elapsed: Date().timeIntervalSince(start),
        cacheSummary: summarize(cache)
    )
}

private func measureGeneration(
    container: ModelContainer,
    prompt: String,
    parameters: GenerateParameters
) async throws -> GenerationResult {
    let input = try await container.prepare(input: UserInput(prompt: prompt))
    let stream = try await container.generate(input: input, parameters: parameters)
    var finalInfo: GenerateCompletionInfo?

    for await update in stream {
        if case .info(let info) = update {
            finalInfo = info
        }
    }

    guard let finalInfo else {
        throw NSError(
            domain: "RotorQuantEval",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "Generation finished without completion info"]
        )
    }

    return GenerationResult(
        promptTokens: finalInfo.promptTokenCount,
        generatedTokens: finalInfo.generationTokenCount,
        prefillTokensPerSecond: finalInfo.promptTokensPerSecond,
        decodeTokensPerSecond: finalInfo.tokensPerSecond
    )
}

private func measureCacheAfterFirstDecode(
    context: ModelContext,
    prompt: String,
    parameters: GenerateParameters
) async throws -> MemoryResult {
    let weights = context.model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
    let input = try await context.processor.prepare(input: UserInput(prompt: prompt))
    var cache = context.model.newCache(parameters: parameters)
    var state: LMOutput.State?
    let startActive = Memory.activeMemory
    Memory.peakMemory = 0

    switch try context.model.prepare(input, cache: cache, windowSize: parameters.prefillStepSize) {
    case .tokens(let tokens):
        let result = context.model(
            tokens[text: .newAxis],
            cache: cache.isEmpty ? nil : cache,
            state: nil
        )
        state = result.state
        eval(result.logits)
    case .logits(let result):
        state = result.state
        eval(result.logits)
    }

    let decodeToken = MLXArray([0]).reshaped([1, 1])
    let decodeResult = context.model(
        LMInput.Text(tokens: decodeToken),
        cache: cache.isEmpty ? nil : cache,
        state: state
    )
    maybeApplyKVCacheCompression(cache: &cache, compression: parameters.resolvedCacheCompression)
    eval(decodeResult.logits)

    // `state` is serialization-oriented and may expose logical slices backed by
    // larger live allocations. Measure the runtime arrays so capacity regressions
    // cannot make the reported KV footprint look artificially small.
    let cacheArrays = cache.flatMap { cacheItem in
        let runtimeArrays = cacheItem.innerState()
        return runtimeArrays.isEmpty ? cacheItem.state : runtimeArrays
    }
    if !cacheArrays.isEmpty {
        eval(cacheArrays)
    }

    let kvBytes = cacheArrays.reduce(0) { $0 + $1.nbytes }
    let peakActive = max(Memory.peakMemory, startActive)
    return MemoryResult(
        kvBytes: kvBytes,
        peakActiveBytes: peakActive,
        workspaceBytes: max(0, peakActive - weights - kvBytes),
        cacheSummary: summarize(cache)
    )
}

private func measureMemory(
    container: ModelContainer,
    prompt: String,
    parameters: GenerateParameters
) async throws -> MemoryResult {
    try await container.perform(values: (prompt, parameters)) { context, values in
        let measurement = try await measureCacheAfterFirstDecode(
            context: context,
            prompt: values.0,
            parameters: values.1
        )
        return MemoryResult(
            kvBytes: measurement.kvBytes,
            peakActiveBytes: measurement.peakActiveBytes,
            workspaceBytes: measurement.workspaceBytes,
            cacheSummary: measurement.cacheSummary
        )
    }
}

private func evaluateScenario(
    _ scenario: Scenario,
    container: ModelContainer,
    options: BenchmarkOptions,
    tokens: [Int]
) async throws -> ScenarioResult {
    let parameters = makeParameters(options: options, compression: scenario.compression)
    logLine("  Measuring perplexity...")
    let ppl = try await container.perform(values: (tokens, parameters)) { context, values in
        try measurePerplexity(context: context, tokens: values.0, parameters: values.1)
    }
    logLine("  Measuring generation...")
    let generation = try await measureGeneration(
        container: container,
        prompt: options.longPrompt,
        parameters: parameters
    )
    logLine("  Measuring cache memory...")
    let memory = try await measureMemory(
        container: container,
        prompt: options.longPrompt,
        parameters: parameters
    )
    return ScenarioResult(
        scenario: scenario,
        ppl: ppl,
        generation: generation,
        memory: memory
    )
}

private func selectedScenarios(_ scenarios: [Scenario], options: BenchmarkOptions) throws -> [Scenario] {
    guard !options.scenarioKeys.isEmpty else { return scenarios }
    let selected = options.scenarioKeys.compactMap { key in
        scenarios.first {
            key == $0.key.lowercased() || key == $0.name.lowercased()
        }
    }
    guard !selected.isEmpty else {
        throw NSError(
            domain: "RotorQuantEval",
            code: 3,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "No scenarios matched ROTORQUANT_EVAL_SCENARIOS=\(options.scenarioKeys.joined(separator: ","))"
            ]
        )
    }
    return selected
}

@main
struct RotorQuantEval {
    static func main() async throws {
        let options = BenchmarkOptions.fromEnvironment()
        logLine("RotorQuant evaluation")
        logLine("  Model:          \(options.modelID)")
        logLine("  PPL tokens:     \(options.pplTokens)")
        logLine("  Prompt target:  \(options.promptTokenTarget)")
        logLine("  Generated:      \(options.generatedTokens)")
        logLine("  Exact buffer:   \(options.rotorExactBufferSize)")
        logLine("  Block tokens:   \(options.rotorAttentionBlockTokens)")
        logLine("  Variant:        \(options.variant.rawValue)")
        if !options.scenarioKeys.isEmpty {
            logLine("  Scenarios:      \(options.scenarioKeys.joined(separator: ","))")
        }

        let hub = HubApi()
        logLine("Loading model container...")
        let container = try await loadModelContainer(
            hub: hub,
            configuration: ModelConfiguration(id: options.modelID)
        )
        logLine("Model loaded.")
        let encoded = await container.encode(options.text)
        let repeatedTokens = Array(
            repeating: encoded,
            count: max(1, options.pplTokens / max(1, encoded.count))
        ).flatMap { $0 }
        let tokens = Array((repeatedTokens + encoded).prefix(options.pplTokens))

        let scenarios = [
            Scenario(key: "dense", name: "dense", compression: nil),
            Scenario(
                key: "legacy",
                name: "legacy-mlx-quantized-kv-4bit",
                compression: .quantized(bits: 4, groupSize: 64, startStep: 0)
            ),
            Scenario(
                key: "rotor",
                name: "rotorquant-\(options.variant.rawValue)-k3-v2",
                compression: .rotorQuant(
                    RotorQuantConfiguration(
                        keyBits: 3,
                        valueBits: 2,
                        seed: 42,
                        exactBufferSize: options.rotorExactBufferSize,
                        attentionBlockTokens: options.rotorAttentionBlockTokens,
                        variant: options.variant
                    )
                )
            ),
        ]

        let scenariosToRun = try selectedScenarios(scenarios, options: options)
        var results: [ScenarioResult] = []
        for scenario in scenariosToRun {
            logLine()
            logLine("Running \(scenario.name)...")
            let result = try await evaluateScenario(
                scenario,
                container: container,
                options: options,
                tokens: tokens
            )
            results.append(result)
            logLine("  loss:       \(String(format: "%.4f", result.ppl.loss))")
            logLine("  ppl:        \(String(format: "%.4f", result.ppl.perplexity))")
            logLine("  ppl time:   \(String(format: "%.2f", result.ppl.elapsed))s")
            logLine("  cache:      \(result.ppl.cacheSummary.description)")
            logLine("  prefill:    \(String(format: "%.1f", result.generation.prefillTokensPerSecond)) tok/s")
            logLine("  decode:     \(String(format: "%.1f", result.generation.decodeTokensPerSecond)) tok/s")
            logLine("  kv memory:  \(formatMiB(result.memory.kvBytes))")
            logLine("  mem cache:  \(result.memory.cacheSummary.description)")
            logLine("  peak:       \(formatMiB(result.memory.peakActiveBytes))")
        }

        guard
            let dense = results.first(where: { $0.scenario.name == "dense" }),
            let legacy = results.first(where: { $0.scenario.name == "legacy-mlx-quantized-kv-4bit" }),
            let rotor = results.first(where: { $0.scenario.name.hasPrefix("rotorquant-") })
        else {
            logLine()
            logLine("Deltas skipped because not all dense, legacy, and rotor scenarios were run.")
            return
        }

        logLine()
        logLine("Deltas")
        logLine("  Rotor PPL - dense:   \(String(format: "%+.4f", rotor.ppl.perplexity - dense.ppl.perplexity))")
        logLine("  Rotor PPL - legacy:  \(String(format: "%+.4f", rotor.ppl.perplexity - legacy.ppl.perplexity))")
        logLine("  Rotor decode/dense:  \(String(format: "%.2fx", rotor.generation.decodeTokensPerSecond / max(0.001, dense.generation.decodeTokensPerSecond)))")
        logLine("  Rotor decode/legacy: \(String(format: "%.2fx", rotor.generation.decodeTokensPerSecond / max(0.001, legacy.generation.decodeTokensPerSecond)))")
        logLine("  Rotor KV/dense:      \(String(format: "%.2f%%", 100.0 * Double(rotor.memory.kvBytes) / Double(max(1, dense.memory.kvBytes))))")
        logLine("  Rotor KV/legacy:     \(String(format: "%.2f%%", 100.0 * Double(rotor.memory.kvBytes) / Double(max(1, legacy.memory.kvBytes))))")
    }
}
