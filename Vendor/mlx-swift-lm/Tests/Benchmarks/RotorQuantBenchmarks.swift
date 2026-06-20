import Foundation
import Hub
import MLX
import MLXLLM
import MLXLMCommon
import Testing

private let rotorQuantBenchmarksEnabled = ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] != nil

@Suite(.serialized)
struct RotorQuantBenchmarks {
    private let hub = HubApi()

    private var modelConfiguration: ModelConfiguration {
        let modelID =
            ProcessInfo.processInfo.environment["ROTORQUANT_BENCHMARK_MODEL"]
            ?? "mlx-community/Qwen3-0.6B-4bit"
        return ModelConfiguration(id: modelID)
    }

    private var promptTokenCount: Int {
        Int(ProcessInfo.processInfo.environment["ROTORQUANT_BENCHMARK_PROMPT_TOKENS"] ?? "")
            ?? 4096
    }

    private var generatedTokenCount: Int {
        Int(ProcessInfo.processInfo.environment["ROTORQUANT_BENCHMARK_GENERATED_TOKENS"] ?? "")
            ?? 64
    }

    private var promptText: String {
        let base =
            "Summarize the engineering tradeoffs of cache compression for long-context inference."
        return Array(repeating: base, count: max(8, promptTokenCount / 12)).joined(separator: " ")
    }

    private func printMeasurementSummary(
        label: String,
        measurement: WiredMemoryMeasurement
    ) {
        let mib = 1024.0 * 1024.0
        print("\(label):")
        print("  Weights:   \(String(format: "%.1f", Double(measurement.weightBytes) / mib)) MiB")
        print("  KV cache:  \(String(format: "%.1f", Double(measurement.kvBytes) / mib)) MiB")
        print("  Workspace: \(String(format: "%.1f", Double(measurement.workspaceBytes) / mib)) MiB")
        print(
            "  Peak:      \(String(format: "%.1f", Double(measurement.peakActiveBytes) / mib)) MiB"
        )
        print("  Tokens:    \(measurement.tokenCount) @ prefill step \(measurement.prefillStepSize)")
    }

    private func measureConfiguration(
        label: String,
        parameters: GenerateParameters,
        context: ModelContext
    ) async throws -> WiredMemoryMeasurement {
        Memory.clearCache()
        let measurement = try await WiredMemoryUtils.tune(
            context: context,
            tokenCount: promptTokenCount,
            parameters: parameters
        )
        printMeasurementSummary(label: label, measurement: measurement)
        return measurement
    }

    private func runGeneration(
        label: String,
        container: ModelContainer,
        parameters: GenerateParameters
    ) async throws {
        let input = try await container.prepare(input: UserInput(prompt: promptText))
        let stream = try await container.generate(input: input, parameters: parameters)

        var finalInfo: GenerateCompletionInfo?
        var chunkCount = 0
        for await update in stream {
            switch update {
            case .chunk:
                chunkCount += 1
            case .info(let info):
                finalInfo = info
            case .toolCall:
                break
            }
        }

        guard let finalInfo else {
            Issue.record("\(label) did not produce generation info")
            return
        }

        print("\(label) decode:")
        print("  Prompt tokens:     \(finalInfo.promptTokenCount)")
        print("  Generated tokens:  \(finalInfo.generationTokenCount)")
        print(
            "  Prompt tok/s:      \(String(format: "%.1f", finalInfo.promptTokensPerSecond))"
        )
        print("  Decode tok/s:      \(String(format: "%.1f", finalInfo.tokensPerSecond))")
        print("  Chunks emitted:    \(chunkCount)")
    }

    @Test(.enabled(if: rotorQuantBenchmarksEnabled))
    func compareMemoryFootprintAgainstDenseAndLegacyQuantizedKV() async throws {
        let context = try await LLMModelFactory.shared.load(
            hub: hub,
            configuration: modelConfiguration
        )

        let dense = GenerateParameters(
            maxTokens: generatedTokenCount,
            prefillStepSize: 256
        )
        let legacyQuantized = GenerateParameters(
            maxTokens: generatedTokenCount,
            cacheCompression: .quantized(bits: 4, groupSize: 64, startStep: 0),
            prefillStepSize: 256
        )
        let rotorQuant = GenerateParameters(
            maxTokens: generatedTokenCount,
            cacheCompression: .rotorQuant(
                RotorQuantConfiguration(
                    keyBits: 3,
                    valueBits: 2,
                    seed: 42,
                    exactBufferSize: 128,
                    attentionBlockTokens: 256,
                    variant: .iso
                )
            ),
            prefillStepSize: 256
        )

        let denseMeasurement = try await measureConfiguration(
            label: "Dense KV",
            parameters: dense,
            context: context
        )
        let legacyMeasurement = try await measureConfiguration(
            label: "Legacy Quantized KV",
            parameters: legacyQuantized,
            context: context
        )
        let rotorMeasurement = try await measureConfiguration(
            label: "RotorQuant KV",
            parameters: rotorQuant,
            context: context
        )

        #expect(rotorMeasurement.kvBytes < denseMeasurement.kvBytes)
        #expect(rotorMeasurement.peakActiveBytes <= denseMeasurement.peakActiveBytes)
        #expect(rotorMeasurement.kvBytes <= legacyMeasurement.kvBytes)
    }

    @Test(.enabled(if: rotorQuantBenchmarksEnabled))
    func longContextDecodeSanityAndRegressionSnapshot() async throws {
        let container = try await LLMModelFactory.shared.loadContainer(
            hub: hub,
            configuration: modelConfiguration
        )

        let rotorQuant = GenerateParameters(
            maxTokens: generatedTokenCount,
            cacheCompression: .rotorQuant(
                RotorQuantConfiguration(
                    keyBits: 3,
                    valueBits: 2,
                    seed: 42,
                    exactBufferSize: 128,
                    attentionBlockTokens: 256,
                    variant: .iso
                )
            ),
            temperature: 0,
            prefillStepSize: 256
        )

        try await runGeneration(
            label: "RotorQuant IsoQuant 3-bit key / 2-bit value",
            container: container,
            parameters: rotorQuant
        )
    }
}
