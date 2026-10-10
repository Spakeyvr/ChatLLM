import Foundation
import MLX
@testable import MLXLMCommon
import Testing

/// MLX kernels share one Metal command queue, so these cases run serially.
@Suite(.serialized)
struct LogitProcessorTests {
    private func values(_ array: MLXArray) -> [Float] {
        array.asType(.float32).reshaped(-1).asArray(Float.self)
    }

    @Test
    func testPresencePenaltyAppliesOnceToPromptAndSampledTokens() {
        var processor = PresenceContext(presencePenalty: 1.5)
        processor.prompt(MLXArray([Int32(1), 1, 3]))
        processor.didSample(token: MLXArray([UInt32(3)]))
        processor.didSample(token: MLXArray(UInt32(5)))

        let processed = processor.process(logits: MLXArray.zeros([1, 8]))

        #expect(values(processed) == [0, -1.5, 0, -1.5, 0, -1.5, 0, 0])
    }

    @Test
    func testRepetitionPenaltyKeepsOnlyTheMostRecentWindow() {
        var processor = RepetitionContext(repetitionPenalty: 2, repetitionContextSize: 2)
        processor.prompt(MLXArray([Int32(0), 1, 2]))
        processor.didSample(token: MLXArray([UInt32(3)]))

        let processed = processor.process(logits: MLXArray([Float](repeating: 4, count: 6), [1, 6]))

        #expect(values(processed) == [4, 4, 2, 2, 4, 4])
    }

    @Test
    func testUnitRepetitionPenaltyBuildsNoProcessor() {
        #expect(GenerateParameters(repetitionPenalty: 1.0).processor() == nil)
        #expect(GenerateParameters(repetitionPenalty: 1.1).processor() is RepetitionContext)
        #expect(
            GenerateParameters(presencePenalty: 1.5, repetitionPenalty: 1.0).processor()
                is PresenceContext)
    }

    @Test
    func testTopKSamplerOnlyReturnsTopCandidates() {
        let sampler = FilteredSampler(temperature: 1, topP: 0.95, topK: 2, minP: 0)
        var logits = [Float](repeating: 0, count: 64)
        logits[11] = 3
        logits[42] = 2.5
        let input = MLXArray(logits, [1, 64])

        let sampled = Set((0 ..< 50).map { _ in sampler.sample(logits: input).item(Int.self) })

        #expect(sampled.isSubset(of: [11, 42]))
    }
}
