// Copyright © 2026 Apple Inc.

import MLX
import XCTest

@testable import MLXVLM

/// Verifies that the fused gated-delta Metal kernel in `Qwen35.swift`
/// produces the same outputs as the per-token ops reference loop.
final class GatedDeltaKernelTests: XCTestCase {

    private struct Inputs {
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        let g: MLXArray
        let beta: MLXArray
        let state: MLXArray
    }

    private func makeInputs(
        batch: Int, tokens: Int, kHeads: Int, vHeads: Int,
        kDim: Int, vDim: Int, dtype: DType
    ) -> Inputs {
        MLXRandom.seed(42)
        let q = MLXRandom.normal([batch, tokens, kHeads, kDim], scale: 0.5).asType(dtype)
        let k = MLXRandom.normal([batch, tokens, kHeads, kDim], scale: 0.5).asType(dtype)
        let v = MLXRandom.normal([batch, tokens, vHeads, vDim], scale: 0.5).asType(dtype)
        // Decay gate in (0, 1) and mixing rate in (0, 1), like the model produces.
        let g = sigmoid(MLXRandom.normal([batch, tokens, vHeads])).asType(dtype)
        let beta = sigmoid(MLXRandom.normal([batch, tokens, vHeads])).asType(dtype)
        let state = MLXRandom.normal([batch, vHeads, vDim, kDim], scale: 0.1).asType(dtype)
        return Inputs(q: q, k: k, v: v, g: g, beta: beta, state: state)
    }

    private func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
    }

    private func assertKernelMatchesOps(
        tokens: Int, dtype: DType, mask: MLXArray?, tolerance: Float,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let inputs = makeInputs(
            batch: 1, tokens: tokens, kHeads: 16, vHeads: 32,
            kDim: 128, vDim: 128, dtype: dtype)

        var (referenceY, referenceState) = gatedDeltaOps(
            q: inputs.q, k: inputs.k, v: inputs.v,
            g: inputs.g, beta: inputs.beta, state: inputs.state, mask: mask)
        var (kernelY, kernelState) = gatedDeltaKernelApply(
            q: inputs.q, k: inputs.k, v: inputs.v,
            g: inputs.g, beta: inputs.beta, state: inputs.state, mask: mask)
        if let mask {
            // Masked positions are padding: the kernel zeroes their outputs
            // while the ops loop leaves computed-but-unused values, so only
            // compare the positions that are actually attended.
            let expanded = mask[.ellipsis, .newAxis, .newAxis]
            referenceY = MLX.where(expanded, referenceY, MLXArray(0).asType(referenceY.dtype))
            kernelY = MLX.where(expanded, kernelY, MLXArray(0).asType(kernelY.dtype))
        }
        eval(referenceY, referenceState, kernelY, kernelState)

        XCTAssertEqual(kernelY.shape, referenceY.shape, file: file, line: line)
        XCTAssertEqual(kernelState.shape, referenceState.shape, file: file, line: line)
        XCTAssertLessThan(
            maxAbsDifference(kernelY, referenceY), tolerance,
            "output mismatch (tokens=\(tokens) dtype=\(dtype))", file: file, line: line)
        XCTAssertLessThan(
            maxAbsDifference(kernelState, referenceState), tolerance,
            "state mismatch (tokens=\(tokens) dtype=\(dtype))", file: file, line: line)
    }

    func testMatchesOpsFloat32Prefill() {
        assertKernelMatchesOps(tokens: 193, dtype: .float32, mask: nil, tolerance: 1e-3)
    }

    func testMatchesOpsFloat32Decode() {
        assertKernelMatchesOps(tokens: 1, dtype: .float32, mask: nil, tolerance: 1e-4)
    }

    func testBFloat16KernelTracksFloat32Reference() {
        // The kernel accumulates in float32 while the ops loop accumulates in
        // bfloat16, so the two bfloat16 implementations legitimately diverge as
        // rounding compounds through the recurrence. The correctness bar is
        // that the kernel is at least as close to the float32 ground truth as
        // the ops loop it replaces.
        let inputs = makeInputs(
            batch: 1, tokens: 64, kHeads: 16, vHeads: 32,
            kDim: 128, vDim: 128, dtype: .bfloat16)

        let (referenceY, referenceState) = gatedDeltaOps(
            q: inputs.q.asType(.float32), k: inputs.k.asType(.float32),
            v: inputs.v.asType(.float32), g: inputs.g.asType(.float32),
            beta: inputs.beta.asType(.float32), state: inputs.state.asType(.float32),
            mask: nil)
        let (opsY, opsState) = gatedDeltaOps(
            q: inputs.q, k: inputs.k, v: inputs.v,
            g: inputs.g, beta: inputs.beta, state: inputs.state, mask: nil)
        let (kernelY, kernelState) = gatedDeltaKernelApply(
            q: inputs.q, k: inputs.k, v: inputs.v,
            g: inputs.g, beta: inputs.beta, state: inputs.state, mask: nil)
        eval(referenceY, referenceState, opsY, opsState, kernelY, kernelState)

        let opsError = maxAbsDifference(opsY, referenceY)
        let kernelError = maxAbsDifference(kernelY, referenceY)
        let opsStateError = maxAbsDifference(opsState, referenceState)
        let kernelStateError = maxAbsDifference(kernelState, referenceState)

        XCTAssertLessThanOrEqual(
            kernelError, max(opsError, 1e-2) * 1.5,
            "kernel output drifts further from float32 truth than the ops loop (kernel=\(kernelError) ops=\(opsError))"
        )
        XCTAssertLessThanOrEqual(
            kernelStateError, max(opsStateError, 1e-2) * 1.5,
            "kernel state drifts further from float32 truth than the ops loop (kernel=\(kernelStateError) ops=\(opsStateError))"
        )
    }

    func testMatchesOpsFloat32Masked() {
        let tokens = 48
        var maskValues = [Bool]()
        for index in 0 ..< tokens {
            maskValues.append(index % 3 != 0)
        }
        let mask = MLXArray(maskValues).reshaped([1, tokens])
        assertKernelMatchesOps(tokens: tokens, dtype: .float32, mask: mask, tolerance: 1e-3)
    }
}
