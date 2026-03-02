//
//  FoundationModelsGate.swift
//  On-Device_LLM_Chat
//
//  Replaces the Bool + sleep-loop pattern in ChatViewModel with a proper
//  continuation-based queue that suspends callers without polling.
//

/// Serializes access to Foundation Models so only one session is active at a time.
/// Callers that arrive while the gate is held are suspended (not spin-waited)
/// and resumed in FIFO order when the gate is released.
actor FoundationModelsGate {
    private var isHeld = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if !isHeld {
            isHeld = true
        } else {
            await withCheckedContinuation { continuation in
                waiters.append(continuation)
            }
        }
    }

    func release() {
        if let next = waiters.first {
            waiters.removeFirst()
            next.resume()
        } else {
            isHeld = false
        }
    }
}
