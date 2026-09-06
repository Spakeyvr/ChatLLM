// Serializes Foundation Models sessions. Cancelled waiters leave the queue
// immediately; only the owner may release an acquired permit.
import Foundation

actor FoundationModelsGate {
    private var isHeld = false
    private var waiters: [(id: UUID, continuation: CheckedContinuation<Void, Error>)] = []

    func acquire() async throws {
        try Task.checkCancellation()
        if !isHeld {
            isHeld = true
            return
        }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append((id, continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        waiters.remove(at: index).continuation.resume(throwing: CancellationError())
    }

    func release() {
        if !waiters.isEmpty {
            waiters.removeFirst().continuation.resume()
        } else {
            isHeld = false
        }
    }
}
