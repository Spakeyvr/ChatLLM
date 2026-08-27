import Foundation
import Observation

enum TavilyKeyValidationStatus: Equatable {
    case idle
    case checking
    case valid
    case invalid(String)
}

@MainActor
@Observable
final class TavilyKeyValidationModel {
    typealias ValidateKey = @Sendable (String) async throws -> Void

    private(set) var status: TavilyKeyValidationStatus = .idle

    @ObservationIgnored private let validateKey: ValidateKey
    @ObservationIgnored private var validationTask: Task<Void, Never>?

    init(validateKey: @escaping ValidateKey = TavilyKeyValidationModel.validateWithTavily) {
        self.validateKey = validateKey
    }

    func validate(_ key: String) {
        let trimmedKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else {
            reset()
            return
        }

        validationTask?.cancel()
        status = .checking
        let validateKey = self.validateKey

        validationTask = Task { [weak self] in
            do {
                try await validateKey(trimmedKey)
                try Task.checkCancellation()
                self?.status = .valid
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled else { return }
                self?.status = .invalid(error.localizedDescription)
            }
        }
    }

    func reset() {
        validationTask?.cancel()
        validationTask = nil
        status = .idle
    }

    deinit {
        validationTask?.cancel()
    }

    private static func validateWithTavily(_ key: String) async throws {
        let service = try TavilySearchService(apiKey: key)
        try await service.validateAPIKey()
    }
}
