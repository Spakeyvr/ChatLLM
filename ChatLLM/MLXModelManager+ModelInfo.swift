//
//  MLXModelManager+ModelInfo
//  ChatLLM
//
//  Split out of MLXModelManager.swift as part of the Type+Concern organization.
//

import Foundation

extension MLXModelManager {
    // MARK: - Model Info

    struct MLXModelInfo: Identifiable, Sendable {
        enum LoadPolicy: Equatable, Sendable {
            case standard
            case qwenMultimodal

            var packageDescription: String {
                switch self {
                case .standard:
                    return "Standard package"
                case .qwenMultimodal:
                    return "Full multimodal package"
                }
            }

            var architectureHint: String? {
                switch self {
                case .standard:
                    return nil
                case .qwenMultimodal:
                    return "Qwen3_5ForConditionalGeneration"
                }
            }

            var defersAutomaticPrewarm: Bool {
                switch self {
                case .standard:
                    return false
                case .qwenMultimodal:
                    return true
                }
            }
        }

        let id: String
        let name: String
        let localDirName: String
        let hfRepoId: String
        let parameters: String
        let downloadSizeLabel: String
        let loadPolicy: LoadPolicy
        let description: String
        let contextLength: Int
        var isAvailable: Bool
        let supportsReasoning: Bool
        let supportsNativeImages: Bool
        let requiredProcessorClass: String?
        let minimumPhoneMemoryBytes: UInt64?
        let minimumPhoneMemoryForToolCallsBytes: UInt64?
        /// Per-device-tier context window override. Keys are minimum RAM in GiB; the
        /// highest matching key wins. Falls through to the device default if no key matches.
        let phoneContextWindowOverride: [Int: Int]?

        init(
            id: String,
            name: String,
            localDirName: String,
            hfRepoId: String,
            parameters: String,
            downloadSizeLabel: String,
            loadPolicy: LoadPolicy = .standard,
            description: String,
            contextLength: Int,
            isAvailable: Bool,
            supportsReasoning: Bool,
            supportsNativeImages: Bool = false,
            requiredProcessorClass: String? = nil,
            minimumPhoneMemoryBytes: UInt64? = nil,
            minimumPhoneMemoryForToolCallsBytes: UInt64? = nil,
            phoneContextWindowOverride: [Int: Int]? = nil
        ) {
            self.id = id
            self.name = name
            self.localDirName = localDirName
            self.hfRepoId = hfRepoId
            self.parameters = parameters
            self.downloadSizeLabel = downloadSizeLabel
            self.loadPolicy = loadPolicy
            self.description = description
            self.contextLength = contextLength
            self.isAvailable = isAvailable
            self.supportsReasoning = supportsReasoning
            self.supportsNativeImages = supportsNativeImages
            self.requiredProcessorClass = requiredProcessorClass
            self.minimumPhoneMemoryBytes = minimumPhoneMemoryBytes
            self.minimumPhoneMemoryForToolCallsBytes = minimumPhoneMemoryForToolCallsBytes
            self.phoneContextWindowOverride = phoneContextWindowOverride
        }

        var displayName: String { "\(name) (\(parameters))" }

        var parameterCount: String {
            parameters.components(separatedBy: " ").first ?? parameters
        }

        var promptIdentity: String {
            let trimmedParameterCount = parameterCount.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedParameterCount.isEmpty else {
                return name
            }
            return "\(name) \(trimmedParameterCount)"
        }
    }
}
