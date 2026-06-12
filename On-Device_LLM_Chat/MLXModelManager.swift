//
//  MLXModelManager.swift
//  On-Device_LLM_Chat
//
//  Manages MLX-based language models downloaded to Documents/Models/.
//

import Foundation
import Combine
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM
import Tokenizers
import OSLog
import UIKit

nonisolated internal enum MLXCachePolicy: Equatable, Sendable {
    case persistentSimple
    case persistentQuantizedSimple
    case boundedRotating(maxKVSize: Int)

    var diagnosticLabel: String {
        switch self {
        case .persistentSimple:
            return "persistent-simple"
        case .persistentQuantizedSimple:
            return "persistent-quantized-simple"
        case .boundedRotating:
            return "bounded-rotating"
        }
    }

    var maxKVSize: Int? {
        switch self {
        case .persistentSimple, .persistentQuantizedSimple:
            return nil
        case .boundedRotating(let maxKVSize):
            return maxKVSize
        }
    }

    var usesDynamicKVQuantization: Bool {
        switch self {
        case .persistentQuantizedSimple:
            return true
        case .persistentSimple, .boundedRotating:
            return false
        }
    }
}

nonisolated internal struct MLXKVBenchmarkMetadata: Sendable, Equatable {
    let cachePolicy: MLXCachePolicy
    let effectiveMaxKVSize: Int?
    let prefillStepSize: Int
    let quantizedKVStart: Int?
}

nonisolated internal struct MLXPerformanceSample: Sendable {
    let conversationID: UUID
    let modelID: String
    let promptTokenCount: Int
    let outputTokenCount: Int
    let toolInvocationCount: Int
    let timeToFirstToken: TimeInterval?
    let totalLatency: TimeInterval
    let promptTokensPerSecond: Double?
    let decodeTokensPerSecond: Double?
    let stopReason: GenerateStopReason?
    let memoryBefore: Memory.Snapshot
    let memoryAfter: Memory.Snapshot
    let peakActiveBytes: Int
    let kvBenchmarkMetadata: MLXKVBenchmarkMetadata
}

// MARK: - MLXModelManager

@MainActor
final class MLXModelManager: ObservableObject {
    private let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM", category: "MLXModelManager")

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

            var simulatorUnsupportedReason: String? {
                switch self {
                case .standard:
                    return nil
                case .qwenMultimodal:
                    return "Qwen multimodal MLX loading is unavailable on the simulator because the upstream MLX/Metal VLM initialization path crashes during load."
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
    }

    // MARK: - Published Properties

    @Published var availableModels: [MLXModelInfo] = []
    @Published var currentModel: MLXModelInfo?
    @Published var isLoading: Bool = false
    @Published var loadError: String?
    @Published private(set) var pendingModelToLoad: MLXModelInfo?

    @Published var downloadProgress: Double = 0
    @Published var isDownloading: Bool = false
    @Published private(set) var activeDownloadModelID: String?
    @Published var downloadError: String?
    @Published private(set) var downloadErrorModelID: String?
    @Published private(set) var latestPerformanceSample: MLXPerformanceSample?

    var supportsNativeThinking: Bool { true }

    // MARK: - Private

    private var container: ModelContainer?
    private var loadTask: Task<Void, Never>?
    private let downloader = ModelDownloader()
    private var downloaderTask: Task<Void, Never>?
    private var memoryWarningCancellable: AnyCancellable?
    private var appWillResignActiveCancellable: AnyCancellable?
    private var appDidEnterBackgroundCancellable: AnyCancellable?
    private var compatibilityErrors: [String: String] = [:]
    private var toolTemplateInspectionCache: [String: ToolTemplateInspection] = [:]
    private var packageMetadataCache: [String: ModelPackageMetadata] = [:]
    private var promptTokenCountCache: [PromptTokenCountCacheKey: Int] = [:]
    private var activeLoadID: UUID?
    private var deferredPrewarmModelID: String?
    private var deferredTuningModelID: String?
    private var prewarmInFlightModelID: String?
    private var tuningTask: Task<Void, Never>?
    private var hasSeenMemoryWarningSinceCurrentLoad = false
    private var memoryMaintenanceTimer: Timer?
    private let deviceSupportProfile: MLXDeviceSupportProfile
    private let inferenceWorker: MLXInferenceWorker
    private var simulatedDownloadedModelIDs: Set<String> = []

    nonisolated private static let aggressiveMemoryCacheLimitBytes = 1 * 1024 * 1024
    nonisolated static let defaultPrefillStepSize = 512
    nonisolated static let minimumAdaptivePrefillStepSize = 128

    /// Scales the tuned prefill chunk size down when the actual prompt is small,
    /// reducing peak memory without changing generation quality. Never exceeds
    /// the tuned value.
    nonisolated static func adaptivePrefillStepSize(
        tuned: Int,
        messages: [Chat.Message]
    ) -> Int {
        let approxTokens = messages.reduce(0) { partial, message in
            partial + max(1, message.content.count / 4)
        }
        if approxTokens <= 0 {
            return max(minimumAdaptivePrefillStepSize, min(tuned, 256))
        }
        var bucket = minimumAdaptivePrefillStepSize
        while bucket < approxTokens && bucket < tuned {
            bucket *= 2
        }
        return max(minimumAdaptivePrefillStepSize, min(tuned, bucket))
    }
    nonisolated static let defaultQuantizedKVStartStep = 256
    nonisolated private static let memoryMaintenanceInterval: TimeInterval = 3
    nonisolated private static let tuningStartupDelayNanoseconds: UInt64 = 1_000_000_000
    nonisolated private static let kvQuantizationBits = 8
    nonisolated private static let kvQuantizationGroupSize = 64
    nonisolated private static let turboQuantKeyTotalBits = 3
    nonisolated private static let turboQuantValueBits = 2
    nonisolated private static let turboQuantSeed: UInt64 = 42
    nonisolated private static let turboQuantExactBufferSize = 128
    nonisolated private static let turboQuantAttentionBlockTokens = 256
    nonisolated private static let turboQuantMediaExactBufferSize = 32
    nonisolated private static let turboQuantMediaAttentionBlockTokens = 64
    nonisolated private static let memoryConstrainedMaxKVSize = 4096
    private static let uiTestFakeDownloadsArgument = "-ui-test-fake-mlx-downloads"

    private static let modelDefinitions: [MLXModelInfo] = [
        MLXModelInfo(
            id: "qwen3.5-4b-mixed36",
            name: "Qwen 3.5",
            localDirName: "Qwen3.5-4B-MLX-mixed36",
            hfRepoId: "Spakie/Qwen3.5-4B-MLX-mixed36",
            parameters: "4B (3/6-bit mixed)",
            downloadSizeLabel: "2.50 GB",
            loadPolicy: .qwenMultimodal,
            description: "Qwen 3.5 4B multimodal model with native reasoning and vision.",
            contextLength: 262144,
            isAvailable: false,
            supportsReasoning: true,
            supportsNativeImages: true,
            requiredProcessorClass: "Qwen3VLProcessor",
            minimumPhoneMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte,
            minimumPhoneMemoryForToolCallsBytes: 12 * MLXDeviceSupportProfile.gibibyte
        ),
        MLXModelInfo(
            id: "qwen3.5-2b-4bit",
            name: "Qwen 3.5",
            localDirName: "Qwen3.5-2B-MLX-4bit",
            hfRepoId: "mlx-community/Qwen3.5-2B-4bit",
            parameters: "2B (4-bit)",
            downloadSizeLabel: "1.75 GB",
            loadPolicy: .qwenMultimodal,
            description: "Qwen 3.5 2B multimodal model with native reasoning and vision.",
            contextLength: 262144,
            isAvailable: false,
            supportsReasoning: true,
            supportsNativeImages: true,
            requiredProcessorClass: "Qwen3VLProcessor",
            minimumPhoneMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte,
            minimumPhoneMemoryForToolCallsBytes: 8 * MLXDeviceSupportProfile.gibibyte
        ),
        MLXModelInfo(
            id: "qwen3.5-0.8b-4bit",
            name: "Qwen 3.5",
            localDirName: "Qwen3.5-0.8B-MLX-4bit",
            hfRepoId: "mlx-community/Qwen3.5-0.8B-4bit",
            parameters: "0.8B (4-bit)",
            downloadSizeLabel: "625 MB",
            loadPolicy: .qwenMultimodal,
            description: "Qwen 3.5 0.8B multimodal model with native reasoning and vision.",
            contextLength: 262144,
            isAvailable: false,
            supportsReasoning: true,
            supportsNativeImages: true,
            requiredProcessorClass: "Qwen3VLProcessor",
            minimumPhoneMemoryBytes: 4 * MLXDeviceSupportProfile.gibibyte,
            minimumPhoneMemoryForToolCallsBytes: 6 * MLXDeviceSupportProfile.gibibyte,
            phoneContextWindowOverride: [8: 4_096, 12: 6_144]
        )
    ]

    private let documentsDirectory: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first ?? FileManager.default.temporaryDirectory

    private struct ModelInstallationStatus {
        let isInstalled: Bool
        let isCompatible: Bool
        let compatibilityError: String?
    }

    private struct ModelPackageMetadata {
        let architecture: String?
        let processorClass: String?
        let modelType: String?
    }

    private struct PromptTokenCountCacheKey: Hashable {
        let modelID: String
        let fingerprint: Int
    }

    struct MLXGenerationResult: Sendable {
        let toolInvocationCount: Int
    }

    struct KVQuantizationConfiguration: Equatable, Sendable {
        let bits: Int
        let groupSize: Int
        let startStep: Int
    }

    enum CacheCompressionMode: Equatable, Sendable {
        case none
        case legacyQuantized(KVQuantizationConfiguration)
        case turboQuant(TurboQuantConfiguration)

        nonisolated var diagnosticLabel: String {
            switch self {
            case .none:
                return "none"
            case .legacyQuantized:
                return "legacy-quantized"
            case .turboQuant(let configuration):
                return "turboquant(k\(configuration.keyTotalBits),v\(configuration.valueBits),exact\(configuration.exactBufferSize),block\(configuration.attentionBlockTokens))"
            }
        }

        nonisolated var usesLegacyQuantization: Bool {
            if case .legacyQuantized = self {
                return true
            }
            return false
        }

        nonisolated func applying(startStep: Int) -> Self {
            switch self {
            case .none:
                return .none
            case .legacyQuantized(let configuration):
                return .legacyQuantized(
                    KVQuantizationConfiguration(
                        bits: configuration.bits,
                        groupSize: configuration.groupSize,
                        startStep: startStep
                    )
                )
            case .turboQuant:
                return self
            }
        }

        nonisolated var generateParametersCompression: KVCacheCompressionMode? {
            switch self {
            case .none:
                return nil
            case .legacyQuantized(let configuration):
                return .quantized(
                    bits: configuration.bits,
                    groupSize: configuration.groupSize,
                    startStep: configuration.startStep
                )
            case .turboQuant(let configuration):
                return .turboQuant(configuration)
            }
        }
    }

    struct GenerationConfiguration: Equatable, Sendable {
        let maxTokens: Int?
        let cachePolicy: MLXCachePolicy
        let cacheCompression: CacheCompressionMode

        var maxKVSize: Int? {
            cachePolicy.maxKVSize
        }

        var usesCompressedPersistentCache: Bool {
            cachePolicy == .persistentQuantizedSimple && cacheCompression != .none
        }

        var kvQuantization: KVQuantizationConfiguration? {
            if case .legacyQuantized(let configuration) = cacheCompression {
                return configuration
            }
            return nil
        }

        var usesQuantizedToolCacheStrategy: Bool {
            usesCompressedPersistentCache
        }
    }

    private enum ToolTemplateSupport: Equatable {
        case supported
        case unsupported(String)
    }

    private struct ToolTemplateInspection {
        let support: ToolTemplateSupport
        let preferredToolCallFormat: ToolCallFormat?
        let usesWrappedXMLToolCalls: Bool
    }

    // MARK: - Model Directory (Documents)

    func modelDirectoryURL(for info: MLXModelInfo) -> URL? {
        let dir = documentsDirectory.appendingPathComponent("Models/\(info.localDirName)")
        let config = dir.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: config.path) ? dir : nil
    }

    private func installationStatus(for info: MLXModelInfo) -> ModelInstallationStatus {
        if Self.isUITestFakeDownloadsEnabled {
            if simulatedDownloadedModelIDs.contains(info.id) {
                return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
            }
            return ModelInstallationStatus(isInstalled: false, isCompatible: true, compatibilityError: nil)
        }

        let dir = documentsDirectory.appendingPathComponent("Models/\(info.localDirName)")
        let fm = FileManager.default
        let hasConfig = fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        let contents = hasConfig ? ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? []) : []
        let hasWeights = contents.contains(where: { $0.hasSuffix(".safetensors") })

        if let deviceIssue = deviceSupportProfile.availabilityIssue(for: info) {
            return ModelInstallationStatus(
                isInstalled: hasConfig && hasWeights,
                isCompatible: false,
                compatibilityError: deviceIssue
            )
        }

        guard hasConfig else {
            #if targetEnvironment(simulator)
            if let simulatorUnsupportedReason = info.loadPolicy.simulatorUnsupportedReason {
                return ModelInstallationStatus(
                    isInstalled: false,
                    isCompatible: false,
                    compatibilityError: simulatorUnsupportedReason
                )
            }
            #endif
            return ModelInstallationStatus(isInstalled: false, isCompatible: false, compatibilityError: nil)
        }
        guard hasWeights else {
            return ModelInstallationStatus(isInstalled: false, isCompatible: false, compatibilityError: nil)
        }

        #if targetEnvironment(simulator)
        if let simulatorUnsupportedReason = info.loadPolicy.simulatorUnsupportedReason {
            return ModelInstallationStatus(
                isInstalled: true,
                isCompatible: false,
                compatibilityError: simulatorUnsupportedReason
            )
        }
        #endif

        guard info.supportsNativeImages else {
            return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
        }

        guard let expectedProcessor = info.requiredProcessorClass else {
            return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
        }

        let detectedProcessorClass = packageMetadata(for: info)?.processorClass

        guard let detectedProcessorClass else {
            return ModelInstallationStatus(
                isInstalled: true,
                isCompatible: false,
                compatibilityError:
                    "Installed model '\(info.localDirName)' is missing processor metadata for native image support. Re-download \(info.localDirName) to enable native images."
            )
        }

        guard detectedProcessorClass == expectedProcessor else {
            return ModelInstallationStatus(
                isInstalled: true,
                isCompatible: false,
                compatibilityError:
                    "Installed model '\(info.localDirName)' is incompatible with native image support (processor '\(detectedProcessorClass)'). Re-download \(info.localDirName) to get the correct multimodal files."
            )
        }

        return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
    }

    private func isModelAvailable(_ info: MLXModelInfo) -> Bool {
        let status = installationStatus(for: info)
        return status.isInstalled && status.isCompatible
    }

    private func compatibilityError(for info: MLXModelInfo) -> String? {
        compatibilityErrors[info.id]
    }

    private static var isUITestFakeDownloadsEnabled: Bool {
        ProcessInfo.processInfo.arguments.contains(uiTestFakeDownloadsArgument)
    }

    func availabilityIssue(for info: MLXModelInfo) -> String? {
        compatibilityError(for: info)
    }

    func supportsToolCalls(for info: MLXModelInfo) -> Bool {
        deviceSupportProfile.supportsToolCalls(for: info)
    }

    func toolCallIssue(for info: MLXModelInfo) -> String? {
        deviceSupportProfile.toolCallIssue(for: info)
    }

    func packageArchitecture(for info: MLXModelInfo) -> String? {
        packageMetadata(for: info)?.architecture
    }

    // MARK: - Init

    init(deviceSupportProfile: MLXDeviceSupportProfile? = nil) {
        self.deviceSupportProfile = deviceSupportProfile ?? MLXDeviceSupportProfile.current
        self.inferenceWorker = MLXInferenceWorker(
            subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM",
            deviceSupportProfile: self.deviceSupportProfile
        )
        availableModels = Self.modelDefinitions.map { definition in
            var model = definition
            let status = installationStatus(for: model)
            model.isAvailable = status.isInstalled && status.isCompatible
            if let error = status.compatibilityError {
                compatibilityErrors[model.id] = error
            }
            return model
        }

        memoryWarningCancellable = NotificationCenter.default.publisher(
            for: UIApplication.didReceiveMemoryWarningNotification
        )
        .sink { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleMemoryWarning()
            }
        }

        appWillResignActiveCancellable = NotificationCenter.default.publisher(
            for: UIApplication.willResignActiveNotification
        )
        .sink { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleApplicationInactivity(reason: "application.will_resign_active")
            }
        }

        appDidEnterBackgroundCancellable = NotificationCenter.default.publisher(
            for: UIApplication.didEnterBackgroundNotification
        )
        .sink { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleApplicationInactivity(reason: "application.did_enter_background")
            }
        }
    }

    deinit {
        memoryWarningCancellable?.cancel()
        appWillResignActiveCancellable?.cancel()
        appDidEnterBackgroundCancellable?.cancel()
    }

    func invalidateConversationSession(_ conversationID: UUID, reason: String) {
        Task {
            await inferenceWorker.invalidateConversation(conversationID, reason: reason)
        }
    }

    nonisolated internal static func promptTokenCountFingerprint(
        role: Chat.Message.Role,
        content: String
    ) -> Int {
        var hasher = Hasher()
        hasher.combine(role.rawValue)
        hasher.combine(content)
        return hasher.finalize()
    }

    func tokenCountForLoadedModelText(
        role: Chat.Message.Role,
        content: String
    ) async -> Int? {
        guard let currentModel, let container else { return nil }
        let cacheKey = PromptTokenCountCacheKey(
            modelID: currentModel.id,
            fingerprint: Self.promptTokenCountFingerprint(role: role, content: content)
        )
        if let cachedCount = promptTokenCountCache[cacheKey] {
            return cachedCount
        }
        let tokenCount = await container.encode(content).count
        promptTokenCountCache[cacheKey] = tokenCount
        return tokenCount
    }

    nonisolated internal static func shouldPersistSessionAcrossTurns(
        enableThinking: Bool,
        hasTools: Bool,
        hasMedia: Bool
    ) -> Bool {
        _ = enableThinking
        _ = hasTools
        _ = hasMedia
        return true
    }

    nonisolated private static func prefillTuningUserDefaultsKey(
        for modelID: String,
        deviceSupportProfile: MLXDeviceSupportProfile
    ) -> String {
        let deviceClass = deviceSupportProfile.isPhone ? "phone" : "other"
        let bytesPerGiB: UInt64 = 1_073_741_824
        let memoryTier = Int(deviceSupportProfile.physicalMemoryBytes / bytesPerGiB)
        return "mlxPrefillStepSize.\(modelID).\(deviceClass).\(memoryTier)"
    }

    nonisolated private static func persistedPrefillStepSize(
        for modelID: String,
        deviceSupportProfile: MLXDeviceSupportProfile
    ) -> Int? {
        let key = prefillTuningUserDefaultsKey(for: modelID, deviceSupportProfile: deviceSupportProfile)
        guard UserDefaults.standard.object(forKey: key) != nil else { return nil }
        let storedValue = UserDefaults.standard.integer(forKey: key)
        guard [256, 512, 1024].contains(storedValue) else { return nil }
        return storedValue
    }

    nonisolated private static func storePersistedPrefillStepSize(
        _ prefillStepSize: Int,
        for modelID: String,
        deviceSupportProfile: MLXDeviceSupportProfile
    ) {
        let key = prefillTuningUserDefaultsKey(for: modelID, deviceSupportProfile: deviceSupportProfile)
        UserDefaults.standard.set(prefillStepSize, forKey: key)
    }

    nonisolated private static func recommendedWiredMemoryCapBytes(
        deviceSupportProfile: MLXDeviceSupportProfile
    ) -> Int? {
        GPU.maxRecommendedWorkingSetBytes() ?? Int(deviceSupportProfile.physicalMemoryBytes)
    }

    nonisolated private static func estimateWeightBytes(for container: ModelContainer) async -> Int {
        await container.perform { context in
            context.model.parameters().flattened().reduce(0) { partialResult, parameter in
                partialResult + parameter.1.nbytes
            }
        }
    }

    nonisolated private static func startReservationTicketIfNeeded(
        weightBytes: Int,
        policy: MLXLMCommon.WiredSumPolicy
    ) async -> MLX.WiredMemoryTicket? {
        guard weightBytes > 0 else { return nil }
        let ticket = MLX.WiredMemoryTicket(size: weightBytes, policy: policy, kind: .reservation)
        _ = await ticket.start()
        return ticket
    }

    nonisolated internal struct TuningBenchmarkResult: Sendable {
        let candidate: Int
        let promptTokensPerSecond: Double
        let decodeTokensPerSecond: Double
        let totalLatency: TimeInterval
        let totalBytes: Int
        let measurement: MLXLMCommon.WiredMemoryMeasurement
    }

    nonisolated private static func quantizedKVStartUserDefaultsKey(
        for modelID: String,
        deviceSupportProfile: MLXDeviceSupportProfile
    ) -> String {
        let deviceClass = deviceSupportProfile.isPhone ? "phone" : "other"
        let bytesPerGiB: UInt64 = 1_073_741_824
        let memoryTier = Int(deviceSupportProfile.physicalMemoryBytes / bytesPerGiB)
        return "mlxQuantizedKVStart.\(modelID).\(deviceClass).\(memoryTier)"
    }

    nonisolated private static func persistedQuantizedKVStart(
        for modelID: String,
        deviceSupportProfile: MLXDeviceSupportProfile
    ) -> Int? {
        let key = quantizedKVStartUserDefaultsKey(for: modelID, deviceSupportProfile: deviceSupportProfile)
        guard UserDefaults.standard.object(forKey: key) != nil else { return nil }
        let storedValue = UserDefaults.standard.integer(forKey: key)
        guard [128, 256, 512].contains(storedValue) else { return nil }
        return storedValue
    }

    nonisolated private static func storePersistedQuantizedKVStart(
        _ quantizedKVStart: Int,
        for modelID: String,
        deviceSupportProfile: MLXDeviceSupportProfile
    ) {
        let key = quantizedKVStartUserDefaultsKey(for: modelID, deviceSupportProfile: deviceSupportProfile)
        UserDefaults.standard.set(quantizedKVStart, forKey: key)
    }

    nonisolated internal static func selectFastestSafeCandidate(
        _ candidates: [TuningBenchmarkResult],
        wiredMemoryCap: Int?,
        preferLargerCandidateOnTie: Bool
    ) -> TuningBenchmarkResult {
        let safeCandidates = candidates.filter { candidate in
            guard let wiredMemoryCap else { return true }
            return candidate.totalBytes <= Int(Double(wiredMemoryCap) * 0.85)
        }
        let performanceCandidates = safeCandidates.isEmpty ? candidates : safeCandidates
        let bestDecodeSpeed = performanceCandidates.map(\.decodeTokensPerSecond).max() ?? 0
        let decodeFloor = bestDecodeSpeed > 0 ? bestDecodeSpeed * 0.95 : 0
        let decodeSafeCandidates = performanceCandidates.filter { candidate in
            bestDecodeSpeed == 0 || candidate.decodeTokensPerSecond >= decodeFloor
        }
        let rankedCandidates = decodeSafeCandidates.isEmpty ? performanceCandidates : decodeSafeCandidates

        return rankedCandidates.max { lhs, rhs in
            if lhs.promptTokensPerSecond != rhs.promptTokensPerSecond {
                return lhs.promptTokensPerSecond < rhs.promptTokensPerSecond
            }
            if lhs.totalLatency != rhs.totalLatency {
                return lhs.totalLatency > rhs.totalLatency
            }
            if lhs.decodeTokensPerSecond != rhs.decodeTokensPerSecond {
                return lhs.decodeTokensPerSecond < rhs.decodeTokensPerSecond
            }
            return preferLargerCandidateOnTie
                ? lhs.candidate < rhs.candidate
                : lhs.candidate > rhs.candidate
        } ?? candidates[0]
    }

    private func makeBenchmarkUserInput(
        container: ModelContainer,
        targetTokenCount: Int = 2_048
    ) async -> UserInput {
        var prompt = " hello"
        var tokenCount = await container.encode(prompt).count
        while tokenCount < targetTokenCount {
            prompt += prompt
            tokenCount = await container.encode(prompt).count
        }
        return UserInput(
            chat: [.user(prompt)],
            processing: UserInput.Processing()
        )
    }

    private func benchmarkGeneration(
        container: ModelContainer,
        userInput: UserInput,
        parameters: GenerateParameters
    ) async throws -> (promptTokensPerSecond: Double, decodeTokensPerSecond: Double, totalLatency: TimeInterval) {
        try Task.checkCancellation()
        let prepared = try await container.prepare(input: userInput)
        let startedAt = Date()
        let stream = try await container.generate(input: prepared, parameters: parameters)
        var completionInfo: GenerateCompletionInfo?
        for await generation in stream {
            try Task.checkCancellation()
            if case .info(let info) = generation {
                completionInfo = info
            }
        }
        let totalLatency = Date().timeIntervalSince(startedAt)
        guard let completionInfo else {
            throw GenerationError.invalidChatHistory
        }
        return (
            promptTokensPerSecond: completionInfo.promptTokensPerSecond,
            decodeTokensPerSecond: completionInfo.tokensPerSecond,
            totalLatency: totalLatency
        )
    }

    private func measureTuningCandidate(
        container: ModelContainer,
        userInput: UserInput,
        parameters: GenerateParameters,
        candidate: Int
    ) async throws -> TuningBenchmarkResult {
        let performanceResult = try await benchmarkGeneration(
            container: container,
            userInput: userInput,
            parameters: parameters
        )
        try Task.checkCancellation()
        aggressivelyFreeMemory(reason: "tuning.candidate.between")
        let memoryMeasurement: MLXLMCommon.WiredMemoryMeasurement = try await container.perform { context in
            try await WiredMemoryUtils.tune(
                userInput: userInput,
                context: context,
                parameters: parameters
            )
        }

        return TuningBenchmarkResult(
            candidate: candidate,
            promptTokensPerSecond: performanceResult.promptTokensPerSecond,
            decodeTokensPerSecond: performanceResult.decodeTokensPerSecond,
            totalLatency: performanceResult.totalLatency,
            totalBytes: memoryMeasurement.totalBytes,
            measurement: memoryMeasurement
        )
    }

    private var prefersBoundedKVCacheAcrossTurns: Bool {
        deviceSupportProfile.hasLowMemoryForPersistentKVCache
    }

    private func configuredContextWindowLimit(for model: MLXModelInfo? = nil) -> Int {
        UserDefaults.standard.mlxContextWindowTokens(
            deviceMaximum: deviceSupportProfile.maxContextWindowTokens(for: model ?? currentModel)
        )
    }

    private func supportsTurboQuant(for model: MLXModelInfo?) -> Bool {
        guard let model else { return true }
        return model.id.hasPrefix("qwen3.5-")
    }

    private func shouldPreferTurboQuant(for model: MLXModelInfo?) -> Bool {
        UserDefaults.standard.mlxEnableTurboQuant && supportsTurboQuant(for: model)
    }

    private func defaultGenerationConfigurationForCurrentDevice(
        model: MLXModelInfo? = nil
    ) -> GenerationConfiguration {
        Self.generationConfiguration(
            isEnabled: UserDefaults.standard.mlxEnableTurboQuant,
            preferTurboQuant: shouldPreferTurboQuant(for: model),
            hasTools: false,
            hasMedia: false,
            memoryConstrained: false,
            prefersBoundedCache: prefersBoundedKVCacheAcrossTurns,
            configuredMaxOutputTokens: UserDefaults.standard.mlxMaxOutputTokensLimit,
            configuredContextWindow: configuredContextWindowLimit(for: model)
        )
    }

    private func schedulePrefillTuningIfNeeded(
        model: MLXModelInfo,
        initialPrefillStepSize: Int
    ) {
        guard tuningNeedsRetry(for: model) else {
            deferredTuningModelID = nil
            return
        }

        let shouldTuneQuantizedKV = defaultGenerationConfigurationForCurrentDevice(model: model).cacheCompression.usesLegacyQuantization
        deferredTuningModelID = model.id
        logger.notice(
            "MLX tuning deferred until after first successful generation: id=\(model.id, privacy: .public) prefill=\(initialPrefillStepSize, privacy: .public) kv_quantization=\(shouldTuneQuantizedKV, privacy: .public)"
        )
    }

    private func startDeferredTuningIfNeeded(
        container: ModelContainer,
        model: MLXModelInfo
    ) async {
        guard deferredTuningModelID == model.id else { return }
        guard currentModel?.id == model.id else { return }
        guard isApplicationActiveForGPUWork else {
            logger.notice(
                "MLX keeping deferred tuning paused until app is active: id=\(model.id, privacy: .public)"
            )
            return
        }
        guard !hasSeenMemoryWarningSinceCurrentLoad else {
            logger.notice(
                "MLX skipping deferred tuning after load-time memory pressure: id=\(model.id, privacy: .public)"
            )
            deferredTuningModelID = nil
            return
        }
        guard tuningTask == nil else { return }

        deferredTuningModelID = nil

        let initialPrefillStepSize = await inferenceWorker.prefillStepSize(for: model.id)
        let persistedPrefill = Self.persistedPrefillStepSize(
            for: model.id,
            deviceSupportProfile: deviceSupportProfile
        )
        let persistedQuantizedKVStart = Self.persistedQuantizedKVStart(
            for: model.id,
            deviceSupportProfile: deviceSupportProfile
        )
        let defaultGenerationConfiguration = defaultGenerationConfigurationForCurrentDevice(model: model)
        let prefillCandidates = persistedPrefill.map { [$0] } ?? Array(Set([256, initialPrefillStepSize, 1024])).sorted()
        let quantizedKVStartCandidates = persistedQuantizedKVStart.map { [$0] } ?? [128, 256, 512]
        let wiredMemoryCap = Self.recommendedWiredMemoryCapBytes(
            deviceSupportProfile: deviceSupportProfile
        )
        let shouldTuneQuantizedKV = defaultGenerationConfiguration.cacheCompression.usesLegacyQuantization

        tuningTask = Task(priority: .utility) { [weak self] in
            guard let self else { return }
            defer {
                Task { @MainActor [weak self] in
                    self?.tuningTask = nil
                }
            }

            try? await Task.sleep(nanoseconds: Self.tuningStartupDelayNanoseconds)
            guard !Task.isCancelled else { return }

            let canUseGPU = await MainActor.run { [weak self] in
                self?.isApplicationActiveForGPUWork == true
            }
            guard canUseGPU else {
                await MainActor.run {
                    if self.tuningNeedsRetry(for: model) {
                        self.deferredTuningModelID = model.id
                    }
                    self.logger.notice(
                        "MLX deferred tuning skipped after app left foreground: id=\(model.id, privacy: .public)"
                    )
                }
                return
            }

            await MainActor.run {
                self.aggressivelyFreeMemory(reason: "tuning.start")
            }

            let benchmarkInput = await self.makeBenchmarkUserInput(container: container)
            var prefillResults: [TuningBenchmarkResult] = []

            for candidate in prefillCandidates {
                guard !Task.isCancelled else { return }
                let parameters = GenerateParameters(
                    maxTokens: 16,
                    maxKVSize: defaultGenerationConfiguration.maxKVSize,
                    prefillStepSize: candidate
                )
                do {
                    let result = try await self.measureTuningCandidate(
                        container: container,
                        userInput: benchmarkInput,
                        parameters: parameters,
                        candidate: candidate
                    )
                    prefillResults.append(result)
                } catch {
                    let logger = Logger(
                        subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM",
                        category: "MLXModelManager"
                    )
                    logger.error(
                        "MLX prefill tuning candidate failed: model=\(model.id, privacy: .public) prefill=\(candidate, privacy: .public) error=\(error.localizedDescription, privacy: .public)"
                    )
                    await MainActor.run {
                        self.aggressivelyFreeMemory(reason: "tuning.prefill.failure")
                    }
                }
            }

            guard !Task.isCancelled else { return }
            guard !prefillResults.isEmpty else { return }

            let selectedPrefill = Self.selectFastestSafeCandidate(
                prefillResults,
                wiredMemoryCap: wiredMemoryCap,
                preferLargerCandidateOnTie: true
            )

            var selectedQuantizedKVStart = persistedQuantizedKVStart ?? Self.defaultQuantizedKVStartStep
            var selectedMeasurement = selectedPrefill.measurement

            if shouldTuneQuantizedKV {
                var quantizedResults: [TuningBenchmarkResult] = []
                for candidate in quantizedKVStartCandidates {
                    guard !Task.isCancelled else { return }
                    let parameters = GenerateParameters(
                        maxTokens: 16,
                        maxKVSize: defaultGenerationConfiguration.maxKVSize,
                        kvBits: Self.kvQuantizationBits,
                        kvGroupSize: Self.kvQuantizationGroupSize,
                        quantizedKVStart: candidate,
                        prefillStepSize: selectedPrefill.candidate
                    )
                    do {
                        let result = try await self.measureTuningCandidate(
                            container: container,
                            userInput: benchmarkInput,
                            parameters: parameters,
                            candidate: candidate
                        )
                        quantizedResults.append(result)
                    } catch {
                        let logger = Logger(
                            subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM",
                            category: "MLXModelManager"
                        )
                        logger.error(
                            "MLX quantized KV start candidate failed: model=\(model.id, privacy: .public) start=\(candidate, privacy: .public) error=\(error.localizedDescription, privacy: .public)"
                        )
                        await MainActor.run {
                            self.aggressivelyFreeMemory(reason: "tuning.quantized.failure")
                        }
                    }
                }

                if let selectedQuantizedResult = quantizedResults.isEmpty
                    ? nil
                    : Self.selectFastestSafeCandidate(
                        quantizedResults,
                        wiredMemoryCap: wiredMemoryCap,
                        preferLargerCandidateOnTie: false
                    ) {
                    selectedQuantizedKVStart = selectedQuantizedResult.candidate
                    selectedMeasurement = selectedQuantizedResult.measurement
                }
            }

            Self.storePersistedPrefillStepSize(
                selectedPrefill.candidate,
                for: model.id,
                deviceSupportProfile: self.deviceSupportProfile
            )
            Self.storePersistedQuantizedKVStart(
                selectedQuantizedKVStart,
                for: model.id,
                deviceSupportProfile: self.deviceSupportProfile
            )
            await self.inferenceWorker.updateLoadedModelTuning(
                modelID: model.id,
                prefillStepSize: selectedPrefill.candidate,
                quantizedKVStart: selectedQuantizedKVStart,
                activeBytesEstimate: selectedMeasurement.kvBytes + selectedMeasurement.workspaceBytes,
                measurement: selectedMeasurement
            )
            await MainActor.run {
                self.logger.notice(
                    "MLX tuning selected: model=\(model.id, privacy: .public) prefill=\(selectedPrefill.candidate, privacy: .public) quantized_kv_start=\(selectedQuantizedKVStart, privacy: .public) active_bytes=\(selectedMeasurement.kvBytes + selectedMeasurement.workspaceBytes, privacy: .public)"
                )
                self.aggressivelyFreeMemory(reason: "tuning.complete")
            }
        }
    }

    private func handleMemoryWarning() {
        logger.notice("MLX received memory warning; clearing caches and invalidating sessions")
        hasSeenMemoryWarningSinceCurrentLoad = true
        deferredTuningModelID = nil
        tuningTask?.cancel()
        tuningTask = nil
        Task {
            await inferenceWorker.invalidateAll(reason: "memory_warning")
        }
        aggressivelyFreeMemory(reason: "memory.warning")
    }

    private var isApplicationActiveForGPUWork: Bool {
        UIApplication.shared.applicationState == .active
    }

    private func tuningNeedsRetry(for model: MLXModelInfo) -> Bool {
        let persistedPrefill = Self.persistedPrefillStepSize(
            for: model.id,
            deviceSupportProfile: deviceSupportProfile
        )
        let persistedQuantizedKVStart = Self.persistedQuantizedKVStart(
            for: model.id,
            deviceSupportProfile: deviceSupportProfile
        )
        let shouldTuneQuantizedKV = defaultGenerationConfigurationForCurrentDevice(model: model)
            .cacheCompression
            .usesLegacyQuantization
        let needsPrefillTuning = persistedPrefill == nil
        let needsQuantizedTuning = shouldTuneQuantizedKV && persistedQuantizedKVStart == nil
        return needsPrefillTuning || needsQuantizedTuning
    }

    private func handleApplicationInactivity(reason: String) {
        guard tuningTask != nil || deferredTuningModelID != nil else { return }

        if let model = currentModel, tuningNeedsRetry(for: model), !hasSeenMemoryWarningSinceCurrentLoad {
            deferredTuningModelID = model.id
        } else {
            deferredTuningModelID = nil
        }

        tuningTask?.cancel()
        tuningTask = nil
        logger.notice("MLX paused deferred tuning: reason=\(reason, privacy: .public)")
    }

    // MARK: - Loading

    func startLoading(modelID: String? = nil, source: String = "unknown") {
        let preferredModel = modelID.flatMap { id in
            availableModels.first(where: { $0.id == id })
        }
        guard let model = preferredModel ?? availableModels.first else {
            return
        }
        guard model.isAvailable else {
            loadError = compatibilityError(for: model) ??
                "\(model.displayName) is not downloaded. Download it before loading."
            return
        }

        if currentModel?.id == model.id, container != nil, !isLoading {
            logger.notice("MLX load request ignored: model already loaded id=\(model.id, privacy: .public) source=\(source, privacy: .public)")
            return
        }

        if pendingModelToLoad?.id == model.id, isLoading {
            logger.notice("MLX load request ignored: model already loading id=\(model.id, privacy: .public) source=\(source, privacy: .public)")
            return
        }

        let loadID = UUID()
        hasSeenMemoryWarningSinceCurrentLoad = false
        logger.notice("MLX load requested: id=\(model.id, privacy: .public) source=\(source, privacy: .public)")

        if let architecture = packageArchitecture(for: model) {
            logger.notice(
                "MLX package metadata: id=\(model.id, privacy: .public) architecture=\(architecture, privacy: .public) package=\(model.loadPolicy.packageDescription, privacy: .public)"
            )
        }

        cancelCurrentLoad(reason: "superseded by \(model.id)", tearDownModel: false)
        tearDownCurrentModel(reason: "preparing to load \(model.id)")
        activeLoadID = loadID
        load(model, loadID: loadID, source: source)
    }

    func model(withID id: String) -> MLXModelInfo? {
        availableModels.first(where: { $0.id == id })
    }

    var activeDownloadModel: MLXModelInfo? {
        guard let activeDownloadModelID else { return nil }
        return model(withID: activeDownloadModelID)
    }

    func isDownloading(modelID: String) -> Bool {
        isDownloading && activeDownloadModelID == modelID
    }

    func hasActiveDownload(excluding modelID: String) -> Bool {
        isDownloading && activeDownloadModelID != modelID
    }

    func downloadError(for modelID: String) -> String? {
        guard downloadErrorModelID == modelID else { return nil }
        return downloadError
    }

    func cancelCurrentLoad(reason: String = "cancelled", tearDownModel: Bool = true) {
        loadTask?.cancel()
        loadTask = nil
        logger.notice("MLX load cancelled: reason=\(reason, privacy: .public)")
        if isLoading {
            isLoading = false
            pendingModelToLoad = nil
        }
        if activeLoadID != nil {
            activeLoadID = nil
        }
        stopMemoryMaintenanceTimer()
        if tearDownModel {
            tearDownCurrentModel(reason: "cancelCurrentLoad(\(reason))")
        }
    }

    func cancelAndLoad(_ model: MLXModelInfo) {
        startLoading(modelID: model.id, source: "manager.cancelAndLoad")
    }

    private func load(_ model: MLXModelInfo, loadID: UUID, source: String) {
        guard model.isAvailable, let modelURL = modelDirectoryURL(for: model) else {
            activeLoadID = nil
            isLoading = false
            pendingModelToLoad = nil
            if let compatibilityError = compatibilityError(for: model) {
                loadError = compatibilityError
            } else {
                loadError = "Model '\(model.localDirName)' not found in Documents/Models/. Use the Download button to fetch it."
            }
            return
        }
        pendingModelToLoad = model
        isLoading = true
        loadError = nil
        let recommendedPrefillStepSize = Self.persistedPrefillStepSize(
            for: model.id,
            deviceSupportProfile: deviceSupportProfile
        ) ?? Self.defaultPrefillStepSize
        let wiredMemoryCap = Self.recommendedWiredMemoryCapBytes(
            deviceSupportProfile: deviceSupportProfile
        )

        loadTask = Task { [weak self] in
            guard let self else { return }
            do {
                await self.inferenceWorker.clearLoadedModel()
                await MainActor.run {
                    self.prepareMemoryForModelLoadTransition()
                }
                await Task.yield()
                self.logger.notice(
                    "MLX container load start: id=\(model.id, privacy: .public) source=\(source, privacy: .public) path=\(modelURL.lastPathComponent, privacy: .public)"
                )
                let loaded = try await loadModelContainer(directory: modelURL)
                await self.applyPreferredToolCallFormatIfNeeded(to: loaded, for: model)
                let weightBytes = await Self.estimateWeightBytes(for: loaded)
                let wiredMemoryPolicy = MLXLMCommon.WiredSumPolicy(cap: wiredMemoryCap)
                let reservationTicket = await Self.startReservationTicketIfNeeded(
                    weightBytes: weightBytes,
                    policy: wiredMemoryPolicy
                )
                await self.inferenceWorker.setLoadedModel(
                    MLXLoadedModelState(
                        container: loaded,
                        model: model,
                        wiredMemoryPolicy: wiredMemoryPolicy,
                        reservationTicket: reservationTicket,
                        weightBytes: weightBytes,
                        activeBytesEstimate: 0,
                        prefillStepSize: recommendedPrefillStepSize,
                        quantizedKVStart: Self.persistedQuantizedKVStart(
                            for: model.id,
                            deviceSupportProfile: deviceSupportProfile
                        ) ?? Self.defaultQuantizedKVStartStep,
                        measurement: nil
                    )
                )
                guard !Task.isCancelled else {
                    await MainActor.run {
                        if self.activeLoadID == loadID {
                            self.isLoading = false
                            self.pendingModelToLoad = nil
                            self.loadTask = nil
                            self.activeLoadID = nil
                            self.cleanupMemoryAfterLoadInterruption()
                        }
                    }
                    return
                }
                guard self.activeLoadID == loadID else {
                    self.logger.notice("MLX container load discarded: stale load id=\(model.id, privacy: .public)")
                    await self.inferenceWorker.clearLoadedModel()
                    await MainActor.run {
                        self.cleanupMemoryAfterLoadInterruption()
                    }
                    return
                }
                await MainActor.run {
                    self.container = loaded
                    self.currentModel = model
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                    self.activeLoadID = nil
                    self.applySteadyStateMemoryCachePolicy()
                    self.startMemoryMaintenanceTimer()
                    self.loadTask = nil
                }
                print("MLX model loaded: \(model.displayName)")
                self.logger.notice("MLX container load finished: id=\(model.id, privacy: .public)")
                await MainActor.run {
                    self.logToolTemplateSupport(for: model)
                    self.schedulePrefillTuningIfNeeded(
                        model: model,
                        initialPrefillStepSize: recommendedPrefillStepSize
                    )
                }
                if model.loadPolicy.defersAutomaticPrewarm {
                    self.deferredPrewarmModelID = model.id
                    self.logger.notice("MLX prewarm deferred until first generation: id=\(model.id, privacy: .public)")
                } else {
                    await self.prewarmModelShaders(for: model, reason: "post-load")
                }
            } catch is CancellationError {
                guard self.activeLoadID == loadID else { return }
                await MainActor.run {
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                    self.loadTask = nil
                    self.activeLoadID = nil
                    self.cleanupMemoryAfterLoadInterruption()
                }
                self.logger.notice("MLX container load cancelled during execution: id=\(model.id, privacy: .public)")
            } catch {
                guard self.activeLoadID == loadID else { return }
                await MainActor.run {
                    self.isLoading = false
                    self.pendingModelToLoad = nil
                    self.loadTask = nil
                    self.activeLoadID = nil
                    self.loadError = "Failed to load model: \(error.localizedDescription)"
                }
                self.tearDownCurrentModel(reason: "load failure for \(model.id)")
                print("Error: MLX model load error: \(error)")
                self.logger.error("MLX container load failed: id=\(model.id, privacy: .public) error=\(error.localizedDescription, privacy: .public)")
            }
        }
    }

    // MARK: - Download

    func startDownload(for model: MLXModelInfo) {
        guard !isDownloading else { return }
        if let issue = availabilityIssue(for: model) {
            downloadError = issue
            downloadErrorModelID = model.id
            return
        }
        isDownloading = true
        activeDownloadModelID = model.id
        downloadProgress = 0
        downloadError = nil
        downloadErrorModelID = nil

        if Self.isUITestFakeDownloadsEnabled {
            startSimulatedDownload(for: model)
            return
        }

        let targetDir = documentsDirectory.appendingPathComponent("Models/\(model.localDirName)")

        downloaderTask = Task { [weak self] in
            guard let self else { return }
            do {
                if FileManager.default.fileExists(atPath: targetDir.path) {
                    try FileManager.default.removeItem(at: targetDir)
                }
                try await self.downloader.download(
                    repoId: model.hfRepoId,
                    to: targetDir,
                    onProgress: { [weak self] progress in
                        Task { @MainActor [weak self] in
                            self?.downloadProgress = progress
                        }
                    }
                )
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self.downloadProgress = 1.0
                    self.isDownloading = false
                    self.activeDownloadModelID = nil
                    self.downloadErrorModelID = nil
                    self.refreshModelAvailability()
                }
                await MainActor.run {
                    self.startLoading(modelID: model.id, source: "download_complete")
                }
                print("Model downloaded: \(model.displayName)")
            } catch is CancellationError {
                await MainActor.run {
                    self.isDownloading = false
                    self.activeDownloadModelID = nil
                    self.downloadProgress = 0
                    self.downloadError = nil
                    self.downloadErrorModelID = nil
                }
                self.cleanupPartialDownload(at: targetDir)
            } catch {
                await MainActor.run {
                    self.isDownloading = false
                    self.activeDownloadModelID = nil
                    self.downloadError = error.localizedDescription
                    self.downloadErrorModelID = model.id
                }
                self.cleanupPartialDownload(at: targetDir)
                print("Error: Download error: \(error)")
            }
        }
    }

    func cancelDownload() {
        let model = activeDownloadModel
        downloaderTask?.cancel()
        downloaderTask = nil
        isDownloading = false
        activeDownloadModelID = nil
        downloadProgress = 0
        downloadError = nil
        downloadErrorModelID = nil
        if let model {
            let targetDir = documentsDirectory.appendingPathComponent("Models/\(model.localDirName)")
            cleanupPartialDownload(at: targetDir)
        }
    }

    private func startSimulatedDownload(for model: MLXModelInfo) {
        downloaderTask = Task { [weak self] in
            guard let self else { return }

            do {
                let progressSteps: [(progress: Double, delay: Duration)] = [
                    (0.2, .milliseconds(120)),
                    (0.55, .seconds(5)),
                    (1.0, .milliseconds(120))
                ]
                for step in progressSteps {
                    try await Task.sleep(for: step.delay)
                    guard !Task.isCancelled else {
                        throw CancellationError()
                    }

                    await MainActor.run {
                        self.downloadProgress = step.progress
                    }
                }

                await MainActor.run {
                    self.simulatedDownloadedModelIDs.insert(model.id)
                    self.isDownloading = false
                    self.activeDownloadModelID = nil
                    self.downloadProgress = 1.0
                    self.downloadErrorModelID = nil
                    self.refreshModelAvailability()
                }
            } catch is CancellationError {
                await MainActor.run {
                    self.isDownloading = false
                    self.activeDownloadModelID = nil
                    self.downloadProgress = 0
                    self.downloadError = nil
                    self.downloadErrorModelID = nil
                }
            } catch {
                await MainActor.run {
                    self.isDownloading = false
                    self.activeDownloadModelID = nil
                    self.downloadError = error.localizedDescription
                    self.downloadErrorModelID = model.id
                }
            }
        }
    }

    private func cleanupPartialDownload(at dir: URL) {
        let fm = FileManager.default
        let items = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        for item in items where item.pathExtension == "download" {
            try? fm.removeItem(at: item)
        }
        try? fm.removeItem(at: dir)
    }

    // MARK: - Shader Pre-warming

    /// Runs a minimal text forward pass immediately after model load to pre-compile all LM Metal
    /// shaders (SSM, full-attention, MoE layers). Without this the compiler spike hits on the
    /// first real inference, when the device is also holding a user image in memory.
    /// Skipped on the simulator — the Metal compute stack there can't handle a full LM forward
    /// pass and triggers EXC_BAD_ACCESS.
    private func prewarmModelShaders(for model: MLXModelInfo, reason: String) async {
        #if targetEnvironment(simulator)
        print("Warning: Skipping LM shader pre-warm on simulator")
        return
        #else
        guard let container else { return }
        guard currentModel?.id == model.id else { return }
        if deferredPrewarmModelID != model.id && prewarmInFlightModelID == nil && reason != "post-load" {
            return
        }
        if prewarmInFlightModelID == model.id {
            return
        }
        prewarmInFlightModelID = model.id
        logger.notice("MLX prewarm start: id=\(model.id, privacy: .public) reason=\(reason, privacy: .public)")
        print("Pre-warming LM Metal shaders...")
        // Free every cached Metal buffer before shader compilation so the compilation
        // spike has the maximum possible headroom on top of the ~3 GB model weights.
        prepareMemoryForPrewarm()
        do {
            let warmupMsg = Chat.Message.user("Hi")
            let userInput = UserInput(chat: [warmupMsg], processing: UserInput.Processing())
            let input = try await container.prepare(input: userInput)
            let params = GenerateParameters(maxTokens: 1, temperature: 1.0, topP: 1.0)
            let stream = try await container.generate(input: input, parameters: params)
            for await _ in stream {
                if Task.isCancelled {
                    throw CancellationError()
                }
            }
            cleanupMemoryAfterPrewarm()
            deferredPrewarmModelID = nil
            prewarmInFlightModelID = nil
            print("LM Metal shaders pre-warmed")
            logger.notice("MLX prewarm finished: id=\(model.id, privacy: .public)")
        } catch is CancellationError {
            cleanupMemoryAfterPrewarm()
            prewarmInFlightModelID = nil
            logger.notice("MLX prewarm cancelled: id=\(model.id, privacy: .public)")
        } catch {
            cleanupMemoryAfterPrewarm()
            prewarmInFlightModelID = nil
            print("Warning: Shader pre-warm failed (non-fatal): \(error.localizedDescription)")
            logger.error("MLX prewarm failed: id=\(model.id, privacy: .public) error=\(error.localizedDescription, privacy: .public)")
        }
        #endif
    }

    // MARK: - Generation

    func generateTextStream(
        conversationID: UUID,
        messages: [Chat.Message],
        enableThinking: Bool,
        memoryConstrained: Bool = false,
        tools: [MLXToolSpec] = [],
        toolDispatch: (@Sendable (MLXToolCall) async throws -> String)? = nil,
        onToken: @escaping @Sendable (String) -> Void,
        onToolCall: @escaping @Sendable (MLXToolCall) -> Void = { _ in }
    ) async throws -> MLXGenerationResult {
        guard let container, let currentModel else {
            throw GenerationError.modelNotLoaded
        }
        let includesMedia = messages.contains { !$0.images.isEmpty || !$0.videos.isEmpty }
        let configuredMaxOutputTokens = UserDefaults.standard.mlxMaxOutputTokensLimit
        let configuredContextWindow = configuredContextWindowLimit(for: currentModel)
        let turboQuantRequested = UserDefaults.standard.mlxEnableTurboQuant
        let generationConfiguration = Self.generationConfiguration(
            isEnabled: turboQuantRequested,
            preferTurboQuant: shouldPreferTurboQuant(for: currentModel),
            hasTools: !tools.isEmpty,
            hasMedia: includesMedia,
            memoryConstrained: memoryConstrained,
            prefersBoundedCache: prefersBoundedKVCacheAcrossTurns,
            configuredMaxOutputTokens: configuredMaxOutputTokens,
            configuredContextWindow: configuredContextWindow
        )
        if deferredPrewarmModelID == currentModel.id && !memoryConstrained {
            await prewarmModelShaders(for: currentModel, reason: "first-generation")
        }
        logger.notice(
            "MLX generation start: model=\(currentModel.localDirName, privacy: .public) conversation=\(conversationID.uuidString, privacy: .public) messages=\(messages.count, privacy: .public) thinking=\(enableThinking, privacy: .public) memory_constrained=\(memoryConstrained, privacy: .public) tools=\(tools.count, privacy: .public) media=\(includesMedia, privacy: .public) cache_policy=\(generationConfiguration.cachePolicy.diagnosticLabel, privacy: .public) cache_compression=\(generationConfiguration.cacheCompression.diagnosticLabel, privacy: .public) max_kv=\(generationConfiguration.maxKVSize ?? -1, privacy: .public)"
        )
        if turboQuantRequested,
           generationConfiguration.cachePolicy.usesDynamicKVQuantization,
           generationConfiguration.cacheCompression.usesLegacyQuantization {
            logger.notice("MLX TurboQuant fell back to legacy KV quantization for model=\(currentModel.id, privacy: .public)")
        }
        let toolCallFormat = await container.configuration.toolCallFormat
        let suppressWrappedXMLToolMarkup =
            !tools.isEmpty &&
            toolCallFormat == .xmlFunction &&
            usesWrappedXMLToolCalls(for: currentModel)
        if suppressWrappedXMLToolMarkup {
            logger.notice("MLX wrapped XML tool-call filter enabled: model=\(currentModel.localDirName, privacy: .public)")
        }
        prepareMemoryForGeneration()

        let tunedPrefillStepSize = await inferenceWorker.prefillStepSize(for: currentModel.id)
        let prefillStepSize = Self.adaptivePrefillStepSize(
            tuned: tunedPrefillStepSize,
            messages: messages
        )
        let quantizedKVStart = await inferenceWorker.quantizedKVStart(for: currentModel.id)
        let tunedCacheCompression = generationConfiguration.cacheCompression.applying(startStep: quantizedKVStart)
        let params = Self.makeGenerateParameters(
            maxTokens: generationConfiguration.maxTokens,
            maxKVSize: generationConfiguration.maxKVSize,
            cacheCompression: tunedCacheCompression,
            enableThinking: enableThinking,
            includesMedia: includesMedia,
            currentModelID: currentModel.id,
            prefillStepSize: prefillStepSize,
            repetitionPenalty: UserDefaults.standard.mlxRepetitionPenaltyValue
        )
        let additionalContext: [String: any Sendable]? = ["enable_thinking": enableThinking]
        let processing = UserInput.Processing()

        if !tools.isEmpty {
            try validateToolTemplateSupport(for: currentModel)
        }

        let sessionKey = MLXSessionKey(
            conversationID: conversationID,
            modelID: currentModel.id,
            enableThinking: enableThinking,
            toolsEnabled: !tools.isEmpty,
            includesMedia: includesMedia,
            instructionFingerprint: messages.first(where: { $0.role == .system })?.content.hashValue ?? 0
        )
        let request = MLXInferenceRequest(
            conversationID: conversationID,
            sessionKey: sessionKey,
            model: currentModel,
            messages: messages,
            enableThinking: enableThinking,
            additionalContext: additionalContext,
            processing: processing,
            tools: tools,
            params: params,
            suppressWrappedXMLToolMarkup: suppressWrappedXMLToolMarkup
        )

        do {
            let response = try await inferenceWorker.generate(
                request: request,
                toolDispatch: toolDispatch,
                onToken: onToken,
                onToolCall: onToolCall
            )
            latestPerformanceSample = response.performanceSample
            cleanupMemoryAfterGeneration()
            if !memoryConstrained {
                await startDeferredTuningIfNeeded(container: container, model: currentModel)
            }
            return MLXGenerationResult(toolInvocationCount: response.toolInvocationCount)
        } catch {
            logger.error("MLX generation failed: \(error.localizedDescription, privacy: .public)")
            cleanupMemoryAfterGenerationError()
            throw error
        }
    }

    // MARK: - Diagnostics / Management

    func refreshModelAvailability() {
        compatibilityErrors = [:]
        toolTemplateInspectionCache = [:]
        packageMetadataCache = [:]
        availableModels = availableModels.map { model in
            var updated = model
            let status = installationStatus(for: model)
            updated.isAvailable = status.isInstalled && status.isCompatible
            if let error = status.compatibilityError {
                compatibilityErrors[model.id] = error
            }
            return updated
        }
        if let currentModel,
           let refreshed = availableModels.first(where: { $0.id == currentModel.id }),
           !refreshed.isAvailable {
            tearDownCurrentModel(reason: "availability refresh invalidated \(currentModel.id)")
        }
    }

    func deleteModel(_ model: MLXModelInfo) throws {
        if isDownloading(modelID: model.id) {
            cancelDownload()
        }
        if pendingModelToLoad?.id == model.id {
            cancelCurrentLoad(reason: "delete \(model.id)", tearDownModel: false)
        }
        if currentModel?.id == model.id {
            tearDownCurrentModel(reason: "delete \(model.id)")
        }
        simulatedDownloadedModelIDs.remove(model.id)
        let dir = documentsDirectory.appendingPathComponent("Models/\(model.localDirName)")
        do {
            try FileManager.default.removeItem(at: dir)
            refreshModelAvailability()
            print("Deleted model '\(model.localDirName)' from Documents/Models/")
        } catch let error as NSError where error.code == NSFileNoSuchFileError {
            refreshModelAvailability()
            print("Model '\(model.localDirName)' not found in Documents/Models/ - nothing to delete.")
        }
    }

    func loadModel(_ model: MLXModelInfo) async {
        startLoading(modelID: model.id, source: "manager.loadModel")
    }

    func unloadAllModels() {
        cancelCurrentLoad(reason: "unloadAllModels", tearDownModel: false)
        tearDownCurrentModel(reason: "unloadAllModels")
        print("Unloaded all MLX models")
    }

    func printModelLocations() {
        for model in availableModels {
            if let url = modelDirectoryURL(for: model) {
                print("Model file \(model.name): \(url.path) - available: \(model.isAvailable)")
            } else {
                print("Model file \(model.name): not found in Documents/Models/ - available: \(model.isAvailable)")
            }
        }
    }

    func printModelInputOutputInfo() {
        guard let current = currentModel else {
            print("Warning: No model loaded")
            return
        }
        print("Current model: \(current.displayName)")
        print("   Context length: \(current.contextLength) tokens")
        print("   Reasoning: \(current.supportsReasoning)")
    }

    // MARK: - Errors

    enum GenerationError: LocalizedError {
        case modelNotLoaded
        case invalidChatHistory
        case missingToolDispatch
        case unsupportedToolTemplate(String)

        var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                return "No MLX model is loaded. Select the MLX backend and wait for the model to finish loading."
            case .invalidChatHistory:
                return "The MLX chat history is empty, so generation cannot start."
            case .missingToolDispatch:
                return "The model emitted a tool call, but no MLX tool dispatcher was configured."
            case .unsupportedToolTemplate(let message):
                return message
            }
        }
    }

    nonisolated internal static let maxToolInvocationsPerResponse = 6

    nonisolated internal static func shouldDisableTools(after toolResult: String) -> Bool {
        AppWebSearchToolBridge.isSearchLimitToolResponse(toolResult)
    }

    nonisolated internal static func excessiveToolCallToolResponse(maximum: Int) -> String {
        "[tool internal error: limit reached after \(maximum) tool invocations for this response. Do not call tools again. Continue with the available information already available.]"
    }

    nonisolated internal static func generationConfiguration(
        isEnabled: Bool,
        preferTurboQuant: Bool,
        hasTools: Bool,
        hasMedia: Bool,
        memoryConstrained: Bool,
        prefersBoundedCache: Bool,
        configuredMaxOutputTokens: Int?,
        configuredContextWindow: Int
    ) -> GenerationConfiguration {
        let maxTokens: Int? = if hasTools {
            4096
        } else if memoryConstrained {
            configuredMaxOutputTokens.map { min($0, 1024) }
        } else {
            configuredMaxOutputTokens
        }
        let cachePolicy = cachePolicy(
            isEnabled: isEnabled,
            hasTools: hasTools,
            hasMedia: hasMedia,
            prefersBoundedCache: prefersBoundedCache,
            memoryConstrained: memoryConstrained
        )
        let clampedCachePolicy = clampedCachePolicy(
            cachePolicy,
            configuredContextWindow: configuredContextWindow
        )
        return GenerationConfiguration(
            maxTokens: maxTokens,
            cachePolicy: clampedCachePolicy,
            cacheCompression: cacheCompressionMode(
                cachePolicy: clampedCachePolicy,
                preferTurboQuant: preferTurboQuant,
                hasMedia: hasMedia
            )
        )
    }

    nonisolated internal static func cachePolicy(
        isEnabled: Bool,
        hasTools: Bool,
        hasMedia: Bool,
        prefersBoundedCache: Bool,
        memoryConstrained: Bool
    ) -> MLXCachePolicy {
        _ = hasTools
        _ = hasMedia
        if memoryConstrained || prefersBoundedCache {
            return .boundedRotating(maxKVSize: memoryConstrainedMaxKVSize)
        }

        return isEnabled ? .persistentQuantizedSimple : .persistentSimple
    }

    nonisolated internal static func effectiveMaxKVSize(
        isEnabled: Bool,
        preferTurboQuant: Bool,
        hasTools: Bool,
        hasMedia: Bool,
        prefersBoundedCache: Bool,
        memoryConstrained: Bool
    ) -> Int? {
        cachePolicy(
            isEnabled: isEnabled,
            hasTools: hasTools,
            hasMedia: hasMedia,
            prefersBoundedCache: prefersBoundedCache,
            memoryConstrained: memoryConstrained
        ).maxKVSize
    }

    nonisolated internal static func cachePolicy(for parameters: GenerateParameters) -> MLXCachePolicy {
        if let maxKVSize = parameters.maxKVSize {
            return .boundedRotating(maxKVSize: maxKVSize)
        }
        if parameters.resolvedCacheCompression != nil {
            return .persistentQuantizedSimple
        }
        return .persistentSimple
    }

    nonisolated private static func clampedCachePolicy(
        _ cachePolicy: MLXCachePolicy,
        configuredContextWindow: Int
    ) -> MLXCachePolicy {
        switch cachePolicy {
        case .boundedRotating(let maxKVSize):
            return .boundedRotating(maxKVSize: min(maxKVSize, configuredContextWindow))
        case .persistentSimple, .persistentQuantizedSimple:
            return cachePolicy
        }
    }

    nonisolated internal static func cacheCompressionMode(
        cachePolicy: MLXCachePolicy,
        preferTurboQuant: Bool,
        hasMedia: Bool = false
    ) -> CacheCompressionMode {
        guard cachePolicy.usesDynamicKVQuantization else {
            return .none
        }

        if preferTurboQuant {
            let exactBufferSize = hasMedia ? turboQuantMediaExactBufferSize : turboQuantExactBufferSize
            let attentionBlockTokens =
                hasMedia ? turboQuantMediaAttentionBlockTokens : turboQuantAttentionBlockTokens
            return .turboQuant(
                TurboQuantConfiguration(
                    keyTotalBits: turboQuantKeyTotalBits,
                    valueBits: turboQuantValueBits,
                    seed: turboQuantSeed,
                    exactBufferSize: exactBufferSize,
                    attentionBlockTokens: attentionBlockTokens
                )
            )
        }

        return .legacyQuantized(
            KVQuantizationConfiguration(
                bits: kvQuantizationBits,
                groupSize: kvQuantizationGroupSize,
                startStep: defaultQuantizedKVStartStep
            )
        )
    }

    nonisolated internal static func kvQuantizationConfiguration(
        cachePolicy: MLXCachePolicy
    ) -> KVQuantizationConfiguration? {
        if case .persistentQuantizedSimple = cachePolicy {
            return KVQuantizationConfiguration(
                bits: kvQuantizationBits,
                groupSize: kvQuantizationGroupSize,
                startStep: defaultQuantizedKVStartStep
            )
        }
        return nil
    }

    nonisolated private static func makeGenerateParameters(
        maxTokens: Int?,
        maxKVSize: Int?,
        cacheCompression: CacheCompressionMode,
        enableThinking: Bool,
        includesMedia: Bool,
        currentModelID: String,
        prefillStepSize: Int,
        repetitionPenalty: Float?
    ) -> GenerateParameters {
        let legacyKVQuantization: KVQuantizationConfiguration? = if case .legacyQuantized(let configuration) = cacheCompression {
            configuration
        } else {
            nil
        }

        let isQwen35Model = currentModelID.hasPrefix("qwen3.5-")

        if enableThinking && isQwen35Model {
            let qwen35ThinkingTemperature: Float = includesMedia ? 0.6 : 1.0
            let qwen35ThinkingPresencePenalty: Float = includesMedia ? 0.0 : 1.5
            return GenerateParameters(
                maxTokens: maxTokens,
                maxKVSize: maxKVSize,
                kvBits: legacyKVQuantization?.bits,
                kvGroupSize: legacyKVQuantization?.groupSize ?? Self.kvQuantizationGroupSize,
                quantizedKVStart: legacyKVQuantization?.startStep ?? Self.defaultQuantizedKVStartStep,
                cacheCompression: cacheCompression.generateParametersCompression,
                temperature: qwen35ThinkingTemperature,
                topP: 0.95,
                topK: 20,
                minP: 0.0,
                presencePenalty: qwen35ThinkingPresencePenalty,
                repetitionPenalty: 1.0,
                prefillStepSize: prefillStepSize
            )
        } else if enableThinking {
            return GenerateParameters(
                maxTokens: maxTokens,
                maxKVSize: maxKVSize,
                kvBits: legacyKVQuantization?.bits,
                kvGroupSize: legacyKVQuantization?.groupSize ?? Self.kvQuantizationGroupSize,
                quantizedKVStart: legacyKVQuantization?.startStep ?? Self.defaultQuantizedKVStartStep,
                cacheCompression: cacheCompression.generateParametersCompression,
                temperature: 0.6,
                topP: 0.95,
                topK: nil,
                minP: 0.0,
                presencePenalty: nil,
                repetitionPenalty: repetitionPenalty,
                prefillStepSize: prefillStepSize
            )
        } else if isQwen35Model {
            let qwen35NonThinkingTemperature: Float = includesMedia ? 0.7 : 1.0
            let qwen35NonThinkingTopP: Float = includesMedia ? 0.8 : 1.0
            let qwen35NonThinkingPresencePenalty: Float = includesMedia ? 1.5 : 2.0
            return GenerateParameters(
                maxTokens: maxTokens,
                maxKVSize: maxKVSize,
                kvBits: legacyKVQuantization?.bits,
                kvGroupSize: legacyKVQuantization?.groupSize ?? Self.kvQuantizationGroupSize,
                quantizedKVStart: legacyKVQuantization?.startStep ?? Self.defaultQuantizedKVStartStep,
                cacheCompression: cacheCompression.generateParametersCompression,
                temperature: qwen35NonThinkingTemperature,
                topP: qwen35NonThinkingTopP,
                topK: 20,
                minP: 0.0,
                presencePenalty: qwen35NonThinkingPresencePenalty,
                repetitionPenalty: 1.0,
                prefillStepSize: prefillStepSize
            )
        } else {
            return GenerateParameters(
                maxTokens: maxTokens,
                maxKVSize: maxKVSize,
                kvBits: legacyKVQuantization?.bits,
                kvGroupSize: legacyKVQuantization?.groupSize ?? Self.kvQuantizationGroupSize,
                quantizedKVStart: legacyKVQuantization?.startStep ?? Self.defaultQuantizedKVStartStep,
                cacheCompression: cacheCompression.generateParametersCompression,
                temperature: 0.7,
                topP: 0.8,
                topK: nil,
                minP: 0.0,
                presencePenalty: nil,
                repetitionPenalty: repetitionPenalty,
                prefillStepSize: prefillStepSize
            )
        }
    }

    nonisolated internal static func wrappedToolResponseContent(_ content: String) -> String {
        "<tool_response>\n\(content)\n</tool_response>"
    }

    nonisolated internal static func toolResponsePromptContent(
        for model: MLXModelInfo,
        toolResult: String
    ) -> String {
        guard prefersUserWrappedToolResponses(for: model) else {
            return toolResult
        }
        return wrappedToolResponseContent(toolResult)
    }

    nonisolated internal static func toolResponsePromptRole(for model: MLXModelInfo) -> Chat.Message.Role {
        prefersUserWrappedToolResponses(for: model) ? .user : .tool
    }

    nonisolated internal static func prefersUserWrappedToolResponses(for model: MLXModelInfo) -> Bool {
        model.id.hasPrefix("qwen3.5-")
    }

    nonisolated internal static func requiresExplicitMessageHistoryForToolLoop(for model: MLXModelInfo) -> Bool {
        prefersUserWrappedToolResponses(for: model)
    }

    nonisolated internal static func normalizedToolArguments(
        _ arguments: [String: any Sendable]
    ) -> [String: any Sendable] {
        arguments.reduce(into: [String: any Sendable]()) { partialResult, entry in
            partialResult[entry.key] = normalizedToolArgumentValue(entry.value)
        }
    }

    nonisolated private static func normalizedToolArgumentValue(_ value: any Sendable) -> any Sendable {
        switch value {
        case let jsonValue as JSONValue:
            return normalized(jsonValue)
        case let string as String:
            return string
        case let bool as Bool:
            return bool
        case let int as Int:
            return int
        case let double as Double:
            return double
        case let array as [JSONValue]:
            return array.map { normalized($0) }
        case let object as [String: JSONValue]:
            return object.mapValues { normalized($0) }
        default:
            return String(describing: value)
        }
    }

    nonisolated private static func normalized(_ value: JSONValue) -> any Sendable {
        switch value {
        case .null:
            return "null"
        case .bool(let bool):
            return bool
        case .int(let int):
            return int
        case .double(let double):
            return double
        case .string(let string):
            return string
        case .array(let array):
            return array.map { normalized($0) }
        case .object(let object):
            return object.mapValues { normalized($0) }
        }
    }

    private func applyPreferredToolCallFormatIfNeeded(
        to loaded: ModelContainer,
        for model: MLXModelInfo
    ) async {
        guard let preferredToolCallFormat = preferredToolCallFormat(for: model) else {
            return
        }

        let currentToolCallFormat = await loaded.configuration.toolCallFormat
        guard currentToolCallFormat != preferredToolCallFormat else {
            logger.notice(
                "MLX tool-call format retained: id=\(model.id, privacy: .public) format=\(preferredToolCallFormat.rawValue, privacy: .public)"
            )
            return
        }

        await loaded.update { context in
            context.configuration.toolCallFormat = preferredToolCallFormat
        }
        logger.notice(
            "MLX tool-call format override applied: id=\(model.id, privacy: .public) format=\(preferredToolCallFormat.rawValue, privacy: .public)"
        )
    }

    private func validateToolTemplateSupport(for model: MLXModelInfo) throws {
        switch toolTemplateSupport(for: model) {
        case .supported:
            return
        case .unsupported(let message):
            throw GenerationError.unsupportedToolTemplate(message)
        }
    }

    private func logToolTemplateSupport(for model: MLXModelInfo) {
        let inspection = toolTemplateInspection(for: model)
        switch inspection.support {
        case .supported:
            print("MLX tool template support confirmed for \(model.displayName)")
            if let preferredToolCallFormat = inspection.preferredToolCallFormat {
                logger.notice(
                    "MLX tool template inspection: id=\(model.id, privacy: .public) format=\(preferredToolCallFormat.rawValue, privacy: .public) wrapped_xml=\(inspection.usesWrappedXMLToolCalls, privacy: .public)"
                )
            }
        case .unsupported(let message):
            print("Warning: MLX tool template support unavailable: \(message)")
        }
    }

    private func toolTemplateSupport(for model: MLXModelInfo) -> ToolTemplateSupport {
        toolTemplateInspection(for: model).support
    }

    private func preferredToolCallFormat(for model: MLXModelInfo) -> ToolCallFormat? {
        toolTemplateInspection(for: model).preferredToolCallFormat
    }

    private func usesWrappedXMLToolCalls(for model: MLXModelInfo) -> Bool {
        toolTemplateInspection(for: model).usesWrappedXMLToolCalls
    }

    private func toolTemplateInspection(for model: MLXModelInfo) -> ToolTemplateInspection {
        if let cached = toolTemplateInspectionCache[model.id] {
            return cached
        }

        guard let modelURL = modelDirectoryURL(for: model) else {
            let inspection = ToolTemplateInspection(
                support: .unsupported(
                    "Installed model '\(model.localDirName)' could not be inspected for tool support. Re-download the model package to restore the tokenizer chat template."
                ),
                preferredToolCallFormat: nil,
                usesWrappedXMLToolCalls: false
            )
            toolTemplateInspectionCache[model.id] = inspection
            return inspection
        }

        let candidateFiles = [
            "tokenizer_config.json",
            "tokenizer.json",
            "chat_template.jinja",
            "chat_template.json",
            "processor_config.json",
            "preprocessor_config.json"
        ]

        let combinedContents = candidateFiles.compactMap { fileName -> String? in
            let url = modelURL.appendingPathComponent(fileName)
            guard FileManager.default.fileExists(atPath: url.path),
                  let data = try? Data(contentsOf: url),
                  let text = String(data: data, encoding: .utf8) else {
                return nil
            }
            return text
        }.joined(separator: "\n")

        let lowered = combinedContents.lowercased()
        let hasToolsContext = lowered.contains("tools")
        let hasToolRole = lowered.contains("\"tool\"") ||
            lowered.contains("'tool'") ||
            lowered.contains("role == 'tool'") ||
            lowered.contains("role != 'tool'")
        let hasToolCalls = lowered.contains("tool_call") || lowered.contains("tool_calls")
        let preferredToolCallFormat = Self.inferToolCallFormat(
            packageContents: combinedContents,
            modelType: packageMetadata(for: model)?.modelType
        )
        let usesWrappedXMLToolCalls = Self.usesWrappedXMLToolCallTemplate(packageContents: combinedContents)

        let support: ToolTemplateSupport
        if hasToolsContext && (hasToolRole || hasToolCalls) {
            support = .supported
        } else {
            support = .unsupported(
                "Installed model '\(model.localDirName)' does not expose a tool-aware chat template. Re-download or update this Qwen 3.5 MLX model package to enable MLX tool calling."
            )
        }

        let inspection = ToolTemplateInspection(
            support: support,
            preferredToolCallFormat: preferredToolCallFormat,
            usesWrappedXMLToolCalls: usesWrappedXMLToolCalls
        )
        toolTemplateInspectionCache[model.id] = inspection
        return inspection
    }

    private func packageMetadata(for info: MLXModelInfo) -> ModelPackageMetadata? {
        if let cached = packageMetadataCache[info.id] {
            return cached
        }

        guard let modelURL = modelDirectoryURL(for: info) else {
            return nil
        }

        let configURL = modelURL.appendingPathComponent("config.json")
        let configJSON = (
            (try? Data(contentsOf: configURL))
                .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        ) ?? [:]
        let architecture = (configJSON["architectures"] as? [String])?.first
        let modelType = configJSON["model_type"] as? String

        let processorURLCandidates = [
            modelURL.appendingPathComponent("preprocessor_config.json"),
            modelURL.appendingPathComponent("processor_config.json")
        ]

        var processorClass: String?
        for url in processorURLCandidates {
            guard let data = try? Data(contentsOf: url),
                  let rawObject = try? JSONSerialization.jsonObject(with: data),
                  let json = rawObject as? [String: Any],
                  let detected = json["processor_class"] as? String else {
                continue
            }
            processorClass = detected
            break
        }

        let metadata = ModelPackageMetadata(
            architecture: architecture,
            processorClass: processorClass,
            modelType: modelType
        )
        packageMetadataCache[info.id] = metadata
        return metadata
    }

    nonisolated internal static func inferToolCallFormat(
        packageContents: String,
        modelType: String?
    ) -> ToolCallFormat? {
        let lowered = packageContents.lowercased()

        if lowered.contains("<function=") && lowered.contains("<parameter=") {
            return .xmlFunction
        }

        let hasJSONToolCallExample =
            lowered.contains("<tool_call>") &&
            lowered.contains("\"name\"") &&
            lowered.contains("\"arguments\"")
        if hasJSONToolCallExample {
            return .json
        }

        if let modelType,
           modelType.lowercased().hasPrefix("qwen3_5"),
           lowered.contains("tool_call") &&
           lowered.contains("function") {
            return .xmlFunction
        }

        return nil
    }

    nonisolated internal static func usesWrappedXMLToolCallTemplate(packageContents: String) -> Bool {
        let lowered = packageContents.lowercased()
        return lowered.contains("<tool_call>") &&
            lowered.contains("<function=") &&
            lowered.contains("<parameter=")
    }

    private func resetMemoryCaches() {
        Memory.cacheLimit = Self.aggressiveMemoryCacheLimitBytes
        URLCache.shared.removeAllCachedResponses()
    }

    private func applySteadyStateMemoryCachePolicy() {
        Memory.cacheLimit = Self.aggressiveMemoryCacheLimitBytes
    }

    private func prepareMemoryForPrewarm() {
        aggressivelyFreeMemory(reason: "prewarm.prepare")
    }

    private func cleanupMemoryAfterPrewarm() {
        applySteadyStateMemoryCachePolicy()
    }

    private func prepareMemoryForGeneration() {
        applySteadyStateMemoryCachePolicy()
    }

    private func cleanupMemoryAfterGeneration() {
        applySteadyStateMemoryCachePolicy()
    }

    private func cleanupMemoryAfterGenerationError() {
        aggressivelyFreeMemory(reason: "generation.error")
        applySteadyStateMemoryCachePolicy()
    }

    private func cleanupMemoryAfterLoadInterruption() {
        aggressivelyFreeMemory(reason: "load.interrupted")
        applySteadyStateMemoryCachePolicy()
    }

    private func prepareMemoryForModelLoadTransition() {
        aggressivelyFreeMemory(reason: "model.load.transition")
    }

    private func tearDownCurrentModel(reason: String) {
        if container != nil || currentModel != nil || deferredPrewarmModelID != nil || deferredTuningModelID != nil || prewarmInFlightModelID != nil {
            logger.notice("MLX model teardown: reason=\(reason, privacy: .public)")
        }
        stopMemoryMaintenanceTimer()
        container = nil
        currentModel = nil
        promptTokenCountCache.removeAll(keepingCapacity: false)
        deferredPrewarmModelID = nil
        deferredTuningModelID = nil
        prewarmInFlightModelID = nil
        tuningTask?.cancel()
        tuningTask = nil
        hasSeenMemoryWarningSinceCurrentLoad = false
        Task {
            await inferenceWorker.clearLoadedModel()
        }
        cleanupMemoryAfterUnload()
    }

    private func cleanupMemoryAfterUnload() {
        aggressivelyFreeMemory(reason: "model.unload")
        applySteadyStateMemoryCachePolicy()
    }

    private func aggressivelyFreeMemory(reason: String) {
        logger.notice("MLX memory cleanup: reason=\(reason, privacy: .public)")
        Memory.clearCache()
        Memory.cacheLimit = Self.aggressiveMemoryCacheLimitBytes
        URLCache.shared.removeAllCachedResponses()
    }

    private func startMemoryMaintenanceTimer() {
        stopMemoryMaintenanceTimer()
        let timer = Timer.scheduledTimer(withTimeInterval: Self.memoryMaintenanceInterval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.performPeriodicMemoryMaintenance()
            }
        }
        timer.tolerance = 1
        memoryMaintenanceTimer = timer
    }

    private func stopMemoryMaintenanceTimer() {
        memoryMaintenanceTimer?.invalidate()
        memoryMaintenanceTimer = nil
    }

    private func performPeriodicMemoryMaintenance() {
        guard currentModel != nil || isLoading else {
            stopMemoryMaintenanceTimer()
            return
        }
        applySteadyStateMemoryCachePolicy()
    }
}
