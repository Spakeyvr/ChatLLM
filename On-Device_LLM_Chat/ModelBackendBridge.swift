//
//  ModelBackendBridge.swift
//  On-Device_LLM_Chat
//
//  Bridges between chat settings and backend-specific model behavior
//

import Foundation
import SwiftUI
import Combine
import FoundationModels
import SwiftData

/// Manages backend/model selection and backend-specific capabilities.
@MainActor
class ModelBackendBridge: ObservableObject {
    private static let legacyModelIDMap: [String: String] = [
        "qwen3.5-4b-4bit": "qwen3.5-4b-mixed36"
    ]

    // MARK: - Published Properties

    /// Currently selected backend
    @Published var selectedBackend: Backend = .foundationModels

    /// Currently selected model ID (if using MLX backend)
    @Published var selectedModelID: String?

    /// Model manager for MLX models
    @Published var modelManager: MLXModelManager?

    weak var activeConversation: Conversation?

    // MARK: - Backend Types

    enum Backend: String, CaseIterable, Identifiable {
        case foundationModels = "foundationModels"
        case mlx = "mlx"

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .foundationModels:
                return "Apple Intelligence"
            case .mlx:
                return "MLX Models"
            }
        }

        var icon: String {
            switch self {
            case .foundationModels:
                return "apple.logo"
            case .mlx:
                return "brain.head.profile"
            }
        }

        var description: String {
            switch self {
            case .foundationModels:
                return "Uses Apple's on-device language model with system-level optimization"
            case .mlx:
                return "Run MLX-converted models locally via mlx-swift-lm"
            }
        }
    }

    // MARK: - Initialization

    /// Shared singleton instance for consistent state across the app
    static let shared = ModelBackendBridge()

    private var cancellables = Set<AnyCancellable>()

    init() {
        // Migrate legacy "customCoreML" → "mlx"
        if let saved = UserDefaults.standard.string(forKey: "selectedLLMBackend"),
           saved == "customCoreML" {
            UserDefaults.standard.set("mlx", forKey: "selectedLLMBackend")
        }

        // Load saved backend preference
        if let savedBackend = UserDefaults.standard.string(forKey: "selectedLLMBackend"),
           let backend = Backend(rawValue: savedBackend) {
            self.selectedBackend = backend
        }

        // Load saved model ID
        if let savedModelID = UserDefaults.standard.string(forKey: "selectedCustomModelID") {
            let normalizedModelID = Self.legacyModelIDMap[savedModelID] ?? savedModelID
            if normalizedModelID != savedModelID {
                UserDefaults.standard.set(normalizedModelID, forKey: "selectedCustomModelID")
            }
            self.selectedModelID = normalizedModelID
        }

        // Always create the manager so model availability can be checked
        // in the picker regardless of which backend is currently selected.
        let manager = MLXModelManager()
        self.modelManager = manager

        // Keep selectedModelID in sync with whatever model actually loads.
        manager.$currentModel
            .compactMap { $0?.id }
            .receive(on: DispatchQueue.main)
            .sink { [weak self] id in
                self?.selectedModelID = id
            }
            .store(in: &cancellables)

        // Forward all modelManager @Published changes so views observing this bridge
        // (e.g. NavigationTitleView) re-render when isLoading or pendingModelToLoad changes.
        manager.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)

        // Only load if the user previously chose this backend.
        if self.selectedBackend == .mlx {
            manager.startLoading(modelID: self.selectedModelID, source: "bridge.init")
        }
    }

    func bindConversation(_ conversation: Conversation?) {
        activeConversation = conversation
        guard let conversation else { return }

        let preferredBackend = conversation.preferredBackendRawValue.flatMap(Backend.init(rawValue:))
            ?? Backend(rawValue: UserDefaults.standard.selectedLLMBackend)
            ?? .foundationModels
        let preferredModelID = conversation.preferredModelID ?? UserDefaults.standard.selectedCustomModelID

        conversation.preferredBackendRawValue = preferredBackend.rawValue
        if conversation.preferredModelID == nil {
            conversation.preferredModelID = preferredModelID
        }

        if preferredBackend == .mlx, let preferredModelID {
            switchToMLXModel(preferredModelID, source: "conversation.bind")
        } else {
            selectBackend(.foundationModels, source: "conversation.bind")
        }
    }

    // MARK: - Backend Management

    /// Switch to a different backend
    func selectBackend(_ backend: Backend, source: String = "unknown") {
        switch backend {
        case .foundationModels:
            let shouldResetPipelines =
                selectedBackend != .foundationModels ||
                modelManager?.currentModel != nil ||
                modelManager?.isLoading == true
            selectedBackend = .foundationModels
            UserDefaults.standard.set(backend.rawValue, forKey: "selectedLLMBackend")
            persistSelectionToActiveConversation()
            guard shouldResetPipelines else { return }
            notifyPipelineReset(reason: "backend.foundationModels")
            modelManager?.unloadAllModels()
        case .mlx:
            let modelIDToLoad = selectedModelID ?? modelManager?.availableModels.first(where: \.isAvailable)?.id
            guard let modelIDToLoad else {
                selectedBackend = .mlx
                UserDefaults.standard.set(backend.rawValue, forKey: "selectedLLMBackend")
                persistSelectionToActiveConversation()
                return
            }
            switchToMLXModel(modelIDToLoad, source: source)
        }
    }

    /// Select a specific model by ID
    func selectModel(_ modelID: String, source: String = "unknown") {
        if selectedBackend == .mlx {
            switchToMLXModel(modelID, source: source)
            return
        }

        selectedModelID = modelID
        UserDefaults.standard.set(modelID, forKey: "selectedCustomModelID")
        persistSelectionToActiveConversation()
    }

    func switchToMLXModel(_ modelID: String, source: String = "unknown") {
        selectedModelID = modelID
        selectedBackend = .mlx
        UserDefaults.standard.set(modelID, forKey: "selectedCustomModelID")
        UserDefaults.standard.set(Backend.mlx.rawValue, forKey: "selectedLLMBackend")
        persistSelectionToActiveConversation()

        guard let manager = modelManager,
              let model = manager.model(withID: modelID),
              model.isAvailable else {
            return
        }

        let isAlreadyLoaded = manager.currentModel?.id == modelID && !manager.isLoading
        let isAlreadyPending = manager.pendingModelToLoad?.id == modelID && manager.isLoading
        guard !isAlreadyLoaded, !isAlreadyPending else { return }

        notifyPipelineReset(reason: "backend.mlx.\(modelID)")
        manager.startLoading(modelID: modelID, source: source)
    }

    private func notifyPipelineReset(reason: String) {
        NotificationCenter.default.post(
            name: .modelPipelineWillReset,
            object: self,
            userInfo: ["reason": reason]
        )
    }

    private func persistSelectionToActiveConversation() {
        activeConversation?.preferredBackendRawValue = selectedBackend.rawValue
        activeConversation?.preferredModelID = selectedModelID
        do {
            try activeConversation?.modelContext?.save()
        } catch {
            print("Failed to persist backend selection: \(error)")
        }
    }

    // MARK: - Model-Specific Features

    /// Known model families that support reasoning/thinking mode.
    /// Used as fallback when modelManager isn't available but a model ID is stored.
    private static let knownReasoningPrefixes = ["qwen"]

    /// Reasoning is only exposed for MLX models that support it.
    var reasoningAvailable: Bool {
        guard selectedBackend == .mlx else {
            return false
        }
        if let manager = modelManager, manager.supportsNativeThinking {
            return true
        }
        guard let modelID = selectedModelID else { return false }
        return Self.knownReasoningPrefixes.contains(where: { modelID.contains($0) })
    }

    var foundationModelsAvailable: Bool {
        SystemLanguageModel.default.availability == .available
    }

    /// Get the current model's display name
    var currentModelDisplayName: String? {
        if let model = modelManager?.currentModel {
            return "\(model.name) (\(model.parameters))"
        }
        guard let modelID = selectedModelID else { return nil }
        switch modelID {
        case "qwen3.5-4b-mixed36":
            return "Qwen 3.5 (4B)"
        case "qwen3.5-2b-4bit":
            return "Qwen 3.5 (2B)"
        default:
            return modelID
        }
    }

    private var selectedMLXModel: MLXModelManager.MLXModelInfo? {
        if let model = modelManager?.currentModel {
            return model
        }
        guard let modelID = selectedModelID else {
            return nil
        }
        return modelManager?.model(withID: modelID)
    }

    var toolCallsAvailableForCurrentBackend: Bool {
        switch selectedBackend {
        case .foundationModels:
            return true
        case .mlx:
            guard let model = selectedMLXModel, let manager = modelManager else {
                return false
            }
            return manager.supportsToolCalls(for: model)
        }
    }

    var toolCallAvailabilityIssueForCurrentBackend: String? {
        guard selectedBackend == .mlx,
              let model = selectedMLXModel,
              let manager = modelManager else {
            return nil
        }
        return manager.toolCallIssue(for: model)
    }

    func shouldEnableThinking(
        reasoningMode: Bool,
        smartReasoningMode: Bool,
        messageReasoningMode: Bool
    ) -> Bool {
        guard reasoningAvailable else { return false }
        if reasoningMode { return true }
        if smartReasoningMode && messageReasoningMode { return true }
        return false
    }
}

// MARK: - UserDefaults Extension

extension Notification.Name {
    static let modelPipelineWillReset = Notification.Name("ModelPipelineWillReset")
}

extension UserDefaults {
    var mlxMaxOutputTokens: Int {
        get {
            guard object(forKey: "mlxMaxOutputTokens") != nil else {
                return 1024
            }
            let storedValue = integer(forKey: "mlxMaxOutputTokens")
            if storedValue <= 0 {
                return 0
            }
            return min(max(storedValue, 512), 1024)
        }
        set {
            let clampedValue = newValue <= 0 ? 0 : min(max(newValue, 512), 1024)
            set(clampedValue, forKey: "mlxMaxOutputTokens")
        }
    }

    var mlxMaxOutputTokensLimit: Int? {
        let storedValue = mlxMaxOutputTokens
        return storedValue == 0 ? nil : storedValue
    }

    func mlxContextWindowTokens(deviceMaximum: Int) -> Int {
        let minimum = 512

        guard object(forKey: "mlxContextWindowTokens") != nil else {
            return deviceMaximum
        }

        let storedValue = integer(forKey: "mlxContextWindowTokens")
        if storedValue <= 0 {
            return deviceMaximum
        }

        return min(max(storedValue, minimum), deviceMaximum)
    }

    var mlxEnableTurboQuant: Bool {
        get {
            if object(forKey: AppSettingsKeys.mlxEnableTurboQuant) != nil {
                return bool(forKey: AppSettingsKeys.mlxEnableTurboQuant)
            }
            if object(forKey: AppSettingsKeys.mlxEnableKVCacheQuantization) != nil {
                return bool(forKey: AppSettingsKeys.mlxEnableKVCacheQuantization)
            }
            return true
        }
        set { set(newValue, forKey: AppSettingsKeys.mlxEnableTurboQuant) }
    }

    var mlxRepetitionPenalty: Double {
        get {
            guard object(forKey: AppSettingsKeys.mlxRepetitionPenalty) != nil else {
                return 1.0
            }
            let storedValue = double(forKey: AppSettingsKeys.mlxRepetitionPenalty)
            return min(max(storedValue, 1.0), 1.5)
        }
        set {
            let clampedValue = min(max(newValue, 1.0), 1.5)
            set(clampedValue, forKey: AppSettingsKeys.mlxRepetitionPenalty)
        }
    }

    var mlxRepetitionPenaltyValue: Float? {
        let penalty = Float(mlxRepetitionPenalty)
        return penalty > 1.0 ? penalty : nil
    }

    var selectedLLMBackend: String {
        get { string(forKey: "selectedLLMBackend") ?? ModelBackendBridge.Backend.foundationModels.rawValue }
        set { set(newValue, forKey: "selectedLLMBackend") }
    }

    var selectedCustomModelID: String? {
        get { string(forKey: "selectedCustomModelID") }
        set { set(newValue, forKey: "selectedCustomModelID") }
    }

    var disableToolCalls: Bool {
        get { bool(forKey: "disableToolCalls") }
        set { set(newValue, forKey: "disableToolCalls") }
    }
}

// MARK: - Preview Support

#if DEBUG
extension ModelBackendBridge {
    static var preview: ModelBackendBridge {
        let bridge = ModelBackendBridge()
        bridge.selectedBackend = .mlx
        bridge.selectedModelID = "qwen3.5-4b-mixed36"
        return bridge
    }
}
#endif
