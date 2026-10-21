//
//  ModelBackendBridge.swift
//  ChatLLM
//
//  Bridges between chat settings and backend-specific model behavior
//

import Foundation
import SwiftUI
import Combine
import FoundationModels
import SwiftData
import OSLog

nonisolated private let bridgeLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM", category: "ModelBackendBridge")

/// Manages backend/model selection and backend-specific capabilities.
@MainActor
class ModelBackendBridge: ObservableObject {
    private static let legacyModelIDMap: [String: String] = [
        "qwen3.5-4b-4bit": "qwen3.5-4b-4bit-hybrid",
        "qwen3.5-4b-mixed36": "qwen3.5-4b-4bit-hybrid",
        "smollm3-3b-4bit": "lfm2.5-2.6b-4bit"
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
    }

    // MARK: - Initialization

    /// Shared singleton instance for consistent state across the app
    static let shared = ModelBackendBridge()

    private var cancellables = Set<AnyCancellable>()

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard, modelManager: MLXModelManager? = nil) {
        self.defaults = defaults
        // Migrate legacy "customCoreML" → "mlx"
        if let saved = defaults.string(forKey: AppSettingsKeys.selectedLLMBackend),
           saved == "customCoreML" {
            defaults.set("mlx", forKey: AppSettingsKeys.selectedLLMBackend)
        }

        // Load saved backend preference
        if let savedBackend = defaults.string(forKey: AppSettingsKeys.selectedLLMBackend),
           let backend = Backend(rawValue: savedBackend) {
            self.selectedBackend = backend
        }

        // Load saved model ID
        if let savedModelID = defaults.string(forKey: AppSettingsKeys.selectedCustomModelID) {
            let normalizedModelID = Self.legacyModelIDMap[savedModelID] ?? savedModelID
            if normalizedModelID != savedModelID {
                defaults.set(normalizedModelID, forKey: AppSettingsKeys.selectedCustomModelID)
            }
            self.selectedModelID = normalizedModelID
        }

        // Always create the manager so model availability can be checked
        // in the picker regardless of which backend is currently selected.
        let manager = modelManager ?? MLXModelManager(defaults: defaults)
        self.modelManager = manager

        // Keep selectedModelID in sync with whatever model actually loads.
        manager.$currentModel
            .compactMap { $0?.id }
            .receive(on: DispatchQueue.main)
            .sink { [weak self] id in
                self?.selectedModelID = id
            }
            .store(in: &cancellables)

        manager.downloadCompleted
            .receive(on: DispatchQueue.main)
            .sink { [weak self] id in
                self?.loadDownloadedModelIfSelected(id)
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

        let hasNonSystemMessages = conversation.messages.contains { $0.role != .system }
        let preferredBackend = conversation.preferredBackendRawValue.flatMap(Backend.init(rawValue:))
            ?? (hasNonSystemMessages ? .foundationModels : Backend(rawValue: defaults.selectedLLMBackend))
            ?? .foundationModels
        let savedModelID = conversation.preferredModelID ?? (hasNonSystemMessages ? nil : defaults.selectedCustomModelID)
        let preferredModelID = savedModelID.map { Self.legacyModelIDMap[$0] ?? $0 }
        if preferredModelID != savedModelID {
            conversation.preferredModelID = preferredModelID
        }

        if conversation.preferredBackendRawValue != preferredBackend.rawValue && !hasNonSystemMessages {
            conversation.preferredBackendRawValue = preferredBackend.rawValue
        }
        if conversation.preferredModelID == nil && !hasNonSystemMessages {
            conversation.preferredModelID = preferredModelID
        }

        if preferredBackend == .mlx {
            // Restore the chat's selection even when its model is unavailable.
            // Reopening a migrated chat must still wait for the new download.
            selectedModelID = preferredModelID
            selectedBackend = .mlx
            if let preferredModelID {
                switchToMLXModel(preferredModelID, source: "conversation.bind")
            } else {
                selectBackend(.mlx, source: "conversation.bind")
            }
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
            defaults.set(backend.rawValue, forKey: AppSettingsKeys.selectedLLMBackend)
            persistSelectionToActiveConversation()
            guard shouldResetPipelines else { return }
            notifyPipelineReset(reason: "backend.foundationModels")
            modelManager?.unloadAllModels()
        case .mlx:
            let modelIDToLoad = selectedModelID ?? modelManager?.availableModels.first(where: \.isAvailable)?.id
            guard let modelIDToLoad else {
                selectedBackend = .mlx
                defaults.set(backend.rawValue, forKey: AppSettingsKeys.selectedLLMBackend)
                persistSelectionToActiveConversation()
                return
            }
            switchToMLXModel(modelIDToLoad, source: source)
        }
    }

    /// Select a specific model by ID
    func selectModel(_ modelID: String, source: String = "unknown") {
        let modelID = Self.legacyModelIDMap[modelID] ?? modelID
        if selectedBackend == .mlx {
            switchToMLXModel(modelID, source: source)
            return
        }

        selectedModelID = modelID
        defaults.set(modelID, forKey: AppSettingsKeys.selectedCustomModelID)
        persistSelectionToActiveConversation()
    }

    func switchToMLXModel(_ modelID: String, source: String = "unknown") {
        let modelID = Self.legacyModelIDMap[modelID] ?? modelID
        guard let manager = modelManager,
              let model = manager.model(withID: modelID) else {
            modelManager?.unloadAllModels()
            modelManager?.loadError = String(localized: "Selected MLX model '\(modelID)' is no longer known. Choose another model.", bundle: .appLocalized)
            return
        }
        guard model.isAvailable else {
            notifyPipelineReset(reason: "backend.mlx.unavailable.\(modelID)")
            manager.unloadAllModels()
            manager.loadError = manager.availabilityIssue(for: model) ??
                String(localized: "Model '\(model.displayName)' isn't downloaded yet. Download it in Manage Models before selecting it.", bundle: .appLocalized)
            return
        }

        let isRetryingSelectedModel = selectedBackend == .mlx && selectedModelID == modelID
            && manager.currentModel == nil
        selectedModelID = modelID
        selectedBackend = .mlx
        defaults.set(modelID, forKey: AppSettingsKeys.selectedCustomModelID)
        defaults.set(Backend.mlx.rawValue, forKey: AppSettingsKeys.selectedLLMBackend)
        persistSelectionToActiveConversation()

        let isAlreadyLoaded = manager.currentModel?.id == modelID && !manager.isLoading
        let isAlreadyPending = manager.pendingModelToLoad?.id == modelID && manager.isLoading
        guard !isAlreadyLoaded, !isAlreadyPending else { return }

        // Retrying a failed load swaps no pipeline, and the only generation it could
        // affect is a send waiting on this very load.
        if !isRetryingSelectedModel {
            notifyPipelineReset(reason: "backend.mlx.\(modelID)")
        }
        manager.startLoading(modelID: modelID, source: source)
    }

    /// Deletes a downloaded model. If it was selected, the chat moves to Apple
    /// Intelligence when that's available, or back to choosing a model, rather than
    /// keeping a selection that can no longer reply.
    func deleteModel(_ model: MLXModelManager.MLXModelInfo) throws {
        guard let manager = modelManager else { return }
        let wasSelected = selectedBackend == .mlx && selectedModelID == model.id
        try manager.deleteModel(model)
        guard wasSelected else { return }
        if foundationModelsAvailable {
            selectBackend(.foundationModels, source: "model.deleted")
        } else {
            selectedModelID = nil
            defaults.removeObject(forKey: AppSettingsKeys.selectedCustomModelID)
            persistSelectionToActiveConversation()
        }
    }

    /// Loads a model that just finished downloading only when the chat is waiting
    /// for it. Loading it otherwise would replace the chat's model, or sit beside
    /// Apple Intelligence in memory.
    private func loadDownloadedModelIfSelected(_ modelID: String) {
        guard selectedBackend == .mlx,
              (selectedModelID ?? modelID) == modelID,
              modelManager?.currentModel?.id != modelID else { return }
        switchToMLXModel(modelID, source: "download_complete")
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
            bridgeLogger.error("Failed to persist backend selection: \((error as NSError).localizedDescription, privacy: .public)")
        }
    }

    // MARK: - Model-Specific Features

    /// Known model families that support reasoning/thinking mode.
    /// Used as fallback when modelManager isn't available but a model ID is stored.
    private static let knownReasoningPrefixes = ["qwen", "lfm2.5"]

    /// Reasoning is only exposed for MLX models that support it.
    var reasoningAvailable: Bool {
        guard selectedBackend == .mlx else {
            return false
        }
        if let model = selectedMLXModel {
            return model.supportsReasoning
        }
        guard let modelID = selectedModelID else { return false }
        return Self.knownReasoningPrefixes.contains(where: { modelID.contains($0) })
    }

    var reasoningRequired: Bool {
        selectedBackend == .mlx &&
            (selectedMLXModel?.requiresReasoning ?? (selectedModelID == "lfm2.5-2.6b-4bit"))
    }

    var foundationModelsAvailable: Bool {
        SystemLanguageModel.default.availability == .available
    }

    /// Tokens the Apple Intelligence model can hold in one request, including
    /// instructions, history, images and the reply.
    nonisolated static var foundationModelsContextSize: Int {
        SystemLanguageModel.default.contextSize
    }

    /// Whether the selected model reads images. Image attachments are only
    /// offered when it does.
    var nativeImageInputAvailable: Bool {
        switch selectedBackend {
        case .foundationModels:
            return SystemLanguageModel.default.capabilities.contains(.vision)
        case .mlx:
            return selectedMLXModel?.supportsNativeImages ?? false
        }
    }

    /// Get the current model's display name
    var currentModelDisplayName: String? {
        guard let modelID = selectedModelID else { return nil }
        if let model = modelManager?.model(withID: modelID)
            ?? modelManager?.currentModel.flatMap({ $0.id == modelID ? $0 : nil })
            ?? MLXModelManager.modelDefinitions.first(where: { $0.id == modelID }) {
            return model.displayName
        }
        return modelID
    }

    private var selectedMLXModel: MLXModelManager.MLXModelInfo? {
        if let modelID = selectedModelID {
            if let current = modelManager?.currentModel, current.id == modelID {
                return current
            }
            return modelManager?.model(withID: modelID)
        }
        return modelManager?.currentModel
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
}

extension Notification.Name {
    static let modelPipelineWillReset = Notification.Name("ModelPipelineWillReset")
}

// MARK: - UserDefaults Extension

extension UserDefaults {
    var mlxMaxOutputTokens: Int {
        get {
            guard object(forKey: AppSettingsKeys.mlxMaxOutputTokens) != nil else {
                return 0
            }
            let storedValue = integer(forKey: AppSettingsKeys.mlxMaxOutputTokens)
            if storedValue <= 0 {
                return 0
            }
            return min(max(storedValue, 512), 1024)
        }
        set {
            let clampedValue = newValue <= 0 ? 0 : min(max(newValue, 512), 1024)
            set(clampedValue, forKey: AppSettingsKeys.mlxMaxOutputTokens)
        }
    }

    var mlxMaxOutputTokensLimit: Int? {
        let storedValue = mlxMaxOutputTokens
        return storedValue == 0 ? nil : storedValue
    }

    func mlxContextWindowTokens(deviceMaximum: Int) -> Int {
        let minimum = 512

        guard object(forKey: AppSettingsKeys.mlxContextWindowTokens) != nil else {
            return deviceMaximum
        }

        let storedValue = integer(forKey: AppSettingsKeys.mlxContextWindowTokens)
        if storedValue <= 0 {
            return deviceMaximum
        }

        return min(max(storedValue, minimum), deviceMaximum)
    }

    var mlxEnableRotorQuant: Bool {
        get {
            if object(forKey: AppSettingsKeys.mlxEnableRotorQuant) != nil {
                return bool(forKey: AppSettingsKeys.mlxEnableRotorQuant)
            }
            return true
        }
        set { set(newValue, forKey: AppSettingsKeys.mlxEnableRotorQuant) }
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
        // Only an unset preference defers to the model's default. An explicit
        // 1.0 must survive as the user's request to disable the penalty.
        guard object(forKey: AppSettingsKeys.mlxRepetitionPenalty) != nil else { return nil }
        return Float(mlxRepetitionPenalty)
    }

    var selectedLLMBackend: String {
        get { string(forKey: AppSettingsKeys.selectedLLMBackend) ?? ModelBackendBridge.Backend.foundationModels.rawValue }
        set { set(newValue, forKey: AppSettingsKeys.selectedLLMBackend) }
    }

    var selectedCustomModelID: String? {
        get { string(forKey: AppSettingsKeys.selectedCustomModelID) }
        set { set(newValue, forKey: AppSettingsKeys.selectedCustomModelID) }
    }

    var disableToolCalls: Bool {
        get { bool(forKey: AppSettingsKeys.disableToolCalls) }
        set { set(newValue, forKey: AppSettingsKeys.disableToolCalls) }
    }
}
