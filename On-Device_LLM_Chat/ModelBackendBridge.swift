//
//  ModelBackendBridge.swift
//  On-Device_LLM_Chat
//
//  Bridges between ChatViewModel's reasoning mode and model-specific implementations
//

import Foundation
import SwiftUI
import Combine

/// Manages the integration between reasoning mode settings and model backends
@MainActor
class ModelBackendBridge: ObservableObject {

    // MARK: - Published Properties

    /// Currently selected backend
    @Published var selectedBackend: Backend = .foundationModels

    /// Currently selected model ID (if using MLX backend)
    @Published var selectedModelID: String?

    /// Model manager for MLX models
    @Published var modelManager: MLXModelManager?

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
        self.selectedModelID = UserDefaults.standard.string(forKey: "selectedCustomModelID")

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
            manager.startLoading()
        }
    }

    // MARK: - Backend Management

    /// Switch to a different backend
    func selectBackend(_ backend: Backend) {
        selectedBackend = backend
        UserDefaults.standard.set(backend.rawValue, forKey: "selectedLLMBackend")

        if backend == .mlx {
            modelManager?.startLoading()
        }
    }

    /// Select a specific model by ID
    func selectModel(_ modelID: String) {
        selectedModelID = modelID
        UserDefaults.standard.set(modelID, forKey: "selectedCustomModelID")
    }

    // MARK: - Model-Specific Features

    /// Reasoning is available when using the MLX backend with a loaded reasoning-capable model
    var reasoningAvailable: Bool {
        guard selectedBackend == .mlx else {
            return false
        }
        if let manager = modelManager, manager.supportsNativeThinking {
            return true
        }
        guard let modelID = selectedModelID else { return false }
        return modelID.contains("qwen")
    }

    var supportsNativeThinking: Bool {
        return reasoningAvailable
    }

    /// Get the current model's display name
    var currentModelDisplayName: String? {
        if let model = modelManager?.currentModel {
            return "\(model.name) (\(model.parameters))"
        }
        guard let modelID = selectedModelID else { return nil }
        switch modelID {
        case "qwen3.5-4b-4bit":
            return "Qwen 3.5 (4B)"
        default:
            return modelID
        }
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

    // MARK: - Model Information

    func getThinkingModeInfo() -> ThinkingModeInfo {
        if reasoningAvailable && selectedBackend == .mlx {
            return ThinkingModeInfo(
                available: true,
                implementation: .native,
                description: "This Qwen model has built-in thinking capabilities that can be enabled via reasoning mode.",
                recommendation: "Enable 'Smart Reasoning' to automatically use thinking for complex questions, or 'Manual Reasoning' to use it for all responses."
            )
        } else if selectedBackend == .foundationModels {
            return ThinkingModeInfo(
                available: false,
                implementation: .unavailable,
                description: "Apple Foundation Models do not support reasoning mode. Switch to MLX Models (Qwen 3.5) to enable this feature.",
                recommendation: "To use reasoning mode, switch to MLX Models backend and ensure the Qwen 3.5 model is placed in Documents/Models/."
            )
        } else {
            return ThinkingModeInfo(
                available: false,
                implementation: .unavailable,
                description: "Reasoning mode is only available with Qwen models.",
                recommendation: "Ensure the Qwen 3.5 model is placed in Documents/Models/Qwen3.5-4B-MLX-4bit/."
            )
        }
    }

    struct ThinkingModeInfo {
        let available: Bool
        let implementation: Implementation
        let description: String
        let recommendation: String

        enum Implementation {
            case native
            case promptBased
            case unavailable
        }
    }
}

// MARK: - UserDefaults Extension

extension UserDefaults {
    var selectedLLMBackend: String {
        get { string(forKey: "selectedLLMBackend") ?? ModelBackendBridge.Backend.foundationModels.rawValue }
        set { set(newValue, forKey: "selectedLLMBackend") }
    }

    var selectedCustomModelID: String? {
        get { string(forKey: "selectedCustomModelID") }
        set { set(newValue, forKey: "selectedCustomModelID") }
    }
}

// MARK: - Preview Support

#if DEBUG
extension ModelBackendBridge {
    static var preview: ModelBackendBridge {
        let bridge = ModelBackendBridge()
        bridge.selectedBackend = .mlx
        bridge.selectedModelID = "qwen3.5-4b-4bit"
        return bridge
    }
}
#endif
