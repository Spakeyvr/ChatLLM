//
//  ModelBackendBridge.swift
//  On-Device_LLM_Chat
//
//  Created by AI Assistant
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
    
    /// Currently selected custom model (if using custom backend)
    @Published var selectedModelID: String?
    
    /// Model manager for custom CoreML models
    @Published var modelManager: LLMModelManager?
    
    // MARK: - Backend Types
    
    enum Backend: String, CaseIterable, Identifiable {
        case foundationModels = "foundationModels"
        case customCoreML = "customCoreML"
        
        var id: String { rawValue }
        
        var displayName: String {
            switch self {
            case .foundationModels:
                return "Apple Intelligence"
            case .customCoreML:
                return "Custom Models"
            }
        }
        
        var icon: String {
            switch self {
            case .foundationModels:
                return "apple.logo"
            case .customCoreML:
                return "brain.head.profile"
            }
        }
        
        var description: String {
            switch self {
            case .foundationModels:
                return "Uses Apple's on-device language model with system-level optimization"
            case .customCoreML:
                return "Run downloaded CoreML models locally with full control"
            }
        }
    }
    
    // MARK: - Initialization

    /// Shared singleton instance for consistent state across the app
    static let shared = ModelBackendBridge()

    private var cancellables = Set<AnyCancellable>()

    init() {
        // Load saved backend preference
        if let savedBackend = UserDefaults.standard.string(forKey: "selectedLLMBackend"),
           let backend = Backend(rawValue: savedBackend) {
            self.selectedBackend = backend
        }

        // Load saved model ID
        self.selectedModelID = UserDefaults.standard.string(forKey: "selectedCustomModelID")

        // Always create the manager so model availability can be checked
        // in the picker regardless of which backend is currently selected.
        let manager = LLMModelManager()
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
        if self.selectedBackend == .customCoreML {
            manager.startLoading()
        }
    }

    // MARK: - Backend Management

    /// Switch to a different backend
    func selectBackend(_ backend: Backend) {
        selectedBackend = backend
        UserDefaults.standard.set(backend.rawValue, forKey: "selectedLLMBackend")

        // Initialize model manager if switching to custom
        if backend == .customCoreML && modelManager == nil {
            let manager = LLMModelManager()
            modelManager = manager
            manager.startLoading()
        }
    }
    
    /// Select a specific custom model
    func selectModel(_ modelID: String) {
        selectedModelID = modelID
        UserDefaults.standard.set(modelID, forKey: "selectedCustomModelID")
    }
    
    // MARK: - Model-Specific Features
    
    /// Determine if reasoning mode is available for the current backend
    /// ⚠️ Reasoning is ONLY available for Qwen models via Custom CoreML
    /// Foundation Models do NOT support reasoning mode
    var reasoningAvailable: Bool {
        guard selectedBackend == .customCoreML else {
            return false  // ❌ Foundation Models do NOT support reasoning
        }
        
        // Check if the model manager reports native thinking support (if model is loaded)
        if let manager = modelManager, manager.supportsNativeThinking {
            return true
        }
        
        // Fallback: Check model ID for known models with native thinking
        // This is crucial for when the model hasn't been loaded yet
        guard let modelID = selectedModelID else {
            return false
        }
        
        // ✅ Only Qwen models support native thinking parameter
        return modelID.contains("qwen")
    }
    
    /// Determine if the current model supports native thinking parameter
    /// DEPRECATED: Use reasoningAvailable instead for clarity
    var supportsNativeThinking: Bool {
        return reasoningAvailable
    }
    
    /// Get the current model's display name
    var currentModelDisplayName: String? {
        // Derive directly from the loaded model when available
        if let model = modelManager?.currentModel {
            return "\(model.name) (\(model.parameters))"
        }
        guard let modelID = selectedModelID else { return nil }
        // Fallback for known model IDs before the model finishes loading
        switch modelID {
        case "qwen3-4b-2bit":
            return "Qwen 3 Instruct (4B)"
        default:
            return modelID
        }
    }
    
    /// Determine if thinking mode should be enabled for the current model
    /// ⚠️ This respects user's reasoning mode preferences
    /// NOTE: Only works with Qwen models - Foundation Models always return false
    func shouldEnableThinking(
        reasoningMode: Bool,
        smartReasoningMode: Bool,
        messageReasoningMode: Bool
    ) -> Bool {
        // CRITICAL: Only enable thinking if using Qwen model
        guard reasoningAvailable else {
            return false  // ❌ Foundation Models never use reasoning
        }
        
        // If manual reasoning is enabled, always use thinking
        if reasoningMode {
            return true
        }
        
        // If smart reasoning is enabled, use message-level determination
        if smartReasoningMode && messageReasoningMode {
            return true
        }
        
        return false
    }
    
    // MARK: - Model Information
    
    /// Get detailed information about thinking mode for current model
    /// ⚠️ Foundation Models do NOT support reasoning - returns unavailable
    func getThinkingModeInfo() -> ThinkingModeInfo {
        if reasoningAvailable && selectedBackend == .customCoreML {
            // ✅ Qwen model with native thinking support
            return ThinkingModeInfo(
                available: true,
                implementation: .native,
                description: "This Qwen model has built-in thinking capabilities that can be enabled via reasoning mode.",
                recommendation: "Enable 'Smart Reasoning' to automatically use thinking for complex questions, or 'Manual Reasoning' to use it for all responses."
            )
        } else if selectedBackend == .foundationModels {
            // ❌ Foundation Models do NOT support reasoning
            return ThinkingModeInfo(
                available: false,
                implementation: .unavailable,
                description: "Apple Foundation Models do not support reasoning mode. Switch to Custom Models (Qwen 3) to enable this feature.",
                recommendation: "To use reasoning mode, switch to Custom Models backend and select Qwen 3."
            )
        } else {
            // ❌ Custom backend but no Qwen model selected
            return ThinkingModeInfo(
                available: false,
                implementation: .unavailable,
                description: "Reasoning mode is only available with Qwen models.",
                recommendation: "Select Qwen 3 model to enable reasoning mode."
            )
        }
    }
    
    struct ThinkingModeInfo {
        let available: Bool
        let implementation: Implementation
        let description: String
        let recommendation: String
        
        enum Implementation {
            case native       // Model has native thinking parameter
            case promptBased  // Thinking via prompt engineering
            case unavailable  // No thinking support
        }
    }
}

// MARK: - UserDefaults Extension

extension UserDefaults {
    /// Convenience accessor for backend selection
    var selectedLLMBackend: String {
        get {
            string(forKey: "selectedLLMBackend") ?? ModelBackendBridge.Backend.foundationModels.rawValue
        }
        set {
            set(newValue, forKey: "selectedLLMBackend")
        }
    }
    
    /// Convenience accessor for model ID selection
    var selectedCustomModelID: String? {
        get {
            string(forKey: "selectedCustomModelID")
        }
        set {
            set(newValue, forKey: "selectedCustomModelID")
        }
    }
}

// MARK: - Preview Support

#if DEBUG
extension ModelBackendBridge {
    static var preview: ModelBackendBridge {
        let bridge = ModelBackendBridge()
        bridge.selectedBackend = .customCoreML
        bridge.selectedModelID = "qwen3-4b-instruct-6bit"
        return bridge
    }
}
#endif
