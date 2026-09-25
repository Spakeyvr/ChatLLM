//
//  ChatInputView.swift
//  ChatLLM
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI

// MARK: - Enhanced composer view

struct ComposerView: View {
    @Binding var text: String
    let placeholder: String
    let focusRequest: Int
    let isEditing: Bool
    let canSend: Bool
    let isGenerating: Bool
    let onSend: () -> Void
    let onStop: () -> Void
    let onClear: () -> Void
    let onCancelEditing: () -> Void
    let onCamera: () -> Void
    let onPhotosPicker: () -> Void
    let onFileImporter: () -> Void
    @Binding var forceSearch: Bool
    var searchAvailable: Bool = true
    @Binding var disableToolCalls: Bool
    var toolCallsLockedDisabled: Bool = false
    @Binding var isReasoningEnabled: Bool
    @Binding var isSmartReasoningEnabled: Bool
    var reasoningAvailable: Bool = false
    var reasoningRequired: Bool = false

    var body: some View {
        SimpleTextComposer(
            text: $text,
            placeholder: placeholder,
            focusRequest: focusRequest,
            isEditing: isEditing,
            onSend: onSend,
            onStop: onStop,
            onClear: onClear,
            onCancelEditing: onCancelEditing,
            canSend: canSend,
            isGenerating: isGenerating,
            onCamera: onCamera,
            onPhotosPicker: onPhotosPicker,
            onFileImporter: onFileImporter,
            forceSearch: $forceSearch,
            searchAvailable: searchAvailable,
            disableToolCalls: $disableToolCalls,
            toolCallsLockedDisabled: toolCallsLockedDisabled,
            isReasoningEnabled: $isReasoningEnabled,
            isSmartReasoningEnabled: $isSmartReasoningEnabled,
            reasoningAvailable: reasoningAvailable,
            reasoningRequired: reasoningRequired
        )
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Message composer")
    }
}

// MARK: - Navigation title component - Liquid Glass & Centered Fix

struct NavigationTitleView: View {
    @Environment(\.locale) private var locale
    let title: String
    let isReasoningEnabled: Bool
    let isSmartReasoningEnabled: Bool
    let reasoningAvailable: Bool
    let hasMessages: Bool

    @ObservedObject var modelBackendBridge: ModelBackendBridge

    @State private var showModelManagement = false

    private var modelManager: MLXModelManager? { modelBackendBridge.modelManager }

    private var isLoading: Bool {
        modelBackendBridge.modelManager?.isLoading ?? false
    }

    private var displayName: String {
        if modelBackendBridge.selectedBackend == .foundationModels {
            return "Apple Foundation"
        } else if let current = modelBackendBridge.currentModelDisplayName {
            return current
        } else {
            return String(localized: "Select Model", bundle: .appLocalized, locale: locale)
        }
    }

    var body: some View {
        if hasMessages {
            // Mid-conversation: model is locked, show non-interactive capsule
            modelCapsule(interactive: false)
        } else {
            // No messages yet: allow model selection
            Menu {
                // Apple Foundation
                if modelBackendBridge.foundationModelsAvailable {
                    Button {
                        modelBackendBridge.selectBackend(.foundationModels, source: "chat-input.foundation")
                    } label: {
                        HStack {
                            Text("Apple Foundation")
                            if modelBackendBridge.selectedBackend == .foundationModels {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }

                // MLX Models (only downloaded, compatible ones; the rest live in Manage Models)
                ForEach(modelManager?.installedModels ?? [], id: \.id) { model in
                    Button {
                        modelBackendBridge.switchToMLXModel(model.id, source: "chat-input.model")
                    } label: {
                        HStack {
                            Text("\(model.name) (\(model.parameters))")
                            if modelBackendBridge.selectedBackend == .mlx && modelBackendBridge.selectedModelID == model.id {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }

                Divider()

                Button {
                    showModelManagement = true
                } label: {
                    Label("Manage Models", systemImage: "square.stack.3d.down.forward")
                }
            } label: {
                modelCapsule(interactive: true)
            }
            .menuOrder(.fixed)
            .sheet(isPresented: $showModelManagement) {
                ModelSelectionView()
            }
        }
    }

    private var isSwitching: Bool {
        isLoading && (modelBackendBridge.modelManager?.pendingModelToLoad != nil)
    }

    @ViewBuilder
    private func modelCapsule(interactive: Bool) -> some View {
        HStack(spacing: 4) {
            if isLoading {
                ProgressView()
                    .controlSize(.mini)
            }

            if isSwitching {
                LoadingDotsLabel(font: .headline)
            } else {
                Text(displayName)
                    .font(.headline)
                    .foregroundStyle(.primary)
            }

            if interactive {
                Image(systemName: "chevron.down")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.secondary.opacity(0.7))
            }
        }
    }
}
