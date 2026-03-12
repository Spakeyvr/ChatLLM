//
//  ChatInputView.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI

// MARK: - Enhanced composer view

struct ComposerView: View {
    @Binding var text: String
    let placeholder: String
    let canSend: Bool
    let isGenerating: Bool
    let onSend: () -> Void
    let onStop: () -> Void
    let onClear: () -> Void
    let onCamera: () -> Void
    let onPhotosPicker: () -> Void
    let onFileImporter: () -> Void
    @Binding var forceSearch: Bool
    var searchAvailable: Bool = true
    @Binding var disableToolCalls: Bool
    @Binding var isReasoningEnabled: Bool
    @Binding var isSmartReasoningEnabled: Bool
    var reasoningAvailable: Bool = false

    var body: some View {
        SimpleTextComposer(
            text: $text,
            placeholder: placeholder,
            onSend: onSend,
            onStop: onStop,
            onClear: onClear,
            canSend: canSend,
            isGenerating: isGenerating,
            onCamera: onCamera,
            onPhotosPicker: onPhotosPicker,
            onFileImporter: onFileImporter,
            forceSearch: $forceSearch,
            searchAvailable: searchAvailable,
            disableToolCalls: $disableToolCalls,
            isReasoningEnabled: $isReasoningEnabled,
            isSmartReasoningEnabled: $isSmartReasoningEnabled,
            reasoningAvailable: reasoningAvailable
        )
        .accessibilityElement(children: .contain)
        .accessibilityLabel(String(localized: "Message composer"))
    }
}

// MARK: - Navigation title component - Liquid Glass & Centered Fix

struct NavigationTitleView: View {
    let title: String
    let isReasoningEnabled: Bool
    let isSmartReasoningEnabled: Bool
    let reasoningAvailable: Bool
    let hasMessages: Bool

    @ObservedObject var modelBackendBridge: ModelBackendBridge
    @AppStorage("selectedLLMBackend") private var selectedBackend: String = "foundationModels"

    @State private var showModelManagement = false

    private var modelManager: MLXModelManager? { modelBackendBridge.modelManager }

    private var isLoading: Bool {
        modelBackendBridge.modelManager?.isLoading ?? false
    }

    private var displayName: String {
        if selectedBackend == "foundationModels" {
            return "Apple Foundation"
        } else if let current = modelBackendBridge.currentModelDisplayName {
            return current
        } else {
            return "Select Model"
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
                Button {
                    modelBackendBridge.selectBackend(.foundationModels, source: "chat-input.foundation")
                } label: {
                    HStack {
                        Text("Apple Foundation")
                        if !modelBackendBridge.foundationModelsAvailable {
                            Image(systemName: "exclamationmark.triangle")
                                .foregroundStyle(.orange)
                        }
                        if selectedBackend == "foundationModels" {
                            Image(systemName: "checkmark")
                        }
                    }
                }
                .disabled(!modelBackendBridge.foundationModelsAvailable)

                // MLX Models
                ForEach(modelManager?.availableModels ?? [], id: \.id) { model in
                    if model.isAvailable {
                        Button {
                            modelBackendBridge.switchToMLXModel(model.id, source: "chat-input.model")
                        } label: {
                            HStack {
                                Text("\(model.name) (\(model.parameters))")
                                if selectedBackend == "mlx" && modelBackendBridge.selectedModelID == model.id {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    } else {
                        Button { } label: {
                            HStack {
                                Text("\(model.name) (\(model.parameters))")
                                Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
                            }
                        }
                        .disabled(true)
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
                if let mm = modelManager {
                    if #available(iOS 16.0, *) {
                        ModelManagementView(modelManager: mm)
                    } else {
                        Text("Model management requires iOS 16+")
                    }
                }
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
                Text("Switching…")
                    .font(.headline)
                    .foregroundStyle(.secondary)
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

// MARK: - Model Management View

@available(iOS 16.0, *)
struct ModelManagementView: View {
    @ObservedObject var modelManager: MLXModelManager
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(modelManager.availableModels, id: \.id) { model in
                    ModelManagementRow(model: model, modelManager: modelManager)
                }
            }
            .navigationTitle("Manage Models")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

@available(iOS 16.0, *)
struct ModelManagementRow: View {
    let model: MLXModelManager.MLXModelInfo
    @ObservedObject var modelManager: MLXModelManager

    @State private var showDeleteAlert = false
    @State private var showInfoSheet = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.displayName)
                        .font(.headline)

                    Text(model.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                Spacer()

                if model.isAvailable {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }

                Button {
                    showInfoSheet = true
                } label: {
                    Image(systemName: "info.circle.fill")
                        .foregroundStyle(.blue)
                        .imageScale(.large)
                }
                .buttonStyle(.plain)
            }

            // Action buttons
            HStack(spacing: 12) {
                if model.isAvailable {
                    Button(role: .destructive) {
                        showDeleteAlert = true
                    } label: {
                        Label("Delete", systemImage: "trash")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.bordered)
                } else if modelManager.isDownloading {
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: modelManager.downloadProgress)
                        Text("\(Int(modelManager.downloadProgress * 100))%")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Button("Cancel") {
                        modelManager.cancelDownload()
                    }
                    .foregroundStyle(.red)
                    .font(.caption)
                } else {
                    VStack(alignment: .leading, spacing: 4) {
                        Button("Download (~3.9 GB)") {
                            modelManager.startDownload(for: model)
                        }
                        .buttonStyle(.borderedProminent)
                        .font(.caption)
                        if let error = modelManager.downloadError {
                            Text(error)
                                .font(.caption2)
                                .foregroundStyle(.red)
                                .lineLimit(2)
                        }
                    }
                }
            }
        }
        .padding(.vertical, 4)
        .alert("Delete Model?", isPresented: $showDeleteAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                try? modelManager.deleteModel(model)
            }
        } message: {
            Text("This will permanently delete \(model.displayName) from your device.")
        }
        .sheet(isPresented: $showInfoSheet) {
            ModelInfoSheet(model: model)
        }
    }
}

// MARK: - Toolbar buttons component

struct ToolbarButtonsView: View {
    let isReasoningEnabled: Bool
    let isSmartReasoningEnabled: Bool
    let reasoningAvailable: Bool
    let onReasoningTap: () -> Void

    var body: some View {
        Button(action: onReasoningTap) {
            if reasoningAvailable && isReasoningEnabled {
                Image(systemName: "brain.head.profile")
                    .foregroundStyle(.blue)
            } else if reasoningAvailable && isSmartReasoningEnabled {
                Image(systemName: "sparkles")
                    .foregroundStyle(.purple)
            } else {
                Image(systemName: "brain.head.profile")
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(!reasoningAvailable)
        .accessibilityLabel(String(localized: "Reasoning Mode"))
        .accessibilityHint(String(localized: "Configure reasoning settings"))
    }
}
