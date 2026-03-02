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
    let onPhotosPicker: () -> Void
    let onFileImporter: () -> Void
    @Binding var forceSearch: Bool
    var searchAvailable: Bool = true

    var body: some View {
        SimpleTextComposer(
            text: $text,
            placeholder: placeholder,
            onSend: onSend,
            onStop: onStop,
            onClear: onClear,
            canSend: canSend,
            isGenerating: isGenerating,
            onPhotosPicker: onPhotosPicker,
            onFileImporter: onFileImporter,
            forceSearch: $forceSearch,
            searchAvailable: searchAvailable
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
    @State private var showDownloadAlert = false
    @State private var modelToDownload: LLMModelManager.LLMModelInfo?

    // modelBackendBridge.modelManager is always set in ModelBackendBridge.init
    private var modelManager: LLMModelManager { modelBackendBridge.modelManager! }

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
                    selectedBackend = "foundationModels"
                    modelBackendBridge.selectBackend(.foundationModels)
                    modelManager.cancelCurrentLoad()
                } label: {
                    Text("Apple Foundation")
                    if selectedBackend == "foundationModels" {
                        Image(systemName: "checkmark")
                    }
                }

                // Custom CoreML Models
                ForEach(modelManager.availableModels, id: \.id) { model in
                    let isComingSoon = model.id == "llama-3.2-3b-instruct"

                    if model.isAvailable {
                        Button {
                            selectedBackend = "customCoreML"
                            modelBackendBridge.selectedBackend = .customCoreML
                            modelBackendBridge.selectModel(model.id)
                            modelManager.cancelAndLoad(model)
                        } label: {
                            HStack {
                                Text("\(model.name) (\(model.parameters))")
                                if selectedBackend == "customCoreML" && modelBackendBridge.selectedModelID == model.id {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    } else if isComingSoon {
                        Button { } label: {
                            HStack {
                                Text("\(model.name) (\(model.parameters))")
                                Image(systemName: "clock").foregroundStyle(.purple)
                                Text("Coming Soon").font(.caption).foregroundStyle(.purple)
                            }
                        }
                        .disabled(true)
                    } else if model.downloadURL != nil {
                        Button {
                            modelToDownload = model
                            showDownloadAlert = true
                        } label: {
                            HStack {
                                Text("\(model.name) (\(model.parameters))")
                                Image(systemName: "arrow.down.circle")
                                if let size = model.fileSizeFormatted {
                                    Text(size).font(.caption).foregroundStyle(.secondary)
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
                if #available(iOS 16.0, *) {
                    ModelManagementView(modelManager: modelManager)
                } else {
                    Text("Model management requires iOS 16+")
                }
            }
            .alert("Download Model?", isPresented: $showDownloadAlert) {
                Button("Cancel", role: .cancel) { }
                Button("Download") {
                    if let model = modelToDownload, let modelManager = modelBackendBridge.modelManager {
                        Task {
                            do { try await modelManager.downloadModel(model) }
                            catch { print("Download failed: \(error)") }
                        }
                    }
                }
            } message: {
                if let model = modelToDownload {
                    Text("Download \(model.displayName)? This will use \(model.fileSizeFormatted ?? "~1.8 GB") of storage.")
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
    @ObservedObject var modelManager: LLMModelManager
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
    let model: LLMModelManager.LLMModelInfo
    @ObservedObject var modelManager: LLMModelManager

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

                if let isDownloading = modelManager.isDownloading[model.id], isDownloading {
                    ProgressView()
                } else if model.isAvailable {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }

                // Info button (always visible)
                Button {
                    showInfoSheet = true
                } label: {
                    Image(systemName: "info.circle.fill")
                        .foregroundStyle(.blue)
                        .imageScale(.large)
                }
                .buttonStyle(.plain)
            }

            // Download progress
            if let progress = modelManager.downloadProgress[model.id], progress > 0, progress < 1 {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                Text("\(Int(progress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Action buttons
            HStack(spacing: 12) {
                // Check if this is a coming soon model
                let isComingSoon = model.id == "llama-3.2-3b-instruct"

                if model.isAvailable {
                    Button(role: .destructive) {
                        showDeleteAlert = true
                    } label: {
                        Label("Delete", systemImage: "trash")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                } else if let isDownloading = modelManager.isDownloading[model.id], isDownloading {
                    Button {
                        modelManager.cancelDownload(for: model.id)
                    } label: {
                        Label("Cancel", systemImage: "xmark")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                } else if isComingSoon {
                    // Coming Soon - show disabled badge
                    HStack(spacing: 6) {
                        Image(systemName: "clock")
                            .foregroundStyle(.purple)
                        Text("Coming Soon")
                            .foregroundStyle(.purple)
                    }
                    .font(.caption)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background {
                        Capsule()
                            .fill(Color.purple.opacity(0.15))
                            .overlay {
                                Capsule()
                                    .strokeBorder(Color.purple.opacity(0.3), lineWidth: 1)
                            }
                    }
                } else if model.downloadURL != nil {
                    // Has download URL - show download button
                    Button {
                        Task {
                            try? await modelManager.downloadModel(model)
                        }
                    } label: {
                        HStack {
                            Image(systemName: "arrow.down.circle")
                            Text("Download")
                            if let size = model.fileSizeFormatted {
                                Text("(\(size))")
                            }
                        }
                        .font(.caption)
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    // No download URL and not available - unavailable
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle")
                            .foregroundStyle(.orange)
                        Text("Unavailable")
                            .foregroundStyle(.secondary)
                    }
                    .font(.caption)
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

// MARK: - Reasoning Mode Settings View

struct ReasoningModeSettings: View {
    @Binding var isEnabled: Bool
    @Binding var isSmartEnabled: Bool
    let reasoningAvailable: Bool
    let thinkingModeInfo: ModelBackendBridge.ThinkingModeInfo
    var onDismiss: () -> Void

    @AppStorage("selectedLLMBackend") private var selectedBackend: String = "foundationModels"

    var body: some View {
        Form {
            // Show unavailability warning if reasoning is not available
            if !reasoningAvailable {
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        Label {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Reasoning Mode Unavailable")
                                    .font(.headline)
                                    .foregroundStyle(.primary)

                                Text(thinkingModeInfo.description)
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }
                        } icon: {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                                .font(.title2)
                        }

                        Divider()

                        Text(thinkingModeInfo.recommendation)
                            .font(.callout)
                            .foregroundStyle(.secondary)

                        if selectedBackend == "foundationModels" {
                            Button {
                                selectedBackend = "customCoreML"
                                onDismiss()
                            } label: {
                                Label("Switch to Custom Models", systemImage: "arrow.right.circle.fill")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                            }
                            .buttonStyle(.borderedProminent)
                            .padding(.top, 8)
                        }
                    }
                    .padding(.vertical, 4)
                } header: {
                    Text("Status")
                }
            } else {
                // Original reasoning settings when available
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        Toggle(String(localized: "Always Use Reasoning"), isOn: $isEnabled)
                            .tint(.blue)
                            .disabled(isSmartEnabled)
                            .accessibilityLabel(String(localized: "Always Use Reasoning"))
                            .accessibilityHint(isSmartEnabled ? String(localized: "Disabled when Smart Reasoning Mode is enabled") : String(localized: "The AI will always show its reasoning process"))

                        Toggle(String(localized: "Smart Reasoning Mode"), isOn: $isSmartEnabled)
                            .padding(.top, 6)
                            .tint(.purple)
                            .disabled(isEnabled)
                            .accessibilityLabel(String(localized: "Smart Reasoning Mode"))
                            .accessibilityHint(isEnabled ? String(localized: "Disabled when Always Use Reasoning is enabled") : String(localized: "The AI will automatically decide when to show reasoning"))
                    }
                } header: {
                    HStack(spacing: 8) {
                        Image(systemName: "brain.head.profile")
                            .foregroundStyle(.blue)
                        Text(String(localized: "Reasoning Mode"))
                    }
                    .font(.headline)
                    .textCase(nil)
                } footer: {
                    if isEnabled {
                        Text(String(localized: "The AI will always show its reasoning process before providing the final answer."))
                            .font(.footnote)
                    } else if isSmartEnabled {
                        Text(String(localized: "The AI will automatically decide whether to show reasoning based on the complexity of your question."))
                            .font(.footnote)
                    } else {
                        Text(String(localized: "The AI will provide direct answers without showing its reasoning process."))
                            .font(.footnote)
                    }
                }

                Section {
                    if isEnabled {
                        VStack(alignment: .leading, spacing: 12) {
                            Label(String(localized: "Always shows thinking process"), systemImage: "brain")
                                .foregroundStyle(.blue)
                            Label(String(localized: "Explains decision making"), systemImage: "text.bubble")
                                .foregroundStyle(.blue)
                            Label(String(localized: "Helps verify answers"), systemImage: "checkmark.circle")
                                .foregroundStyle(.blue)
                            Label(String(localized: "May take longer to respond"), systemImage: "clock")
                                .foregroundStyle(.orange)
                        }
                        .font(.subheadline)
                    } else if isSmartEnabled {
                        VStack(alignment: .leading, spacing: 12) {
                            Label(String(localized: "Intelligently uses reasoning"), systemImage: "sparkles")
                                .foregroundStyle(.purple)
                            Label(String(localized: "Shows reasoning for complex tasks"), systemImage: "brain")
                                .foregroundStyle(.purple)
                            Label(String(localized: "Direct answers for simple questions"), systemImage: "message")
                                .foregroundStyle(.purple)
                            Label(String(localized: "Optimizes response speed"), systemImage: "speedometer")
                                .foregroundStyle(.green)
                        }
                        .font(.subheadline)
                    } else {
                        VStack(alignment: .leading, spacing: 12) {
                            Label(String(localized: "Fast direct responses"), systemImage: "bolt")
                                .foregroundStyle(.green)
                            Label(String(localized: "No reasoning shown"), systemImage: "eye.slash")
                                .foregroundStyle(.secondary)
                            Label(String(localized: "Suitable for quick questions"), systemImage: "message")
                                .foregroundStyle(.secondary)
                        }
                        .font(.subheadline)
                    }
                } header: {
                    Text(String(localized: "Features"))
                        .textCase(nil)
                }
            } // End of reasoningAvailable else block
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button(String(localized: "Done")) {
                    onDismiss()
                }
                .fontWeight(.semibold)
            }
        }
    }
}
