//
//  ModelSelectionView.swift
//  ChatLLM
//
//  UI for selecting and managing MLX language models
//

import SwiftUI

struct ModelSelectionView: View {
    @ObservedObject private var modelBackendBridge = ModelBackendBridge.shared
    @Environment(\.dismiss) private var dismiss

    @State private var selectedModelForInfo: MLXModelManager.MLXModelInfo?

    private var modelManager: MLXModelManager? { modelBackendBridge.modelManager }

    var body: some View {
        NavigationStack {
            Group {
                if let modelManager {
                    listContent(modelManager: modelManager)
                } else {
                    LoadingIndicator()
                }
            }
                .navigationTitle("Language Models")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Done") {
                            dismiss()
                        }
                    }
                }
                .alert("Model Error", isPresented: errorBinding) {
                    Button("OK") {
                        modelManager?.loadError = nil
                    }
                } message: {
                    if let error = modelManager?.loadError {
                        Text(error)
                    }
                }
                .sheet(item: $selectedModelForInfo) { model in
                    ModelInfoSheet(
                        model: model,
                        isSupportedOnDevice: modelManager?.availabilityIssue(for: model) == nil
                    )
                }
        }
    }

    private func listContent(modelManager: MLXModelManager) -> some View {
        List {
            availableModelsSection(modelManager: modelManager)
        }
    }

    private func availableModelsSection(modelManager: MLXModelManager) -> some View {
        Section("Available Models") {
            ForEach(modelManager.availableModels) { model in
                modelRow(for: model, modelManager: modelManager)
            }
        }
    }

    private func modelRow(for model: MLXModelManager.MLXModelInfo, modelManager: MLXModelManager) -> some View {
        ModelRow(
            model: model,
            modelManager: modelManager
        ) {
            selectedModelForInfo = model
        } onDelete: {
            deleteModel(model)
        }
    }

    private var errorBinding: Binding<Bool> {
        Binding(
            get: { modelManager?.loadError != nil },
            set: { _ in }
        )
    }

    private func deleteModel(_ model: MLXModelManager.MLXModelInfo) {
        guard let modelManager else { return }
        Task {
            do {
                try modelBackendBridge.deleteModel(model)
            } catch {
                modelManager.loadError = String(localized: "Delete failed: \(error.localizedDescription)", bundle: .appLocalized)
            }
        }
    }
}

// MARK: - Model Row

struct ModelRow: View {
    let model: MLXModelManager.MLXModelInfo
    @ObservedObject var modelManager: MLXModelManager
    let onInfo: () -> Void
    let onDelete: () -> Void

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @Environment(\.locale) private var locale
    @State private var showDeleteConfirmation = false

    private var availabilityIssue: String? {
        modelManager.availabilityIssue(for: model)
    }

    private var isDownloadingThisModel: Bool {
        modelManager.isDownloading(modelID: model.id)
    }

    private var anotherDownloadInProgress: Bool {
        modelManager.hasActiveDownload(excluding: model.id)
    }

    var body: some View {
        Group {
            if dynamicTypeSize.isAccessibilitySize {
                // Side by side, the text column shrinks to a word or two per line.
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 12) {
                        modelIcon
                        Spacer()
                        actions
                    }
                    modelInfo
                }
            } else {
                HStack(spacing: 12) {
                    modelIcon
                    modelInfo
                    Spacer()
                    actions
                }
            }
        }
        .padding(.vertical, 4)
        .alert("Delete Model?", isPresented: $showDeleteConfirmation) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive, action: onDelete)
        } message: {
            Text("This will permanently delete \(model.name) \(model.parameterCount) from your device.")
        }
    }

    private var modelInfo: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(model.name) \(model.parameterCount)")
                .font(.headline)

            // Short, but translations run longer than English; don't cut them off.
            Text(model.description)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if model.isAvailable {
                // A Label in a List row snaps its icon to the list's icon column.
                HStack(spacing: 4) {
                    Image(systemName: "checkmark.circle.fill")
                    Text("Downloaded")
                }
                .font(.caption2)
                .foregroundStyle(.green)
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("modelSelection.downloaded.\(model.id)")
            } else {
                if isDownloadingThisModel {
                    VStack(alignment: .leading, spacing: 2) {
                        ProgressView(value: modelManager.downloadProgress)
                            .frame(maxWidth: 180)
                            .accessibilityIdentifier("modelSelection.progress.\(model.id)")
                        HStack(spacing: 8) {
                            Text(modelManager.downloadProgress, format: .percent.precision(.fractionLength(0)))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            Button("Cancel") {
                                modelManager.cancelDownload()
                            }
                            .font(.caption2)
                            .foregroundStyle(.red)
                            .buttonStyle(.plain)
                        }
                    }
                } else if let availabilityIssue {
                    Text(availabilityIssue)
                        .font(.caption2)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    VStack(alignment: .leading, spacing: 2) {
                        Button("Download (\(model.downloadSize(locale: locale)))") {
                            modelManager.startDownload(for: model)
                        }
                        .font(.caption2)
                        .buttonStyle(.borderedProminent)
                        .disabled(anotherDownloadInProgress)
                        .accessibilityIdentifier("modelSelection.download.\(model.id)")
                        if anotherDownloadInProgress, let activeModel = modelManager.activeDownloadModel {
                            Text("\(activeModel.displayName) is downloading")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        } else if let error = modelManager.downloadError(for: model.id) {
                            Text(error)
                                .font(.caption2)
                                .foregroundStyle(.red)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
        }
    }

    private var actions: some View {
        HStack(spacing: 12) {
            if model.isAvailable {
                Button {
                    showDeleteConfirmation = true
                } label: {
                    Image(systemName: "trash.fill")
                        .foregroundStyle(.red)
                        .imageScale(.large)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Delete")
            }

            // Info Button
            Button(action: onInfo) {
                Image(systemName: "info.circle.fill")
                    .foregroundStyle(.blue)
                    .imageScale(.large)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Model Information")
        }
    }

    @ViewBuilder
    private var modelIcon: some View {
        if model.id.hasPrefix("qwen") {
            Image("Qwen-logo")
                .resizable()
                .scaledToFill()
                .frame(width: 44, height: 44)
                .clipShape(Circle())
        } else {
            Image(systemName: "brain.head.profile")
                .font(.title2)
                .foregroundStyle(.white)
                .frame(width: 44, height: 44)
                .background(Color.accentColor, in: Circle())
        }
    }

}

// MARK: - Model Info Sheet

struct ModelInfoSheet: View {
    let model: MLXModelManager.MLXModelInfo
    var isSupportedOnDevice = true
    @Environment(\.dismiss) private var dismiss
    @Environment(\.locale) private var locale

    private var statusText: String {
        Self.statusText(isAvailable: model.isAvailable, isSupportedOnDevice: isSupportedOnDevice)
    }

    static func statusText(isAvailable: Bool, isSupportedOnDevice: Bool) -> String {
        if isAvailable {
            return String(localized: "Downloaded", bundle: .appLocalized)
        }
        if !isSupportedOnDevice {
            return String(localized: "Not supported on this device", bundle: .appLocalized)
        }
        return String(localized: "Not downloaded", bundle: .appLocalized)
    }

    var body: some View {
        NavigationStack {
            List {
                Section("Overview") {
                    InfoRow(label: "Name", value: model.name)
                    InfoRow(label: "Parameters", value: model.parameters)
                    InfoRow(label: "Context Length", value: String(localized: "\(model.contextLength) tokens", bundle: .appLocalized, locale: locale))
                    InfoRow(label: "Status", value: statusText)
                    InfoRow(label: "Reasoning", value: model.supportsReasoning ? String(localized: "Supported", bundle: .appLocalized) : String(localized: "Not supported", bundle: .appLocalized))
                    InfoRow(label: "Package", value: model.loadPolicy.packageDescription)
                }

                Section("Description") {
                    Text(model.description)
                        .font(.body)
                }

                Section("Technical Details") {
                    InfoRow(label: "Model ID", value: model.id)
                    InfoRow(label: "HF Repository", value: model.hfRepoId)
                    InfoRow(label: "Directory", value: model.localDirName)
                    if let architectureHint = model.loadPolicy.architectureHint {
                        InfoRow(label: "Architecture", value: architectureHint)
                    }
                }
            }
            .navigationTitle("Model Information")
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

struct InfoRow: View {
    let label: LocalizedStringKey
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .foregroundStyle(.primary)
        }
    }
}

// MARK: - Loading Overlay

// MARK: - Previews

#Preview {
    ModelSelectionView()
}

#Preview("Model Row - Available") {
    @Previewable @State var showingInfo = false
    let mgr = MLXModelManager()

    List {
        ModelRow(
            model: MLXModelManager.MLXModelInfo(
                id: "qwen3.5-4b-4bit-hybrid",
                name: "Qwen 3.5",
                localDirName: "Qwen3.5-4B-MLX-4bit-hybrid",
                hfRepoId: "Spakie/Qwen3.5-4B-MLX-4bit-hybrid",
                parameters: "4B (4-bit hybrid)",
                downloadSizeLabel: "2.66 GB",
                loadPolicy: .qwenMultimodal,
                description: "Qwen 3.5 4B multimodal model with native reasoning and vision.",
                contextLength: 262144,
                isAvailable: true,
                supportsReasoning: true
            ),
            modelManager: mgr,
            onInfo: { showingInfo = true },
            onDelete: {}
        )
    }
    .sheet(isPresented: $showingInfo) {
        ModelInfoSheet(
            model: MLXModelManager.MLXModelInfo(
                id: "qwen3.5-4b-4bit-hybrid",
                name: "Qwen 3.5",
                localDirName: "Qwen3.5-4B-MLX-4bit-hybrid",
                hfRepoId: "Spakie/Qwen3.5-4B-MLX-4bit-hybrid",
                parameters: "4B (4-bit hybrid)",
                downloadSizeLabel: "2.66 GB",
                loadPolicy: .qwenMultimodal,
                description: "Qwen 3.5 4B multimodal model with native reasoning and vision.",
                contextLength: 262144,
                isAvailable: true,
                supportsReasoning: true
            )
        )
    }
}

#Preview("Model Row - Not Available") {
    let mgr = MLXModelManager()

    List {
        ModelRow(
            model: MLXModelManager.MLXModelInfo(
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
                supportsReasoning: true
            ),
            modelManager: mgr,
            onInfo: {},
            onDelete: {}
        )
    }
}
