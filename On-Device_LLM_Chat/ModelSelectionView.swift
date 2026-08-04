//
//  ModelSelectionView.swift
//  On-Device_LLM_Chat
//
//  UI for selecting and managing MLX language models
//

import SwiftUI

struct ModelSelectionView: View {
    @ObservedObject private var modelBackendBridge = ModelBackendBridge.shared
    @Environment(\.dismiss) private var dismiss

    @State private var showingModelInfo = false
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
                .sheet(isPresented: $showingModelInfo) {
                    if let model = selectedModelForInfo {
                        ModelInfoSheet(model: model)
                    }
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
            showingModelInfo = true
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
                try modelManager.deleteModel(model)
            } catch {
                modelManager.loadError = "Delete failed: \(error.localizedDescription)"
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

    @State private var showDeleteConfirmation = false

    private var availabilityIssue: String? {
        modelManager.availabilityIssue(for: model)
    }

    private var toolCallIssue: String? {
        modelManager.toolCallIssue(for: model)
    }

    private var isDownloadingThisModel: Bool {
        modelManager.isDownloading(modelID: model.id)
    }

    private var anotherDownloadInProgress: Bool {
        modelManager.hasActiveDownload(excluding: model.id)
    }

    var body: some View {
        HStack(spacing: 12) {
                modelIcon

                // Model Info
                VStack(alignment: .leading, spacing: 4) {
                    Text("\(model.name) \(model.parameterCount)")
                        .font(.headline)
                        .foregroundStyle(model.isAvailable ? .primary : .secondary)

                    Text(model.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)

                    if !model.isAvailable {
                        if isDownloadingThisModel {
                            VStack(alignment: .leading, spacing: 2) {
                                ProgressView(value: modelManager.downloadProgress)
                                    .frame(maxWidth: 180)
                                    .accessibilityIdentifier("modelSelection.progress.\(model.id)")
                                HStack(spacing: 8) {
                                    Text("\(Int(modelManager.downloadProgress * 100))%")
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
                                .lineLimit(3)
                        } else {
                            VStack(alignment: .leading, spacing: 2) {
                                Button("Download (\(model.downloadSizeLabel))") {
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
                                        .lineLimit(2)
                                }
                            }
                        }
                    }
                }

                Spacer()

                // Status and Actions
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
                    }

                    // Info Button
                    Button(action: onInfo) {
                        Image(systemName: "info.circle.fill")
                            .foregroundStyle(.blue)
                            .imageScale(.large)
                    }
                    .buttonStyle(.plain)
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
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Overview") {
                    InfoRow(label: "Name", value: model.name)
                    InfoRow(label: "Parameters", value: model.parameters)
                    InfoRow(label: "Context Length", value: "\(model.contextLength) tokens")
                    InfoRow(label: "Status", value: model.isAvailable ? "Available" : "Not Available")
                    InfoRow(label: "Reasoning", value: model.supportsReasoning ? "Supported" : "Not supported")
                    InfoRow(label: "Package", value: model.loadPolicy.packageDescription)
                    if let architectureHint = model.loadPolicy.architectureHint {
                        InfoRow(label: "Architecture", value: architectureHint)
                    }
                }

                Section("Description") {
                    Text(model.description)
                        .font(.body)
                }

                Section("Technical Details") {
                    InfoRow(label: "Model ID", value: model.id)
                    InfoRow(label: "HF Repository", value: model.hfRepoId)
                    InfoRow(label: "Directory", value: model.localDirName)
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
    let label: String
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
