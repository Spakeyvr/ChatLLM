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
                    ProgressView()
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
                .overlay {
                    if modelManager?.isLoading == true {
                        LoadingOverlay()
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
            currentModelSection(modelManager: modelManager)
            availableModelsSection(modelManager: modelManager)
            diagnosticsSection(modelManager: modelManager)
        }
    }

    @ViewBuilder
    private func currentModelSection(modelManager: MLXModelManager) -> some View {
        if let currentModel = modelManager.currentModel {
            Section {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Current Model")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text(currentModel.displayName)
                            .font(.headline)
                    }

                    Spacer()

                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }
                .padding(.vertical, 4)
            }
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
            modelManager: modelManager,
            isSelected: modelManager.currentModel?.id == model.id,
            isLoading: modelManager.isLoading && modelManager.currentModel?.id != model.id
        ) {
            selectModel(model)
        } onInfo: {
            selectedModelForInfo = model
            showingModelInfo = true
        } onDelete: {
            deleteModel(model)
        }
    }

    private func diagnosticsSection(modelManager: MLXModelManager) -> some View {
        Section {
            Button {
                modelManager.refreshModelAvailability()
            } label: {
                Label("Refresh Model Availability", systemImage: "arrow.clockwise")
            }

            Button {
                modelManager.printModelLocations()
            } label: {
                Label("Print Model Locations", systemImage: "folder")
            }

            Button {
                modelManager.printModelInputOutputInfo()
            } label: {
                Label("Print Model Info to Console", systemImage: "info.circle")
            }
            .disabled(modelManager.currentModel == nil)

            Button(role: .destructive) {
                modelManager.unloadAllModels()
            } label: {
                Label("Unload All Models", systemImage: "trash")
            }
            .disabled(modelManager.currentModel == nil)
        } header: {
            Text("Diagnostics")
        } footer: {
            Text("Use these tools to debug model issues")
        }
    }

    private var errorBinding: Binding<Bool> {
        Binding(
            get: { modelManager?.loadError != nil },
            set: { _ in }
        )
    }

    private func selectModel(_ model: MLXModelManager.MLXModelInfo) {
        guard model.isAvailable else { return }
        modelBackendBridge.switchToMLXModel(model.id, source: "model-selection-sheet")
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
    let isSelected: Bool
    let isLoading: Bool
    let onSelect: () -> Void
    let onInfo: () -> Void
    let onDelete: () -> Void

    private var availabilityIssue: String? {
        modelManager.availabilityIssue(for: model)
    }

    private var toolCallIssue: String? {
        modelManager.toolCallIssue(for: model)
    }

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: 12) {
                // Model Icon
                ZStack {
                    Circle()
                        .fill(iconBackgroundColor)
                        .frame(width: 44, height: 44)

                    Image(systemName: iconName)
                        .font(.system(size: 20))
                        .foregroundStyle(iconForegroundColor)
                }

                // Model Info
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.name)
                        .font(.headline)
                        .foregroundStyle(model.isAvailable ? .primary : .secondary)

                    Text(model.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)

                    HStack(spacing: 8) {
                        Label(model.parameters, systemImage: "cpu")
                            .font(.caption2)
                            .foregroundStyle(.secondary)

                        Label("\(model.contextLength) tokens", systemImage: "doc.text")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    Label(model.loadPolicy.packageDescription, systemImage: "square.stack.3d.up.fill")
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    if let toolCallIssue {
                        Text(toolCallIssue)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                    }

                    if !model.isAvailable {
                        if modelManager.isDownloading {
                            VStack(alignment: .leading, spacing: 2) {
                                ProgressView(value: modelManager.downloadProgress)
                                    .frame(maxWidth: 180)
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

                Spacer()

                // Status and Actions
                HStack(spacing: 12) {
                    if model.isAvailable {
                        Menu {
                            Button(role: .destructive, action: onDelete) {
                                Label("Delete", systemImage: "trash")
                            }
                        } label: {
                            Image(systemName: "ellipsis.circle.fill")
                                .foregroundStyle(.secondary)
                                .imageScale(.large)
                        }
                        .buttonStyle(.plain)
                    }

                    // Status Indicator
                    if isLoading {
                        ProgressView()
                    } else if isSelected {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    } else if !model.isAvailable {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                    } else {
                        Image(systemName: "circle")
                            .foregroundStyle(.secondary.opacity(0.3))
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
        }
        .disabled(!model.isAvailable || isLoading)
    }

    private var iconName: String {
        if !model.isAvailable {
            return "exclamationmark.triangle"
        } else {
            return "brain.head.profile"
        }
    }

    private var iconBackgroundColor: Color {
        if !model.isAvailable {
            return .orange.opacity(0.2)
        } else if isSelected {
            return .green.opacity(0.2)
        } else {
            return .blue.opacity(0.2)
        }
    }

    private var iconForegroundColor: Color {
        if !model.isAvailable {
            return .orange
        } else if isSelected {
            return .green
        } else {
            return .blue
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
                    InfoRow(label: "Directory", value: model.localDirName)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Location")
                            .foregroundStyle(.secondary)
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

struct LoadingOverlay: View {
    var body: some View {
        ZStack {
            Color.black.opacity(0.3)
                .ignoresSafeArea()

            VStack(spacing: 16) {
                ProgressView()
                    .scaleEffect(1.5)

                Text("Loading Model...")
                    .font(.headline)
                    .foregroundStyle(.white)
            }
            .padding(32)
            .background {
                RoundedRectangle(cornerRadius: 16)
                    .fill(.ultraThinMaterial)
            }
        }
    }
}

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
                id: "qwen3.5-4b-4bit",
                name: "Qwen 3.5",
                localDirName: "Qwen3.5-4B-MLX-4bit",
                hfRepoId: "mlx-community/Qwen3.5-4B-MLX-4bit",
                parameters: "4B (4-bit)",
                downloadSizeLabel: "3.03 GB",
                loadPolicy: .qwenMultimodal,
                description: "Qwen 3.5 4B multimodal model with native reasoning and image support",
                contextLength: 262144,
                isAvailable: true,
                supportsReasoning: true
            ),
            modelManager: mgr,
            isSelected: false,
            isLoading: false,
            onSelect: {},
            onInfo: { showingInfo = true },
            onDelete: {}
        )
    }
    .sheet(isPresented: $showingInfo) {
        ModelInfoSheet(
            model: MLXModelManager.MLXModelInfo(
                id: "qwen3.5-4b-4bit",
                name: "Qwen 3.5",
                localDirName: "Qwen3.5-4B-MLX-4bit",
                hfRepoId: "mlx-community/Qwen3.5-4B-MLX-4bit",
                parameters: "4B (4-bit)",
                downloadSizeLabel: "3.03 GB",
                loadPolicy: .qwenMultimodal,
                description: "Qwen 3.5 4B multimodal model with native reasoning and image support",
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
                description: "Qwen 3.5 2B multimodal model with native reasoning and image support",
                contextLength: 262144,
                isAvailable: false,
                supportsReasoning: true
            ),
            modelManager: mgr,
            isSelected: false,
            isLoading: false,
            onSelect: {},
            onInfo: {},
            onDelete: {}
        )
    }
}

#Preview("Model Row - Selected") {
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
                description: "Qwen 3.5 2B multimodal model with native reasoning and image support",
                contextLength: 262144,
                isAvailable: true,
                supportsReasoning: true
            ),
            modelManager: mgr,
            isSelected: true,
            isLoading: false,
            onSelect: {},
            onInfo: {},
            onDelete: {}
        )
    }
}
