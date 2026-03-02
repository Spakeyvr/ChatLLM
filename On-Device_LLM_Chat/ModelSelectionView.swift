//
//  ModelSelectionView.swift
//  On-Device_LLM_Chat
//
//  Created by Xcode Assistant
//  UI for selecting and managing LLM models
//

import SwiftUI

struct ModelSelectionView: View {
    @StateObject private var modelManager = LLMModelManager()
    @Environment(\.dismiss) private var dismiss
    
    @State private var showingModelInfo = false
    @State private var selectedModelForInfo: LLMModelManager.LLMModelInfo?
    @State private var showingDownloadConfirmation = false
    @State private var modelToDownload: LLMModelManager.LLMModelInfo?
    
    var body: some View {
        NavigationStack {
            listContent
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
                    if modelManager.isLoading {
                        LoadingOverlay()
                    }
                }
                .alert("Model Error", isPresented: errorBinding) {
                    Button("OK") {
                        modelManager.loadError = nil
                    }
                } message: {
                    if let error = modelManager.loadError {
                        Text(error)
                    }
                }
                .sheet(isPresented: $showingModelInfo) {
                    if let model = selectedModelForInfo {
                        ModelInfoSheet(model: model)
                    }
                }
                .confirmationDialog(
                    "Download Model?",
                    isPresented: $showingDownloadConfirmation,
                    presenting: modelToDownload
                ) { model in
                    Button("Download \(model.fileSizeFormatted ?? "Large File")") {
                        downloadModel(model)
                    }
                    Button("Cancel", role: .cancel) { }
                } message: { model in
                    Text("This will download \(model.fileSizeFormatted ?? "a large file"). Make sure you have enough storage and are connected to Wi-Fi.")
                }
        }
    }
    
    private var listContent: some View {
        List {
            currentModelSection
            availableModelsSection
            diagnosticsSection
        }
    }
    
    @ViewBuilder
    private var currentModelSection: some View {
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
    
    private var availableModelsSection: some View {
        Section("Available Models") {
            ForEach(modelManager.availableModels) { model in
                modelRow(for: model)
            }
        }
    }
    
    private func modelRow(for model: LLMModelManager.LLMModelInfo) -> some View {
        ModelRow(
            model: model,
            isSelected: modelManager.currentModel?.id == model.id,
            isLoading: modelManager.isLoading && modelManager.currentModel?.id != model.id,
            isDownloading: modelManager.isDownloading[model.id] ?? false,
            downloadProgress: modelManager.downloadProgress[model.id] ?? 0.0,
            downloadStatus: modelManager.downloadStatus[model.id] ?? .idle,
            isComingSoon: false // All models are available
        ) {
            selectModel(model)
        } onInfo: {
            selectedModelForInfo = model
            showingModelInfo = true
        } onDownload: {
            promptDownload(model)
        } onDelete: {
            deleteModel(model)
        }
    }
    
    private var diagnosticsSection: some View {
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
            get: { modelManager.loadError != nil },
            set: { _ in }
        )
    }
    
    private func selectModel(_ model: LLMModelManager.LLMModelInfo) {
        guard model.isAvailable else { return }
        Task {
            await modelManager.loadModel(model)
        }
    }
    
    private func downloadModel(_ model: LLMModelManager.LLMModelInfo) {
        Task {
            do {
                try await modelManager.downloadModel(model)
            } catch {
                modelManager.loadError = "Download failed: \(error.localizedDescription)"
            }
        }
    }
    
    private func promptDownload(_ model: LLMModelManager.LLMModelInfo) {
        modelToDownload = model
        showingDownloadConfirmation = true
    }
    
    private func deleteModel(_ model: LLMModelManager.LLMModelInfo) {
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
    let model: LLMModelManager.LLMModelInfo
    let isSelected: Bool
    let isLoading: Bool
    let isDownloading: Bool
    let downloadProgress: Double
    let downloadStatus: LLMModelManager.DownloadStatus
    let isComingSoon: Bool // New parameter for coming soon state
    let onSelect: () -> Void
    let onInfo: () -> Void
    let onDownload: () -> Void
    let onDelete: () -> Void
    
    var body: some View {
        VStack(spacing: 0) {
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
                            
                            if let fileSize = model.fileSizeFormatted {
                                Label(fileSize, systemImage: "arrow.down.circle")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    
                    Spacer()
                    
                    // Status and Actions
                    HStack(spacing: 12) {
                        // Download/Delete Button
                        if isComingSoon {
                            // Coming Soon indicator - can be customized for future models
                            HStack(spacing: 6) {
                                Image(systemName: "clock.fill")
                                    .foregroundStyle(.purple)
                                    .imageScale(.small)
                                Text("Coming Soon")
                                    .font(.caption)
                                    .fontWeight(.semibold)
                                    .foregroundStyle(.purple)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background {
                                Capsule()
                                    .fill(.purple.opacity(0.15))
                                    .overlay {
                                        Capsule()
                                            .strokeBorder(.purple.opacity(0.3), lineWidth: 1)
                                    }
                            }
                        } else if !model.isAvailable && model.downloadURL != nil && !isComingSoon {
                            Button(action: onDownload) {
                                Image(systemName: "arrow.down.circle.fill")
                                    .foregroundStyle(.blue)
                                    .imageScale(.large)
                            }
                            .buttonStyle(.plain)
                        } else if !model.isAvailable && model.downloadURL == nil && !isComingSoon {
                            // Model not available and no download URL - show unavailable state
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.gray)
                                .imageScale(.large)
                                .opacity(0.4)
                        } else if model.isAvailable {
                            Menu {
                                Button(role: .destructive, action: onDelete) {
                                    Label("Delete Download", systemImage: "trash")
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
                        } else if isComingSoon {
                            Image(systemName: "hourglass")
                                .foregroundStyle(.purple)
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
            .disabled((!model.isAvailable || isLoading || isComingSoon) && !isDownloading)
            
            // Download Progress Bar
            if isDownloading {
                VStack(spacing: 8) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(downloadStatus.message)
                                .font(.caption)
                                .foregroundStyle(.primary)
                            
                            if case .downloading = downloadStatus {
                                Text("\(Int(downloadProgress * 100))%")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        
                        Spacer()
                        
                        // Show spinner for extracting/initializing phases
                        if case .extracting = downloadStatus {
                            ProgressView()
                                .controlSize(.small)
                        } else if case .initializing = downloadStatus {
                            ProgressView()
                                .controlSize(.small)
                        }
                    }
                    
                    ProgressView(value: downloadProgress)
                        .progressViewStyle(.linear)
                        .tint(progressTint)
                }
                .padding(.top, 8)
                .padding(.horizontal, 4)
            }
            
            // Completed status
            if case .completed = downloadStatus {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .imageScale(.small)
                    Text("Download complete!")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
                .padding(.top, 8)
            }
            
            // Error status
            if case .failed(let error) = downloadStatus {
                HStack(alignment: .top, spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .imageScale(.small)
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .lineLimit(2)
                }
                .padding(.top, 8)
            }
        }
    }
    
    private var iconName: String {
        if isComingSoon {
            return "clock"
        } else if !model.isAvailable {
            return "exclamationmark.triangle"
        } else {
            return "brain.head.profile"
        }
    }
    
    private var iconBackgroundColor: Color {
        if isComingSoon {
            return .purple.opacity(0.2)
        } else if !model.isAvailable {
            return .orange.opacity(0.2)
        } else if isSelected {
            return .green.opacity(0.2)
        } else {
            return .blue.opacity(0.2)
        }
    }
    
    private var iconForegroundColor: Color {
        if isComingSoon {
            return .purple
        } else if !model.isAvailable {
            return .orange
        } else if isSelected {
            return .green
        } else {
            return .blue
        }
    }
    
    private var progressTint: Color {
        switch downloadStatus {
        case .downloading:
            return .blue
        case .extracting, .initializing:
            return .orange
        case .completed:
            return .green
        case .failed:
            return .red
        case .idle:
            return .blue
        }
    }
}

// MARK: - Model Info Sheet

struct ModelInfoSheet: View {
    let model: LLMModelManager.LLMModelInfo
    @Environment(\.dismiss) private var dismiss
    
    // Check if model is coming soon (none currently)
    private var isComingSoon: Bool {
        false // All models are available
    }
    
    var body: some View {
        NavigationStack {
            List {
                // Coming Soon Banner
                if isComingSoon {
                    Section {
                        HStack(spacing: 12) {
                            Image(systemName: "clock.fill")
                                .font(.title2)
                                .foregroundStyle(.purple)
                            
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Coming Soon")
                                    .font(.headline)
                                    .foregroundStyle(.purple)
                                Text("This model will be available for download in a future update")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.vertical, 8)
                    }
                }
                
                Section("Overview") {
                    InfoRow(label: "Name", value: model.name)
                    InfoRow(label: "Parameters", value: model.parameters)
                    InfoRow(label: "Context Length", value: "\(model.contextLength) tokens")
                    InfoRow(label: "Status", value: isComingSoon ? "Coming Soon" : (model.isAvailable ? "Available" : "Not Available"))
                    
                    if let fileSize = model.fileSizeFormatted {
                        InfoRow(label: "File Size", value: fileSize)
                    }
                }
                
                Section("Description") {
                    Text(model.description)
                        .font(.body)
                }
                
                Section("Technical Details") {
                    InfoRow(label: "Model ID", value: model.id)
                    InfoRow(label: "File Name", value: model.modelFileName)
                    
                    if let downloadURL = model.downloadURL, !isComingSoon {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Download URL")
                                .foregroundStyle(.secondary)
                            Link(destination: URL(string: downloadURL)!) {
                                Text(downloadURL)
                                    .font(.caption)
                                    .lineLimit(2)
                            }
                        }
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
    
    List {
        ModelRow(
            model: LLMModelManager.LLMModelInfo(
                id: "qwen3-4b-2bit",
                name: "Qwen 3 Instruct",
                modelFileName: "Qwen3-4B-2bit",
                description: "Alibaba's instruction-tuned model with thinking support",
                parameters: "4B (2-bit)",
                contextLength: 32768,
                isAvailable: true,
                downloadURL: nil,
                fileSize: 2_500_000_000,  // 2.5 GB in bytes
                supportsReasoning: true
            ),
            isSelected: false,
            isLoading: false,
            isDownloading: false,
            downloadProgress: 0.0,
            downloadStatus: .idle,
            isComingSoon: false,
            onSelect: {},
            onInfo: { showingInfo = true },
            onDownload: {},
            onDelete: {}
        )
    }
    .sheet(isPresented: $showingInfo) {
        ModelInfoSheet(
            model: LLMModelManager.LLMModelInfo(
                id: "qwen3-4b-2bit",
                name: "Qwen 3 Instruct",
                modelFileName: "Qwen3-4B-2bit",
                description: "Alibaba's instruction-tuned model with thinking support and 32K context",
                parameters: "4B (2-bit)",
                contextLength: 32768,
                isAvailable: true,
                downloadURL: nil,
                fileSize: 3_000_000_000,  // ~3.0 GB in bytes
                supportsReasoning: true
            )
        )
    }
}

#Preview("Model Row - Coming Soon") {
    List {
        ModelRow(
            model: LLMModelManager.LLMModelInfo(
                id: "qwen3-4b-2bit",
                name: "Qwen 3 Instruct",
                modelFileName: "Qwen3-4B-2bit",
                description: "Alibaba's instruction-tuned model with thinking support",
                parameters: "4B (2-bit)",
                contextLength: 32768,
                isAvailable: false,
                downloadURL: nil,
                fileSize: 2_500_000_000,  // 2.5 GB in bytes
                supportsReasoning: true
            ),
            isSelected: false,
            isLoading: false,
            isDownloading: false,
            downloadProgress: 0.0,
            downloadStatus: .idle,
            isComingSoon: true,
            onSelect: {},
            onInfo: {},
            onDownload: {},
            onDelete: {}
        )
    }
}

#Preview("Model Row - Selected") {
    List {
        ModelRow(
            model: LLMModelManager.LLMModelInfo(
                id: "qwen3-4b-2bit",
                name: "Qwen 3 Instruct",
                modelFileName: "Qwen3-4B-2bit",
                description: "Alibaba's instruction-tuned model with thinking support",
                parameters: "4B (2-bit)",
                contextLength: 32768,
                isAvailable: true,
                downloadURL: nil,
                fileSize: 2_500_000_000,  // 2.5 GB in bytes
                supportsReasoning: true
            ),
            isSelected: true,
            isLoading: false,
            isDownloading: false,
            downloadProgress: 0.0,
            downloadStatus: .idle,
            isComingSoon: false,
            onSelect: {},
            onInfo: {},
            onDownload: {},
            onDelete: {}
        )
    }
}

#Preview("Model Row - Not Available") {
    List {
        ModelRow(
            model: LLMModelManager.LLMModelInfo(
                id: "qwen3-4b-2bit",
                name: "Qwen 3 Instruct",
                modelFileName: "Qwen3-4B-2bit",
                description: "Alibaba's instruction-tuned model with thinking support",
                parameters: "4B (2-bit)",
                contextLength: 32768,
                isAvailable: false,
                downloadURL: nil,
                fileSize: 2_500_000_000,  // 2.5 GB in bytes
                supportsReasoning: true
            ),
            isSelected: false,
            isLoading: false,
            isDownloading: false,
            downloadProgress: 0.0,
            downloadStatus: .idle,
            isComingSoon: false,
            onSelect: {},
            onInfo: {},
            onDownload: {},
            onDelete: {}
        )
    }
}

#Preview("Model Row - Downloading") {
    List {
        ModelRow(
            model: LLMModelManager.LLMModelInfo(
                id: "qwen3-4b-2bit",
                name: "Qwen 3 Instruct",
                modelFileName: "Qwen3-4B-2bit",
                description: "Alibaba's instruction-tuned model with thinking support",
                parameters: "4B (2-bit)",
                contextLength: 32768,
                isAvailable: false,
                downloadURL: nil,
                fileSize: 2_500_000_000,  // 2.5 GB in bytes
                supportsReasoning: true
            ),
            isSelected: false,
            isLoading: false,
            isDownloading: true,
            downloadProgress: 0.45,
            downloadStatus: .downloading(bytesDownloaded: 1_125_000_000, totalBytes: 2_500_000_000),
            isComingSoon: false,
            onSelect: {},
            onInfo: {},
            onDownload: {},
            onDelete: {}
        )
    }
}

#Preview("Model Row - Extracting") {
    List {
        ModelRow(
            model: LLMModelManager.LLMModelInfo(
                id: "qwen3-4b-2bit",
                name: "Qwen 3 Instruct",
                modelFileName: "Qwen3-4B-2bit",
                description: "Alibaba's instruction-tuned model with thinking support",
                parameters: "4B (2-bit)",
                contextLength: 32768,
                isAvailable: false,
                downloadURL: nil,
                fileSize: 2_500_000_000,  // 2.5 GB in bytes
                supportsReasoning: true
            ),
            isSelected: false,
            isLoading: false,
            isDownloading: true,
            downloadProgress: 0.9,
            downloadStatus: .extracting,
            isComingSoon: false,
            onSelect: {},
            onInfo: {},
            onDownload: {},
            onDelete: {}
        )
    }
}

#Preview("Model Row - Initializing") {
    List {
        ModelRow(
            model: LLMModelManager.LLMModelInfo(
                id: "qwen3-4b-2bit",
                name: "Qwen 3 Instruct",
                modelFileName: "Qwen3-4B-2bit",
                description: "Alibaba's instruction-tuned model with thinking support",
                parameters: "4B (2-bit)",
                contextLength: 32768,
                isAvailable: false,
                downloadURL: nil,
                fileSize: 2_500_000_000,  // 2.5 GB in bytes
                supportsReasoning: true
            ),
            isSelected: false,
            isLoading: false,
            isDownloading: true,
            downloadProgress: 0.95,
            downloadStatus: .initializing,
            isComingSoon: false,
            onSelect: {},
            onInfo: {},
            onDownload: {},
            onDelete: {}
        )
    }
}
