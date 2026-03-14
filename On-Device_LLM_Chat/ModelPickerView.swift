//
//  ModelPickerView.swift
//  On-Device_LLM_Chat
//
//  Simple inline model picker for chat navigation bar
//

import SwiftUI

/// Model picker that appears in the navigation bar
struct ModelPickerView: View {
    @State private var showingModelManagement = false
    @ObservedObject private var bridge = ModelBackendBridge.shared

    var body: some View {
        Menu {
            Button {
                bridge.selectBackend(.foundationModels, source: "nav-picker.foundation")
            } label: {
                HStack {
                    Text("Apple Foundation")
                    if !bridge.foundationModelsAvailable {
                        Image(systemName: "exclamationmark.triangle")
                            .foregroundStyle(.orange)
                    }
                    if bridge.selectedBackend == .foundationModels {
                        Image(systemName: "checkmark")
                    }
                }
            }
            .disabled(!bridge.foundationModelsAvailable)

            ForEach(bridge.modelManager?.availableModels.filter(\.isAvailable) ?? [], id: \.id) { model in
                Button {
                    bridge.switchToMLXModel(model.id, source: "nav-picker.model")
                } label: {
                    HStack {
                        Text("\(model.name) (\(model.parameters))")
                        if bridge.selectedBackend == .mlx && bridge.selectedModelID == model.id {
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }

            Divider()

            Button {
                showingModelManagement = true
            } label: {
                Label("Manage models", systemImage: "gearshape")
            }
        } label: {
            HStack(spacing: 4) {
                Text(displayName)
                Image(systemName: "chevron.right")
                    .font(.caption)
            }
        }
        .sheet(isPresented: $showingModelManagement) {
            ModelSelectionView()
        }
    }

    private var displayName: String {
        if bridge.selectedBackend == .mlx,
           let model = bridge.modelManager?.currentModel {
            return "\(model.name) (\(model.parameters))"
        }
        return "Apple Foundation"
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        Text("Chat Content")
            .toolbar {
                ToolbarItem(placement: .principal) {
                    ModelPickerView()
                }
            }
    }
}
