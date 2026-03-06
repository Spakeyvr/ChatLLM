//
//  ModelPickerView.swift
//  On-Device_LLM_Chat
//
//  Simple inline model picker for chat navigation bar
//

import SwiftUI

/// Model picker that appears in the navigation bar
struct ModelPickerView: View {
    @AppStorage("selectedLLMBackend") private var selectedBackend: String = "foundationModels"
    @State private var showingModelManagement = false

    var body: some View {
        Menu {
            Button {
                ModelBackendBridge.shared.selectBackend(.foundationModels, source: "nav-picker.foundation")
            } label: {
                HStack {
                    Text("Apple Foundation")
                    if selectedBackend == "foundationModels" {
                        Image(systemName: "checkmark")
                    }
                }
            }

            let bridge = ModelBackendBridge.shared
            ForEach(bridge.modelManager?.availableModels.filter(\.isAvailable) ?? [], id: \.id) { model in
                Button {
                    bridge.selectModel(model.id, source: "nav-picker.model")
                    bridge.selectBackend(.mlx, source: "nav-picker.model")
                } label: {
                    HStack {
                        Text("\(model.name) (\(model.parameters))")
                        if selectedBackend == "mlx" && bridge.selectedModelID == model.id {
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
        if selectedBackend == "mlx",
           let model = ModelBackendBridge.shared.modelManager?.currentModel {
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
