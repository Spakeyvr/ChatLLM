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
                selectedBackend = "foundationModels"
            } label: {
                HStack {
                    Text("Apple Foundation")
                    if selectedBackend == "foundationModels" {
                        Image(systemName: "checkmark")
                    }
                }
            }

            // Qwen 3 Instruct option
            let bridge = ModelBackendBridge.shared
            if let qwenModel = bridge.modelManager?.availableModels.first(where: { $0.isAvailable }) {
                Button {
                    selectedBackend = "customCoreML"
                    bridge.selectBackend(.customCoreML)
                    bridge.selectModel(qwenModel.id)
                    // selectBackend creates the canonical manager and calls startLoading().
                    // No direct loadModel call — that would create a second concurrent load.
                } label: {
                    HStack {
                        Text(qwenModel.name)
                        if selectedBackend == "customCoreML" && bridge.modelManager?.currentModel?.id == qwenModel.id {
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
        if selectedBackend == "customCoreML",
           let model = ModelBackendBridge.shared.modelManager?.currentModel {
            return model.name
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
