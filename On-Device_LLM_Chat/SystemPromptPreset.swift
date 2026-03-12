//
//  SystemPromptPreset.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI
import Foundation

// MARK: - Model

struct SystemPromptPreset: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var name: String
    var text: String
    var createdAt: Date = Date()
}

// MARK: - Preset Editor View

struct PresetEditorView: View {
    @State var preset: SystemPromptPreset
    var onCancel: () -> Void
    var onSave: (SystemPromptPreset) -> Void

    @State private var isPresetTextFocused: Bool = false
    @AppStorage("useMonospacedEditors") private var useMonospacedEditors: Bool = true

    var body: some View {
        Form {
            Section(header: Text(String(localized: "Name"))) {
                TextField(String(localized: "Preset name"), text: $preset.name)
            }
            Section(header: Text(String(localized: "Text"))) {
                NativePromptEditor(
                    text: $preset.text,
                    placeholder: String(localized: "Write the system prompt text for this preset…"),
                    isFocused: $isPresetTextFocused,
                    minHeight: 160,
                    maxHeight: 280,
                    isMonospaced: useMonospacedEditors
                )
            }
        }
        .navigationTitle(preset.name.isEmpty ? String(localized: "New Preset") : String(localized: "Edit Preset"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button(String(localized: "Cancel"), action: onCancel)
                    .buttonStyle(.glass)
            }
            ToolbarItem(placement: .confirmationAction) {
                Button(String(localized: "Save")) {
                    var toSave = preset
                    if toSave.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        toSave.name = String(localized: "Untitled Preset")
                    }
                    onSave(toSave)
                }
                .disabled(preset.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .buttonStyle(.glass)
            }
        }
    }
}
