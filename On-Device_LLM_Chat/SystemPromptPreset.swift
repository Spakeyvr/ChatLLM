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

    // Dynamic height for the preset editor
    @State private var presetEditorHeight: CGFloat = 160
    @AppStorage("useMonospacedEditors") private var useMonospacedEditors: Bool = true

    var body: some View {
        Form {
            Section(header: Text(String(localized: "Name"))) {
                TextField(String(localized: "Preset name"), text: $preset.name)
            }
            Section(header: Text(String(localized: "Text"))) {
                GeometryReader { geometry in
                    ZStack(alignment: .topLeading) {
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(.ultraThinMaterial)
                            .overlay(
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .strokeBorder(Color.secondary.opacity(0.15))
                            )

                        AutoSizingTextEditor(
                            text: $preset.text,
                            minHeight: 160,
                            dynamicHeight: $presetEditorHeight,
                            availableWidth: geometry.size.width,
                            isMonospaced: useMonospacedEditors,
                            font: useMonospacedEditors ? .body.monospaced() : .body
                        )
                        .padding(10)
                        .frame(height: max(160, presetEditorHeight))
                        .background(Color.clear)

                        if preset.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Text(String(localized: "Write the system prompt text for this preset…"))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 14)
                        }
                    }
                }
                .frame(minHeight: 160)
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
