import SwiftUI

struct ChatSettingsView: View {
    @Binding var settings: AppSettingsDraft
    @State private var showingPreferencesEditor = false

    var body: some View {
        Form {
            Section {
                Toggle("Send on Return", isOn: $settings.sendOnReturn)
                    .accessibilityIdentifier("settings.sendOnReturn")
            } footer: {
                Text("Use the Return key to send a message instead of adding a new line.")
            }
            Section {
                Toggle("Haptic Feedback", isOn: $settings.enableHaptics)
                    .accessibilityIdentifier("settings.haptics")
            } footer: {
                Text("Play subtle vibrations for actions on supported devices.")
            }
            Section {
                Toggle("Reasoning by Default", isOn: $settings.reasoningModeDefault)
                    .accessibilityIdentifier("settings.reasoning")
                Button {
                    showingPreferencesEditor = true
                } label: {
                    HStack {
                        LabeledContent {
                            Text(settings.chatPreferences.isEmpty
                                 ? String(localized: "Not Set") : String(localized: "Custom"))
                                .foregroundStyle(.secondary)
                        } label: {
                            Text("Response Preferences")
                        }
                        Image(systemName: "chevron.right")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(.tertiary)
                    }
                    .foregroundStyle(.primary)
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("settings.responsePreferences")
            } header: {
                Text("New Chats")
            } footer: {
                Text("Start new chats with reasoning enabled on supported models. Response preferences tell the assistant how you like answers written; existing chats keep their original preferences.")
            }
        }
        .navigationTitle("Chat")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showingPreferencesEditor) {
            ResponsePreferencesEditor(preferences: $settings.chatPreferences)
        }
    }
}

private struct ResponsePreferencesEditor: View {
    @Binding var preferences: String
    @State private var text: String
    @Environment(\.dismiss) private var dismiss

    init(preferences: Binding<String>) {
        _preferences = preferences
        _text = State(initialValue: preferences.wrappedValue)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextEditor(text: $text)
                        .frame(minHeight: 200)
                        .accessibilityLabel("Response Preferences")
                        .accessibilityIdentifier("settings.preferencesEditor")
                } header: {
                    Text("How should the assistant respond?")
                } footer: {
                    Text("For example: Keep answers concise, use metric units, and explain technical terms. Applies to new chats only.")
                }
                Section {
                    Button("Clear Preferences", role: .destructive) { text = "" }
                        .disabled(text.isEmpty)
                }
            }
            .scrollDismissesKeyboard(.interactively)
            .navigationTitle("Response Preferences")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        preferences = text.trimmingCharacters(in: .whitespacesAndNewlines)
                        dismiss()
                    }
                    .accessibilityIdentifier("settings.savePreferences")
                }
            }
        }
        .interactiveDismissDisabled(text != preferences)
    }
}
