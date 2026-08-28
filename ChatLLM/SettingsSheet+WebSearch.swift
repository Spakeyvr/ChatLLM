import SwiftUI

struct WebSearchSettingsView: View {
    @Binding var settings: AppSettingsDraft
    @State private var validation = TavilyKeyValidationModel()
    @State private var showingKeyEditor = false
    @State private var showingRemoveConfirmation = false

    var body: some View {
        Form {
            Section {
                LabeledContent("Provider", value: "Tavily")
                Button {
                    showingKeyEditor = true
                } label: {
                    LabeledContent("API Key", value: settings.tavilyApiKey.isEmpty
                                   ? String(localized: "Set Up") : String(localized: "Edit"))
                }
                .accessibilityIdentifier("settings.editAPIKey")
                if !settings.tavilyApiKey.isEmpty {
                    Button {
                        validation.validate(settings.tavilyApiKey)
                    } label: {
                        HStack {
                            Text("Test Key")
                            if validation.status == .checking {
                                Spacer()
                                ProgressView()
                            }
                        }
                    }
                    .disabled(validation.status == .checking)
                    .accessibilityIdentifier("settings.testAPIKey")
                }
                switch validation.status {
                case .idle, .checking:
                    EmptyView()
                case .valid:
                    Label("Key is valid", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                case .invalid(let message):
                    Label(message, systemImage: "exclamationmark.circle.fill")
                        .foregroundStyle(.red)
                }
            } footer: {
                Text("Connect Tavily to search the web in conversations. Your API key is stored securely in the device’s Keychain.")
            }
            Section {
                Link(destination: URL(string: "https://app.tavily.com/")!) {
                    Label("Get or Manage API Key", systemImage: "arrow.up.right.square")
                }
            } footer: {
                Text("Web searches send queries to Tavily. Local chats do not need an API key.")
            }
            if !settings.tavilyApiKey.isEmpty {
                Section {
                    Button("Remove API Key", role: .destructive) { showingRemoveConfirmation = true }
                        .accessibilityIdentifier("settings.removeAPIKey")
                }
            }
        }
        .navigationTitle("Web Search")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showingKeyEditor) {
            APIKeySettingsEditor(apiKey: $settings.tavilyApiKey)
        }
        .alert("Remove API Key?", isPresented: $showingRemoveConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Remove", role: .destructive) { settings.tavilyApiKey = "" }
        } message: {
            Text("Web search will be unavailable until you add a key again.")
        }
        .onChange(of: settings.tavilyApiKey) { validation.reset() }
        .onDisappear { validation.reset() }
    }
}

private struct APIKeySettingsEditor: View {
    @Binding var apiKey: String
    @State private var text: String
    @Environment(\.dismiss) private var dismiss

    init(apiKey: Binding<String>) {
        _apiKey = apiKey
        _text = State(initialValue: apiKey.wrappedValue)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    SecureField("Enter your Tavily API key", text: $text)
                        .textContentType(.password)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .accessibilityLabel("Tavily API Key")
                        .accessibilityIdentifier("settings.apiKeyEditor")
                } footer: {
                    Text("Your existing key is unchanged until you tap Save.")
                }
            }
            .navigationTitle("API Key")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        apiKey = text.trimmingCharacters(in: .whitespacesAndNewlines)
                        dismiss()
                    }
                    .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityIdentifier("settings.saveAPIKey")
                }
            }
        }
        .interactiveDismissDisabled(text != apiKey)
    }
}
