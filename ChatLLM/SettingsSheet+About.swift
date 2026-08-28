import SwiftUI

struct AboutSettingsView: View {
    private var version: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "—"
    }
    private var build: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "—"
    }

    var body: some View {
        Form {
            Section("ChatLLM") {
                LabeledContent("Version", value: version)
                LabeledContent("Build", value: build)
            }
            Section {
                Link(destination: URL(string: "https://discord.gg/PGNCC4Vy7T")!) {
                    Label("Join Discord Community", systemImage: "bubble.left.and.bubble.right")
                }
                Link(destination: URL(string: "mailto:chatllm@icloud.com")!) {
                    Label("Email Support", systemImage: "envelope")
                }
            } header: {
                Text("Support")
            } footer: {
                Text("Get help, report a problem, or share an idea for ChatLLM.")
            }
            Section {
                LabeledContent("Release Channel", value: String(localized: "Beta"))
            } footer: {
                Text("ChatLLM is in beta. Features may change, and you may encounter bugs. Your feedback helps improve the app.")
            }
        }
        .navigationTitle("About & Support")
        .navigationBarTitleDisplayMode(.inline)
    }
}
