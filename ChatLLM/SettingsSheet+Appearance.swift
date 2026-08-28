import SwiftUI

struct AppearanceSettingsView: View {
    @Binding var settings: AppSettingsDraft

    var body: some View {
        Form {
            Section {
                Picker("Color Scheme", selection: $settings.appAppearance) {
                    Text("System").tag("system")
                    Text("Light").tag("light")
                    Text("Dark").tag("dark")
                }
                .accessibilityIdentifier("settings.colorScheme")
            } footer: {
                Text("System follows your device’s appearance.")
            }
            Section("Message Preview") {
                VStack(alignment: .leading, spacing: 12) {
                    Text("How can I make my day a little simpler?")
                        .foregroundStyle(.secondary)
                    Text("Start with one thing that matters. Break it into small steps, and take them one at a time.")
                }
                .font(.system(size: settings.messageFontSize))
                .padding(.vertical, 8)
                .accessibilityIdentifier("settings.messagePreview")
            }
            Section {
                LabeledContent("Message Text Size", value: String(localized: "\(Int(settings.messageFontSize)) pt"))
                Slider(value: $settings.messageFontSize, in: 12...22, step: 1) {
                    Text("Message Text Size")
                } minimumValueLabel: {
                    Image(systemName: "textformat.size.smaller").accessibilityHidden(true)
                } maximumValueLabel: {
                    Image(systemName: "textformat.size.larger").accessibilityHidden(true)
                }
                .accessibilityValue(String(localized: "\(Int(settings.messageFontSize)) points"))
                .accessibilityIdentifier("settings.messageTextSize")
                Button("Reset Text Size") { settings.messageFontSize = 16 }
                    .disabled(settings.messageFontSize == 16)
            } footer: {
                Text("Adjusts user and assistant messages. Settings and other controls follow your device’s text size.")
            }
        }
        .navigationTitle("Appearance")
        .navigationBarTitleDisplayMode(.inline)
    }
}
