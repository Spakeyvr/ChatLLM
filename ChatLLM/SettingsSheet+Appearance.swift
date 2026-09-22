import SwiftUI

struct AppearanceSettingsView: View {
    @Binding var settings: AppSettingsDraft
    @Environment(\.locale) private var locale

    var body: some View {
        Form {
            Section {
                Picker("Color Scheme", selection: $settings.appAppearance) {
                    Text("System").tag("system")
                    Text("Light").tag("light")
                    Text("Dark").tag("dark")
                }
                .pickerStyle(.segmented)
                .accessibilityIdentifier("settings.colorScheme")
            } header: {
                Text("Color Scheme")
            } footer: {
                Text("System follows your device’s appearance.")
            }
            Section {
                Picker("Language", selection: $settings.appLanguage) {
                    ForEach(AppLanguage.allCases) { language in
                        Text(verbatim: language.nativeName).tag(language.rawValue)
                    }
                }
                .accessibilityIdentifier("settings.language")
            }
            Section("Message Preview") {
                VStack(alignment: .leading, spacing: 12) {
                    Text("How can I make my day a little simpler?")
                        .padding(12)
                        .background(Color.accentColor.opacity(0.12), in: .rect(cornerRadius: 16))
                        .frame(maxWidth: .infinity, alignment: .trailing)
                        .padding(.leading, 24)
                    Text("Start with one thing that matters. Break it into small steps, and take them one at a time.")
                        .padding(12)
                        .background(Color(uiColor: .tertiarySystemGroupedBackground), in: .rect(cornerRadius: 16))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.trailing, 24)
                }
                .font(.system(size: settings.messageFontSize))
                .padding(.vertical, 8)
                .accessibilityIdentifier("settings.messagePreview")
            }
            Section {
                LabeledContent("Message Text Size", value: String(localized: "\(Int(settings.messageFontSize)) pt", bundle: .appLocalized))
                Slider(value: $settings.messageFontSize, in: 12...22, step: 1) {
                    Text("Message Text Size")
                } minimumValueLabel: {
                    Image(systemName: "textformat.size.smaller").accessibilityHidden(true)
                } maximumValueLabel: {
                    Image(systemName: "textformat.size.larger").accessibilityHidden(true)
                }
                .accessibilityValue("\(Int(settings.messageFontSize)) points")
                .accessibilityIdentifier("settings.messageTextSize")
                Button("Reset Text Size") { settings.messageFontSize = 16 }
                    .disabled(settings.messageFontSize == 16)
            } footer: {
                Text("Adjusts user and assistant messages. Settings and other controls follow your device’s text size.")
            }
        }
        .navigationTitle(String(localized: "Appearance", bundle: .appLocalized, locale: locale))
        .navigationBarTitleDisplayMode(.inline)
    }
}
