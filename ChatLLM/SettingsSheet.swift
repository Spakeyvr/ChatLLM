import SwiftUI

struct SettingsSheet: View {
    static let mlxRotorQuantInfoMessage = String(localized: "Enabled by default for persistent MLX text and image chats, RotorQuant uses IsoQuant block rotations with 3-bit keys, 2-bit values, exact prefill buffering, and per-layer deterministic rotation parameters to reduce memory use. Tool and low-memory turns use safer uncompressed or bounded caches.")
    static let mlxRotorQuantAccessibilityHint = String(localized: "Enabled by default for supported persistent MLX text and image chats. Tool and low-memory turns use safer cache modes.")
    static let mlxRotorQuantExperimentalTitle = String(localized: "RotorQuant Experimental")
    static let mlxRotorQuantExperimentalMessage = String(localized: "RotorQuant is very early and in beta. It can reduce KV-cache memory, but speed and stability are still being tuned.")

    @Binding var settings: AppSettingsDraft
    let hasChats: Bool
    let canDeleteAllExceptCurrent: Bool
    let onDeleteAll: () -> Void
    let onDeleteAllExceptCurrent: () -> Void
    let onExportChats: () -> Void

    private enum Destination: Hashable {
        case appearance, chat, webSearch, privacy, advanced, about
    }

    var body: some View {
        List {
            Section {
                NavigationLink(value: Destination.appearance) {
                    SettingsRow("Appearance", systemImage: "paintpalette.fill", color: .indigo,
                                value: appearanceSummary)
                }
                .accessibilityIdentifier("settings.appearance")
                NavigationLink(value: Destination.chat) {
                    SettingsRow("Chat", systemImage: "bubble.left.and.bubble.right.fill", color: .green)
                }
                .accessibilityIdentifier("settings.chat")
                NavigationLink(value: Destination.webSearch) {
                    SettingsRow("Web Search", systemImage: "globe", color: .blue,
                                value: settings.tavilyApiKey.isEmpty
                                    ? String(localized: "Set Up") : String(localized: "Configured"))
                }
                .accessibilityIdentifier("settings.webSearch")
            }
            Section {
                NavigationLink(value: Destination.privacy) {
                    SettingsRow("Privacy & Data", systemImage: "hand.raised.fill", color: .blue)
                }
                .accessibilityIdentifier("settings.privacy")
            }
            Section {
                NavigationLink(value: Destination.advanced) {
                    SettingsRow("Advanced", systemImage: "gearshape.2.fill", color: .gray)
                }
                .accessibilityIdentifier("settings.advanced")
                NavigationLink(value: Destination.about) {
                    SettingsRow("About & Support", systemImage: "info.circle.fill", color: .gray)
                }
                .accessibilityIdentifier("settings.about")
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.large)
        .navigationDestination(for: Destination.self) { destination in
            switch destination {
            case .appearance:
                AppearanceSettingsView(settings: $settings)
            case .chat:
                ChatSettingsView(settings: $settings)
            case .webSearch:
                WebSearchSettingsView(settings: $settings)
            case .privacy:
                PrivacySettingsView(settings: $settings, hasChats: hasChats,
                                    canDeleteAllExceptCurrent: canDeleteAllExceptCurrent,
                                    onDeleteAll: onDeleteAll, onDeleteAllExceptCurrent: onDeleteAllExceptCurrent,
                                    onExportChats: onExportChats)
            case .advanced:
                AdvancedSettingsView(settings: $settings)
            case .about:
                AboutSettingsView()
            }
        }
    }

    private var appearanceSummary: String {
        switch settings.appAppearance {
        case "light": String(localized: "Light")
        case "dark": String(localized: "Dark")
        default: String(localized: "System")
        }
    }
}

// Only category icons are custom. Rows and navigation retain system behavior.
struct SettingsRow: View {
    let title: LocalizedStringKey
    let systemImage: String
    let color: Color
    var value: String?

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @ScaledMetric(relativeTo: .body) private var scaledIconSize = 28.0

    // Oversized decorative icons steal label width at accessibility sizes.
    private var iconSize: Double { min(scaledIconSize, 32) }

    init(_ title: LocalizedStringKey, systemImage: String, color: Color, value: String? = nil) {
        self.title = title
        self.systemImage = systemImage
        self.color = color
        self.value = value
    }

    var body: some View {
        HStack(spacing: 12) {
            if dynamicTypeSize.isAccessibilitySize {
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).foregroundStyle(.primary)
                    if let value { Text(value).font(.subheadline).foregroundStyle(.secondary) }
                }
            } else {
                Image(systemName: systemImage)
                    .font(.system(size: iconSize * 0.57, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: iconSize, height: iconSize)
                    .background(color, in: .rect(cornerRadius: iconSize * 0.23))
                    .accessibilityHidden(true)
                Text(title).foregroundStyle(.primary)
                if let value {
                    Spacer(minLength: 8)
                    Text(value).foregroundStyle(.secondary)
                }
            }
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .combine)
    }
}
