import Foundation

/// The interface language is independent of prompts and conversation content.
nonisolated enum AppLanguage: String, CaseIterable, Identifiable, Sendable {
    case english = "en"
    case german = "de"
    case spanish = "es"

    var id: String { rawValue }

    // Keep names in their own language so users can always find their way back.
    var nativeName: String {
        switch self {
        case .english: "English"
        case .german: "Deutsch"
        case .spanish: "Español"
        }
    }

    var locale: Locale { Locale(identifier: rawValue) }

    static func resolve(_ identifier: String) -> AppLanguage {
        AppLanguage(rawValue: identifier) ?? .english
    }

    static var current: AppLanguage {
        resolve(UserDefaults.standard.string(forKey: "appLanguage") ?? "en")
    }

    var bundle: Bundle {
        Self.bundles[self] ?? .main
    }

    private static let bundles: [AppLanguage: Bundle] = Dictionary(
        uniqueKeysWithValues: allCases.compactMap { language in
            guard let path = Bundle.main.path(forResource: language.rawValue, ofType: "lproj"),
                  let bundle = Bundle(path: path) else { return nil }
            return (language, bundle)
        }
    )

    /// A draft may have been created before the user changed the interface language.
    static func isDefaultConversationTitle(_ title: String) -> Bool {
        title == "New Chat" || allCases.contains {
            title == String(localized: "New Chat", bundle: $0.bundle, locale: $0.locale)
        }
    }
}

extension Bundle {
    /// Foundation/UIKit strings need the selected bundle as well as SwiftUI's locale.
    nonisolated static var appLocalized: Bundle { AppLanguage.current.bundle }
}
