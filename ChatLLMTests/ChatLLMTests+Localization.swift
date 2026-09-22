import Foundation
import Testing
@testable import ChatLLM

extension ChatLLMTests {
    @Test func interfaceLanguagesResolveAndFallBackToEnglish() throws {
        #expect(AppLanguage.resolve("de") == .german)
        #expect(AppLanguage.resolve("es") == .spanish)
        #expect(AppLanguage.resolve("unsupported") == .english)
        let defaults = try makeScratchDefaults(#function)
        defaults.set("unsupported", forKey: AppSettingsKeys.appLanguage)
        #expect(AppSettingsDraft.load(from: defaults).appLanguage == "en")
    }

    @Test func interfaceTranslationsAreBundledForBothLanguages() {
        #expect(String(localized: "Settings", bundle: AppLanguage.german.bundle) == "Einstellungen")
        #expect(String(localized: "Settings", bundle: AppLanguage.spanish.bundle) == "Ajustes")
        #expect(String(localized: "New Chat", bundle: AppLanguage.german.bundle) == "Neuer Chat")
        #expect(String(localized: "New Chat", bundle: AppLanguage.spanish.bundle) == "Nuevo chat")
        #expect(String(localized: "Settings", bundle: AppLanguage.english.bundle) == "Settings")
    }

    @Test func sourceCountsUseLocalizedSingularAndPlural() {
        for (language, singular, plural) in [
            (AppLanguage.english, "1 source", "2 sources"),
            (.german, "1 Quelle", "2 Quellen"),
            (.spanish, "1 fuente", "2 fuentes")
        ] {
            let one = 1
            let two = 2
            #expect(String(localized: "\(one) sources", bundle: language.bundle, locale: language.locale) == singular)
            #expect(String(localized: "\(two) sources", bundle: language.bundle, locale: language.locale) == plural)
        }
    }

    @Test func languageSwitchUpdatesEagerStringsWithoutChangingChatPreferences() throws {
        let defaults = UserDefaults.standard
        let previousLanguage = defaults.object(forKey: AppSettingsKeys.appLanguage)
        defer { defaults.set(previousLanguage, forKey: AppSettingsKeys.appLanguage) }
        let preferences = defaults.string(forKey: AppSettingsKeys.chatPreferences)

        defaults.set("de", forKey: AppSettingsKeys.appLanguage)
        #expect(Strings.thinking == "Denke nach…")
        #expect(TavilySearchError.invalidAPIKey.errorDescription == "Der Tavily-API-Schlüssel fehlt oder ist ungültig.")
        defaults.set("es", forKey: AppSettingsKeys.appLanguage)
        #expect(Strings.thinking == "Pensando…")
        #expect(TavilySearchError.invalidAPIKey.errorDescription == "La clave de API de Tavily falta o no es válida.")
        #expect(defaults.string(forKey: AppSettingsKeys.chatPreferences) == preferences)
    }

    @Test func defaultChatTitlesRemainRecognizableAcrossLanguages() {
        for title in ["New Chat", "Neuer Chat", "Nuevo chat"] {
            #expect(AppLanguage.isDefaultConversationTitle(title))
        }
        #expect(!AppLanguage.isDefaultConversationTitle("My holiday plans"))
    }
}
