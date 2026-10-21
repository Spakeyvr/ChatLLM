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

    @Test func imageOnlyMessagePromptFollowsTheAppLanguage() {
        #expect(String(localized: "Describe what you see", bundle: AppLanguage.german.bundle) == "Beschreibe, was du siehst")
        #expect(String(localized: "Describe what you see", bundle: AppLanguage.spanish.bundle) == "Describe lo que ves")
    }

    @Test func undownloadedModelErrorIsTranslated() {
        let name = "Qwen 3.5"
        #expect(String(localized: "Model '\(name)' isn't downloaded yet. Download it in Manage Models before selecting it.", bundle: AppLanguage.german.bundle)
            == "Das Modell „Qwen 3.5“ ist noch nicht heruntergeladen. Lade es unter „Modelle verwalten“ herunter, bevor du es auswählst.")
    }

    @Test func missingModelFilesErrorIsTranslated() {
        let name = "Qwen 3.5 2B (4-bit)"
        #expect(String(localized: "\(name)'s files are missing. Download it again in Manage Models.", bundle: AppLanguage.german.bundle)
            == "Die Dateien von Qwen 3.5 2B (4-bit) fehlen. Lade es unter „Modelle verwalten“ erneut herunter.")
    }

    @Test func tokenCountsGroupDigitsForTheAppLanguage() {
        let tokens = 6_144
        #expect(String(localized: "\(tokens) tokens", bundle: AppLanguage.english.bundle, locale: AppLanguage.english.locale) == "6,144 tokens")
        #expect(String(localized: "\(tokens) tokens", bundle: AppLanguage.german.bundle, locale: AppLanguage.german.locale) == "6.144 Tokens")
    }

    @Test func modelDownloadSizesUseTheAppLanguagesDecimalSeparator() throws {
        let model = try #require(MLXModelManager.modelDefinitions.first { $0.id == "qwen3.5-4b-4bit-hybrid" })
        #expect(model.downloadSize(locale: AppLanguage.english.locale) == "2.66 GB")
        #expect(model.downloadSize(locale: AppLanguage.german.locale) == "2,66\u{00A0}GB")
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

    @Test func chatExportIsWrittenInTheAppLanguage() {
        let conversation = Conversation(title: "Reise")
        conversation.lastUpdated = Date(timeIntervalSince1970: 1_790_000_000)
        conversation.messages = [
            ChatLLM.Message(role: .user, text: "Hallo", order: 0, conversation: conversation, isFinal: true),
            ChatLLM.Message(role: .assistant, text: "Hi!", order: 1, conversation: conversation, isFinal: true)
        ]

        let text = ChatExportFormatter.text(
            for: [conversation],
            language: .german,
            exportedAt: conversation.lastUpdated
        )

        #expect(text.hasPrefix("# Chat-Export\n"))
        #expect(text.contains("Unterhaltungen: 1\n"))
        #expect(text.contains("Zuletzt aktualisiert: 21. Sept. 2026"))
        #expect(text.contains("**Nutzer:**\nHallo\n"))
        #expect(text.contains("**Assistent:**\nHi!\n"))
        #expect(!text.contains("Last Updated"))
    }

    @Test func messageTextSizeFollowsDynamicType() {
        #expect(RichMarkdownView.scaledFontSize(16, bodyPointSize: 17) == 16)
        #expect(RichMarkdownView.scaledFontSize(16, bodyPointSize: 34) == 32)
        #expect(RichMarkdownView.scaledFontSize(20, bodyPointSize: 15) < 20)
    }

    @Test func defaultChatTitlesRemainRecognizableAcrossLanguages() {
        for title in ["New Chat", "Neuer Chat", "Nuevo chat"] {
            #expect(AppLanguage.isDefaultConversationTitle(title))
        }
        #expect(!AppLanguage.isDefaultConversationTitle("My holiday plans"))
    }
}
