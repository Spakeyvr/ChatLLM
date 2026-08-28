import Foundation

enum AppSettingsKeys {
    static let chatPreferences = "chatPreferences"
    static let legacyDefaultSystemPrompt = "defaultSystemPrompt"
    static let appAppearance = "appAppearance"
    static let appLanguage = "appLanguage"
    static let sendOnReturn = "sendOnReturn"
    static let enableHaptics = "enableHaptics"
    static let reasoningModeDefault = "reasoningModeDefault"
    static let messageFontSize = "messageFontSize"
    static let mlxMaxOutputTokens = "mlxMaxOutputTokens"
    static let mlxContextWindowTokens = "mlxContextWindowTokens"
    static let mlxEnableRotorQuant = "mlxEnableRotorQuant"
    static let mlxRepetitionPenalty = "mlxRepetitionPenalty"
    static let autoDeleteOldChats = "autoDeleteOldChats"
    static let autoDeleteDays = "autoDeleteDays"
    static let developerModeEnabled = "developerModeEnabled"
    static let disableRAMPrecautions = "disableRAMPrecautions"
    static let visionConfidenceThreshold = "visionConfidenceThreshold"
    static let disableToolCalls = "disableToolCalls"
}

struct AppSettingsDraft: Equatable {
    var chatPreferences: String
    var appAppearance: String
    var appLanguage: String
    var tavilyApiKey: String
    var sendOnReturn: Bool
    var enableHaptics: Bool
    var reasoningModeDefault: Bool
    var messageFontSize: Double
    var mlxMaxOutputTokens: Int
    var mlxContextWindowTokens: Int
    var mlxEnableRotorQuant: Bool
    var mlxRepetitionPenalty: Double
    var autoDeleteOldChats: Bool
    var autoDeleteDays: Int
    var developerModeEnabled: Bool
    var disableRAMPrecautions: Bool
    var visionConfidenceThreshold: Double
    var disableToolCalls: Bool

    static func load(from defaults: UserDefaults = .standard) -> Self {
        Self(
            chatPreferences: defaults.string(forKey: AppSettingsKeys.chatPreferences)
                ?? defaults.string(forKey: AppSettingsKeys.legacyDefaultSystemPrompt)
                ?? "",
            appAppearance: defaults.string(forKey: AppSettingsKeys.appAppearance) ?? "system",
            appLanguage: defaults.string(forKey: AppSettingsKeys.appLanguage) ?? "en",
            tavilyApiKey: TavilyAPIKeyStore.currentKey(
                userDefaults: defaults,
                service: TavilyAPIKeyStore.service,
                account: TavilyAPIKeyStore.account
            ) ?? "",
            sendOnReturn: defaults.sendOnReturn,
            enableHaptics: defaults.enableHapticsPreference,
            reasoningModeDefault: defaults.bool(forKey: AppSettingsKeys.reasoningModeDefault),
            messageFontSize: defaults.object(forKey: AppSettingsKeys.messageFontSize) as? Double ?? 16.0,
            mlxMaxOutputTokens: defaults.mlxMaxOutputTokens,
            mlxContextWindowTokens: defaults.object(forKey: AppSettingsKeys.mlxContextWindowTokens) != nil
                ? defaults.integer(forKey: AppSettingsKeys.mlxContextWindowTokens)
                : 0,
            mlxEnableRotorQuant: defaults.mlxEnableRotorQuant,
            mlxRepetitionPenalty: defaults.mlxRepetitionPenalty,
            autoDeleteOldChats: defaults.bool(forKey: AppSettingsKeys.autoDeleteOldChats),
            autoDeleteDays: defaults.object(forKey: AppSettingsKeys.autoDeleteDays) as? Int ?? 30,
            developerModeEnabled: defaults.bool(forKey: AppSettingsKeys.developerModeEnabled),
            disableRAMPrecautions: defaults.bool(forKey: AppSettingsKeys.disableRAMPrecautions),
            visionConfidenceThreshold: defaults.object(forKey: AppSettingsKeys.visionConfidenceThreshold) as? Double ?? 0.5,
            disableToolCalls: defaults.bool(forKey: AppSettingsKeys.disableToolCalls)
        )
    }

    static func defaults() -> Self {
        Self(
            chatPreferences: "",
            appAppearance: "system",
            appLanguage: "en",
            tavilyApiKey: "",
            sendOnReturn: false,
            enableHaptics: true,
            reasoningModeDefault: false,
            messageFontSize: 16.0,
            mlxMaxOutputTokens: 0,
            mlxContextWindowTokens: 0,
            mlxEnableRotorQuant: true,
            mlxRepetitionPenalty: 1.0,
            autoDeleteOldChats: false,
            autoDeleteDays: 30,
            developerModeEnabled: false,
            disableRAMPrecautions: false,
            visionConfidenceThreshold: 0.5,
            disableToolCalls: false
        )
    }

    mutating func resetToDefaults() {
        self = Self.defaults()
    }

    // Live settings edits pass the prior snapshot so an unrelated preference
    // never rewrites credentials or overwrites values changed elsewhere.
    func persist(to defaults: UserDefaults = .standard, comparedTo previous: Self? = nil) {
        if previous?.chatPreferences != chatPreferences || defaults.object(forKey: AppSettingsKeys.legacyDefaultSystemPrompt) != nil {
            defaults.set(chatPreferences, forKey: AppSettingsKeys.chatPreferences)
            defaults.removeObject(forKey: AppSettingsKeys.legacyDefaultSystemPrompt)
        }
        if previous?.appAppearance != appAppearance {
            defaults.set(appAppearance, forKey: AppSettingsKeys.appAppearance)
        }
        if previous?.appLanguage != appLanguage {
            defaults.set(appLanguage, forKey: AppSettingsKeys.appLanguage)
        }
        if previous?.sendOnReturn != sendOnReturn {
            defaults.sendOnReturn = sendOnReturn
        }
        if previous?.enableHaptics != enableHaptics {
            defaults.enableHapticsPreference = enableHaptics
        }
        if previous?.reasoningModeDefault != reasoningModeDefault {
            defaults.set(reasoningModeDefault, forKey: AppSettingsKeys.reasoningModeDefault)
        }
        if previous?.messageFontSize != messageFontSize {
            defaults.set(messageFontSize, forKey: AppSettingsKeys.messageFontSize)
        }
        if previous?.mlxMaxOutputTokens != mlxMaxOutputTokens {
            defaults.mlxMaxOutputTokens = mlxMaxOutputTokens
        }
        if previous?.mlxContextWindowTokens != mlxContextWindowTokens {
            defaults.set(mlxContextWindowTokens, forKey: AppSettingsKeys.mlxContextWindowTokens)
        }
        if previous?.mlxEnableRotorQuant != mlxEnableRotorQuant {
            defaults.mlxEnableRotorQuant = mlxEnableRotorQuant
        }
        if previous?.mlxRepetitionPenalty != mlxRepetitionPenalty {
            defaults.mlxRepetitionPenalty = mlxRepetitionPenalty
        }
        if previous?.autoDeleteOldChats != autoDeleteOldChats {
            defaults.set(autoDeleteOldChats, forKey: AppSettingsKeys.autoDeleteOldChats)
        }
        if previous?.autoDeleteDays != autoDeleteDays {
            defaults.set(autoDeleteDays, forKey: AppSettingsKeys.autoDeleteDays)
        }
        if previous?.developerModeEnabled != developerModeEnabled {
            defaults.set(developerModeEnabled, forKey: AppSettingsKeys.developerModeEnabled)
        }
        if previous?.disableRAMPrecautions != disableRAMPrecautions {
            defaults.set(disableRAMPrecautions, forKey: AppSettingsKeys.disableRAMPrecautions)
        }
        if previous?.visionConfidenceThreshold != visionConfidenceThreshold {
            defaults.set(visionConfidenceThreshold, forKey: AppSettingsKeys.visionConfidenceThreshold)
        }
        if previous?.disableToolCalls != disableToolCalls {
            defaults.disableToolCalls = disableToolCalls
        }

        let trimmedKey = tavilyApiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard previous == nil || previous?.tavilyApiKey.trimmingCharacters(in: .whitespacesAndNewlines) != trimmedKey else {
            return
        }
        if trimmedKey.isEmpty {
            TavilyAPIKeyStore.clear(
                userDefaults: defaults,
                service: TavilyAPIKeyStore.service,
                account: TavilyAPIKeyStore.account
            )
        } else {
            TavilyAPIKeyStore.save(
                trimmedKey,
                userDefaults: defaults,
                service: TavilyAPIKeyStore.service,
                account: TavilyAPIKeyStore.account
            )
        }
    }

}

extension UserDefaults {
    var sendOnReturn: Bool {
        get { bool(forKey: AppSettingsKeys.sendOnReturn) }
        set { set(newValue, forKey: AppSettingsKeys.sendOnReturn) }
    }

    var enableHapticsPreference: Bool {
        get {
            guard object(forKey: AppSettingsKeys.enableHaptics) != nil else {
                return true
            }
            return bool(forKey: AppSettingsKeys.enableHaptics)
        }
        set { set(newValue, forKey: AppSettingsKeys.enableHaptics) }
    }
}
