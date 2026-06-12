import Foundation

enum AppSettingsKeys {
    static let defaultSystemPrompt = "defaultSystemPrompt"
    static let appAppearance = "appAppearance"
    static let appLanguage = "appLanguage"
    static let customSystemPromptPresets = "customSystemPromptPresets"
    static let sendOnReturn = "sendOnReturn"
    static let enableHaptics = "enableHaptics"
    static let reasoningModeDefault = "reasoningModeDefault"
    static let messageFontSize = "messageFontSize"
    static let mlxMaxOutputTokens = "mlxMaxOutputTokens"
    static let mlxContextWindowTokens = "mlxContextWindowTokens"
    static let mlxEnableTurboQuant = "mlxEnableTurboQuant"
    static let mlxEnableKVCacheQuantization = "mlxEnableKVCacheQuantization"
    static let mlxRepetitionPenalty = "mlxRepetitionPenalty"
    static let autoDeleteOldChats = "autoDeleteOldChats"
    static let autoDeleteDays = "autoDeleteDays"
    static let developerModeEnabled = "developerModeEnabled"
    static let visionConfidenceThreshold = "visionConfidenceThreshold"
    static let disableToolCalls = "disableToolCalls"
}

struct AppSettingsDraft: Equatable {
    var defaultSystemPrompt: String
    var appAppearance: String
    var appLanguage: String
    var customPresets: [SystemPromptPreset]
    var tavilyApiKey: String
    var sendOnReturn: Bool
    var enableHaptics: Bool
    var reasoningModeDefault: Bool
    var messageFontSize: Double
    var mlxMaxOutputTokens: Int
    var mlxContextWindowTokens: Int
    var mlxEnableTurboQuant: Bool
    var mlxRepetitionPenalty: Double
    var autoDeleteOldChats: Bool
    var autoDeleteDays: Int
    var developerModeEnabled: Bool
    var visionConfidenceThreshold: Double
    var disableToolCalls: Bool

    static func load(from defaults: UserDefaults = .standard) -> Self {
        Self(
            defaultSystemPrompt: defaults.string(forKey: AppSettingsKeys.defaultSystemPrompt) ?? "",
            appAppearance: defaults.string(forKey: AppSettingsKeys.appAppearance) ?? "system",
            appLanguage: defaults.string(forKey: AppSettingsKeys.appLanguage) ?? "en",
            customPresets: decodePresets(from: defaults.data(forKey: AppSettingsKeys.customSystemPromptPresets) ?? Data()),
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
            mlxEnableTurboQuant: defaults.mlxEnableTurboQuant,
            mlxRepetitionPenalty: defaults.mlxRepetitionPenalty,
            autoDeleteOldChats: defaults.bool(forKey: AppSettingsKeys.autoDeleteOldChats),
            autoDeleteDays: defaults.object(forKey: AppSettingsKeys.autoDeleteDays) as? Int ?? 30,
            developerModeEnabled: defaults.bool(forKey: AppSettingsKeys.developerModeEnabled),
            visionConfidenceThreshold: defaults.object(forKey: AppSettingsKeys.visionConfidenceThreshold) as? Double ?? 0.5,
            disableToolCalls: defaults.bool(forKey: AppSettingsKeys.disableToolCalls)
        )
    }

    static func defaults() -> Self {
        Self(
            defaultSystemPrompt: "",
            appAppearance: "system",
            appLanguage: "en",
            customPresets: [],
            tavilyApiKey: "",
            sendOnReturn: false,
            enableHaptics: true,
            reasoningModeDefault: false,
            messageFontSize: 16.0,
            mlxMaxOutputTokens: 1024,
            mlxContextWindowTokens: 0,
            mlxEnableTurboQuant: true,
            mlxRepetitionPenalty: 1.0,
            autoDeleteOldChats: false,
            autoDeleteDays: 30,
            developerModeEnabled: false,
            visionConfidenceThreshold: 0.5,
            disableToolCalls: false
        )
    }

    mutating func resetToDefaults() {
        self = Self.defaults()
    }

    func persist(to defaults: UserDefaults = .standard) {
        defaults.set(defaultSystemPrompt, forKey: AppSettingsKeys.defaultSystemPrompt)
        defaults.set(appAppearance, forKey: AppSettingsKeys.appAppearance)
        defaults.set(appLanguage, forKey: AppSettingsKeys.appLanguage)
        defaults.set(encodePresets(customPresets), forKey: AppSettingsKeys.customSystemPromptPresets)
        defaults.sendOnReturn = sendOnReturn
        defaults.enableHapticsPreference = enableHaptics
        defaults.set(reasoningModeDefault, forKey: AppSettingsKeys.reasoningModeDefault)
        defaults.set(messageFontSize, forKey: AppSettingsKeys.messageFontSize)
        defaults.mlxMaxOutputTokens = mlxMaxOutputTokens
        defaults.set(mlxContextWindowTokens, forKey: AppSettingsKeys.mlxContextWindowTokens)
        defaults.mlxEnableTurboQuant = mlxEnableTurboQuant
        defaults.mlxRepetitionPenalty = mlxRepetitionPenalty
        defaults.set(autoDeleteOldChats, forKey: AppSettingsKeys.autoDeleteOldChats)
        defaults.set(autoDeleteDays, forKey: AppSettingsKeys.autoDeleteDays)
        defaults.set(developerModeEnabled, forKey: AppSettingsKeys.developerModeEnabled)
        defaults.set(visionConfidenceThreshold, forKey: AppSettingsKeys.visionConfidenceThreshold)
        defaults.disableToolCalls = disableToolCalls

        let trimmedKey = tavilyApiKey.trimmingCharacters(in: .whitespacesAndNewlines)
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

    private static func decodePresets(from data: Data) -> [SystemPromptPreset] {
        (try? JSONDecoder().decode([SystemPromptPreset].self, from: data)) ?? []
    }

    private func encodePresets(_ presets: [SystemPromptPreset]) -> Data {
        (try? JSONEncoder().encode(presets)) ?? Data()
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
