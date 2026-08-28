//
//  ChatLLMTests+AppSettings.swift
//  ChatLLMTests
//
//  Split out of ChatLLMTests.swift; part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
import MLXLMCommon
import SwiftUI
import SwiftData
import WebKit
import UIKit
import FoundationModels
@testable import ChatLLM

extension ChatLLMTests {
    @Test func settingsSheetDescribesAdaptiveKVCacheBehavior() {
        #expect(SettingsSheet.mlxRotorQuantInfoMessage.contains("Enabled by default"))
        #expect(SettingsSheet.mlxRotorQuantInfoMessage.contains("IsoQuant"))
        #expect(SettingsSheet.mlxRotorQuantInfoMessage.contains("3-bit keys"))
        #expect(SettingsSheet.mlxRotorQuantInfoMessage.contains("2-bit values"))
        #expect(SettingsSheet.mlxRotorQuantInfoMessage.contains("text and image"))
        #expect(SettingsSheet.mlxRotorQuantInfoMessage.contains("Tool and low-memory"))
        #expect(SettingsSheet.mlxRotorQuantAccessibilityHint.contains("safer cache modes"))
        #expect(SettingsSheet.mlxRotorQuantExperimentalTitle.contains("Experimental"))
        #expect(SettingsSheet.mlxRotorQuantExperimentalMessage.contains("very early"))
        #expect(SettingsSheet.mlxRotorQuantExperimentalMessage.contains("beta"))
    }

    @Test func generationConfigurationUsesStructuredRotorQuantWhenPreferred() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: true,
            preferRotorQuant: true,
            hasTools: false,
            hasMedia: false,
            memoryConstrained: false,
            prefersBoundedCache: false,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 32768
        )

        guard case .rotorQuant(let rotorConfiguration) = configuration.cacheCompression else {
            Issue.record("Expected RotorQuant cache compression")
            return
        }

        #expect(rotorConfiguration.variant == .iso)
        #expect(rotorConfiguration.keyBits == 3)
        #expect(rotorConfiguration.valueBits == 2)
        #expect(rotorConfiguration.exactBufferSize == 128)
        #expect(rotorConfiguration.attentionBlockTokens == 128)
    }

    @Test func generationConfigurationUsesMediaTunedRotorQuantProfile() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: true,
            preferRotorQuant: true,
            hasTools: false,
            hasMedia: true,
            memoryConstrained: false,
            prefersBoundedCache: false,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 32768
        )

        #expect(configuration.cachePolicy == .persistentRotorQuant)
        guard case .rotorQuant(let rotorConfiguration) = configuration.cacheCompression else {
            Issue.record("Expected RotorQuant cache compression")
            return
        }

        #expect(rotorConfiguration.variant == .iso)
        #expect(rotorConfiguration.keyBits == 3)
        #expect(rotorConfiguration.valueBits == 2)
        #expect(rotorConfiguration.exactBufferSize == 32)
        #expect(rotorConfiguration.attentionBlockTokens == 64)
    }

    @Test func userDefaultsRotorQuantDefaultsToEnabledWhenUnset() {
        let key = AppSettingsKeys.mlxEnableRotorQuant
        let defaults = UserDefaults.standard
        let hadExistingValue = defaults.object(forKey: key) != nil
        let previousValue = defaults.bool(forKey: key)

        defaults.removeObject(forKey: key)
        defer {
            if hadExistingValue {
                defaults.set(previousValue, forKey: key)
            } else {
                defaults.removeObject(forKey: key)
            }
        }

        #expect(defaults.mlxEnableRotorQuant)
    }

    @Test func unlimitedOutputDefaultsApplyOnFirstLoadAndReset() throws {
        let suiteName = "settings-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        #expect(defaults.mlxMaxOutputTokensLimit == nil)
        var draft = AppSettingsDraft.load(from: defaults)
        #expect(draft.mlxMaxOutputTokens == 0)

        draft.mlxMaxOutputTokens = 768
        draft.resetToDefaults()
        #expect(draft.mlxMaxOutputTokens == 0)
    }

    @Test func unlimitedOutputDefaultPreservesSavedLimits() throws {
        let suiteName = "settings-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        defaults.mlxMaxOutputTokens = 768
        #expect(defaults.mlxMaxOutputTokensLimit == 768)
        #expect(AppSettingsDraft.load(from: defaults).mlxMaxOutputTokens == 768)

        defaults.mlxMaxOutputTokens = 0
        #expect(defaults.mlxMaxOutputTokensLimit == nil)
        #expect(AppSettingsDraft.load(from: defaults).mlxMaxOutputTokens == 0)
    }

    @Test func resetSettingsRestoresRotorQuantDefault() {
        var draft = AppSettingsDraft.defaults()
        draft.mlxEnableRotorQuant = false

        draft.resetToDefaults()

        #expect(draft.mlxEnableRotorQuant)
    }

    @Test func resetSettingsRestoresDeveloperModeDefault() {
        var draft = AppSettingsDraft.defaults()
        draft.developerModeEnabled = true

        draft.resetToDefaults()

        #expect(!draft.developerModeEnabled)
    }

    @Test func resetSettingsRestoresContextWindowAndToolCallDefaults() {
        var draft = AppSettingsDraft.defaults()
        draft.mlxContextWindowTokens = 4_096
        draft.disableToolCalls = true
        draft.sendOnReturn = true

        draft.resetToDefaults()

        #expect(draft.mlxContextWindowTokens == 0)
        #expect(!draft.disableToolCalls)
        #expect(!draft.sendOnReturn)
    }

    @Test func appSettingsDraftDoesNotPersistUntilExplicitCommit() throws {
        let suiteName = "settings-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }

        defaults.set("Existing preferences", forKey: AppSettingsKeys.chatPreferences)
        defaults.set(false, forKey: AppSettingsKeys.developerModeEnabled)
        defaults.set(false, forKey: AppSettingsKeys.disableToolCalls)

        var draft = AppSettingsDraft.load(from: defaults)
        draft.chatPreferences = "Draft only"
        draft.developerModeEnabled = true
        draft.disableToolCalls = true

        #expect(defaults.string(forKey: AppSettingsKeys.chatPreferences) == "Existing preferences")
        #expect(defaults.bool(forKey: AppSettingsKeys.developerModeEnabled) == false)
        #expect(defaults.bool(forKey: AppSettingsKeys.disableToolCalls) == false)

        draft.persist(to: defaults)

        #expect(defaults.string(forKey: AppSettingsKeys.chatPreferences) == "Draft only")
        #expect(defaults.bool(forKey: AppSettingsKeys.developerModeEnabled))
        #expect(defaults.bool(forKey: AppSettingsKeys.disableToolCalls))
    }

    @Test func appSettingsDraftMigratesLegacyDefaultPromptToChatPreferences() throws {
        let suiteName = "settings-migration-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }

        defaults.set("Prefer concise answers.", forKey: AppSettingsKeys.legacyDefaultSystemPrompt)

        let draft = AppSettingsDraft.load(from: defaults)
        #expect(draft.chatPreferences == "Prefer concise answers.")

        draft.persist(to: defaults)

        #expect(defaults.string(forKey: AppSettingsKeys.chatPreferences) == "Prefer concise answers.")
        #expect(defaults.object(forKey: AppSettingsKeys.legacyDefaultSystemPrompt) == nil)
    }

    @Test func incrementalSettingsSaveOnlyWritesChangedPreferences() throws {
        let suiteName = "settings-incremental-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let previous = AppSettingsDraft.defaults()
        var updated = previous
        updated.sendOnReturn = true
        // A value changed elsewhere must not be overwritten by an unrelated edit.
        defaults.set(22.0, forKey: AppSettingsKeys.messageFontSize)
        // An unchanged credential must not trigger key migration, removal or save.
        defaults.set("test-key-sentinel", forKey: TavilyAPIKeyStore.userDefaultsKey)

        updated.persist(to: defaults, comparedTo: previous)

        #expect(defaults.sendOnReturn)
        #expect(defaults.double(forKey: AppSettingsKeys.messageFontSize) == 22)
        #expect(defaults.string(forKey: TavilyAPIKeyStore.userDefaultsKey) == "test-key-sentinel")
        #expect(defaults.object(forKey: AppSettingsKeys.appAppearance) == nil)
    }

    @Test func incrementalSettingsSaveMigratesLegacyPreferencesWithoutChangingThem() throws {
        let suiteName = "settings-incremental-migration-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        var previous = AppSettingsDraft.defaults()
        previous.chatPreferences = "Keep answers concise."
        defaults.set(previous.chatPreferences, forKey: AppSettingsKeys.legacyDefaultSystemPrompt)
        var updated = previous
        updated.enableHaptics = false

        updated.persist(to: defaults, comparedTo: previous)

        #expect(defaults.string(forKey: AppSettingsKeys.chatPreferences) == previous.chatPreferences)
        #expect(defaults.object(forKey: AppSettingsKeys.legacyDefaultSystemPrompt) == nil)
        #expect(!defaults.enableHapticsPreference)
    }

    @Test func tavilyKeyMigratesOutOfPlaintextUserDefaults() throws {
        let suiteName = "tavily-migration-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        let service = "test.service.\(UUID().uuidString)"
        let account = "test.account"
        defaults.removePersistentDomain(forName: suiteName)
        defer {
            defaults.removePersistentDomain(forName: suiteName)
            TavilyAPIKeyStore.clear(userDefaults: defaults, service: service, account: account, postNotification: false)
        }

        defaults.set(" legacy-key ", forKey: TavilyAPIKeyStore.userDefaultsKey)

        let migrated = TavilyAPIKeyStore.currentKey(
            userDefaults: defaults,
            service: service,
            account: account
        )

        #expect(migrated == "legacy-key")
        #expect(defaults.object(forKey: TavilyAPIKeyStore.userDefaultsKey) == nil)
        #expect(TavilyAPIKeyStore.currentKey(userDefaults: defaults, service: service, account: account) == "legacy-key")
    }

    @Test func tavilySaveDoesNotLeavePlaintextCopyBehind() throws {
        let suiteName = "tavily-save-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        let service = "test.service.\(UUID().uuidString)"
        let account = "test.account"
        defaults.removePersistentDomain(forName: suiteName)
        defer {
            defaults.removePersistentDomain(forName: suiteName)
            TavilyAPIKeyStore.clear(userDefaults: defaults, service: service, account: account, postNotification: false)
        }

        TavilyAPIKeyStore.save(
            "test-secret",
            userDefaults: defaults,
            service: service,
            account: account
        )

        #expect(defaults.object(forKey: TavilyAPIKeyStore.userDefaultsKey) == nil)
        #expect(TavilyAPIKeyStore.currentKey(userDefaults: defaults, service: service, account: account) == "test-secret")
    }
    @Test func settingsLoadFallsBackToDocumentedDefaults() throws {
        let defaults = try makeScratchDefaults(#function)

        let loaded = AppSettingsDraft.load(from: defaults)

        #expect(loaded.appAppearance == "system")
        #expect(loaded.appLanguage == "en")
        #expect(loaded.messageFontSize == 16.0)
        #expect(loaded.autoDeleteDays == 30)
        #expect(loaded.visionConfidenceThreshold == 0.5)
        // Haptics default on even though `bool(forKey:)` would say false.
        #expect(loaded.enableHaptics)
        #expect(!loaded.sendOnReturn)
        #expect(!loaded.autoDeleteOldChats)
    }

    @Test func settingsMigrateTheLegacyDefaultSystemPromptKey() throws {
        let defaults = try makeScratchDefaults(#function)
        defaults.set("Be concise.", forKey: AppSettingsKeys.legacyDefaultSystemPrompt)

        let loaded = AppSettingsDraft.load(from: defaults)
        #expect(loaded.chatPreferences == "Be concise.")

        // Persisting moves it onto the current key and retires the old one.
        loaded.persist(to: defaults, comparedTo: loaded)
        #expect(defaults.string(forKey: AppSettingsKeys.chatPreferences) == "Be concise.")
        #expect(defaults.object(forKey: AppSettingsKeys.legacyDefaultSystemPrompt) == nil)
    }

    @Test func settingsPersistRoundTripsEveryStoredField() throws {
        let defaults = try makeScratchDefaults(#function)
        var draft = AppSettingsDraft.load(from: defaults)
        let baseline = draft

        draft.chatPreferences = "Answer in German."
        draft.appAppearance = "dark"
        draft.appLanguage = "de"
        draft.sendOnReturn = true
        draft.enableHaptics = false
        draft.reasoningModeDefault = true
        draft.messageFontSize = 20
        draft.mlxMaxOutputTokens = 768
        draft.mlxContextWindowTokens = 4_096
        draft.mlxEnableRotorQuant = false
        draft.mlxRepetitionPenalty = 1.15
        draft.autoDeleteOldChats = true
        draft.autoDeleteDays = 7
        draft.developerModeEnabled = true
        draft.disableRAMPrecautions = true
        draft.visionConfidenceThreshold = 0.8
        draft.disableToolCalls = true

        draft.persist(to: defaults, comparedTo: baseline)

        var reloaded = AppSettingsDraft.load(from: defaults)
        // The key lives in the Keychain rather than this scratch domain.
        reloaded.tavilyApiKey = draft.tavilyApiKey
        #expect(reloaded == draft)
    }

    @Test func mlxTuningPreferencesClampToTheirSupportedRange() throws {
        let defaults = try makeScratchDefaults(#function)

        // Output-token budget: 0 means "no limit", anything else lands in 512...1024.
        #expect(defaults.mlxMaxOutputTokens == 0)
        defaults.mlxMaxOutputTokens = 2_048
        #expect(defaults.mlxMaxOutputTokens == 1_024)
        defaults.mlxMaxOutputTokens = 100
        #expect(defaults.mlxMaxOutputTokens == 512)
        #expect(defaults.mlxMaxOutputTokensLimit == 512)
        defaults.mlxMaxOutputTokens = 0
        #expect(defaults.mlxMaxOutputTokens == 0)
        #expect(defaults.mlxMaxOutputTokensLimit == nil)

        // Repetition penalty lives in 1.0...1.5, and 1.0 reads back as "off".
        #expect(defaults.mlxRepetitionPenalty == 1.0)
        defaults.mlxRepetitionPenalty = 2.0
        #expect(defaults.mlxRepetitionPenalty == 1.5)
        #expect(defaults.mlxRepetitionPenaltyValue == Float(1.5))
        defaults.mlxRepetitionPenalty = 0.5
        #expect(defaults.mlxRepetitionPenalty == 1.0)
        #expect(defaults.mlxRepetitionPenaltyValue == nil)

        // The context window defers to the device maximum until it is set,
        // then stays inside 512...deviceMaximum.
        #expect(defaults.mlxContextWindowTokens(deviceMaximum: 8_192) == 8_192)
        defaults.set(100, forKey: AppSettingsKeys.mlxContextWindowTokens)
        #expect(defaults.mlxContextWindowTokens(deviceMaximum: 8_192) == 512)
        defaults.set(99_999, forKey: AppSettingsKeys.mlxContextWindowTokens)
        #expect(defaults.mlxContextWindowTokens(deviceMaximum: 8_192) == 8_192)
        defaults.set(4_096, forKey: AppSettingsKeys.mlxContextWindowTokens)
        #expect(defaults.mlxContextWindowTokens(deviceMaximum: 8_192) == 4_096)
    }

    @Test func settingsDoNotRoundTripAnOutOfRangeOutputTokenBudget() throws {
        // Worth stating plainly: this field is the one place where what you
        // save and what you load back differ, because the store clamps it.
        let defaults = try makeScratchDefaults(#function)
        let baseline = AppSettingsDraft.load(from: defaults)
        var draft = baseline
        draft.mlxMaxOutputTokens = 4_096

        draft.persist(to: defaults, comparedTo: baseline)

        #expect(AppSettingsDraft.load(from: defaults).mlxMaxOutputTokens == 1_024)
    }

    @Test func settingsPersistOnlyWritesKeysThatActuallyChanged() throws {
        let defaults = try makeScratchDefaults(#function)
        let baseline = AppSettingsDraft.load(from: defaults)
        var draft = baseline
        draft.messageFontSize = 19

        draft.persist(to: defaults, comparedTo: baseline)

        #expect(defaults.object(forKey: AppSettingsKeys.messageFontSize) as? Double == 19)
        // Untouched preferences must not be materialised, so their defaults
        // keep applying instead of being frozen at today's values.
        #expect(defaults.object(forKey: AppSettingsKeys.appAppearance) == nil)
        #expect(defaults.object(forKey: AppSettingsKeys.developerModeEnabled) == nil)
        #expect(defaults.object(forKey: AppSettingsKeys.autoDeleteDays) == nil)
    }

    @Test func settingsResetReturnsEveryFieldToItsDefault() {
        var draft = AppSettingsDraft.defaults()
        draft.messageFontSize = 22
        draft.developerModeEnabled = true

        draft.resetToDefaults()

        #expect(draft == AppSettingsDraft.defaults())
    }

    // MARK: - Attachments
}
