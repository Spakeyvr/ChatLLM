//
//  On_Device_LLM_ChatTests.swift
//  On-Device_LLM_ChatTests
//
//  Created by Nevio on 10/24/25.
//

import Testing
import Foundation
import MLXLMCommon
import SwiftUI
import SwiftData
@testable import On_Device_LLM_Chat

@MainActor
@Suite(.serialized)
struct On_Device_LLM_ChatTests {

    @Test func webSearchBridgeStoresInvocationFromDirectExecution() async throws {
        let bridge = try makeSearchBridge()

        let result = try await bridge.executeSearch(query: "latest swift release")

        #expect(result.contains("Swift 6.2 Released"))
        #expect(result.contains("https://example.com/swift"))
        #expect(bridge.allInvocations.count == 1)
        #expect(bridge.allInvocations.first?.query == "latest swift release")
    }

    @Test func webSearchBridgeExecutesMLXToolCall() async throws {
        let bridge = try makeSearchBridge()
        let toolName = AppWebSearchToolBridge.toolName
        let toolCall = MLXToolCall(function: .init(name: toolName, arguments: [
            "query": "qwen tool calling"
        ]))

        let result = try await bridge.dispatchMLXToolCall(toolCall)

        #expect(result.contains("Swift 6.2 Released"))
        #expect(bridge.allInvocations.count == 1)
        #expect(bridge.allInvocations.first?.query == "qwen tool calling")
        let function = bridge.mlxToolSpec["function"] as? [String: any Sendable]
        #expect(function?["name"] as? String == toolName)
    }

    @Test func tavilySearchServiceUsesAuthorizationHeader() async throws {
        let service = try makeSearchService()

        _ = try await service.search(query: " latest swift release ")

        let request = try #require(MockTavilyURLProtocol.lastRequest)
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer test-key")
        #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
        let body = try #require(MockTavilyURLProtocol.lastRequestBody)
        let payload = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(payload["api_key"] == nil)
        #expect(payload["query"] as? String == "latest swift release")
        #expect(payload["max_results"] as? Int == 2)
        #expect(payload["topic"] as? String == "news")
        #expect(payload["time_range"] as? String == "week")
        #expect(payload["search_depth"] as? String == "basic")
        #expect(payload["auto_parameters"] as? Bool == true)
        #expect(payload["include_answer"] as? String == "basic")
        #expect(payload["include_raw_content"] as? Bool == false)
    }

    @Test func tavilySearchServiceKeepsCompactShapeForExactMatchQueries() async throws {
        let service = try makeSearchService()

        _ = try await service.search(query: #" "swift 6.2 release notes" "#)

        let body = try #require(MockTavilyURLProtocol.lastRequestBody)
        let payload = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(payload["search_depth"] as? String == "basic")
        #expect(payload["exact_match"] == nil)
        #expect(payload["chunks_per_source"] == nil)
        #expect(payload["include_answer"] as? String == "basic")
    }

    @Test func tavilySearchServiceMapsUnauthorizedResponsesToInvalidAPIKey() async throws {
        MockTavilyURLProtocol.responseStatusCode = 401
        MockTavilyURLProtocol.responseData = """
        {
          "detail": "Invalid API key"
        }
        """.data(using: .utf8)!

        let service = try makeSearchService()
        var receivedInvalidAPIKey = false

        do {
            _ = try await service.search(query: "latest swift release")
        } catch let error as TavilySearchError {
            if case .invalidAPIKey = error {
                receivedInvalidAPIKey = true
            }
        }

        #expect(receivedInvalidAPIKey)
    }

    @Test func webSearchBridgeEnforcesInvocationLimit() async throws {
        let bridge = try makeSearchBridge()

        for index in 0..<AppWebSearchToolBridge.maxInvocations {
            let result = try await bridge.executeSearch(query: "query \(index)")
            #expect(result.contains("Swift 6.2 Released"))
        }

        let limited = try await bridge.executeSearch(query: "query overflow")

        #expect(bridge.allInvocations.count == AppWebSearchToolBridge.maxInvocations)
        #expect(bridge.searchLimitReached)
        #expect(AppWebSearchToolBridge.isSearchLimitToolResponse(limited))
    }

    @Test func webSearchBridgeReturnsRecoverableErrorWhenSearchFails() async throws {
        MockTavilyURLProtocol.responseStatusCode = 500
        MockTavilyURLProtocol.responseData = """
        {
          "detail": "Upstream search unavailable"
        }
        """.data(using: .utf8)!

        let service = try makeSearchService()
        let bridge = AppWebSearchToolBridge(searchService: service)

        let result = try await bridge.executeSearch(query: "latest swift release")

        #expect(AppWebSearchToolBridge.isInternalToolErrorResponse(result))
        #expect(result.contains("Upstream search unavailable"))
        #expect(bridge.allInvocations.isEmpty)
    }

    @Test func qwenToolTemplateInspectionPrefersXMLFunctionFormat() {
        let packageContents = """
        <tool_call>
        <function=webSearch>
        <parameter=query>
        latest OpenAI model
        </parameter>
        </function>
        </tool_call>
        """

        let inferredFormat = MLXModelManager.inferToolCallFormat(
            packageContents: packageContents,
            modelType: "qwen3_5"
        )

        #expect(inferredFormat == .xmlFunction)
        #expect(MLXModelManager.usesWrappedXMLToolCallTemplate(packageContents: packageContents))
    }

    @Test func wrappedXMLToolCallStreamFilterSuppressesToolMarkup() async {
        let filter = WrappedXMLToolCallStreamFilter()

        let firstChunk = await filter.consume("I will check that.\n<tool")
        let secondChunk = await filter.consume("_call>\n<function=webSearch>")
        let thirdChunk = await filter.consume("\n<parameter=query>\nlatest OpenAI model\n</parameter>")
        await filter.didDispatchToolCall()
        let trailingChunk = await filter.finish()

        #expect(firstChunk == "I will check that.\n")
        #expect(secondChunk == nil)
        #expect(thirdChunk == nil)
        #expect(trailingChunk == nil)
    }

    @Test func qwenToolArgumentNormalizationUnwrapsJSONValueStrings() {
        let normalized = MLXModelManager.normalizedToolArguments([
            "query": JSONValue.string("OpenAI latest model")
        ])

        #expect(normalized["query"] as? String == "OpenAI latest model")
    }

    @Test func mlxToolLoopDisablesToolsOnlyAfterSearchLimitResponse() {
        #expect(
            MLXModelManager.shouldDisableTools(
                after: AppWebSearchToolBridge.searchLimitToolResponse(limit: AppWebSearchToolBridge.maxInvocations)
            )
        )
        #expect(
            !MLXModelManager.shouldDisableTools(
                after: AppWebSearchToolBridge.invalidArgumentsToolResponse()
            )
        )
    }

    @Test func qwenWrappedToolResponseContentUsesExpectedEnvelope() {
        let content = MLXModelManager.wrappedToolResponseContent("Search result body")

        #expect(content == "<tool_response>\nSearch result body\n</tool_response>")
        #expect(content.starts(with: "<tool_response>"))
        #expect(content.hasSuffix("</tool_response>"))
    }

    @Test func qwenToolResponsesReuseWrappedUserPromptCompatibilityPath() {
        let manager = MLXModelManager()
        let model = try! #require(manager.model(withID: "qwen3.5-2b-4bit"))

        let content = MLXModelManager.toolResponsePromptContent(
            for: model,
            toolResult: "Search result body"
        )
        let role = MLXModelManager.toolResponsePromptRole(for: model)

        #expect(content == "<tool_response>\nSearch result body\n</tool_response>")
        #expect(role == .user)
    }

    @Test func qwenToolResponsesRequireExplicitMessageHistoryAcrossToolLoop() {
        let manager = MLXModelManager()
        let model = try! #require(manager.model(withID: "qwen3.5-2b-4bit"))

        #expect(MLXModelManager.requiresExplicitMessageHistoryForToolLoop(for: model))
    }

    @Test func nonQwenModelsKeepNativeToolRoleForToolResponses() {
        let nonQwenModel = MLXModelManager.MLXModelInfo(
            id: "test-model",
            name: "Test",
            localDirName: "TestModel",
            hfRepoId: "example/test-model",
            parameters: "1B",
            downloadSizeLabel: "1 GB",
            loadPolicy: .qwenMultimodal,
            description: "Test model",
            contextLength: 4096,
            isAvailable: true,
            supportsReasoning: false,
            supportsNativeImages: false,
            requiredProcessorClass: nil,
            minimumPhoneMemoryBytes: nil,
            minimumPhoneMemoryForToolCallsBytes: nil
        )

        let content = MLXModelManager.toolResponsePromptContent(
            for: nonQwenModel,
            toolResult: "Search result body"
        )
        let role = MLXModelManager.toolResponsePromptRole(for: nonQwenModel)

        #expect(content == "Search result body")
        #expect(role == .tool)
        #expect(!MLXModelManager.requiresExplicitMessageHistoryForToolLoop(for: nonQwenModel))
    }

    @Test func mlxToolLoopHasHardStopResponse() {
        let response = MLXModelManager.excessiveToolCallToolResponse(
            maximum: MLXModelManager.maxToolInvocationsPerResponse
        )

        #expect(response.contains("limit reached"))
        #expect(response.contains("\(MLXModelManager.maxToolInvocationsPerResponse)"))
    }

    @Test func kvQuantizationConfigurationIsDisabledForPersistentSimpleCachePolicy() {
        let config = MLXModelManager.kvQuantizationConfiguration(
            cachePolicy: .persistentSimple
        )

        #expect(config == nil)
    }

    @Test func kvQuantizationConfigurationUsesExpectedDefaultsOnQuantizedPersistentCachePolicy() {
        let config = MLXModelManager.kvQuantizationConfiguration(
            cachePolicy: .persistentQuantizedSimple
        )

        #expect(config?.bits == 8)
        #expect(config?.groupSize == 64)
        #expect(config?.startStep == MLXModelManager.defaultQuantizedKVStartStep)
    }

    @Test func kvQuantizationConfigurationIsDisabledForBoundedRotatingCachePolicy() {
        let config = MLXModelManager.kvQuantizationConfiguration(
            cachePolicy: .boundedRotating(maxKVSize: 4096)
        )

        #expect(config == nil)
    }

    @Test func cachePolicyKeepsNormalTurnsOnPersistentSimpleCacheWhenQuantizationIsOffOnHighMemoryDevices() {
        let cachePolicy = MLXModelManager.cachePolicy(
            isEnabled: false,
            hasTools: true,
            hasMedia: false,
            prefersBoundedCache: false,
            memoryConstrained: false
        )

        #expect(cachePolicy == .persistentSimple)
    }

    @Test func cachePolicyUsesBoundedRotatingFallbackWhenMemoryConstrained() {
        let cachePolicy = MLXModelManager.cachePolicy(
            isEnabled: true,
            hasTools: true,
            hasMedia: false,
            prefersBoundedCache: false,
            memoryConstrained: true
        )

        #expect(cachePolicy == .boundedRotating(maxKVSize: 4096))
    }

    @Test func cachePolicyUsesBoundedRotatingCacheOnLowMemoryDevices() {
        let cachePolicy = MLXModelManager.cachePolicy(
            isEnabled: true,
            hasTools: false,
            hasMedia: false,
            prefersBoundedCache: true,
            memoryConstrained: false
        )

        #expect(cachePolicy == .boundedRotating(maxKVSize: 4096))
    }

    @Test func generationConfigurationUsesQuantizedPersistentCacheOnHighMemoryDevices() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: true,
            hasTools: true,
            hasMedia: false,
            memoryConstrained: false,
            prefersBoundedCache: false,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 32768
        )

        #expect(configuration.maxTokens == 4096)
        #expect(configuration.maxKVSize == nil)
        #expect(configuration.kvQuantization?.bits == 8)
        #expect(configuration.cachePolicy == .persistentQuantizedSimple)
        #expect(configuration.usesQuantizedToolCacheStrategy)
    }

    @Test func generationConfigurationKeepsToolTurnsUnboundedWhenQuantizationIsOffOnHighMemoryDevices() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: false,
            hasTools: true,
            hasMedia: false,
            memoryConstrained: false,
            prefersBoundedCache: false,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 32768
        )

        #expect(configuration.maxTokens == 4096)
        #expect(configuration.maxKVSize == nil)
        #expect(configuration.kvQuantization == nil)
        #expect(configuration.cachePolicy == .persistentSimple)
        #expect(!configuration.usesQuantizedToolCacheStrategy)
    }

    @Test func generationConfigurationUsesBoundedRotatingCacheOnLowMemoryDevicesWithTools() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: true,
            hasTools: true,
            hasMedia: false,
            memoryConstrained: false,
            prefersBoundedCache: true,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 8192
        )

        #expect(configuration.maxTokens == 4096)
        #expect(configuration.maxKVSize == 4096)
        #expect(configuration.kvQuantization == nil)
        #expect(configuration.cachePolicy == .boundedRotating(maxKVSize: 4096))
        #expect(!configuration.usesQuantizedToolCacheStrategy)
    }

    @Test func generationConfigurationUsesBoundedRotatingCacheOnLowMemoryDevicesWithMedia() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: true,
            hasTools: false,
            hasMedia: true,
            memoryConstrained: false,
            prefersBoundedCache: true,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 8192
        )

        #expect(configuration.maxTokens == 2048)
        #expect(configuration.maxKVSize == 4096)
        #expect(configuration.kvQuantization == nil)
        #expect(configuration.cachePolicy == .boundedRotating(maxKVSize: 4096))
    }

    @Test func generationConfigurationClampsLowMemoryMaxKVSizeToConfiguredContextWindow() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: true,
            hasTools: false,
            hasMedia: false,
            memoryConstrained: false,
            prefersBoundedCache: true,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 2048
        )

        #expect(configuration.maxKVSize == 2048)
        #expect(configuration.cachePolicy == .boundedRotating(maxKVSize: 2048))
        #expect(configuration.kvQuantization == nil)
    }

    @Test func promptTokenCostFallsBackToHeuristicWhenTokenizerCountUnavailable() {
        let cost = ChatViewModel.promptTokenCost(
            tokenizedContentTokenCount: nil,
            fallbackContent: "1234567890"
        )

        #expect(cost == 24)
    }

    @Test func promptTokenCostUsesTokenizerCountWhenAvailable() {
        let cost = ChatViewModel.promptTokenCost(
            tokenizedContentTokenCount: 40,
            fallbackContent: "short"
        )

        #expect(cost == 52)
    }

    @Test func qwen4BRequiresEightGigabytesOnIPhone() {
        let profile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let model = try! #require(manager.model(withID: "qwen3.5-4b-mixed36"))

        #expect(!profile.supportsModel(model))
        #expect(profile.availabilityIssue(for: model)?.contains("8 GB") == true)
    }

    @Test func qwen4BToolCallsRequireTwelveGigabytesOnIPhone() {
        let profile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let model = try! #require(manager.model(withID: "qwen3.5-4b-mixed36"))

        #expect(profile.supportsModel(model))
        #expect(!manager.supportsToolCalls(for: model))
        #expect(manager.toolCallIssue(for: model)?.contains("12 GB") == true)
    }

    @Test func qwen2BAllowsSixGigabyteIPhonesButBlocksToolCalls() {
        let profile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let model = try! #require(manager.model(withID: "qwen3.5-2b-4bit"))

        #expect(profile.supportsModel(model))
        #expect(!manager.supportsToolCalls(for: model))
        #expect(manager.toolCallIssue(for: model)?.contains("8 GB") == true)
    }

    @Test func qwen2BEnablesToolCallsAtEightGigabytesOnIPhone() {
        let profile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let model = try! #require(manager.model(withID: "qwen3.5-2b-4bit"))

        #expect(profile.supportsModel(model))
        #expect(manager.supportsToolCalls(for: model))
        #expect(manager.toolCallIssue(for: model) == nil)
    }

    @Test func qwenEightGigabyteTierTreatsSlightlyUnderreportedPhoneAsEightGigabytes() {
        let marketedEightGigabytesButReportedBelowEightGiB: UInt64 = 7_950_000_000
        let profile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: marketedEightGigabytesButReportedBelowEightGiB
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let twoBModel = try! #require(manager.model(withID: "qwen3.5-2b-4bit"))
        let fourBModel = try! #require(manager.model(withID: "qwen3.5-4b-mixed36"))

        #expect(profile.supportsModel(twoBModel))
        #expect(manager.supportsToolCalls(for: twoBModel))
        #expect(profile.supportsModel(fourBModel))
        #expect(!manager.supportsToolCalls(for: fourBModel))
    }

    @Test func twelveGigabyteTierDisablesLowMemoryKVFallbackAfterNormalization() {
        let marketedTwelveGigabytesButReportedBelowTwelveGiB: UInt64 = 11_900_000_000
        let profile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: marketedTwelveGigabytesButReportedBelowTwelveGiB
        )

        #expect(!profile.hasLowMemoryForPersistentKVCache)
    }

    @Test func eightGigabyteTierStillUsesLowMemoryKVFallback() {
        let profile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )

        #expect(profile.hasLowMemoryForPersistentKVCache)
    }

    @Test func contextWindowMaximumTracksSixEightAndTwelveGigabyteTiers() {
        let sixGigabyteProfile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let eightGigabyteProfile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )
        let twelveGigabyteProfile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 12 * MLXDeviceSupportProfile.gibibyte
        )

        #expect(sixGigabyteProfile.maxContextWindowTokens == 1_024)
        #expect(eightGigabyteProfile.maxContextWindowTokens == 2_048)
        #expect(twelveGigabyteProfile.maxContextWindowTokens == 6_144)
    }

    @Test func userDefaultsContextWindowClampsToFiveHundredTwelveTokenMinimum() {
        let defaults = UserDefaults.standard
        let key = "mlxContextWindowTokens"
        let hadExistingValue = defaults.object(forKey: key) != nil
        let previousValue = defaults.integer(forKey: key)

        defaults.set(128, forKey: key)
        defer {
            if hadExistingValue {
                defaults.set(previousValue, forKey: key)
            } else {
                defaults.removeObject(forKey: key)
            }
        }

        #expect(defaults.mlxContextWindowTokens(deviceMaximum: 1_024) == 512)
        #expect(defaults.mlxContextWindowTokens(deviceMaximum: 2_048) == 512)
        #expect(defaults.mlxContextWindowTokens(deviceMaximum: 6_144) == 512)
    }

    @Test func qwenIPhoneMemoryLimitsDoNotApplyToNonPhoneDevices() {
        let profile = MLXDeviceSupportProfile(
            isPhone: false,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let fourBModel = try! #require(manager.model(withID: "qwen3.5-4b-mixed36"))

        #expect(profile.supportsModel(fourBModel))
        #expect(manager.supportsToolCalls(for: fourBModel))
        #expect(profile.availabilityIssue(for: fourBModel) == nil)
    }

    @Test func settingsSheetDescribesAdaptiveKVCacheBehavior() {
        #expect(SettingsSheet.mlxKVCacheInfoMessage.contains("Enabled by default"))
        #expect(SettingsSheet.mlxKVCacheInfoMessage.contains("less than 12 GB"))
        #expect(SettingsSheet.mlxKVCacheInfoMessage.contains("does not support quantizing rotating KV caches"))
        #expect(SettingsSheet.mlxKVCacheAccessibilityHint.contains("Low-memory devices"))
    }

    @Test func userDefaultsKVQuantizationDefaultsToEnabledWhenUnset() {
        let key = "mlxEnableKVCacheQuantization"
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

        #expect(defaults.mlxEnableKVCacheQuantization)
    }

    @Test func resetSettingsRestoresKVQuantizationDefault() {
        var draft = AppSettingsDraft.defaults()
        draft.mlxEnableKVCacheQuantization = false

        draft.resetToDefaults()

        #expect(draft.mlxEnableKVCacheQuantization)
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

    @Test func appSettingsDraftDoesNotPersistUntilSave() throws {
        let suiteName = "settings-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }

        defaults.set("Existing prompt", forKey: AppSettingsKeys.defaultSystemPrompt)
        defaults.set(false, forKey: AppSettingsKeys.developerModeEnabled)
        defaults.set(false, forKey: AppSettingsKeys.disableToolCalls)

        var draft = AppSettingsDraft.load(from: defaults)
        draft.defaultSystemPrompt = "Draft only"
        draft.developerModeEnabled = true
        draft.disableToolCalls = true

        #expect(defaults.string(forKey: AppSettingsKeys.defaultSystemPrompt) == "Existing prompt")
        #expect(defaults.bool(forKey: AppSettingsKeys.developerModeEnabled) == false)
        #expect(defaults.bool(forKey: AppSettingsKeys.disableToolCalls) == false)

        draft.persist(to: defaults)

        #expect(defaults.string(forKey: AppSettingsKeys.defaultSystemPrompt) == "Draft only")
        #expect(defaults.bool(forKey: AppSettingsKeys.developerModeEnabled))
        #expect(defaults.bool(forKey: AppSettingsKeys.disableToolCalls))
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

    @Test func resetForRegenerationClearsDeveloperMetadata() {
        let conversation = Conversation(title: "Test")
        let message = Message(
            role: .assistant,
            text: "Visible",
            order: 1,
            conversation: conversation,
            isFinal: true,
            rawText: "<thinking>raw</thinking>",
            generationBackend: "MLX",
            generationModelName: "Qwen",
            generationStartedAt: Date(timeIntervalSince1970: 10),
            generationCompletedAt: Date(timeIntervalSince1970: 12)
        )

        message.resetForRegeneration()

        #expect(message.rawText == nil)
        #expect(message.generationBackend == nil)
        #expect(message.generationModelName == nil)
        #expect(message.generationStartedAt == nil)
        #expect(message.generationCompletedAt == nil)
    }

    @Test func searchInvocationMergePreservesAnchorsAcrossFinalization() {
        let preservedID = UUID()
        let newID = UUID()
        let liveInvocations = [
            SearchInvocation(
                id: preservedID,
                query: "swift release date",
                results: "Result A"
            ),
            SearchInvocation(
                id: newID,
                query: "swift evolution",
                results: "Result B"
            )
        ]
        let existingInvocations = [
            SearchInvocation(
                id: preservedID,
                query: "swift release date",
                results: "Result A",
                anchorStepNumber: 2
            )
        ]

        let merged = ChatViewModel.mergeSearchInvocations(
            liveInvocations,
            preservingAnchorsFrom: existingInvocations,
            currentChunkCount: 3
        )

        #expect(merged.count == 2)
        #expect(merged.first(where: { $0.id == preservedID })?.anchorStepNumber == 2)
        #expect(merged.first(where: { $0.id == newID })?.anchorStepNumber == 3)
    }

    @Test func clearingSearchInvocationsAlsoClearsLegacySearchQuery() {
        let conversation = Conversation(title: "Test")
        let message = Message(
            role: .assistant,
            text: "Answer",
            order: 1,
            conversation: conversation
        )

        message.searchInvocations = [
            SearchInvocation(
                query: "latest swift release",
                results: "Swift 6.2 Released"
            )
        ]
        #expect(message.searchQuery == "latest swift release")

        message.searchInvocations = nil

        #expect(message.searchInvocations == nil)
        #expect(message.searchQuery == nil)
    }

    @Test func assistantPlaceholderTracksForcedWebSearchSeparatelyFromSearchHistory() throws {
        let viewModel = try makeViewModel()

        let forced = viewModel.appendAssistantPlaceholder(
            isReasoningMode: false,
            searchQuery: "latest swift release",
            requiresWebSearch: true
        )
        let autonomous = viewModel.appendAssistantPlaceholder(isReasoningMode: false)
        autonomous.searchInvocations = [
            SearchInvocation(
                query: "latest swift release",
                results: "Swift 6.2 Released"
            )
        ]

        #expect(forced.requiresWebSearch == true)
        #expect(forced.searchQuery == "latest swift release")
        #expect(autonomous.requiresWebSearch != true)
        #expect(autonomous.searchQuery == "latest swift release")
    }

    @Test func currentDateTimeContextIncludesExactDateAndYearHint() {
        let referenceDate = Date(timeIntervalSince1970: 1_772_884_800) // 2026-03-07 12:00:00 UTC
        let utc = TimeZone(secondsFromGMT: 0)!

        let context = ChatViewModel.currentDateTimeContext(
            referenceDate: referenceDate,
            timeZone: utc
        )

        #expect(context.contains("Saturday, March 7, 2026 at 12:00 UTC"))
        #expect(context.contains("current year"))
        #expect(context.contains("exact date"))
    }

    @Test func mlxSessionReusePolicyAllowsKeyedSessionReuseAcrossModes() {
        #expect(MLXModelManager.shouldPersistSessionAcrossTurns(enableThinking: false, hasTools: false, hasMedia: false))
        #expect(MLXModelManager.shouldPersistSessionAcrossTurns(enableThinking: true, hasTools: false, hasMedia: false))
        #expect(MLXModelManager.shouldPersistSessionAcrossTurns(enableThinking: false, hasTools: true, hasMedia: false))
        #expect(MLXModelManager.shouldPersistSessionAcrossTurns(enableThinking: false, hasTools: false, hasMedia: true))
    }

    @Test func mlxTimeContextGateSkipsOrdinaryTurns() {
        let shouldInject = ChatViewModel.shouldInjectMLXCurrentDateTimeContext(
            latestUserText: "Explain Swift actors simply.",
            forceWebSearch: false,
            webSearchEnabled: true,
            referenceDate: Date(timeIntervalSince1970: 1_772_884_800)
        )

        #expect(!shouldInject)
    }

    @Test func mlxTimeContextGateIncludesTimeSensitiveTurns() {
        let shouldInject = ChatViewModel.shouldInjectMLXCurrentDateTimeContext(
            latestUserText: "What is the latest Swift release today?",
            forceWebSearch: false,
            webSearchEnabled: false,
            referenceDate: Date(timeIntervalSince1970: 1_772_884_800)
        )

        #expect(shouldInject)
    }

    @Test func mlxTimeContextGateIncludesForcedWebSearchTurns() {
        let shouldInject = ChatViewModel.shouldInjectMLXCurrentDateTimeContext(
            latestUserText: "Compare the newest Xcode beta to stable.",
            forceWebSearch: true,
            webSearchEnabled: true,
            referenceDate: Date(timeIntervalSince1970: 1_772_884_800)
        )

        #expect(shouldInject)
    }

    @Test func mlxTimeContextGateDoesNotTreatEmbeddedSubstringsAsTimeSensitive() {
        let shouldInject = ChatViewModel.shouldInjectMLXCurrentDateTimeContext(
            latestUserText: "Explain how snow forms in clouds.",
            forceWebSearch: false,
            webSearchEnabled: true,
            referenceDate: Date(timeIntervalSince1970: 1_772_884_800)
        )

        #expect(!shouldInject)
    }

    @Test func requiresWebSearchDoesNotTriggerOnEmbeddedKeywordSubstrings() throws {
        let viewModel = try makeViewModel()

        #expect(!viewModel.requiresWebSearch(for: "Explain how snow forms in clouds."))
    }

    @Test func webSearchSystemPromptUsesCurrentCalendarYear() {
        let year = Calendar.current.component(.year, from: Date())
        let sentinelYear = year == 2026 ? 2025 : 2026

        let prompt = ChatViewModel.webSearchSystemPrompt(
            reasoningEnabled: false,
            forceSearchRequired: false
        )

        #expect(prompt.contains("add \(year) to the search query"))
        #expect(!prompt.contains("add \(sentinelYear) to the search query"))
    }

    @Test func webSearchSystemPromptMatchesSearchInvocationLimit() {
        let prompt = ChatViewModel.webSearchSystemPrompt(
            reasoningEnabled: true,
            forceSearchRequired: false
        )

        #expect(prompt.contains("up to and only up to \(AppWebSearchToolBridge.maxInvocations) searches"))
    }

    @Test func conversationSearchableVisibleTextUsesDisplayedContent() {
        let conversation = Conversation(title: "Vision")
        let user = Message(
            role: .user,
            text: """
            [HIDDEN_IMAGE_CONTEXT]ocr: secret token[/HIDDEN_IMAGE_CONTEXT]
            What does the sign say?
            """,
            order: 0,
            conversation: conversation,
            isFinal: true
        )
        let assistant = Message(
            role: .assistant,
            text: "intermediate",
            order: 1,
            conversation: conversation,
            isFinal: true,
            reasoning: "internal reasoning",
            finalAnswer: "The sign says Welcome.",
            isReasoningMode: true
        )
        conversation.messages = [user, assistant]

        #expect(conversation.searchableVisibleText.contains("What does the sign say?"))
        #expect(conversation.searchableVisibleText.contains("The sign says Welcome."))
        #expect(!conversation.searchableVisibleText.contains("secret token"))
        #expect(!conversation.searchableVisibleText.contains("internal reasoning"))
    }

    @Test func chatViewOnlyInterceptsWebURLsForInAppBrowser() throws {
        #expect(ChatView.shouldPresentInAppBrowser(for: try #require(URL(string: "https://example.com"))))
        #expect(!ChatView.shouldPresentInAppBrowser(for: try #require(URL(string: "mailto:test@example.com"))))
        #expect(!ChatView.shouldPresentInAppBrowser(for: try #require(URL(string: "tel:+123456"))))
    }

    @Test func networkMonitorStartsOfflineUntilFirstPathResolves() async {
        let monitor = FakePathMonitor()
        let networkMonitor = NetworkMonitor(
            monitor: monitor,
            queue: DispatchQueue(label: "test.network.monitor")
        )

        #expect(!networkMonitor.isConnected)

        monitor.emit(.satisfied)
        await Task.yield()
        try? await Task.sleep(for: .milliseconds(20))

        #expect(networkMonitor.isConnected)
    }

    @Test func composerReturnKeyBehaviorSendsOnlyWhenEnabledAndSendable() {
        #expect(
            ComposerReturnKeyBehavior.action(
                sendOnReturn: true,
                canSend: true,
                isGenerating: false,
                replacementText: "\n"
            ) == .send
        )
        #expect(
            ComposerReturnKeyBehavior.action(
                sendOnReturn: false,
                canSend: true,
                isGenerating: false,
                replacementText: "\n"
            ) == .insertNewline
        )
        #expect(
            ComposerReturnKeyBehavior.action(
                sendOnReturn: true,
                canSend: false,
                isGenerating: false,
                replacementText: "\n"
            ) == .insertNewline
        )
    }

    @Test func disabledHapticsSuppressFeedbackDispatch() throws {
        let suiteName = "haptics-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }

        defaults.enableHapticsPreference = false
        var invoked = false

        let didPerform = AppHaptics.performIfEnabled(defaults: defaults) {
            invoked = true
        }

        #expect(!didPerform)
        #expect(!invoked)
    }

    @Test func searchSessionParserKeepsBulletAnswerAfterThinkingCloses() {
        let parsed = ChatViewModel.parseReasoningResponseForSearchSessionText(
            """
            I should verify the release notes.
            </think>

            Based on the search, Swift 6.2 is the latest stable release.

            - Adds more concurrency fixes
            - Available now
            """
        )

        #expect(parsed.reasoning == "I should verify the release notes.")
        #expect(parsed.finalAnswer == """
        Based on the search, Swift 6.2 is the latest stable release.

        - Adds more concurrency fixes
        - Available now
        """)
    }

    @Test func streamingReasoningUpdateKeepsPostSearchReasoningOutOfAnswerBubble() throws {
        let originalBackend = ModelBackendBridge.shared.selectedBackend
        ModelBackendBridge.shared.selectedBackend = .mlx
        defer { ModelBackendBridge.shared.selectedBackend = originalBackend }

        let viewModel = try makeViewModel()
        let conversation = viewModel.conversation
        let message = Message(
            role: .assistant,
            text: "",
            order: 1,
            conversation: conversation,
            isReasoningMode: true
        )
        message.searchInvocations = [
            SearchInvocation(
                query: "latest swift release",
                results: "Swift 6.2 Released"
            )
        ]
        message.streamingReasoningPhase = .postToolReasoning

        viewModel.updateMessageWithReasoningContent(
            message,
            fullText: """
            I should verify the release notes.
            </think>

            I should cross-check whether 6.2 is stable or still in beta.
            """,
            finalize: false
        )

        #expect(message.reasoning == """
        I should verify the release notes.

        I should cross-check whether 6.2 is stable or still in beta.
        """)
        #expect(message.finalAnswer == nil)
        #expect(message.text.isEmpty)
        #expect(message.streamingReasoningPhase == .postToolReasoning)
    }

    @Test func streamingReasoningUpdateShowsAnswerAfterPostSearchReasoningTurnsIntoAnswer() throws {
        let originalBackend = ModelBackendBridge.shared.selectedBackend
        ModelBackendBridge.shared.selectedBackend = .mlx
        defer { ModelBackendBridge.shared.selectedBackend = originalBackend }

        let viewModel = try makeViewModel()
        let conversation = viewModel.conversation
        let message = Message(
            role: .assistant,
            text: "",
            order: 1,
            conversation: conversation,
            isReasoningMode: true
        )
        message.searchInvocations = [
            SearchInvocation(
                query: "latest swift release",
                results: "Swift 6.2 Released"
            )
        ]
        message.streamingReasoningPhase = .postToolReasoning

        viewModel.updateMessageWithReasoningContent(
            message,
            fullText: """
            I should verify the release notes.
            </think>

            Based on the search, Swift 6.2 is the latest stable release.
            """,
            finalize: false
        )

        #expect(message.reasoning == "I should verify the release notes.")
        #expect(message.finalAnswer == "Based on the search, Swift 6.2 is the latest stable release.")
        #expect(message.text == "Based on the search, Swift 6.2 is the latest stable release.")
        #expect(message.streamingReasoningPhase == .finalAnswer)
    }

    @Test func buildPromptExcludesHistoricalReasoningFromContext() throws {
        let viewModel = try makeViewModel()
        viewModel.conversation.reasoningMode = true

        let user = Message(
            role: .user,
            text: "What changed in Swift 6.2?",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        let assistant = Message(
            role: .assistant,
            text: "Swift 6.2 is now available.",
            order: 1,
            conversation: viewModel.conversation,
            isFinal: true,
            reasoning: "I should think through the release timeline before answering.",
            finalAnswer: "Swift 6.2 is now available.",
            isReasoningMode: true
        )
        let nextUser = Message(
            role: .user,
            text: "Should I upgrade now?",
            order: 2,
            conversation: viewModel.conversation,
            isFinal: true
        )

        viewModel.conversation.messages.append(user)
        viewModel.conversation.messages.append(assistant)
        viewModel.conversation.messages.append(nextUser)

        let prompt = viewModel.buildPrompt(upToOrderExclusive: 3, currentReasoningActive: true)

        #expect(prompt.contains("Assistant: Swift 6.2 is now available."))
        #expect(!prompt.contains("I should think through the release timeline"))
        #expect(!prompt.contains("<thinking>"))
    }

    @Test func buildPromptIncludesPersistedSystemPrompt() throws {
        let viewModel = try makeViewModel()

        let system = Message(
            role: .system,
            text: "Always answer like a pirate.",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        let user = Message(
            role: .user,
            text: "Say hello",
            order: 1,
            conversation: viewModel.conversation,
            isFinal: true
        )

        viewModel.conversation.messages.append(system)
        viewModel.conversation.messages.append(user)

        let prompt = viewModel.buildPrompt(upToOrderExclusive: 2)

        #expect(prompt.contains("Always answer like a pirate."))
    }

    @Test func buildQwenMessagesExcludeHistoricalReasoningFromContext() async throws {
        let viewModel = try makeViewModel()
        viewModel.conversation.reasoningMode = true

        let user = Message(
            role: .user,
            text: "What changed in Swift 6.2?",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        let assistant = Message(
            role: .assistant,
            text: "Swift 6.2 is now available.",
            order: 1,
            conversation: viewModel.conversation,
            isFinal: true,
            reasoning: "I should think through the release timeline before answering.",
            finalAnswer: "Swift 6.2 is now available.",
            isReasoningMode: true
        )
        let nextUser = Message(
            role: .user,
            text: "Should I upgrade now?",
            order: 2,
            conversation: viewModel.conversation,
            isFinal: true
        )

        viewModel.conversation.messages.append(user)
        viewModel.conversation.messages.append(assistant)
        viewModel.conversation.messages.append(nextUser)

        let messages = try await viewModel.buildQwenMessages(upToOrderExclusive: 3)
        let assistantMessage = try #require(messages.first(where: { $0.role == .assistant }))

        #expect(assistantMessage.content == "Swift 6.2 is now available.")
        #expect(!assistantMessage.content.contains("I should think through the release timeline"))
        #expect(!assistantMessage.content.contains("<think>"))
    }

    @Test func buildQwenMessagesIncludePersistedSystemPrompt() async throws {
        let viewModel = try makeViewModel()

        let system = Message(
            role: .system,
            text: "Respond in terse bullet points.",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        let user = Message(
            role: .user,
            text: "Summarize SwiftData",
            order: 1,
            conversation: viewModel.conversation,
            isFinal: true
        )

        viewModel.conversation.messages.append(system)
        viewModel.conversation.messages.append(user)

        let messages = try await viewModel.buildQwenMessages(upToOrderExclusive: 2)
        let systemMessage = try #require(messages.first(where: { $0.role == .system }))

        #expect(systemMessage.content.contains("Respond in terse bullet points."))
    }

    @Test func buildQwenMessagesKeepsOrdinaryMLXPromptStable() async throws {
        let viewModel = try makeViewModel()
        let user = Message(
            role: .user,
            text: "Explain value types in Swift.",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        viewModel.conversation.messages.append(user)

        let messages = try await viewModel.buildQwenMessages(
            upToOrderExclusive: 1,
            toolsAvailable: true,
            forceWebSearch: false
        )
        let systemMessage = try #require(messages.first(where: { $0.role == .system }))

        #expect(!systemMessage.content.contains("Current date and time:"))
    }

    @Test func buildQwenMessagesAddsTimeContextForTimeSensitiveMLXTurn() async throws {
        let viewModel = try makeViewModel()
        let user = Message(
            role: .user,
            text: "What is the latest Swift release today?",
            order: 0,
            conversation: viewModel.conversation,
            isFinal: true
        )
        viewModel.conversation.messages.append(user)

        let messages = try await viewModel.buildQwenMessages(
            upToOrderExclusive: 1,
            toolsAvailable: true,
            forceWebSearch: false
        )
        let systemMessage = try #require(messages.first(where: { $0.role == .system }))

        #expect(systemMessage.content.contains("Current date and time:"))
    }

    @Test func reasoningOnlyStreamingIsNotTreatedAsEmptyOutput() async throws {
        let schema = Schema([Conversation.self, Message.self, MessageAttachment.self])
        let container = try ModelContainer(
            for: schema,
            configurations: [ModelConfiguration(isStoredInMemoryOnly: true)]
        )
        let context = ModelContext(container)
        let conversation = Conversation(title: "Test")
        let viewModel = ChatViewModel(
            generator: PartialReasoningLLMGenerator(),
            context: context,
            conversation: conversation
        )

        let userMessage = Message(
            role: .user,
            text: "Explain your thinking.",
            order: 0,
            conversation: conversation,
            isFinal: true
        )
        conversation.messages.append(userMessage)

        let assistantMessage = Message(
            role: .assistant,
            text: "",
            order: 1,
            conversation: conversation,
            isReasoningMode: true
        )
        conversation.messages.append(assistantMessage)

        let outcome = await viewModel.streamAssistant(
            into: assistantMessage,
            basedOnHistoryUpTo: assistantMessage.order
        )

        #expect(outcome == .succeeded)
        #expect(assistantMessage.generationError == nil)
        #expect(assistantMessage.reasoning == "Analyze the request carefully.")
        #expect(assistantMessage.finalAnswer == nil)
        #expect(assistantMessage.isFinal)
    }

    @Test func longFirstMessageFallsBackToGeneratedTitle() async throws {
        let viewModel = try makeViewModel()
        let longInput = String(repeating: "swiftdata ", count: 80)

        await viewModel.generateChatTitle(fromUserMessage: longInput)

        #expect(viewModel.conversation.title != "New Chat")
        #expect(viewModel.conversation.hasAutoGeneratedTitle)
    }

    @Test func deleteMessageAndMaybeTrimDeletesPersistentModel() async throws {
        let schema = Schema([Conversation.self, Message.self, MessageAttachment.self])
        let container = try ModelContainer(
            for: schema,
            configurations: [ModelConfiguration(isStoredInMemoryOnly: true)]
        )
        let context = ModelContext(container)
        let conversation = Conversation(title: "Test")
        context.insert(conversation)

        let message = Message(
            role: .user,
            text: "Delete me",
            order: 0,
            conversation: conversation,
            isFinal: true
        )
        conversation.messages.append(message)
        try context.save()

        let viewModel = ChatViewModel(
            generator: TestLLMGenerator(),
            context: context,
            conversation: conversation
        )

        await viewModel.deleteMessageAndMaybeTrim(message)

        let remainingMessages = try context.fetch(FetchDescriptor<Message>())
        #expect(!conversation.messages.contains(where: { $0.id == message.id }))
        #expect(!remainingMessages.contains(where: { $0.id == message.id }))
    }

    @Test func selectingModelPersistsChoiceToConversationStore() throws {
        let schema = Schema([Conversation.self, Message.self, MessageAttachment.self])
        let container = try ModelContainer(
            for: schema,
            configurations: [ModelConfiguration(isStoredInMemoryOnly: true)]
        )
        let context = ModelContext(container)
        let conversation = Conversation(title: "Persisted")
        context.insert(conversation)
        try context.save()

        let bridge = ModelBackendBridge.shared
        let originalBackend = bridge.selectedBackend
        let originalModelID = bridge.selectedModelID
        let originalConversation = bridge.activeConversation
        defer {
            bridge.activeConversation = originalConversation
            bridge.selectedBackend = originalBackend
            bridge.selectedModelID = originalModelID
        }

        bridge.bindConversation(conversation)
        bridge.selectedBackend = .foundationModels
        bridge.selectModel("persisted-model", source: "test")

        let verificationContext = ModelContext(container)
        let storedConversation = try #require(
            verificationContext.fetch(FetchDescriptor<Conversation>()).first
        )

        #expect(storedConversation.preferredModelID == "persisted-model")
    }

    @Test func messageDisplayTextStripsThinkTags() {
        let message = Message(
            role: .assistant,
            text: """
            <think>
            Analyze the request carefully.
            </think>

            Final answer: The tags should not be visible.
            """,
            order: 0
        )

        #expect(message.displayText == "The tags should not be visible.")
    }

    @Test func cancelledGenerationBeforeFirstTokenRemovesEmptyAssistantPlaceholder() async throws {
        let schema = Schema([Conversation.self, Message.self, MessageAttachment.self])
        let container = try ModelContainer(
            for: schema,
            configurations: [ModelConfiguration(isStoredInMemoryOnly: true)]
        )
        let context = ModelContext(container)
        let conversation = Conversation(title: "Test")
        let viewModel = ChatViewModel(
            generator: BlockingLLMGenerator(),
            context: context,
            conversation: conversation
        )

        let sendTask = Task { await viewModel.send(userText: "Hello") }

        for _ in 0..<50 {
            if viewModel.isGenerating { break }
            try? await Task.sleep(for: .milliseconds(20))
        }

        #expect(viewModel.isGenerating)
        #expect(conversation.messages.contains(where: { $0.role == .assistant }))

        viewModel.cancelGeneration()
        await sendTask.value

        #expect(!viewModel.isGenerating)
        #expect(conversation.messages.count == 1)
        #expect(conversation.messages.allSatisfy { $0.role == .user })
    }

    @Test func streamingChunkMergePreservesSingleCharacterNumericChunks() {
        let merged = ChatViewModel.mergedStreamingChunk(
            currentText: "10",
            newText: "0"
        )

        #expect(merged == "100")
    }

    @Test func streamingChunkMergeStillCollapsesRealPrefixOverlap() {
        let merged = ChatViewModel.mergedStreamingChunk(
            currentText: "Hello wor",
            newText: "world"
        )

        #expect(merged == "Hello world")
    }

    @Test func webSearchBridgeReturnsRecoverableErrorForMissingQueryArgument() async throws {
        let bridge = try makeSearchBridge()
        let toolCall = MLXToolCall(function: .init(
            name: AppWebSearchToolBridge.toolName,
            arguments: [:]
        ))

        let result = try await bridge.dispatchMLXToolCall(toolCall)

        #expect(AppWebSearchToolBridge.isInternalToolErrorResponse(result))
        #expect(bridge.allInvocations.isEmpty)
    }

    @Test func taskBackedAsyncThrowingStreamCancelsProducerWhenConsumerStops() async {
        let probe = CancellationProbe()
        let stream: AsyncThrowingStream<String, Error> = TaskBackedAsyncThrowingStream.make { continuation in
            Task {
                do {
                    try await Task.sleep(for: .seconds(30))
                    continuation.finish()
                } catch is CancellationError {
                    await probe.markCancelled()
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        let consumer = Task {
            var iterator = stream.makeAsyncIterator()
            _ = try? await iterator.next()
        }

        try? await Task.sleep(for: .milliseconds(50))
        consumer.cancel()
        _ = await consumer.result

        #expect(await waitForCancellation(probe))
    }

    @Test func modelDownloaderRejectsRemoteTraversalPaths() {
        var rejected = false

        do {
            _ = try ModelDownloader.destinationURLs(
                for: "../escape.txt",
                inside: URL(fileURLWithPath: "/tmp/models", isDirectory: true)
            )
        } catch let error as ModelDownloader.DownloadError {
            if case .invalidRemotePath(let file) = error {
                rejected = file == "../escape.txt"
            }
        } catch {}

        #expect(rejected)
    }

    @Test func modelDownloaderKeepsNestedFilesInsideTargetDirectory() throws {
        let targetDir = URL(fileURLWithPath: "/tmp/models", isDirectory: true)
        let urls = try ModelDownloader.destinationURLs(
            for: "nested/config.json",
            inside: targetDir
        )

        #expect(urls.destination.standardizedFileURL.path == "/tmp/models/nested/config.json")
        #expect(urls.temporary.standardizedFileURL.path == "/tmp/models/nested/config.json.download")
    }

    private func makeSearchBridge() throws -> AppWebSearchToolBridge {
        MockTavilyURLProtocol.responseStatusCode = 200
        MockTavilyURLProtocol.responseData = """
        {
          "query": "latest swift release",
          "answer": "Swift 6.2 is the latest stable release.",
          "auto_parameters": {
            "topic": "news",
            "search_depth": "basic",
            "time_range": "week"
          },
          "results": [
            {
              "title": "Swift 6.2 Released",
              "url": "https://example.com/swift",
              "content": "Swift 6.2 adds more concurrency fixes.",
              "score": 0.99,
              "published_date": "2026-02-28"
            }
          ]
        }
        """.data(using: .utf8)!

        let service = try makeSearchService()
        return AppWebSearchToolBridge(searchService: service)
    }

    private func makeSearchService() throws -> TavilySearchService {
        MockTavilyURLProtocol.lastRequest = nil
        MockTavilyURLProtocol.lastRequestBody = nil
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockTavilyURLProtocol.self]
        let session = URLSession(configuration: configuration)
        return try TavilySearchService(apiKey: "test-key", session: session)
    }

    private func makeViewModel(generator: LLMGenerator = TestLLMGenerator()) throws -> ChatViewModel {
        let schema = Schema([Conversation.self, Message.self, MessageAttachment.self])
        let container = try ModelContainer(
            for: schema,
            configurations: [ModelConfiguration(isStoredInMemoryOnly: true)]
        )
        let context = ModelContext(container)
        let conversation = Conversation(title: "Test")
        return ChatViewModel(
            generator: generator,
            context: context,
            conversation: conversation
        )
    }

}

private final class MockTavilyURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var responseData = Data()
    nonisolated(unsafe) static var responseStatusCode = 200
    nonisolated(unsafe) static var lastRequest: URLRequest?
    nonisolated(unsafe) static var lastRequestBody: Data?

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "api.tavily.com"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        Self.lastRequest = request
        Self.lastRequestBody = Self.requestBody(from: request)
        let response = HTTPURLResponse(
            url: request.url ?? URL(string: "https://api.tavily.com/search")!,
            statusCode: Self.responseStatusCode,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.responseData)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func requestBody(from request: URLRequest) -> Data? {
        if let body = request.httpBody {
            return body
        }

        guard let stream = request.httpBodyStream else {
            return nil
        }

        stream.open()
        defer { stream.close() }

        let bufferSize = 1024
        var data = Data()
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }

        while stream.hasBytesAvailable {
            let read = stream.read(buffer, maxLength: bufferSize)
            if read < 0 {
                return data.isEmpty ? nil : data
            }
            if read == 0 {
                break
            }
            data.append(buffer, count: read)
        }

        return data.isEmpty ? nil : data
    }
}

private actor CancellationProbe {
    private var cancelled = false

    func markCancelled() {
        cancelled = true
    }

    func isCancelled() -> Bool {
        cancelled
    }
}

private final class FakePathMonitor: PathMonitoring {
    nonisolated(unsafe) var pathUpdateHandler: ((NetworkPathStatus) -> Void)?

    func start(queue: DispatchQueue) {
        _ = queue
    }

    func cancel() {}

    func emit(_ status: NetworkPathStatus) {
        pathUpdateHandler?(status)
    }
}

private func waitForCancellation(_ probe: CancellationProbe) async -> Bool {
    for _ in 0..<20 {
        if await probe.isCancelled() {
            return true
        }
        try? await Task.sleep(for: .milliseconds(25))
    }
    return await probe.isCancelled()
}

private struct TestLLMGenerator: LLMGenerator {
    func isAvailable() -> Bool { true }

    func respond(to prompt: String, tools: [any FoundationModelTool]) async throws -> String {
        ""
    }

    func streamResponse(to prompt: String, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }
}

private struct BlockingLLMGenerator: LLMGenerator {
    func isAvailable() -> Bool { true }

    func respond(to prompt: String, tools: [any FoundationModelTool]) async throws -> String {
        ""
    }

    func streamResponse(to prompt: String, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    try await Task.sleep(for: .seconds(30))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                producer.cancel()
            }
        }
    }
}

private struct PartialReasoningLLMGenerator: LLMGenerator {
    func isAvailable() -> Bool { true }

    func respond(to prompt: String, tools: [any FoundationModelTool]) async throws -> String {
        "<thinking>\nAnalyze the request carefully.\n</thinking>"
    }

    func streamResponse(to prompt: String, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield("<thinking>\nAnalyze the request carefully.\n</thinking>")
            continuation.finish()
        }
    }
}
