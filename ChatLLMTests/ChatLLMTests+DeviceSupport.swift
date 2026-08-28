//
//  ChatLLMTests+DeviceSupport
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
    @Test func qwen4BRequiresEightGigabytesOnIPhone() {
        let profile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let model = try! #require(manager.model(withID: "qwen3.5-4b-4bit-hybrid"))

        #expect(!profile.supportsModel(model))
        #expect(profile.availabilityIssue(for: model)?.contains("8 GB") == true)
    }

    @Test func qwen4BEnablesToolCallsAtEightGigabytesOnIPhone() {
        let profile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let model = try! #require(manager.model(withID: "qwen3.5-4b-4bit-hybrid"))

        #expect(profile.supportsModel(model))
        #expect(manager.supportsToolCalls(for: model))
        #expect(manager.toolCallIssue(for: model) == nil)
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

    @Test func smolLM3ModelDefinitionMatchesDownloadMetadata() {
        let manager = MLXModelManager()
        let model = try! #require(manager.model(withID: "smollm3-3b-4bit"))

        #expect(model.name == "SmolLM3")
        #expect(model.localDirName == "SmolLM3-3B-MLX-4bit")
        #expect(model.hfRepoId == "mlx-community/SmolLM3-3B-4bit")
        #expect(model.parameters == "3B (4-bit)")
        #expect(model.downloadSizeLabel == "1.75 GB")
        #expect(model.loadPolicy == .standard)
        #expect(model.contextLength == 65_536)
        #expect(model.supportsReasoning)
        #expect(!model.supportsNativeImages)
        #expect(model.requiredProcessorClass == nil)
    }

    @Test func smolLM3PhoneMemoryLimitsGateModelAndToolCalls() {
        let fourGigabyteProfile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 4 * MLXDeviceSupportProfile.gibibyte
        )
        let sixGigabyteProfile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let eightGigabyteProfile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )

        let fourGigabyteManager = MLXModelManager(deviceSupportProfile: fourGigabyteProfile)
        let sixGigabyteManager = MLXModelManager(deviceSupportProfile: sixGigabyteProfile)
        let eightGigabyteManager = MLXModelManager(deviceSupportProfile: eightGigabyteProfile)
        let fourGigabyteModel = try! #require(fourGigabyteManager.model(withID: "smollm3-3b-4bit"))
        let sixGigabyteModel = try! #require(sixGigabyteManager.model(withID: "smollm3-3b-4bit"))
        let eightGigabyteModel = try! #require(eightGigabyteManager.model(withID: "smollm3-3b-4bit"))

        #expect(!fourGigabyteProfile.supportsModel(fourGigabyteModel))
        #expect(fourGigabyteProfile.availabilityIssue(for: fourGigabyteModel)?.contains("6 GB") == true)
        #expect(sixGigabyteProfile.supportsModel(sixGigabyteModel))
        #expect(!sixGigabyteManager.supportsToolCalls(for: sixGigabyteModel))
        #expect(sixGigabyteManager.toolCallIssue(for: sixGigabyteModel)?.contains("8 GB") == true)
        #expect(eightGigabyteProfile.supportsModel(eightGigabyteModel))
        #expect(eightGigabyteManager.supportsToolCalls(for: eightGigabyteModel))
        #expect(eightGigabyteManager.toolCallIssue(for: eightGigabyteModel) == nil)
    }

    @Test func qwenEightGigabyteTierTreatsSlightlyUnderreportedPhoneAsEightGigabytes() {
        let marketedEightGigabytesButReportedBelowEightGiB: UInt64 = 7_950_000_000
        let profile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: marketedEightGigabytesButReportedBelowEightGiB
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let twoBModel = try! #require(manager.model(withID: "qwen3.5-2b-4bit"))
        let fourBModel = try! #require(manager.model(withID: "qwen3.5-4b-4bit-hybrid"))

        #expect(profile.supportsModel(twoBModel))
        #expect(manager.supportsToolCalls(for: twoBModel))
        #expect(profile.supportsModel(fourBModel))
        #expect(manager.supportsToolCalls(for: fourBModel))
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
        let fourBModel = try! #require(manager.model(withID: "qwen3.5-4b-4bit-hybrid"))

        #expect(profile.supportsModel(fourBModel))
        #expect(manager.supportsToolCalls(for: fourBModel))
        #expect(profile.availabilityIssue(for: fourBModel) == nil)
    }

    #if targetEnvironment(simulator)
    @Test func qwenModelsAreEligibleForDownloadOnIPhoneSimulator() {
        let profile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 16 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let qwenModels = manager.availableModels.filter { $0.id.hasPrefix("qwen") }

        #expect(qwenModels.count == 3)
        #expect(qwenModels.allSatisfy { manager.availabilityIssue(for: $0) == nil })
        #expect(qwenModels.allSatisfy { !$0.isAvailable })
    }
    #endif

    @Test func simulatorMLXArchitectureUsesHostAppleSiliconGenerationAndVariant() {
        #expect(MLXRuntimeConfiguration.metalGPUArchitecture(for: "Apple M1") == "applegpu_g13g")
        #expect(MLXRuntimeConfiguration.metalGPUArchitecture(for: "Apple M3 Pro") == "applegpu_g15g")
        #expect(MLXRuntimeConfiguration.metalGPUArchitecture(for: "Apple M5 Max") == "applegpu_g17s")
        #expect(MLXRuntimeConfiguration.metalGPUArchitecture(for: "Apple M2 Ultra") == "applegpu_g14d")
        #expect(MLXRuntimeConfiguration.metalGPUArchitecture(for: "Generic Simulator GPU") == nil)
    }

    @Test func phoneContextOverridesDoNotClampNonPhoneDevices() {
        let profile = MLXDeviceSupportProfile(
            isPhone: false,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let compactModel = try! #require(manager.model(withID: "qwen3.5-0.8b-4bit"))

        #expect(profile.maxContextWindowTokens(for: compactModel) == 2_048)
    }

    @Test func modelManagerRejectsIncompatibleProgrammaticDownloads() {
        let profile = MLXDeviceSupportProfile(
            isPhone: true,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let model = try! #require(manager.model(withID: "qwen3.5-4b-4bit-hybrid"))

        manager.startDownload(for: model)

        #expect(!manager.isDownloading)
        #expect(manager.downloadErrorModelID == model.id)
        #expect(manager.downloadError?.contains("8 GB") == true)
    }

    @Test func modelManagerReportsUndownloadedModelLoadError() {
        let profile = MLXDeviceSupportProfile(
            isPhone: false,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = MLXModelManager(deviceSupportProfile: profile)
        let model = MLXModelManager.MLXModelInfo(
            id: "undownloaded-test-model",
            name: "Undownloaded Test",
            localDirName: "UndownloadedTest",
            hfRepoId: "local/undownloaded",
            parameters: "1B",
            downloadSizeLabel: "1 GB",
            description: "A compatible model that has not been downloaded.",
            contextLength: 4096,
            isAvailable: false,
            supportsReasoning: true
        )
        manager.availableModels = [model]

        manager.startLoading(modelID: model.id, source: "test")

        #expect(manager.loadError?.contains("not downloaded") == true)
    }
}
