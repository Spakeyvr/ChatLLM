//
//  ChatLLMTests+DeviceSupport
//  ChatLLMTests
//
//  Split out of ChatLLMTests.swift; part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Combine
import Foundation
import SwiftUI
import SwiftData
import WebKit
import UIKit
import FoundationModels
@testable import ChatLLM

extension ChatLLMTests {
    @Test func qwen4BRequiresEightGigabytesOnIPhone() {
        let profile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = makeModelManager(deviceSupportProfile: profile)
        let model = try! #require(manager.model(withID: "qwen3.5-4b-4bit-hybrid"))

        #expect(!profile.supportsModel(model))
        #expect(profile.availabilityIssue(for: model)?.contains("8 GB") == true)
    }

    @Test func modelPickerListsOnlyInstalledModels() {
        let manager = makeModelManager()
        #expect(!manager.availableModels.isEmpty)
        #expect(manager.installedModels.isEmpty)

        let installedID = manager.availableModels[0].id
        manager.availableModels[0].isAvailable = true

        #expect(manager.installedModels.map(\.id) == [installedID])
    }

    @Test func qwen4BEnablesToolCallsAtEightGigabytesOnIPhone() {
        let profile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = makeModelManager(deviceSupportProfile: profile)
        let model = try! #require(manager.model(withID: "qwen3.5-4b-4bit-hybrid"))

        #expect(profile.supportsModel(model))
        #expect(manager.supportsToolCalls(for: model))
        #expect(manager.toolCallIssue(for: model) == nil)
    }

    @Test func qwen2BAllowsSixGigabyteIPhonesButBlocksToolCalls() {
        let profile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = makeModelManager(deviceSupportProfile: profile)
        let model = try! #require(manager.model(withID: "qwen3.5-2b-4bit"))

        #expect(profile.supportsModel(model))
        #expect(!manager.supportsToolCalls(for: model))
        #expect(manager.toolCallIssue(for: model)?.contains("8 GB") == true)
    }

    @Test func qwen2BEnablesToolCallsAtEightGigabytesOnIPhone() {
        let profile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = makeModelManager(deviceSupportProfile: profile)
        let model = try! #require(manager.model(withID: "qwen3.5-2b-4bit"))

        #expect(profile.supportsModel(model))
        #expect(manager.supportsToolCalls(for: model))
        #expect(manager.toolCallIssue(for: model) == nil)
    }

    @Test func lfm25ModelDefinitionMatchesDownloadMetadata() {
        let manager = makeModelManager()
        let model = try! #require(manager.model(withID: "lfm2.5-2.6b-4bit"))

        #expect(model.name == "LFM2.5")
        #expect(model.localDirName == "LFM2.5-2.6B-MLX-4bit")
        #expect(model.hfRepoId == "LiquidAI/LFM2.5-2.6B-MLX-4bit")
        #expect(model.parameters == "2.6B (4-bit)")
        #expect(model.downloadSizeLabel == "1.60 GB")
        #expect(model.loadPolicy == .standard)
        #expect(model.contextLength == 131_072)
        #expect(model.supportsReasoning)
        #expect(!model.supportsNativeImages)
        #expect(model.requiredProcessorClass == nil)
    }

    @Test func lfm25PhoneMemoryLimitsGateModelAndToolCalls() {
        let fourGigabyteProfile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 4 * MLXDeviceSupportProfile.gibibyte
        )
        let sixGigabyteProfile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let eightGigabyteProfile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )

        let fourGigabyteManager = makeModelManager(deviceSupportProfile: fourGigabyteProfile)
        let sixGigabyteManager = makeModelManager(deviceSupportProfile: sixGigabyteProfile)
        let eightGigabyteManager = makeModelManager(deviceSupportProfile: eightGigabyteProfile)
        let fourGigabyteModel = try! #require(fourGigabyteManager.model(withID: "lfm2.5-2.6b-4bit"))
        let sixGigabyteModel = try! #require(sixGigabyteManager.model(withID: "lfm2.5-2.6b-4bit"))
        let eightGigabyteModel = try! #require(eightGigabyteManager.model(withID: "lfm2.5-2.6b-4bit"))

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
        let profile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: marketedEightGigabytesButReportedBelowEightGiB
        )
        let manager = makeModelManager(deviceSupportProfile: profile)
        let twoBModel = try! #require(manager.model(withID: "qwen3.5-2b-4bit"))
        let fourBModel = try! #require(manager.model(withID: "qwen3.5-4b-4bit-hybrid"))

        #expect(profile.supportsModel(twoBModel))
        #expect(manager.supportsToolCalls(for: twoBModel))
        #expect(profile.supportsModel(fourBModel))
        #expect(manager.supportsToolCalls(for: fourBModel))
    }

    @Test func twelveGigabyteTierDisablesLowMemoryKVFallbackAfterNormalization() {
        let marketedTwelveGigabytesButReportedBelowTwelveGiB: UInt64 = 11_900_000_000
        let profile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: marketedTwelveGigabytesButReportedBelowTwelveGiB
        )

        #expect(!profile.hasLowMemoryForPersistentKVCache)
    }

    @Test func eightGigabyteTierStillUsesLowMemoryKVFallback() {
        let profile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )

        #expect(profile.hasLowMemoryForPersistentKVCache)
    }

    @Test func contextWindowMaximumTracksSixEightAndTwelveGigabyteTiers() {
        let sixGigabyteProfile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let eightGigabyteProfile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )
        let twelveGigabyteProfile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 12 * MLXDeviceSupportProfile.gibibyte
        )

        #expect(sixGigabyteProfile.maxContextWindowTokens == 1_024)
        #expect(eightGigabyteProfile.maxContextWindowTokens == 2_048)
        #expect(twelveGigabyteProfile.maxContextWindowTokens == 6_144)
    }

    @Test func userDefaultsContextWindowClampsToFiveHundredTwelveTokenMinimum() {
        let defaults = testEnvironment.defaults
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
        let profile = makeDeviceProfile(
            isPhone: false,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = makeModelManager(deviceSupportProfile: profile)
        let fourBModel = try! #require(manager.model(withID: "qwen3.5-4b-4bit-hybrid"))

        #expect(profile.supportsModel(fourBModel))
        #expect(manager.supportsToolCalls(for: fourBModel))
        #expect(profile.availabilityIssue(for: fourBModel) == nil)
    }

    #if targetEnvironment(simulator)
    @Test func qwenModelsAreEligibleForDownloadOnIPhoneSimulator() {
        let profile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 16 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = makeModelManager(deviceSupportProfile: profile)
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
        let profile = makeDeviceProfile(
            isPhone: false,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = makeModelManager(deviceSupportProfile: profile)
        let compactModel = try! #require(manager.model(withID: "qwen3.5-0.8b-4bit"))

        #expect(profile.maxContextWindowTokens(for: compactModel) == 2_048)
    }

    @Test func modelManagerRejectsIncompatibleProgrammaticDownloads() {
        let profile = makeDeviceProfile(
            isPhone: true,
            physicalMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = makeModelManager(deviceSupportProfile: profile)
        let model = try! #require(manager.model(withID: "qwen3.5-4b-4bit-hybrid"))

        manager.startDownload(for: model)

        #expect(!manager.isDownloading)
        #expect(manager.downloadErrorModelID == model.id)
        #expect(manager.downloadError?.contains("8 GB") == true)
    }

    @Test func modelManagerReportsUndownloadedModelLoadError() {
        let profile = makeDeviceProfile(
            isPhone: false,
            physicalMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )
        let manager = makeModelManager(deviceSupportProfile: profile)
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

    @Test func sendingWithAModelThatFailedToLoadRetriesAndExplainsWhy() async throws {
        let manager = makeModelManager(deviceSupportProfile: makeDeviceProfile(isPhone: false))
        let model = MLXModelManager.MLXModelInfo(
            id: "missing-files-test-model",
            name: "Missing Files",
            localDirName: "MissingFilesTest",
            hfRepoId: "local/missing-files",
            parameters: "1B",
            downloadSizeLabel: "1 GB",
            description: "Marked downloaded, but its files are gone.",
            contextLength: 4096,
            isAvailable: true,
            supportsReasoning: false
        )
        manager.availableModels = [model]
        let bridge = testEnvironment.bridge
        bridge.modelManager = manager
        bridge.selectedBackend = .mlx
        bridge.selectedModelID = model.id

        let viewModel = try makeViewModel()
        await viewModel.send(userText: "Hello")

        let reply = try #require(viewModel.conversation.messages.first { $0.role == .assistant })
        let error = try #require(reply.generationError)
        #expect(error.contains("Missing Files 1B"))
        #expect(!error.contains("Documents/"))
        #expect(manager.loadError != nil)
    }

    @Test func finishedDownloadLoadsOnlyTheModelTheChatIsWaitingFor() async throws {
        let bridge = makeBackendBridge()
        let manager = try #require(bridge.modelManager)
        let modelID = try #require(manager.availableModels.first?.id)

        // A chat on Apple Intelligence keeps it; the new model isn't loaded beside it.
        bridge.selectedBackend = .foundationModels
        manager.loadError = nil
        manager.downloadCompleted.send(modelID)
        try await Task.sleep(for: .milliseconds(100))
        #expect(bridge.selectedBackend == .foundationModels)
        #expect(manager.loadError == nil && !manager.isLoading)

        // A chat that uses another MLX model keeps it too.
        bridge.selectedBackend = .mlx
        bridge.selectedModelID = "another-model"
        manager.downloadCompleted.send(modelID)
        try await Task.sleep(for: .milliseconds(100))
        #expect(bridge.selectedModelID == "another-model")
        #expect(manager.loadError == nil && !manager.isLoading)

        // A chat waiting for this model loads it. The test has no model files, so the
        // attempt ends in a load error rather than a loaded model.
        bridge.selectedModelID = modelID
        manager.downloadCompleted.send(modelID)
        try await Task.sleep(for: .milliseconds(100))
        #expect(manager.loadError != nil)
    }

    @Test func deletingTheSelectedModelLeavesNoSelectionThatCannotReply() throws {
        let bridge = makeBackendBridge()
        let manager = try #require(bridge.modelManager)
        let models = Array(manager.availableModels.prefix(2))
        try #require(models.count == 2)
        bridge.selectedBackend = .mlx
        bridge.selectedModelID = models[0].id

        // Deleting another model keeps the selection.
        try bridge.deleteModel(models[1])
        #expect(bridge.selectedBackend == .mlx)
        #expect(bridge.selectedModelID == models[0].id)

        try bridge.deleteModel(models[0])
        if bridge.foundationModelsAvailable {
            #expect(bridge.selectedBackend == .foundationModels)
        } else {
            #expect(bridge.selectedModelID == nil)
            #expect(bridge.currentModelDisplayName == nil)
        }
    }

    @Test func retryingTheSelectedModelsLoadDoesNotResetThePipeline() {
        let manager = makeModelManager(deviceSupportProfile: makeDeviceProfile(isPhone: false))
        let model = MLXModelManager.MLXModelInfo(
            id: "missing-files-test-model",
            name: "Missing Files",
            localDirName: "MissingFilesTest",
            hfRepoId: "local/missing-files",
            parameters: "1B",
            downloadSizeLabel: "1 GB",
            description: "Marked downloaded, but its files are gone.",
            contextLength: 4096,
            isAvailable: true,
            supportsReasoning: false
        )
        manager.availableModels = [model]
        let bridge = makeBackendBridge()
        bridge.modelManager = manager
        bridge.selectedBackend = .mlx
        bridge.selectedModelID = model.id

        // Posted synchronously on the main thread by the bridge.
        let resets = ResetCounter()
        let observer = NotificationCenter.default.addObserver(
            forName: .modelPipelineWillReset, object: bridge, queue: nil
        ) { _ in MainActor.assumeIsolated { resets.count += 1 } }
        defer { NotificationCenter.default.removeObserver(observer) }

        // Reopening the chat retries the failed load; a send waiting on it must survive.
        bridge.switchToMLXModel(model.id, source: "test.retry")
        #expect(resets.count == 0)

        // Replacing a loaded model still resets, so in-flight generation stops first.
        manager.currentModel = MLXModelManager.MLXModelInfo(
            id: "loaded-test-model",
            name: "Loaded",
            localDirName: "LoadedTest",
            hfRepoId: "local/loaded",
            parameters: "1B",
            downloadSizeLabel: "1 GB",
            description: "Currently loaded.",
            contextLength: 4096,
            isAvailable: true,
            supportsReasoning: false
        )
        bridge.switchToMLXModel(model.id, source: "test.switch")
        #expect(resets.count == 1)
    }

    @Test func modelInfoStatusSeparatesUndownloadedFromUnsupported() {
        #expect(ModelInfoSheet.statusText(isAvailable: true, isSupportedOnDevice: true) == "Downloaded")
        #expect(ModelInfoSheet.statusText(isAvailable: false, isSupportedOnDevice: true) == "Not downloaded")
        #expect(ModelInfoSheet.statusText(isAvailable: false, isSupportedOnDevice: false) == "Not supported on this device")
    }
}

@MainActor
private final class ResetCounter {
    var count = 0
}
