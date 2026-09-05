//
//  MLXModelManager.swift
//  ChatLLM
//
//  Manages MLX-based language models downloaded to Documents/Models/.
//
//  The manager body is split into Type+Concern extension files:
//    MLXCachePolicy.swift              Shared cache-policy/benchmark/perf types
//    MLXModelManager+ModelInfo.swift    MLXModelInfo and generation result types
//    MLXModelManager+Catalog.swift      Model catalog, installation status, package metadata
//    MLXModelManager+Downloads.swift    Download lifecycle and partial-download cleanup
//    MLXModelManager+Loading.swift      Load/teardown lifecycle and shader pre-warming
//    MLXModelManager+Tuning.swift       Prefill-step-size benchmarking and persistence
//    MLXModelManager+Generation.swift   Generation configuration and stream orchestration
//    MLXModelManager+ToolCalling.swift  Tool-template inspection and tool-response formatting
//    MLXModelManager+Memory.swift       Memory pressure handling and teardown
//
//  Stored state lives here; it is internal so the extension files can share it.
//  Four published properties (pendingModelToLoad, activeDownloadModelID,
//  downloadErrorModelID, latestPerformanceSample) intentionally lost their
//  private(set) qualifiers for the same reason: Swift has no module-scoped
//  setter, so cross-file extensions must be able to write them.
//

import Foundation
import Combine
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM
import Tokenizers
import OSLog
import UIKit

// MARK: - MLXModelManager

@MainActor
final class MLXModelManager: ObservableObject {
    let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM", category: "MLXModelManager")

    // MARK: - Published Properties

    @Published var availableModels: [MLXModelInfo] = []
    @Published var currentModel: MLXModelInfo?
    @Published var isLoading: Bool = false
    @Published var loadError: String?
    @Published var pendingModelToLoad: MLXModelInfo?

    @Published var downloadProgress: Double = 0
    @Published var isDownloading: Bool = false
    @Published var activeDownloadModelID: String?
    @Published var downloadError: String?
    @Published var downloadErrorModelID: String?
    @Published var latestPerformanceSample: MLXPerformanceSample?


    // MARK: - State
    //
    // Internal (not private) so the Type+Concern extension files can share it.

    var container: ModelContainer?
    var loadTask: Task<Void, Never>?
    let downloader = ModelDownloader()
    var downloaderTask: Task<Void, Never>?
    private var memoryWarningCancellable: AnyCancellable?
    private var appWillResignActiveCancellable: AnyCancellable?
    private var appDidEnterBackgroundCancellable: AnyCancellable?
    private var settingsDidChangeCancellable: AnyCancellable?
    var compatibilityErrors: [String: String] = [:]
    var toolTemplateInspectionCache: [String: ToolTemplateInspection] = [:]
    var packageMetadataCache: [String: ModelPackageMetadata] = [:]
    var promptTokenCountCache: [PromptTokenCountCacheKey: Int] = [:]
    var promptTokenCountCacheOrder: [PromptTokenCountCacheKey] = []
    var activeLoadID: UUID?
    var installedWorkerLoadID: UUID?
    var deferredPrewarmModelID: String?
    var deferredTuningModelID: String?
    var prewarmInFlightModelID: String?
    var tuningTask: Task<Void, Never>?
    var hasSeenMemoryWarningSinceCurrentLoad = false
    let deviceSupportProfile: MLXDeviceSupportProfile
    let inferenceWorker: MLXInferenceWorker
    var simulatedDownloadedModelIDs: Set<String> = []

    let documentsDirectory: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first ?? FileManager.default.temporaryDirectory

    struct ModelInstallationStatus {
        let isInstalled: Bool
        let isCompatible: Bool
        let compatibilityError: String?
    }

    struct ModelPackageMetadata {
        let architecture: String?
        let processorClass: String?
        let modelType: String?
    }

    // MARK: - Init

    init(deviceSupportProfile: MLXDeviceSupportProfile? = nil) {
        MLXRuntimeConfiguration.configureForCurrentProcess()
        self.deviceSupportProfile = deviceSupportProfile ?? MLXDeviceSupportProfile.current
        self.inferenceWorker = MLXInferenceWorker(
            subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM",
            deviceSupportProfile: self.deviceSupportProfile
        )
        removeObsoleteModelDirectories()
        availableModels = Self.modelDefinitions.map { definition in
            var model = definition
            let status = installationStatus(for: model)
            model.isAvailable = status.isInstalled && status.isCompatible
            if let error = status.compatibilityError {
                compatibilityErrors[model.id] = error
            }
            return model
        }

        memoryWarningCancellable = NotificationCenter.default.publisher(
            for: UIApplication.didReceiveMemoryWarningNotification
        )
        .sink { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleMemoryWarning()
            }
        }

        appWillResignActiveCancellable = NotificationCenter.default.publisher(
            for: UIApplication.willResignActiveNotification
        )
        .sink { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleApplicationInactivity(reason: "application.will_resign_active")
            }
        }

        appDidEnterBackgroundCancellable = NotificationCenter.default.publisher(
            for: UIApplication.didEnterBackgroundNotification
        )
        .sink { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.handleApplicationInactivity(reason: "application.did_enter_background")
            }
        }

        // `UserDefaults.didChangeNotification` fires on every defaults write in
        // the process — including this manager's own prefill-tuning writes and
        // each of the ~13 keys a settings save persists. Collapse the stream to
        // actual transitions of the one flag before hopping to the main actor.
        settingsDidChangeCancellable = NotificationCenter.default.publisher(
            for: UserDefaults.didChangeNotification
        )
        .map { _ in
            UserDefaults.standard.bool(forKey: AppSettingsKeys.disableRAMPrecautions)
        }
        .prepend(UserDefaults.standard.bool(forKey: AppSettingsKeys.disableRAMPrecautions))
        .removeDuplicates()
        .dropFirst()
        .sink { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.refreshModelAvailability()
            }
        }

        restoreBackgroundDownloadIfNeeded()
    }

    deinit {
        memoryWarningCancellable?.cancel()
        appWillResignActiveCancellable?.cancel()
        appDidEnterBackgroundCancellable?.cancel()
        settingsDidChangeCancellable?.cancel()
    }

}
