//
//  MLXModelManager+Catalog
//  ChatLLM
//
//  Split out of MLXModelManager.swift as part of the Type+Concern organization.
//

import Foundation
import OSLog

extension MLXModelManager {
    // MARK: - Catalog

    static let uiTestFakeDownloadsArgument = "-ui-test-fake-mlx-downloads"
    /// Model directories left behind by superseded releases. Nothing in
    /// `modelDefinitions` refers to them, so the picker cannot show or delete
    /// them; they are removed at startup instead of stranding gigabytes in
    /// Documents.
    nonisolated private static let obsoleteModelDirNames = [
        "Qwen3.5-4B-MLX-mixed36"
    ]

    static let modelDefinitions: [MLXModelInfo] = [
        MLXModelInfo(
            id: "qwen3.5-4b-4bit-hybrid",
            name: "Qwen 3.5",
            localDirName: "Qwen3.5-4B-MLX-4bit-hybrid",
            hfRepoId: "Spakie/Qwen3.5-4B-MLX-4bit-hybrid",
            parameters: "4B (4-bit hybrid)",
            downloadSizeLabel: "2.66 GB",
            loadPolicy: .qwenMultimodal,
            description: "Qwen 3.5 4B multimodal model with native reasoning and vision.",
            contextLength: 262144,
            isAvailable: false,
            supportsReasoning: true,
            supportsNativeImages: true,
            requiredProcessorClass: "Qwen3VLProcessor",
            minimumPhoneMemoryBytes: 8 * MLXDeviceSupportProfile.gibibyte,
            minimumPhoneMemoryForToolCallsBytes: 8 * MLXDeviceSupportProfile.gibibyte
        ),
        MLXModelInfo(
            id: "qwen3.5-2b-4bit",
            name: "Qwen 3.5",
            localDirName: "Qwen3.5-2B-MLX-4bit",
            hfRepoId: "mlx-community/Qwen3.5-2B-4bit",
            parameters: "2B (4-bit)",
            downloadSizeLabel: "1.75 GB",
            loadPolicy: .qwenMultimodal,
            description: "Qwen 3.5 2B multimodal model with native reasoning and vision.",
            contextLength: 262144,
            isAvailable: false,
            supportsReasoning: true,
            supportsNativeImages: true,
            requiredProcessorClass: "Qwen3VLProcessor",
            minimumPhoneMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte,
            minimumPhoneMemoryForToolCallsBytes: 8 * MLXDeviceSupportProfile.gibibyte
        ),
        MLXModelInfo(
            id: "qwen3.5-0.8b-4bit",
            name: "Qwen 3.5",
            localDirName: "Qwen3.5-0.8B-MLX-4bit",
            hfRepoId: "mlx-community/Qwen3.5-0.8B-4bit",
            parameters: "0.8B (4-bit)",
            downloadSizeLabel: "625 MB",
            loadPolicy: .qwenMultimodal,
            description: "Qwen 3.5 0.8B multimodal model with native reasoning and vision.",
            contextLength: 262144,
            isAvailable: false,
            supportsReasoning: true,
            supportsNativeImages: true,
            requiredProcessorClass: "Qwen3VLProcessor",
            minimumPhoneMemoryBytes: 4 * MLXDeviceSupportProfile.gibibyte,
            minimumPhoneMemoryForToolCallsBytes: 6 * MLXDeviceSupportProfile.gibibyte,
            phoneContextWindowOverride: [8: 4_096, 12: 6_144]
        ),
        MLXModelInfo(
            id: "smollm3-3b-4bit",
            name: "SmolLM3",
            localDirName: "SmolLM3-3B-MLX-4bit",
            hfRepoId: "mlx-community/SmolLM3-3B-4bit",
            parameters: "3B (4-bit)",
            downloadSizeLabel: "1.75 GB",
            loadPolicy: .standard,
            description: "SmolLM3 3B text model with native reasoning.",
            contextLength: 65536,
            isAvailable: false,
            supportsReasoning: true,
            supportsNativeImages: false,
            requiredProcessorClass: nil,
            minimumPhoneMemoryBytes: 6 * MLXDeviceSupportProfile.gibibyte,
            minimumPhoneMemoryForToolCallsBytes: 8 * MLXDeviceSupportProfile.gibibyte
        )
    ]

    // MARK: - Model Directory (Documents)

    func modelDirectoryURL(for info: MLXModelInfo) -> URL? {
        let dir = documentsDirectory.appendingPathComponent("Models/\(info.localDirName)")
        let config = dir.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: config.path) ? dir : nil
    }

    func installationStatus(for info: MLXModelInfo) -> ModelInstallationStatus {
        if Self.isUITestFakeDownloadsEnabled {
            if simulatedDownloadedModelIDs.contains(info.id) {
                return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
            }
            return ModelInstallationStatus(isInstalled: false, isCompatible: true, compatibilityError: nil)
        }

        let dir = documentsDirectory.appendingPathComponent("Models/\(info.localDirName)")
        let fm = FileManager.default
        let hasConfig = fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        let contents = hasConfig ? ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? []) : []
        let hasWeights = contents.contains(where: { $0.hasSuffix(".safetensors") })

        if let deviceIssue = deviceSupportProfile.availabilityIssue(for: info) {
            return ModelInstallationStatus(
                isInstalled: hasConfig && hasWeights,
                isCompatible: false,
                compatibilityError: deviceIssue
            )
        }

        guard hasConfig else {
            return ModelInstallationStatus(isInstalled: false, isCompatible: false, compatibilityError: nil)
        }
        guard hasWeights else {
            return ModelInstallationStatus(isInstalled: false, isCompatible: false, compatibilityError: nil)
        }

        guard info.supportsNativeImages else {
            return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
        }

        guard let expectedProcessor = info.requiredProcessorClass else {
            return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
        }

        let detectedProcessorClass = packageMetadata(for: info)?.processorClass

        guard let detectedProcessorClass else {
            return ModelInstallationStatus(
                isInstalled: true,
                isCompatible: false,
                compatibilityError:
                    "Installed model '\(info.localDirName)' is missing processor metadata for native image support. Re-download \(info.localDirName) to enable native images."
            )
        }

        guard detectedProcessorClass == expectedProcessor else {
            return ModelInstallationStatus(
                isInstalled: true,
                isCompatible: false,
                compatibilityError:
                    "Installed model '\(info.localDirName)' is incompatible with native image support (processor '\(detectedProcessorClass)'). Re-download \(info.localDirName) to get the correct multimodal files."
            )
        }

        return ModelInstallationStatus(isInstalled: true, isCompatible: true, compatibilityError: nil)
    }

    func compatibilityError(for info: MLXModelInfo) -> String? {
        compatibilityErrors[info.id]
    }

    static var isUITestFakeDownloadsEnabled: Bool {
        ProcessInfo.processInfo.arguments.contains(uiTestFakeDownloadsArgument)
    }

    func availabilityIssue(for info: MLXModelInfo) -> String? {
        compatibilityError(for: info)
    }

    func supportsToolCalls(for info: MLXModelInfo) -> Bool {
        deviceSupportProfile.supportsToolCalls(for: info)
    }

    func toolCallIssue(for info: MLXModelInfo) -> String? {
        deviceSupportProfile.toolCallIssue(for: info)
    }

    func packageArchitecture(for info: MLXModelInfo) -> String? {
        packageMetadata(for: info)?.architecture
    }
    func removeObsoleteModelDirectories() {
        let fm = FileManager.default
        let modelsDir = documentsDirectory.appendingPathComponent("Models", isDirectory: true)
        for name in Self.obsoleteModelDirNames {
            let dir = modelsDir.appendingPathComponent(name, isDirectory: true)
            guard fm.fileExists(atPath: dir.path) else { continue }
            do {
                try fm.removeItem(at: dir)
                logger.notice("Removed obsolete model '\(name, privacy: .public)' from Documents/Models/")
            } catch {
                logger.error("Could not remove obsolete model '\(name, privacy: .public)': \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    // MARK: - Diagnostics / Management

    func refreshModelAvailability() {
        compatibilityErrors = [:]
        toolTemplateInspectionCache = [:]
        packageMetadataCache = [:]
        availableModels = availableModels.map { model in
            var updated = model
            let status = installationStatus(for: model)
            updated.isAvailable = status.isInstalled && status.isCompatible
            if let error = status.compatibilityError {
                compatibilityErrors[model.id] = error
            }
            return updated
        }
        if let currentModel,
           let refreshed = availableModels.first(where: { $0.id == currentModel.id }),
           !refreshed.isAvailable {
            tearDownCurrentModel(reason: "availability refresh invalidated \(currentModel.id)")
        }
    }

    func deleteModel(_ model: MLXModelInfo) throws {
        if isDownloading(modelID: model.id) {
            cancelDownload()
        }
        if pendingModelToLoad?.id == model.id {
            cancelCurrentLoad(reason: "delete \(model.id)", tearDownModel: false)
        }
        if currentModel?.id == model.id {
            tearDownCurrentModel(reason: "delete \(model.id)")
        }
        simulatedDownloadedModelIDs.remove(model.id)
        let dir = documentsDirectory.appendingPathComponent("Models/\(model.localDirName)")
        do {
            try FileManager.default.removeItem(at: dir)
            refreshModelAvailability()
            logger.notice("Deleted model '\(model.localDirName, privacy: .public)' from Documents/Models/")
        } catch let error as NSError where error.code == NSFileNoSuchFileError {
            refreshModelAvailability()
            logger.debug("Model '\(model.localDirName, privacy: .public)' was not installed; nothing to delete")
        }
    }
    func printModelLocations() {
        for model in availableModels {
            if let url = modelDirectoryURL(for: model) {
                logger.debug("Model file \(model.name, privacy: .public): \(url.path, privacy: .private) available=\(model.isAvailable, privacy: .public)")
            } else {
                logger.debug("Model file \(model.name, privacy: .public) not found; available=\(model.isAvailable, privacy: .public)")
            }
        }
    }

    func printModelInputOutputInfo() {
        guard let current = currentModel else {
            logger.notice("No MLX model loaded")
            return
        }
        logger.notice("Current model: \(current.displayName, privacy: .public) context=\(current.contextLength, privacy: .public) reasoning=\(current.supportsReasoning, privacy: .public)")
    }
    func packageMetadata(for info: MLXModelInfo) -> ModelPackageMetadata? {
        if let cached = packageMetadataCache[info.id] {
            return cached
        }

        guard let modelURL = modelDirectoryURL(for: info) else {
            return nil
        }

        let configURL = modelURL.appendingPathComponent("config.json")
        let configJSON = (
            (try? Data(contentsOf: configURL))
                .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        ) ?? [:]
        let architecture = (configJSON["architectures"] as? [String])?.first
        let modelType = configJSON["model_type"] as? String

        let processorURLCandidates = [
            modelURL.appendingPathComponent("preprocessor_config.json"),
            modelURL.appendingPathComponent("processor_config.json")
        ]

        var processorClass: String?
        for url in processorURLCandidates {
            guard let data = try? Data(contentsOf: url),
                  let rawObject = try? JSONSerialization.jsonObject(with: data),
                  let json = rawObject as? [String: Any],
                  let detected = json["processor_class"] as? String else {
                continue
            }
            processorClass = detected
            break
        }

        let metadata = ModelPackageMetadata(
            architecture: architecture,
            processorClass: processorClass,
            modelType: modelType
        )
        packageMetadataCache[info.id] = metadata
        return metadata
    }
}
