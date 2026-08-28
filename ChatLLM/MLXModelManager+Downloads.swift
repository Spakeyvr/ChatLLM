//
//  MLXModelManager+Downloads
//  ChatLLM
//
//  Split out of MLXModelManager.swift as part of the Type+Concern organization.
//

import Foundation
import OSLog

extension MLXModelManager {
    // MARK: - Download

    private static let activeBackgroundDownloadModelIDKey = "mlx.activeBackgroundDownloadModelID"
    var activeDownloadModel: MLXModelInfo? {
        guard let activeDownloadModelID else { return nil }
        return model(withID: activeDownloadModelID)
    }

    func isDownloading(modelID: String) -> Bool {
        isDownloading && activeDownloadModelID == modelID
    }

    func hasActiveDownload(excluding modelID: String) -> Bool {
        isDownloading && activeDownloadModelID != modelID
    }

    func downloadError(for modelID: String) -> String? {
        guard downloadErrorModelID == modelID else { return nil }
        return downloadError
    }

    func restoreBackgroundDownloadIfNeeded() {
        guard !Self.isUITestFakeDownloadsEnabled,
              let modelID = UserDefaults.standard.string(
                forKey: Self.activeBackgroundDownloadModelIDKey
              ) else {
            return
        }

        guard let model = model(withID: modelID) else {
            UserDefaults.standard.removeObject(forKey: Self.activeBackgroundDownloadModelIDKey)
            return
        }

        guard !model.isAvailable else {
            UserDefaults.standard.removeObject(forKey: Self.activeBackgroundDownloadModelIDKey)
            return
        }

        logger.notice("Restoring background model download: id=\(modelID, privacy: .public)")
        startDownload(for: model)
    }

    func startDownload(for model: MLXModelInfo) {
        guard !isDownloading else { return }
        if let issue = availabilityIssue(for: model) {
            downloadError = issue
            downloadErrorModelID = model.id
            return
        }
        isDownloading = true
        activeDownloadModelID = model.id
        downloadProgress = 0
        downloadError = nil
        downloadErrorModelID = nil

        if Self.isUITestFakeDownloadsEnabled {
            startSimulatedDownload(for: model)
            return
        }

        UserDefaults.standard.set(model.id, forKey: Self.activeBackgroundDownloadModelIDKey)

        let modelsDir = documentsDirectory.appendingPathComponent("Models", isDirectory: true)
        let targetDir = modelsDir.appendingPathComponent(model.localDirName, isDirectory: true)
        let tempDir = partialDownloadDirectory(for: model, in: modelsDir)

        downloaderTask = Task { [weak self] in
            guard let self else { return }
            do {
                try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
                self.cleanupLegacyPartialDownloads(for: model, in: modelsDir)
                try await self.downloader.download(
                    repoId: model.hfRepoId,
                    to: tempDir,
                    onProgress: { [weak self] progress in
                        Task { @MainActor [weak self] in
                            self?.downloadProgress = progress
                        }
                    }
                )
                guard !Task.isCancelled else { return }
                try self.validateDownloadedModel(at: tempDir)
                if FileManager.default.fileExists(atPath: targetDir.path) {
                    try FileManager.default.removeItem(at: targetDir)
                }
                try FileManager.default.moveItem(at: tempDir, to: targetDir)
                await MainActor.run {
                    self.downloadProgress = 1.0
                    self.isDownloading = false
                    self.activeDownloadModelID = nil
                    self.downloadErrorModelID = nil
                    UserDefaults.standard.removeObject(
                        forKey: Self.activeBackgroundDownloadModelIDKey
                    )
                    self.refreshModelAvailability()
                    self.startLoading(modelID: model.id, source: "download_complete")
                }
                self.logger.notice("Model downloaded: \(model.displayName, privacy: .public)")
            } catch is CancellationError {
                BackgroundModelDownloadSession.shared.cancelTransfers(
                    withIDPrefix: model.hfRepoId + "/"
                )
                await MainActor.run {
                    self.isDownloading = false
                    self.activeDownloadModelID = nil
                    self.downloadProgress = 0
                    self.downloadError = nil
                    self.downloadErrorModelID = nil
                    UserDefaults.standard.removeObject(
                        forKey: Self.activeBackgroundDownloadModelIDKey
                    )
                }
                self.cleanupPartialDownload(at: tempDir)
            } catch {
                BackgroundModelDownloadSession.shared.cancelTransfers(
                    withIDPrefix: model.hfRepoId + "/"
                )
                await MainActor.run {
                    self.isDownloading = false
                    self.activeDownloadModelID = nil
                    self.downloadError = error.localizedDescription
                    self.downloadErrorModelID = model.id
                    UserDefaults.standard.removeObject(
                        forKey: Self.activeBackgroundDownloadModelIDKey
                    )
                }
                self.logger.error("Model download failed: \((error as NSError).localizedDescription, privacy: .public)")
            }
        }
    }

    func cancelDownload() {
        let model = activeDownloadModel
        downloaderTask?.cancel()
        downloaderTask = nil
        isDownloading = false
        activeDownloadModelID = nil
        downloadProgress = 0
        downloadError = nil
        downloadErrorModelID = nil
        UserDefaults.standard.removeObject(forKey: Self.activeBackgroundDownloadModelIDKey)
        if let model {
            let modelsDir = documentsDirectory.appendingPathComponent("Models", isDirectory: true)
            BackgroundModelDownloadSession.shared.cancelTransfers(
                withIDPrefix: model.hfRepoId + "/"
            )
            cleanupPartialDownload(at: partialDownloadDirectory(for: model, in: modelsDir))
            cleanupLegacyPartialDownloads(for: model, in: modelsDir)
        }
    }
    private func partialDownloadDirectory(for model: MLXModelInfo, in modelsDir: URL) -> URL {
        modelsDir.appendingPathComponent(".\(model.localDirName).download", isDirectory: true)
    }

    private func cleanupLegacyPartialDownloads(for model: MLXModelInfo, in modelsDir: URL) {
        let tempPrefix = ".\(model.localDirName).download-"
        let contents = (try? FileManager.default.contentsOfDirectory(at: modelsDir, includingPropertiesForKeys: nil)) ?? []
        for dir in contents where dir.lastPathComponent.hasPrefix(tempPrefix) {
            cleanupPartialDownload(at: dir)
        }
    }

    private func validateDownloadedModel(at dir: URL) throws {
        let fm = FileManager.default
        let hasConfig = fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        let contents = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
        let hasWeights = contents.contains { $0.hasSuffix(".safetensors") }
        guard hasConfig, hasWeights else {
            throw NSError(domain: "MLXModelDownload", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Downloaded model is incomplete. Missing config.json or safetensors weights."
            ])
        }
    }

    private func startSimulatedDownload(for model: MLXModelInfo) {
        downloaderTask = Task { [weak self] in
            guard let self else { return }

            do {
                let progressSteps: [(progress: Double, delay: Duration)] = [
                    (0.2, .milliseconds(120)),
                    (0.55, .seconds(5)),
                    (1.0, .milliseconds(120))
                ]
                for step in progressSteps {
                    try await Task.sleep(for: step.delay)
                    guard !Task.isCancelled else {
                        throw CancellationError()
                    }

                    await MainActor.run {
                        self.downloadProgress = step.progress
                    }
                }

                await MainActor.run {
                    self.simulatedDownloadedModelIDs.insert(model.id)
                    self.isDownloading = false
                    self.activeDownloadModelID = nil
                    self.downloadProgress = 1.0
                    self.downloadErrorModelID = nil
                    self.refreshModelAvailability()
                }
            } catch is CancellationError {
                await MainActor.run {
                    self.isDownloading = false
                    self.activeDownloadModelID = nil
                    self.downloadProgress = 0
                    self.downloadError = nil
                    self.downloadErrorModelID = nil
                }
            } catch {
                await MainActor.run {
                    self.isDownloading = false
                    self.activeDownloadModelID = nil
                    self.downloadError = error.localizedDescription
                    self.downloadErrorModelID = model.id
                }
            }
        }
    }

    private func cleanupPartialDownload(at dir: URL) {
        let fm = FileManager.default
        let items = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        for item in items where item.pathExtension == "download" {
            try? fm.removeItem(at: item)
        }
        try? fm.removeItem(at: dir)
    }
}
