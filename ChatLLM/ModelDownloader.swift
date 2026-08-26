//
//  ModelDownloader.swift
//  ChatLLM
//
//  Downloads MLX model files from HuggingFace to the local Documents directory.
//

import Foundation

actor ModelDownloader {

    private struct FilePlan {
        let name: String
        let size: Int64
        let remoteURL: URL
        let destinationURL: URL
        let temporaryURL: URL
        let transferID: String
    }

    private static let hfApiBase     = "https://huggingface.co/api"
    private static let hfResolveBase = "https://huggingface.co"
    private static let minProgressInterval: Double = 0.01  // report at most every 1% change
    private static let disallowedRemotePathComponents: Set<String> = [".", ".."]
    private let backgroundSession = BackgroundModelDownloadSession.shared

    /// Last value handed to `onProgress`. Actor state rather than a local so the
    /// in-flight delegate callbacks share one throttle with the per-file path.
    private var lastReportedProgress: Double = -1
    private var activeProgressRunID: UUID?
    private var transferBytesByID: [String: Int64] = [:]
    private var transferFractionsByID: [String: Double] = [:]

    // MARK: - File List

    func fetchFileList(repoId: String) async throws -> [(name: String, size: Int64)] {
        guard let url = Self.metadataURL(repoId: repoId) else {
            throw DownloadError.badResponse
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw DownloadError.badResponse
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let siblings = json["siblings"] as? [[String: Any]] else {
            return []
        }
        return siblings.compactMap { item in
            guard let name = item["rfilename"] as? String else { return nil }
            let directSize = item["size"] as? Int64 ?? 0
            let lfsSize = (item["lfs"] as? [String: Any]).flatMap { $0["size"] as? Int64 } ?? 0
            return (name: name, size: max(directSize, lfsSize))
        }
    }

    internal static func metadataURL(repoId: String) -> URL? {
        guard var components = URLComponents(string: "\(hfApiBase)/models/\(repoId)") else {
            return nil
        }
        components.queryItems = [URLQueryItem(name: "blobs", value: "true")]
        return components.url
    }

    // MARK: - Download

    /// Downloads all model files to `targetDir`, streaming each file and reporting
    /// aggregate 0.0–1.0 progress via `onProgress`.
    func download(
        repoId: String,
        to targetDir: URL,
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws {
        let files = try await fetchFileList(repoId: repoId)
        guard !files.isEmpty else { throw DownloadError.noFilesFound }

        let totalExpected = files.reduce(0) { $0 + $1.size }
        let fileCount = files.count
        var totalWritten: Int64 = 0
        var filesCompleted = 0
        lastReportedProgress = -1

        try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)

        let plans = try files.map { file in
            try Task.checkCancellation()

            let remotePathComponents = try Self.validatedRemotePathComponents(for: file.name)
            guard let fileURL = Self.resolveDownloadURL(
                repoId: repoId,
                remotePathComponents: remotePathComponents
            ) else {
                throw DownloadError.badURL(file: file.name)
            }

            let (destURL, tempURL) = try Self.destinationURLs(for: file.name, inside: targetDir)

            let parentDir = destURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

            return FilePlan(
                name: file.name,
                size: file.size,
                remoteURL: fileURL,
                destinationURL: destURL,
                temporaryURL: tempURL,
                transferID: "\(repoId)/\(file.name)"
            )
        }

        var pendingPlans: [FilePlan] = []
        for plan in plans {
            try Task.checkCancellation()

            if try reuseCompletedFileIfPossible(
                destination: plan.destinationURL,
                temporary: plan.temporaryURL,
                expectedSize: plan.size
            ) {
                let actualSize = Self.fileSize(at: plan.destinationURL) ?? plan.size
                totalWritten += actualSize
                filesCompleted += 1
                let progress = computeProgress(
                    totalWritten: totalWritten,
                    totalExpected: totalExpected,
                    filesCompleted: filesCompleted,
                    fileCount: fileCount
                )
                emitProgress(progress, onProgress: onProgress)
                continue
            }

            pendingPlans.append(plan)
        }

        let progressRunID = UUID()
        activeProgressRunID = progressRunID
        transferBytesByID.removeAll(keepingCapacity: true)
        transferFractionsByID.removeAll(keepingCapacity: true)
        defer {
            if activeProgressRunID == progressRunID {
                activeProgressRunID = nil
                transferBytesByID.removeAll(keepingCapacity: true)
                transferFractionsByID.removeAll(keepingCapacity: true)
            }
        }

        let completedBytesBeforeTransfers = totalWritten
        let completedFilesBeforeTransfers = filesCompleted

        // Queue the complete repository up front. If iOS suspends or relaunches
        // the app, URLSession can continue every queued file rather than only
        // the one that happened to be active at that moment.
        await backgroundSession.prepareDownloads(pendingPlans.map { plan in
            BackgroundModelDownloadSession.PreparedDownload(
                remoteURL: plan.remoteURL,
                destinationURL: plan.temporaryURL,
                transferID: plan.transferID,
                onProgress: { [weak self] written, expected in
                    guard let self else { return }
                    Task { @Sendable in
                        await self.reportAggregateProgress(
                            runID: progressRunID,
                            transferID: plan.transferID,
                            bytesWritten: written,
                            serverExpectedBytes: expected,
                            declaredExpectedBytes: plan.size,
                            completedBytesBeforeTransfers: completedBytesBeforeTransfers,
                            completedFilesBeforeTransfers: completedFilesBeforeTransfers,
                            totalExpectedBytes: totalExpected,
                            fileCount: fileCount,
                            onProgress: onProgress
                        )
                    }
                }
            )
        })

        for plan in pendingPlans {
            try Task.checkCancellation()

            _ = try await backgroundSession.download(
                from: plan.remoteURL,
                to: plan.temporaryURL,
                transferID: plan.transferID
            ) { _, _ in }
            try Task.checkCancellation()

            let actualSize = Self.fileSize(at: plan.temporaryURL) ?? plan.size
            if plan.size > 0, actualSize != plan.size {
                try? FileManager.default.removeItem(at: plan.temporaryURL)
                throw DownloadError.incorrectFileSize(
                    file: plan.name,
                    expected: plan.size,
                    actual: actualSize
                )
            }

            try? FileManager.default.removeItem(at: plan.destinationURL)
            try FileManager.default.moveItem(
                at: plan.temporaryURL,
                to: plan.destinationURL
            )
            totalWritten += actualSize
            filesCompleted += 1
            let progress = computeProgress(
                totalWritten: totalWritten,
                totalExpected: totalExpected,
                filesCompleted: filesCompleted,
                fileCount: fileCount
            )
            emitProgress(progress, onProgress: onProgress)
        }

        lastReportedProgress = 1.0
        onProgress(1.0)
    }

    // MARK: - Helpers

    /// Throttled, monotonic progress emission. Shared by the per-file completion
    /// path and the in-flight delegate callbacks.
    private func emitProgress(
        _ progress: Double,
        onProgress: @escaping @Sendable (Double) -> Void
    ) {
        guard progress - lastReportedProgress >= Self.minProgressInterval else { return }
        lastReportedProgress = progress
        onProgress(progress)
    }

    private func reportAggregateProgress(
        runID: UUID,
        transferID: String,
        bytesWritten: Int64,
        serverExpectedBytes: Int64,
        declaredExpectedBytes: Int64,
        completedBytesBeforeTransfers: Int64,
        completedFilesBeforeTransfers: Int,
        totalExpectedBytes: Int64,
        fileCount: Int,
        onProgress: @escaping @Sendable (Double) -> Void
    ) {
        guard activeProgressRunID == runID else { return }

        // `totalBytesExpectedToWrite` is `NSURLSessionTransferSizeUnknown` (-1)
        // when the server omits Content-Length; fall back to the size the
        // HuggingFace API reported for this file.
        let expected = serverExpectedBytes > 0
            ? serverExpectedBytes
            : declaredExpectedBytes
        let clamped = expected > 0 ? min(bytesWritten, expected) : bytesWritten
        transferBytesByID[transferID] = max(0, clamped)
        if expected > 0 {
            transferFractionsByID[transferID] = min(
                Double(max(0, clamped)) / Double(expected),
                1.0
            )
        }

        // Never let an in-flight estimate reach 1.0 -- completion is only
        // reported once every file has actually been moved into place.
        let progress: Double
        if totalExpectedBytes > 0 {
            let transferredBytes = transferBytesByID.values.reduce(0, +)
            progress = min(
                Double(completedBytesBeforeTransfers + transferredBytes)
                    / Double(totalExpectedBytes),
                0.999
            )
        } else if fileCount > 0 {
            let transferredFileEquivalents = transferFractionsByID.values.reduce(0, +)
            progress = min(
                (Double(completedFilesBeforeTransfers) + transferredFileEquivalents)
                    / Double(fileCount),
                0.999
            )
        } else {
            return
        }
        emitProgress(progress, onProgress: onProgress)
    }

    private func reuseCompletedFileIfPossible(
        destination: URL,
        temporary: URL,
        expectedSize: Int64
    ) throws -> Bool {
        if Self.fileMatchesExpectedSize(at: destination, expectedSize: expectedSize) {
            return true
        }
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }

        guard Self.fileMatchesExpectedSize(at: temporary, expectedSize: expectedSize) else {
            if FileManager.default.fileExists(atPath: temporary.path) {
                try FileManager.default.removeItem(at: temporary)
            }
            return false
        }

        try FileManager.default.moveItem(at: temporary, to: destination)
        return true
    }

    private static func fileMatchesExpectedSize(at url: URL, expectedSize: Int64) -> Bool {
        guard FileManager.default.fileExists(atPath: url.path),
              let actualSize = fileSize(at: url) else {
            return false
        }
        return expectedSize > 0 ? actualSize == expectedSize : actualSize > 0
    }

    private static func fileSize(at url: URL) -> Int64? {
        (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init)
    }

    private func computeProgress(
        totalWritten: Int64,
        totalExpected: Int64,
        filesCompleted: Int,
        fileCount: Int
    ) -> Double {
        if totalExpected > 0 {
            return min(Double(totalWritten) / Double(totalExpected), 1.0)
        } else if fileCount > 0 {
            return min(Double(filesCompleted) / Double(fileCount), 1.0)
        }
        return 0
    }

    internal static func validatedRemotePathComponents(for remotePath: String) throws -> [String] {
        let trimmed = remotePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.hasPrefix("/") else {
            throw DownloadError.invalidRemotePath(file: remotePath)
        }

        let components = trimmed
            .split(separator: "/", omittingEmptySubsequences: false)
            .map(String.init)

        guard !components.isEmpty,
              components.allSatisfy({
                  !$0.isEmpty &&
                  !Self.disallowedRemotePathComponents.contains($0) &&
                  !$0.contains("\\")
              }) else {
            throw DownloadError.invalidRemotePath(file: remotePath)
        }

        return components
    }

    internal static func destinationURLs(
        for remotePath: String,
        inside targetDir: URL
    ) throws -> (destination: URL, temporary: URL) {
        let components = try validatedRemotePathComponents(for: remotePath)
        let targetRoot = targetDir.standardizedFileURL

        let destinationDirectory = components
            .dropLast()
            .reduce(targetRoot) { partialResult, component in
                partialResult.appendingPathComponent(component, isDirectory: true)
            }
        let destination = destinationDirectory.appendingPathComponent(
            components.last ?? "",
            isDirectory: false
        )
        let temporary = destination
            .deletingLastPathComponent()
            .appendingPathComponent(destination.lastPathComponent + ".download", isDirectory: false)

        try ensureURLIsInsideTarget(destination, targetDir: targetRoot, originalFile: remotePath)
        try ensureURLIsInsideTarget(temporary, targetDir: targetRoot, originalFile: remotePath)

        return (destination, temporary)
    }

    private static func resolveDownloadURL(
        repoId: String,
        remotePathComponents: [String]
    ) -> URL? {
        let repoComponents = repoId
            .split(separator: "/", omittingEmptySubsequences: true)
            .map(String.init)
        guard !repoComponents.isEmpty,
              let baseURL = URL(string: hfResolveBase) else {
            return nil
        }

        let repoURL = repoComponents.reduce(baseURL) { partialResult, component in
            partialResult.appendingPathComponent(component, isDirectory: false)
        }

        return (["resolve", "main"] + remotePathComponents).reduce(repoURL) { partialResult, component in
            partialResult.appendingPathComponent(component, isDirectory: false)
        }
    }

    private static func ensureURLIsInsideTarget(
        _ url: URL,
        targetDir: URL,
        originalFile: String
    ) throws {
        let targetPath = targetDir.standardizedFileURL.path
        let resolvedPath = url.standardizedFileURL.path
        guard resolvedPath == targetPath || resolvedPath.hasPrefix(targetPath + "/") else {
            throw DownloadError.invalidRemotePath(file: originalFile)
        }
    }

    // MARK: - Errors

    enum DownloadError: LocalizedError {
        case badResponse
        case noFilesFound
        case badURL(file: String)
        case invalidRemotePath(file: String)
        case incorrectFileSize(file: String, expected: Int64, actual: Int64)

        var errorDescription: String? {
            switch self {
            case .badResponse:       return "The server returned an unexpected response. Check your internet connection."
            case .noFilesFound:      return "No files found for this model on HuggingFace."
            case .badURL(let file):  return "Could not construct download URL for file: \(file)"
            case .invalidRemotePath(let file):
                return "The server returned an unexpected file path for: \(file)"
            case .incorrectFileSize(let file, let expected, let actual):
                return "Downloaded file '\(file)' has the wrong size (expected \(expected) bytes, received \(actual) bytes)."
            }
        }
    }
}
