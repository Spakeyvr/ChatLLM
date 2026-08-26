//
//  ModelDownloader.swift
//  ChatLLM
//
//  Downloads MLX model files from HuggingFace to the local Documents directory.
//

import Foundation

actor ModelDownloader {

    private static let hfApiBase     = "https://huggingface.co/api"
    private static let hfResolveBase = "https://huggingface.co"
    private static let minProgressInterval: Double = 0.01  // report at most every 1% change
    private static let disallowedRemotePathComponents: Set<String> = [".", ".."]

    /// Last value handed to `onProgress`. Actor state rather than a local so the
    /// in-flight delegate callbacks share one throttle with the per-file path.
    private var lastReportedProgress: Double = -1

    // MARK: - File List

    func fetchFileList(repoId: String) async throws -> [(name: String, size: Int64)] {
        guard let url = URL(string: "\(Self.hfApiBase)/models/\(repoId)") else {
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

        for file in files {
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

            // `download(from:)` only returns once the whole file has landed, so
            // reporting progress per completed file leaves the UI pinned near 0%
            // for repositories dominated by a single multi-gigabyte weights file.
            // The delegate reports bytes as they arrive instead.
            let bytesSoFar = totalWritten
            let minimumDelegateByteDelta = totalExpected > 0
                ? max(1, Int64(Double(totalExpected) * Self.minProgressInterval))
                : max(1, file.size / 100)
            let progressDelegate = DownloadProgressDelegate(
                minimumByteDelta: minimumDelegateByteDelta
            ) { [weak self] written, expected in
                guard let self else { return }
                Task { @Sendable in
                    await self.reportInFlightProgress(
                        bytesForCurrentFile: written,
                        expectedForCurrentFile: expected,
                        declaredSizeForCurrentFile: file.size,
                        completedBytes: bytesSoFar,
                        totalExpected: totalExpected,
                        onProgress: onProgress
                    )
                }
            }

            let (downloadedURL, response) = try await URLSession.shared.download(
                from: fileURL,
                delegate: progressDelegate
            )
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                try? FileManager.default.removeItem(at: downloadedURL)
                throw DownloadError.badResponse
            }

            try? FileManager.default.removeItem(at: tempURL)
            try FileManager.default.moveItem(at: downloadedURL, to: tempURL)
            try Task.checkCancellation()

            let actualSize = (try? tempURL.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init)
                ?? file.size

            try? FileManager.default.removeItem(at: destURL)
            try FileManager.default.moveItem(at: tempURL, to: destURL)
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

    private func reportInFlightProgress(
        bytesForCurrentFile: Int64,
        expectedForCurrentFile: Int64,
        declaredSizeForCurrentFile: Int64,
        completedBytes: Int64,
        totalExpected: Int64,
        onProgress: @escaping @Sendable (Double) -> Void
    ) {
        guard totalExpected > 0 else { return }

        // `totalBytesExpectedToWrite` is `NSURLSessionTransferSizeUnknown` (-1)
        // when the server omits Content-Length; fall back to the size the
        // HuggingFace API reported for this file.
        let expected = expectedForCurrentFile > 0 ? expectedForCurrentFile : declaredSizeForCurrentFile
        let clamped = expected > 0 ? min(bytesForCurrentFile, expected) : bytesForCurrentFile

        // Never let an in-flight estimate reach 1.0 -- completion is only
        // reported once every file has actually been moved into place.
        let progress = min(Double(completedBytes + clamped) / Double(totalExpected), 0.999)
        emitProgress(progress, onProgress: onProgress)
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

    // MARK: - Progress Delegate

    /// Reports bytes as they arrive for a single `URLSession` download task.
    /// Only stored property is an immutable `Sendable` closure, so the
    /// unchecked conformance is safe despite the `NSObject` base.
    private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        private let onUpdate: @Sendable (_ totalBytesWritten: Int64, _ totalBytesExpected: Int64) -> Void
        private let minimumByteDelta: Int64
        private let progressLock = NSLock()
        private var lastForwardedBytes: Int64 = 0

        init(
            minimumByteDelta: Int64,
            onUpdate: @escaping @Sendable (Int64, Int64) -> Void
        ) {
            self.minimumByteDelta = max(1, minimumByteDelta)
            self.onUpdate = onUpdate
            super.init()
        }

        func urlSession(
            _ session: URLSession,
            downloadTask: URLSessionDownloadTask,
            didWriteData bytesWritten: Int64,
            totalBytesWritten: Int64,
            totalBytesExpectedToWrite: Int64
        ) {
            progressLock.lock()
            let shouldForward = totalBytesWritten - lastForwardedBytes >= minimumByteDelta
            if shouldForward {
                lastForwardedBytes = totalBytesWritten
            }
            progressLock.unlock()

            guard shouldForward else { return }
            onUpdate(totalBytesWritten, totalBytesExpectedToWrite)
        }

        /// Required by `URLSessionDownloadDelegate`. The async
        /// `download(from:delegate:)` API takes care of the finished file, so
        /// there is nothing to do here.
        func urlSession(
            _ session: URLSession,
            downloadTask: URLSessionDownloadTask,
            didFinishDownloadingTo location: URL
        ) {}
    }

    // MARK: - Errors

    enum DownloadError: LocalizedError {
        case badResponse
        case noFilesFound
        case badURL(file: String)
        case invalidRemotePath(file: String)

        var errorDescription: String? {
            switch self {
            case .badResponse:       return "The server returned an unexpected response. Check your internet connection."
            case .noFilesFound:      return "No files found for this model on HuggingFace."
            case .badURL(let file):  return "Could not construct download URL for file: \(file)"
            case .invalidRemotePath(let file):
                return "The server returned an unexpected file path for: \(file)"
            }
        }
    }
}
