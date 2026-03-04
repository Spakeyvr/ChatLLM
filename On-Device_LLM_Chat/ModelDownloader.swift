//
//  ModelDownloader.swift
//  On-Device_LLM_Chat
//
//  Downloads MLX model files from HuggingFace to the local Documents directory.
//

import Foundation

actor ModelDownloader {

    private static let hfApiBase     = "https://huggingface.co/api"
    private static let hfResolveBase = "https://huggingface.co"
    private static let chunkSize     = 256 * 1024
    private static let minProgressInterval: Double = 0.01  // report at most every 1% change

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
        var lastReportedProgress: Double = -1

        try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)

        for file in files {
            try Task.checkCancellation()

            guard let fileURL = URL(string: "\(Self.hfResolveBase)/\(repoId)/resolve/main/\(file.name)") else {
                throw DownloadError.badURL(file: file.name)
            }

            let destURL = targetDir.appendingPathComponent(file.name)
            let tempURL = targetDir.appendingPathComponent("\(file.name).download")

            let parentDir = destURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

            let (asyncBytes, response) = try await URLSession.shared.bytes(from: fileURL)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw DownloadError.badResponse
            }

            // Use Content-Length if HF API reported zero (common for LFS files).
            let contentLength = (http.value(forHTTPHeaderField: "Content-Length")).flatMap(Int64.init) ?? 0
            let effectiveTotal = totalExpected > 0 ? totalExpected
                : (contentLength > 0 ? totalWritten + contentLength + files.dropFirst(filesCompleted + 1).reduce(0) { $0 + $1.size } : 0)

            FileManager.default.createFile(atPath: tempURL.path, contents: nil)
            let handle = try FileHandle(forWritingTo: tempURL)

            do {
                var chunk = Data(capacity: Self.chunkSize)
                for try await byte in asyncBytes {
                    try Task.checkCancellation()
                    chunk.append(byte)
                    if chunk.count >= Self.chunkSize {
                        handle.write(chunk)
                        totalWritten += Int64(chunk.count)
                        chunk.removeAll(keepingCapacity: true)
                        let p = computeProgress(totalWritten: totalWritten, totalExpected: effectiveTotal,
                                                filesCompleted: filesCompleted, fileCount: fileCount)
                        if p - lastReportedProgress >= Self.minProgressInterval {
                            lastReportedProgress = p
                            onProgress(p)
                        }
                    }
                }
                if !chunk.isEmpty {
                    handle.write(chunk)
                    totalWritten += Int64(chunk.count)
                }
                try handle.close()
            } catch {
                try? handle.close()
                try? FileManager.default.removeItem(at: tempURL)
                throw error
            }

            try? FileManager.default.removeItem(at: destURL)
            try FileManager.default.moveItem(at: tempURL, to: destURL)
            filesCompleted += 1
        }

        onProgress(1.0)
    }

    // MARK: - Helpers

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

    // MARK: - Errors

    enum DownloadError: LocalizedError {
        case badResponse
        case noFilesFound
        case badURL(file: String)

        var errorDescription: String? {
            switch self {
            case .badResponse:       return "The server returned an unexpected response. Check your internet connection."
            case .noFilesFound:      return "No files found for this model on HuggingFace."
            case .badURL(let file):  return "Could not construct download URL for file: \(file)"
            }
        }
    }
}
