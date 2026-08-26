//
//  BackgroundModelDownloadSession.swift
//  ChatLLM
//
//  Owns the background URLSession used for large model file transfers.
//

import Foundation

nonisolated final class BackgroundModelDownloadSession: NSObject, @unchecked Sendable {
    static let shared = BackgroundModelDownloadSession()
    static let sessionIdentifier = "Nevio.ChatLLM.model-downloads"

    struct PreparedDownload: Sendable {
        let remoteURL: URL
        let destinationURL: URL
        let transferID: String
        let onProgress: @Sendable (Int64, Int64) -> Void
    }

    private struct TransferDescriptor: Codable, Sendable {
        let id: String
        let destinationPath: String
    }

    private struct Waiter {
        let continuation: CheckedContinuation<URL, any Error>
        let onProgress: @Sendable (Int64, Int64) -> Void
    }

    private struct ActiveTransfer {
        let descriptor: TransferDescriptor
        var waiters: [Waiter]
        var progressObservers: [@Sendable (Int64, Int64) -> Void]
        var stagedResult: Result<URL, any Error>?
    }

    private let stateLock = NSLock()
    private var transfersByTaskID: [Int: ActiveTransfer] = [:]
    private var cancelledTransferIDs: Set<String> = []
    private var backgroundEventsCompletionHandler: (() -> Void)?

    private var session: URLSession!

    private override init() {
        let configuration = URLSessionConfiguration.background(
            withIdentifier: Self.sessionIdentifier
        )
        configuration.sessionSendsLaunchEvents = true
        configuration.isDiscretionary = false
        configuration.waitsForConnectivity = true
        configuration.allowsCellularAccess = true
        configuration.allowsExpensiveNetworkAccess = true
        configuration.allowsConstrainedNetworkAccess = true
        configuration.httpMaximumConnectionsPerHost = 3
        super.init()
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    func download(
        from remoteURL: URL,
        to destinationURL: URL,
        transferID: String,
        onProgress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws -> URL {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let waiter = Waiter(continuation: continuation, onProgress: onProgress)
                attachOrCreateTask(
                    from: remoteURL,
                    to: destinationURL,
                    transferID: transferID,
                    waiter: waiter
                )
            }
        } onCancel: {
            cancelTransfer(withID: transferID)
        }
    }

    /// Enqueues every remaining repository file before the app can be suspended.
    /// The background session keeps these tasks alive even when no in-process
    /// continuation is currently waiting for a particular file.
    func prepareDownloads(_ downloads: [PreparedDownload]) async {
        guard !downloads.isEmpty else { return }

        await withCheckedContinuation { continuation in
            session.getAllTasks { [weak self] tasks in
                guard let self else {
                    continuation.resume()
                    return
                }

                var tasksByTransferID: [String: (
                    task: URLSessionTask,
                    descriptor: TransferDescriptor
                )] = [:]
                for task in tasks {
                    guard let descriptor = task.taskDescription
                        .flatMap(Self.decodeDescriptor) else {
                        continue
                    }
                    tasksByTransferID[descriptor.id] = (task, descriptor)
                }

                for download in downloads {
                    guard !FileManager.default.fileExists(
                        atPath: download.destinationURL.path
                    ) else {
                        continue
                    }

                    if let (task, descriptor) = tasksByTransferID[download.transferID] {
                        self.registerPreparedTask(
                            task,
                            descriptor: descriptor,
                            onProgress: download.onProgress
                        )
                        if task.state == .suspended {
                            task.resume()
                        }
                        continue
                    }

                    let descriptor = TransferDescriptor(
                        id: download.transferID,
                        destinationPath: download.destinationURL.path
                    )
                    guard let encodedDescriptor = Self.encodeDescriptor(descriptor) else {
                        continue
                    }

                    let task = self.session.downloadTask(with: download.remoteURL)
                    task.taskDescription = encodedDescriptor
                    task.priority = URLSessionTask.highPriority
                    self.registerPreparedTask(
                        task,
                        descriptor: descriptor,
                        onProgress: download.onProgress
                    )
                    task.resume()
                }

                continuation.resume()
            }
        }
    }

    func cancelTransfers(withIDPrefix prefix: String) {
        session.getAllTasks { [weak self] tasks in
            guard let self else { return }
            for task in tasks {
                guard let description = task.taskDescription,
                      let descriptor = Self.decodeDescriptor(description),
                      descriptor.id.hasPrefix(prefix) else {
                    continue
                }
                self.cancelTransfer(withID: descriptor.id)
            }
        }
    }

    func handleEventsForBackgroundSession(completionHandler: @escaping () -> Void) {
        _ = session
        stateLock.lock()
        backgroundEventsCompletionHandler = completionHandler
        stateLock.unlock()
    }

    private func attachOrCreateTask(
        from remoteURL: URL,
        to destinationURL: URL,
        transferID: String,
        waiter: Waiter
    ) {
        let descriptor = TransferDescriptor(id: transferID, destinationPath: destinationURL.path)

        session.getAllTasks { [weak self] tasks in
            guard let self else {
                waiter.continuation.resume(throwing: CancellationError())
                return
            }

            self.stateLock.lock()
            let wasCancelled = self.cancelledTransferIDs.remove(transferID) != nil
            self.stateLock.unlock()
            guard !wasCancelled else {
                waiter.continuation.resume(throwing: CancellationError())
                return
            }

            if let existingTask = tasks.first(where: {
                guard let description = $0.taskDescription,
                      let existingDescriptor = Self.decodeDescriptor(description) else {
                    return false
                }
                return existingDescriptor.id == transferID
            }) {
                self.register(waiter, for: existingTask, descriptor: descriptor)
                if existingTask.state == .suspended {
                    existingTask.resume()
                }
                return
            }

            if FileManager.default.fileExists(atPath: descriptor.destinationPath) {
                waiter.continuation.resume(
                    returning: URL(fileURLWithPath: descriptor.destinationPath)
                )
                return
            }

            let task = self.session.downloadTask(with: remoteURL)
            task.taskDescription = Self.encodeDescriptor(descriptor)
            task.priority = URLSessionTask.highPriority
            self.register(waiter, for: task, descriptor: descriptor)
            task.resume()
        }
    }

    private func register(
        _ waiter: Waiter,
        for task: URLSessionTask,
        descriptor: TransferDescriptor
    ) {
        stateLock.lock()
        if var transfer = transfersByTaskID[task.taskIdentifier] {
            transfer.waiters.append(waiter)
            transfersByTaskID[task.taskIdentifier] = transfer
        } else {
            transfersByTaskID[task.taskIdentifier] = ActiveTransfer(
                descriptor: descriptor,
                waiters: [waiter],
                progressObservers: [],
                stagedResult: nil
            )
        }
        stateLock.unlock()
    }

    private func registerPreparedTask(
        _ task: URLSessionTask,
        descriptor: TransferDescriptor,
        onProgress: @escaping @Sendable (Int64, Int64) -> Void
    ) {
        stateLock.lock()
        if var transfer = transfersByTaskID[task.taskIdentifier] {
            transfer.progressObservers.append(onProgress)
            transfersByTaskID[task.taskIdentifier] = transfer
        } else {
            transfersByTaskID[task.taskIdentifier] = ActiveTransfer(
                descriptor: descriptor,
                waiters: [],
                progressObservers: [onProgress],
                stagedResult: nil
            )
        }
        stateLock.unlock()
    }

    private func cancelTransfer(withID transferID: String) {
        stateLock.lock()
        cancelledTransferIDs.insert(transferID)
        let taskIDs = transfersByTaskID.compactMap { taskID, transfer in
            transfer.descriptor.id == transferID ? taskID : nil
        }
        stateLock.unlock()

        session.getAllTasks { tasks in
            for task in tasks where taskIDs.contains(task.taskIdentifier) || {
                guard let description = task.taskDescription,
                      let descriptor = Self.decodeDescriptor(description) else {
                    return false
                }
                return descriptor.id == transferID
            }() {
                task.cancel()
            }
        }
    }

    private func stageDownloadedFile(
        from location: URL,
        for task: URLSessionDownloadTask
    ) -> Result<URL, any Error> {
        guard let http = task.response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            let statusCode = (task.response as? HTTPURLResponse)?.statusCode
            return .failure(BackgroundDownloadError.badResponse(statusCode: statusCode))
        }

        let descriptor: TransferDescriptor?
        if let description = task.taskDescription {
            descriptor = Self.decodeDescriptor(description)
        } else {
            descriptor = nil
        }
        guard let descriptor else {
            return .failure(BackgroundDownloadError.missingTaskDescription)
        }

        let destinationURL = URL(fileURLWithPath: descriptor.destinationPath)
        do {
            let fileManager = FileManager.default
            try fileManager.createDirectory(
                at: destinationURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try? fileManager.removeItem(at: destinationURL)
            try fileManager.moveItem(at: location, to: destinationURL)
            return .success(destinationURL)
        } catch {
            return .failure(error)
        }
    }

    internal static func encodeDescriptor(id: String, destinationPath: String) -> String? {
        encodeDescriptor(TransferDescriptor(id: id, destinationPath: destinationPath))
    }

    internal static func decodeDescriptorForTesting(_ value: String) -> (id: String, destinationPath: String)? {
        guard let descriptor = decodeDescriptor(value) else { return nil }
        return (descriptor.id, descriptor.destinationPath)
    }

    private static func encodeDescriptor(_ descriptor: TransferDescriptor) -> String? {
        guard let data = try? JSONEncoder().encode(descriptor) else { return nil }
        return data.base64EncodedString()
    }

    private static func decodeDescriptor(_ value: String) -> TransferDescriptor? {
        guard let data = Data(base64Encoded: value) else { return nil }
        return try? JSONDecoder().decode(TransferDescriptor.self, from: data)
    }

    private enum BackgroundDownloadError: LocalizedError {
        case badResponse(statusCode: Int?)
        case missingTaskDescription

        var errorDescription: String? {
            switch self {
            case .badResponse(let statusCode):
                if let statusCode {
                    return "The model server returned HTTP \(statusCode)."
                }
                return "The model server returned an unexpected response."
            case .missingTaskDescription:
                return "The background model download lost its destination information."
            }
        }
    }
}

extension BackgroundModelDownloadSession: URLSessionDownloadDelegate, URLSessionTaskDelegate {
    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        stateLock.lock()
        let transfer = transfersByTaskID[downloadTask.taskIdentifier]
        let callbacks = (transfer?.progressObservers ?? [])
            + (transfer?.waiters.map(\.onProgress) ?? [])
        stateLock.unlock()

        for callback in callbacks {
            callback(totalBytesWritten, totalBytesExpectedToWrite)
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        let result = stageDownloadedFile(from: location, for: downloadTask)

        stateLock.lock()
        if var transfer = transfersByTaskID[downloadTask.taskIdentifier] {
            transfer.stagedResult = result
            transfersByTaskID[downloadTask.taskIdentifier] = transfer
        } else if let description = downloadTask.taskDescription,
                  let descriptor = Self.decodeDescriptor(description) {
            transfersByTaskID[downloadTask.taskIdentifier] = ActiveTransfer(
                descriptor: descriptor,
                waiters: [],
                progressObservers: [],
                stagedResult: result
            )
        }
        stateLock.unlock()
    }

    nonisolated func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: (any Error)?
    ) {
        stateLock.lock()
        let transfer = transfersByTaskID.removeValue(forKey: task.taskIdentifier)
        let transferID = transfer?.descriptor.id
            ?? task.taskDescription.flatMap(Self.decodeDescriptor)?.id
        if let transferID {
            cancelledTransferIDs.remove(transferID)
        }
        stateLock.unlock()

        guard let transfer else { return }
        let result: Result<URL, any Error>
        if let error {
            result = .failure(error)
        } else if let stagedResult = transfer.stagedResult {
            result = stagedResult
        } else {
            result = .failure(BackgroundDownloadError.missingTaskDescription)
        }

        for waiter in transfer.waiters {
            waiter.continuation.resume(with: result)
        }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        stateLock.lock()
        let completionHandler = backgroundEventsCompletionHandler
        backgroundEventsCompletionHandler = nil
        stateLock.unlock()

        guard let completionHandler else { return }
        DispatchQueue.main.async {
            completionHandler()
        }
    }
}
