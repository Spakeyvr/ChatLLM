//
//  On_Device_LLM_ChatApp.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI
import SwiftData

@main
struct On_Device_LLM_ChatApp: App {
    private static let appSchema = Schema([
        Conversation.self,
        Message.self,
        MessageAttachment.self
    ])

    init() {
        configureUITestStateIfNeeded()
    }

    var sharedModelContainer: ModelContainer = Self.makeModelContainer()

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(sharedModelContainer)
    }

    private func configureUITestStateIfNeeded() {
        let arguments = ProcessInfo.processInfo.arguments
        guard arguments.contains("-ui-test-reset-app-state") else {
            return
        }

        if let bundleIdentifier = Bundle.main.bundleIdentifier {
            UserDefaults.standard.removePersistentDomain(forName: bundleIdentifier)
        }

        TavilyAPIKeyStore.clear(postNotification: false)
        clearSwiftDataStoreIfNeeded()
        clearAttachmentStorageIfNeeded()
    }

    private static func makeModelConfiguration() -> ModelConfiguration {
        ModelConfiguration(schema: appSchema, isStoredInMemoryOnly: false)
    }

    private static func makeModelContainer() -> ModelContainer {
        let modelConfiguration = makeModelConfiguration()
        do {
            return try ModelContainer(for: appSchema, configurations: [modelConfiguration])
        } catch {
            print("Could not create SwiftData container; attempting recovery: \(error)")
        }

        quarantineSwiftDataStore(at: modelConfiguration.url)
        print("SwiftData store was quarantined; using in-memory store for this launch.")

        let fallbackConfiguration = ModelConfiguration(schema: appSchema, isStoredInMemoryOnly: true)
        do {
            return try ModelContainer(for: appSchema, configurations: [fallbackConfiguration])
        } catch {
            preconditionFailure("Could not create fallback in-memory SwiftData container: \(error)")
        }
    }

    private static func quarantineSwiftDataStore(at storeURL: URL) {
        let fileManager = FileManager.default
        let parentURL = storeURL.deletingLastPathComponent()
        let recoveryURL = parentURL.appendingPathComponent("FailedStores", isDirectory: true)
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        let safeTimestamp = formatter.string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let backupURL = recoveryURL.appendingPathComponent(safeTimestamp, isDirectory: true)

        do {
            try fileManager.createDirectory(at: backupURL, withIntermediateDirectories: true)
        } catch {
            print("Could not create SwiftData recovery directory: \(error)")
            return
        }

        let sidecarURLs = [
            storeURL,
            URL(fileURLWithPath: storeURL.path + "-shm"),
            URL(fileURLWithPath: storeURL.path + "-wal")
        ]

        for url in sidecarURLs where fileManager.fileExists(atPath: url.path) {
            let destinationURL = backupURL.appendingPathComponent(url.lastPathComponent)
            do {
                try fileManager.moveItem(at: url, to: destinationURL)
            } catch {
                print("Could not quarantine SwiftData store file \(url.lastPathComponent): \(error)")
            }
        }
    }

    private func clearSwiftDataStoreIfNeeded() {
        let storeURL = Self.makeModelConfiguration().url
        let sidecarURLs = [
            storeURL,
            URL(fileURLWithPath: storeURL.path + "-shm"),
            URL(fileURLWithPath: storeURL.path + "-wal")
        ]

        for url in sidecarURLs where FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.removeItem(at: url)
        }
    }

    private func clearAttachmentStorageIfNeeded() {
        guard let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return
        }

        let attachmentsURL = documentsURL.appendingPathComponent("Attachments", isDirectory: true)
        guard FileManager.default.fileExists(atPath: attachmentsURL.path) else {
            return
        }

        try? FileManager.default.removeItem(at: attachmentsURL)
    }
}
