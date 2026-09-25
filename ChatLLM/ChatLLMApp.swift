//
//  ChatLLMApp.swift
//  ChatLLM
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI
import SwiftData
import UIKit

final class ChatLLMAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        guard identifier == BackgroundModelDownloadSession.sessionIdentifier else {
            completionHandler()
            return
        }

        BackgroundModelDownloadSession.shared.handleEventsForBackgroundSession(
            completionHandler: completionHandler
        )

        // Recreate the model manager promptly when iOS relaunches the app to
        // deliver background-session events. It reconnects to the persisted task.
        _ = ModelBackendBridge.shared
    }
}

@main
struct ChatLLMApp: App {
    @UIApplicationDelegateAdaptor(ChatLLMAppDelegate.self) private var appDelegate

    private static let appSchema = Schema([
        Conversation.self,
        Message.self,
        MessageAttachment.self
    ])

    init() {
        // Reset test data before opening SQLite, never underneath a live store.
        Self.configureUITestStateIfNeeded()
        sharedModelContainer = Self.makeModelContainer()
    }

    var sharedModelContainer: ModelContainer

    var body: some Scene {
        WindowGroup {
            #if DEBUG
            if ProcessInfo.processInfo.arguments.contains("-ui-test-markdown") {
                MarkdownRenderingPreview()
            } else {
                ContentView()
            }
            #else
            ContentView()
            #endif
        }
        .modelContainer(sharedModelContainer)
    }

    private static func configureUITestStateIfNeeded() {
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

        if arguments.contains("-ui-test-web-search-demo") {
            UserDefaults.standard.set(true, forKey: "hasCompletedOnboarding")
        }
    }

    private static func makeModelConfiguration() -> ModelConfiguration {
        ModelConfiguration(schema: appSchema, isStoredInMemoryOnly: false)
    }

    private static func makeModelContainer() -> ModelContainer {
        let modelConfiguration = makeModelConfiguration()
        var lastError: Error?
        // Retry once: a transient failure (for example a briefly locked file)
        // must not be treated as a corrupt store.
        for _ in 0..<2 {
            do {
                return try ModelContainer(for: appSchema, configurations: [modelConfiguration])
            } catch {
                lastError = error
            }
        }
        print("Could not create SwiftData container; using in-memory store for this launch: \(String(describing: lastError))")

        // Leave the on-disk store where it is so a later launch (or an app
        // update that fixes a migration) can still open it. Keep a copy aside
        // as a backup, and let the user decide whether to start fresh.
        backUpSwiftDataStore(at: modelConfiguration.url)
        PersistentStoreRecovery.markOpenFailed(storeURL: modelConfiguration.url)

        let fallbackConfiguration = ModelConfiguration(schema: appSchema, isStoredInMemoryOnly: true)
        do {
            return try ModelContainer(for: appSchema, configurations: [fallbackConfiguration])
        } catch {
            preconditionFailure("Could not create fallback in-memory SwiftData container: \(error)")
        }
    }

    private static func backUpSwiftDataStore(at storeURL: URL) {
        guard let backupURL = PersistentStoreRecovery.makeBackupDirectory(near: storeURL) else { return }
        for url in PersistentStoreRecovery.storeFileURLs(for: storeURL)
        where FileManager.default.fileExists(atPath: url.path) {
            do {
                try FileManager.default.copyItem(
                    at: url,
                    to: backupURL.appendingPathComponent(url.lastPathComponent)
                )
            } catch {
                print("Could not back up SwiftData store file \(url.lastPathComponent): \(error)")
            }
        }
    }

    private static func clearSwiftDataStoreIfNeeded() {
        let storeURL = Self.makeModelConfiguration().url
        for url in PersistentStoreRecovery.storeFileURLs(for: storeURL) where FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.removeItem(at: url)
        }
    }

    private static func clearAttachmentStorageIfNeeded() {
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

/// Tracks a launch whose on-disk SwiftData store could not be opened, so the
/// UI can tell the user that chats are temporarily not being saved.
@MainActor
enum PersistentStoreRecovery {
    private(set) static var openFailed = false
    private static var storeURL: URL?

    static func markOpenFailed(storeURL: URL) {
        openFailed = true
        self.storeURL = storeURL
    }

    nonisolated static func storeFileURLs(for storeURL: URL) -> [URL] {
        [
            storeURL,
            URL(fileURLWithPath: storeURL.path + "-shm"),
            URL(fileURLWithPath: storeURL.path + "-wal")
        ]
    }

    nonisolated static func makeBackupDirectory(near storeURL: URL) -> URL? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        let safeTimestamp = formatter.string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let backupURL = storeURL.deletingLastPathComponent()
            .appendingPathComponent("FailedStores", isDirectory: true)
            .appendingPathComponent(safeTimestamp, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: backupURL, withIntermediateDirectories: true)
            return backupURL
        } catch {
            print("Could not create SwiftData recovery directory: \(error)")
            return nil
        }
    }

    /// Moves the unreadable store aside (a backup copy already exists) so the
    /// next launch starts with an empty, working store. The store is not open
    /// during this launch, which runs on the in-memory fallback.
    static func discardUnreadableStore() {
        guard openFailed, let storeURL else { return }
        for url in storeFileURLs(for: storeURL) where FileManager.default.fileExists(atPath: url.path) {
            do {
                try FileManager.default.removeItem(at: url)
            } catch {
                print("Could not remove unreadable SwiftData store file \(url.lastPathComponent): \(error)")
            }
        }
    }
}
