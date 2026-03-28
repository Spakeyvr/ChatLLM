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

    var sharedModelContainer: ModelContainer = {
        let schema = Self.appSchema
        let modelConfiguration = Self.makeModelConfiguration()

        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

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
