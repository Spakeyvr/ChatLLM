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
    init() {
        configureUITestStateIfNeeded()
    }

    var sharedModelContainer: ModelContainer = {
        let schema = Schema([
            Conversation.self,
            Message.self,
            MessageAttachment.self
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)

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
    }
}
