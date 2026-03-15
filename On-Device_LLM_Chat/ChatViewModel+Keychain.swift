//
//  ChatViewModel+Keychain.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import os

extension ChatViewModel {

    /// Load API key from Keychain; initializes service if valid.
    /// Syncs with AppStorage for consistency.
    func loadTavilyAPIKey() {
        guard let key = TavilyAPIKeyStore.currentKey() else {
            searchService = nil
            tavilyKeyMissing = true
            return
        }

        do {
            searchService = try TavilySearchService(apiKey: key)
            tavilyKeyMissing = false
            logger.info("✅ Tavily service initialized from shared store")
        } catch {
            logger.error("❌ Invalid Tavily key from shared store: \(error.localizedDescription)")
            searchService = nil
            tavilyKeyMissing = true
        }
    }

    /// Save API key to Keychain and AppStorage, then reinitialize service.
    /// Call this from SettingsSheet when key changes.
    func saveTavilyAPIKey(_ key: String) {
        TavilyAPIKeyStore.save(key)
        loadTavilyAPIKey()
    }
}
