//
//  ChatViewModel+Keychain.swift
//  ChatLLM
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import os

extension ChatViewModel {

    /// Load API key from the shared Keychain-backed store and initialize the search service.
    func loadTavilyAPIKey() {
        guard let key = TavilyAPIKeyStore.currentKey() else {
            searchService = nil
            tavilyKeyMissing = true
            return
        }

        do {
            searchService = try TavilySearchService(apiKey: key)
            tavilyKeyMissing = false
            logger.info("Tavily service initialized from shared store")
        } catch {
            logger.error("Error: Invalid Tavily key from shared store: \(error.localizedDescription)")
            searchService = nil
            tavilyKeyMissing = true
        }
    }
}
