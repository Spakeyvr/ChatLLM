//
//  ChatViewModel+Keychain.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import Security
import os

extension ChatViewModel {

    /// Load API key from Keychain; initializes service if valid.
    /// Syncs with AppStorage for consistency.
    func loadTavilyAPIKey() {
        // AppStorage is the primary source; Keychain is the secure backing store.
        if let appStorageKey = UserDefaults.standard.string(forKey: "tavilyApiKey"),
           !appStorageKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if getKeychainValue(service: tavilyKeyService, account: tavilyKeyAccount) != appStorageKey {
                saveKeychainValue(appStorageKey, service: tavilyKeyService, account: tavilyKeyAccount)
            }
            do {
                searchService = try TavilySearchService(apiKey: appStorageKey)
                tavilyKeyMissing = false
                logger.info("✅ Tavily service initialized from AppStorage")
                return
            } catch {
                logger.error("❌ Invalid Tavily key from AppStorage: \(error.localizedDescription)")
            }
        }

        guard let key = getKeychainValue(service: tavilyKeyService, account: tavilyKeyAccount) else {
            searchService = nil
            tavilyKeyMissing = true
            return
        }
        do {
            searchService = try TavilySearchService(apiKey: key)
            tavilyKeyMissing = false
            // Sync back so the Settings UI can see the stored key
            UserDefaults.standard.set(key, forKey: "tavilyApiKey")
            logger.info("✅ Tavily service initialized from Keychain")
        } catch {
            logger.error("❌ Invalid Tavily key from Keychain: \(error.localizedDescription)")
            searchService = nil
            tavilyKeyMissing = true
        }
    }

    /// Save API key to Keychain and AppStorage, then reinitialize service.
    /// Call this from SettingsSheet when key changes.
    func saveTavilyAPIKey(_ key: String) {
        UserDefaults.standard.set(key, forKey: "tavilyApiKey")

        guard !key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            deleteKeychainValue(service: tavilyKeyService, account: tavilyKeyAccount)
            UserDefaults.standard.removeObject(forKey: "tavilyApiKey")
            searchService = nil
            tavilyKeyMissing = true
            NotificationCenter.default.post(name: NSNotification.Name("TavilyKeyChanged"), object: nil)
            return
        }
        saveKeychainValue(key, service: tavilyKeyService, account: tavilyKeyAccount)
        loadTavilyAPIKey()
        NotificationCenter.default.post(name: NSNotification.Name("TavilyKeyChanged"), object: nil)
    }

    // MARK: - Keychain Helpers (Private)
    func saveKeychainValue(_ value: String, service: String, account: String) {
        guard let valueData = value.data(using: .utf8) else {
            logger.error("❌ Failed to encode Keychain value as UTF-8")
            return
        }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: valueData,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        SecItemDelete(query as CFDictionary)  // Delete first to avoid duplicate-item error
        let status = SecItemAdd(query as CFDictionary, nil)
        if status != errSecSuccess {
            logger.error("❌ Keychain save failed with status: \(status)")
        } else {
            logger.debug("✅ Keychain value saved successfully")
        }
    }

    func getKeychainValue(service: String, account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess,
              let data = item as? Data,
              let value = String(data: data, encoding: .utf8) else {
            if status != errSecItemNotFound {
                logger.error("❌ Keychain read failed with status: \(status)")
            }
            return nil
        }
        logger.debug("✅ Keychain value retrieved successfully")
        return value
    }

    func deleteKeychainValue(service: String, account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        let status = SecItemDelete(query as CFDictionary)
        if status == errSecSuccess {
            logger.debug("✅ Keychain value deleted successfully")
        } else if status != errSecItemNotFound {
            logger.error("❌ Keychain delete failed with status: \(status)")
        }
    }
}
