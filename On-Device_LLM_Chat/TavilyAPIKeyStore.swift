//
//  TavilyAPIKeyStore.swift
//  On-Device_LLM_Chat
//
//  Shared persistence for the optional Tavily API key.
//

import Foundation
import Security

enum TavilyAPIKeyStore {
    static let userDefaultsKey = "tavilyApiKey"
    static let service = "com.yourapp.tavily"
    static let account = "TavilyAPIKey"
    static let didChangeNotification = Notification.Name("TavilyKeyChanged")

    static func currentKey() -> String? {
        if UserDefaults.standard.object(forKey: userDefaultsKey) != nil,
           let storedKey = UserDefaults.standard.string(forKey: userDefaultsKey) {
            let trimmedKey = storedKey.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedKey.isEmpty else {
                deleteKeychainValue(service: service, account: account)
                UserDefaults.standard.removeObject(forKey: userDefaultsKey)
                return nil
            }

            if getKeychainValue(service: service, account: account) != trimmedKey {
                saveKeychainValue(trimmedKey, service: service, account: account)
            }
            return trimmedKey
        }

        guard let key = getKeychainValue(service: service, account: account) else {
            return nil
        }

        UserDefaults.standard.set(key, forKey: userDefaultsKey)
        return key
    }

    static func save(_ key: String) {
        let trimmedKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else {
            clear()
            return
        }

        UserDefaults.standard.set(trimmedKey, forKey: userDefaultsKey)
        saveKeychainValue(trimmedKey, service: service, account: account)
        NotificationCenter.default.post(name: didChangeNotification, object: nil)
    }

    static func clear(postNotification: Bool = true) {
        deleteKeychainValue(service: service, account: account)
        UserDefaults.standard.removeObject(forKey: userDefaultsKey)

        if postNotification {
            NotificationCenter.default.post(name: didChangeNotification, object: nil)
        }
    }

    private static func saveKeychainValue(_ value: String, service: String, account: String) {
        guard let valueData = value.data(using: .utf8) else {
            return
        }

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: valueData,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        SecItemDelete(query as CFDictionary)
        SecItemAdd(query as CFDictionary, nil)
    }

    private static func getKeychainValue(service: String, account: String) -> String? {
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
            return nil
        }

        return value
    }

    private static func deleteKeychainValue(service: String, account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]

        SecItemDelete(query as CFDictionary)
    }
}
