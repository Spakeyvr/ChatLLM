//
//  TavilyAPIKeyStore.swift
//  On-Device_LLM_Chat
//
//  Shared persistence for the optional Tavily API key.
//

import Foundation
import Security
import os.log

enum TavilyAPIKeyStore {
    static let userDefaultsKey = "tavilyApiKey"
    static let service = "com.yourapp.tavily"
    static let account = "TavilyAPIKey"
    static let didChangeNotification = Notification.Name("TavilyKeyChanged")

    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM",
        category: "TavilyAPIKeyStore"
    )

    static func currentKey() -> String? {
        currentKey(
            userDefaults: .standard,
            service: service,
            account: account
        )
    }

    static func save(_ key: String) {
        save(
            key,
            userDefaults: .standard,
            service: service,
            account: account
        )
    }

    static func clear(postNotification: Bool = true) {
        clear(
            userDefaults: .standard,
            service: service,
            account: account,
            postNotification: postNotification
        )
    }

    static func currentKey(
        userDefaults: UserDefaults,
        service: String,
        account: String
    ) -> String? {
        migrateLegacyPlaintextKeyIfNeeded(
            userDefaults: userDefaults,
            service: service,
            account: account
        )

        return getKeychainValue(service: service, account: account)
    }

    static func save(
        _ key: String,
        userDefaults: UserDefaults,
        service: String,
        account: String
    ) {
        let trimmedKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else {
            clear(
                userDefaults: userDefaults,
                service: service,
                account: account
            )
            return
        }

        if saveKeychainValue(trimmedKey, service: service, account: account) {
            userDefaults.removeObject(forKey: userDefaultsKey)
            NotificationCenter.default.post(name: didChangeNotification, object: nil)
        }
    }

    static func clear(
        userDefaults: UserDefaults,
        service: String,
        account: String,
        postNotification: Bool = true
    ) {
        deleteKeychainValue(service: service, account: account)
        userDefaults.removeObject(forKey: userDefaultsKey)

        if postNotification {
            NotificationCenter.default.post(name: didChangeNotification, object: nil)
        }
    }

    private static func migrateLegacyPlaintextKeyIfNeeded(
        userDefaults: UserDefaults,
        service: String,
        account: String
    ) {
        guard userDefaults.object(forKey: userDefaultsKey) != nil else {
            return
        }

        let legacyValue = userDefaults.string(forKey: userDefaultsKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        defer {
            userDefaults.removeObject(forKey: userDefaultsKey)
        }

        guard !legacyValue.isEmpty else {
            return
        }

        if getKeychainValue(service: service, account: account) != legacyValue {
            _ = saveKeychainValue(legacyValue, service: service, account: account)
        }
    }

    @discardableResult
    private static func saveKeychainValue(_ value: String, service: String, account: String) -> Bool {
        guard let valueData = value.data(using: .utf8) else {
            logger.error("Tavily key save failed: UTF-8 encoding returned nil")
            return false
        }

        let matchQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        let updateAttributes: [String: Any] = [
            kSecValueData as String: valueData,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        let updateStatus = SecItemUpdate(matchQuery as CFDictionary, updateAttributes as CFDictionary)
        if updateStatus == errSecSuccess {
            return true
        }
        guard updateStatus == errSecItemNotFound else {
            logger.error("Tavily key SecItemUpdate failed: OSStatus \(updateStatus)")
            return false
        }

        var addQuery = matchQuery
        addQuery[kSecValueData as String] = valueData
        addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly

        let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
        if addStatus != errSecSuccess {
            logger.error("Tavily key SecItemAdd failed: OSStatus \(addStatus)")
            return false
        }
        return true
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

        let status = SecItemDelete(query as CFDictionary)
        if status != errSecSuccess && status != errSecItemNotFound {
            logger.error("Tavily key SecItemDelete failed: OSStatus \(status)")
        }
    }
}
