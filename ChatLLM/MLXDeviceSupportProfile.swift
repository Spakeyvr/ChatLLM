//
//  MLXDeviceSupportProfile.swift
//  ChatLLM
//
//  Centralizes device-specific MLX availability and tool-call limits.
//

import Foundation
import UIKit

struct MLXDeviceSupportProfile: Equatable, Sendable {
    static let gibibyte: UInt64 = 1024 * 1024 * 1024
    static let minimumMemoryForPersistentKVCacheBytes: UInt64 = 12 * gibibyte
    private static let commonPhoneMemoryTiersInGigabytes = [4, 6, 8, 12, 16]
    @MainActor static var current: Self {
        Self(
            isPhone: UIDevice.current.userInterfaceIdiom == .phone,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            ramPrecautionsDisabledOverride: nil
        )
    }

    let isPhone: Bool
    let physicalMemoryBytes: UInt64
    /// When nil the flag is read live from UserDefaults, so toggling the Dev
    /// Settings option takes effect without recreating long-lived holders of
    /// this profile (e.g. MLXModelManager).
    let ramPrecautionsDisabledOverride: Bool?

    init(
        isPhone: Bool,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        ramPrecautionsDisabledOverride: Bool? = nil
    ) {
        self.isPhone = isPhone
        self.physicalMemoryBytes = physicalMemoryBytes
        self.ramPrecautionsDisabledOverride = ramPrecautionsDisabledOverride
    }

    var ramPrecautionsDisabled: Bool {
        ramPrecautionsDisabledOverride
            ?? UserDefaults.standard.bool(forKey: AppSettingsKeys.disableRAMPrecautions)
    }

    func supportsModel(_ model: MLXModelManager.MLXModelInfo) -> Bool {
        guard isPhone, let minimumBytes = model.minimumPhoneMemoryBytes, !ramPrecautionsDisabled else {
            return true
        }
        return normalizedMemoryBytes >= minimumBytes
    }

    func supportsToolCalls(for model: MLXModelManager.MLXModelInfo) -> Bool {
        guard supportsModel(model) else {
            return false
        }
        guard isPhone, let minimumBytes = model.minimumPhoneMemoryForToolCallsBytes, !ramPrecautionsDisabled else {
            return true
        }
        return normalizedMemoryBytes >= minimumBytes
    }

    func availabilityIssue(for model: MLXModelManager.MLXModelInfo) -> String? {
        guard isPhone,
              !ramPrecautionsDisabled,
              let minimumBytes = model.minimumPhoneMemoryBytes,
              normalizedMemoryBytes < minimumBytes else {
            return nil
        }
        return "\(model.displayName) requires an iPhone with at least \(Self.formattedGigabytes(minimumBytes)) GB of RAM."
    }

    func toolCallIssue(for model: MLXModelManager.MLXModelInfo) -> String? {
        guard supportsModel(model),
              isPhone,
              !ramPrecautionsDisabled,
              let minimumBytes = model.minimumPhoneMemoryForToolCallsBytes,
              normalizedMemoryBytes < minimumBytes else {
            return nil
        }
        return "Web search and tool calls for \(model.displayName) require an iPhone with at least \(Self.formattedGigabytes(minimumBytes)) GB of RAM."
    }

    var maxContextWindowTokens: Int {
        maxContextWindowTokens(for: nil)
    }

    func maxContextWindowTokens(for model: MLXModelManager.MLXModelInfo?) -> Int {
        let normalizedGigabytes = Int(normalizedMemoryBytes / Self.gibibyte)
        if isPhone, let overrides = model?.phoneContextWindowOverride {
            let tokens = overrides
                .filter { normalizedGigabytes >= $0.key }
                .max(by: { $0.key < $1.key })
                .map(\.value)
            if let tokens { return tokens }
        }
        if normalizedGigabytes >= 12 { return 6_144 }
        if normalizedGigabytes >= 8 { return 2_048 }
        return 1_024
    }

    var hasLowMemoryForPersistentKVCache: Bool {
        normalizedMemoryBytes < Self.minimumMemoryForPersistentKVCacheBytes
    }

    var normalizedMemoryBytes: UInt64 {
        guard isPhone else {
            return physicalMemoryBytes
        }

        let rawGiB = Double(physicalMemoryBytes) / Double(Self.gibibyte)
        if let bucketedTier = Self.commonPhoneMemoryTiersInGigabytes.first(where: {
            rawGiB >= Double($0) - 1.0 && rawGiB <= Double($0)
        }) {
            return UInt64(bucketedTier) * Self.gibibyte
        }

        return physicalMemoryBytes
    }

    private static func formattedGigabytes(_ bytes: UInt64) -> String {
        String(bytes / gibibyte)
    }
}
