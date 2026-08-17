//
//  MLXRuntimeConfiguration.swift
//  On-Device_LLM_Chat
//
//  Configures MLX before its Metal backend is initialized.
//

import Darwin
import Foundation

nonisolated enum MLXRuntimeConfiguration {
    private static let metalArchitectureEnvironmentKey = "MLX_METAL_GPU_ARCH"

    static func configureForCurrentProcess() {
        #if targetEnvironment(simulator)
        guard ProcessInfo.processInfo.environment[metalArchitectureEnvironmentKey] == nil,
              let deviceName = hostCPUBrandString(),
              let architecture = metalGPUArchitecture(for: deviceName) else {
            return
        }
        setenv(metalArchitectureEnvironmentKey, architecture, 0)
        #endif
    }

    static func metalGPUArchitecture(for deviceName: String) -> String? {
        let components = deviceName.split(whereSeparator: \Character.isWhitespace)
        guard components.contains(where: { $0 == "Apple" }),
              let modelComponent = components.first(where: {
                  $0.first == "M" && $0.dropFirst().allSatisfy(\.isNumber)
              }),
              let modelGeneration = Int(modelComponent.dropFirst()),
              modelGeneration >= 1 else {
            return nil
        }

        let variant: Character
        let lowercasedName = deviceName.lowercased()
        if lowercasedName.contains("ultra") {
            variant = "d"
        } else if lowercasedName.contains("max") {
            variant = "s"
        } else {
            variant = "g"
        }
        // Metal architecture names use the `applegpu_g13g` shape. Apple M-series
        // generations map sequentially from M1/g13 through M5/g17.
        return "applegpu_g\(modelGeneration + 12)\(variant)"
    }

    #if targetEnvironment(simulator)
    private static func hostCPUBrandString() -> String? {
        var byteCount = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &byteCount, nil, 0) == 0,
              byteCount > 1 else {
            return nil
        }

        var bytes = [CChar](repeating: 0, count: byteCount)
        let status = bytes.withUnsafeMutableBufferPointer { buffer in
            sysctlbyname("machdep.cpu.brand_string", buffer.baseAddress, &byteCount, nil, 0)
        }
        guard status == 0 else { return nil }
        return bytes.withUnsafeBufferPointer { buffer in
            buffer.baseAddress.map(String.init(cString:))
        }
    }
    #endif
}
