//
//  NetworkMonitor.swift
//  On-Device_LLM_Chat
//
//  Created by Assistant on 11/07/25.
//

import Foundation
import Network
import Combine

/// Monitors network connectivity status using Apple's Network framework
@MainActor
final class NetworkMonitor: ObservableObject {
    /// Shared singleton instance for app-wide network monitoring
    static let shared = NetworkMonitor()
    
    /// Current network connectivity status
    @Published private(set) var isConnected: Bool = true
    
    private let monitor: NWPathMonitor
    private let queue = DispatchQueue(label: "com.yourapp.networkmonitor")
    
    private init() {
        monitor = NWPathMonitor()
        startMonitoring()
    }
    
    private func startMonitoring() {
        monitor.pathUpdateHandler = { [weak self] path in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                let wasConnected = self.isConnected
                self.isConnected = path.status == .satisfied
                
                // Log connectivity changes
                if wasConnected != self.isConnected {
                    if self.isConnected {
                        print("🌐 Network connection restored")
                    } else {
                        print("📡 Network connection lost")
                    }
                }
            }
        }
        
        monitor.start(queue: queue)
    }
    
    deinit {
        monitor.cancel()
    }
}
