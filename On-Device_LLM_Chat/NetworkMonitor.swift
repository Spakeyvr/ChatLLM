//
//  NetworkMonitor.swift
//  On-Device_LLM_Chat
//
//  Created by Assistant on 11/07/25.
//

import Combine
import Foundation
import Network

enum NetworkPathStatus {
    case satisfied
    case unsatisfied
}

protocol PathMonitoring: AnyObject {
    var pathUpdateHandler: ((NetworkPathStatus) -> Void)? { get set }
    func start(queue: DispatchQueue)
    func cancel()
}

final class SystemPathMonitor: PathMonitoring {
    private let monitor = NWPathMonitor()

    var pathUpdateHandler: ((NetworkPathStatus) -> Void)?

    func start(queue: DispatchQueue) {
        monitor.pathUpdateHandler = { [weak self] path in
            self?.pathUpdateHandler?(path.status == .satisfied ? .satisfied : .unsatisfied)
        }
        monitor.start(queue: queue)
    }

    func cancel() {
        monitor.cancel()
    }
}

/// Monitors network connectivity status using Apple's Network framework
final class NetworkMonitor: ObservableObject {
    /// Shared singleton instance for app-wide network monitoring
    static let shared = NetworkMonitor()

    /// Current network connectivity status
    @Published private(set) var isConnected = false

    private let monitor: PathMonitoring
    private let queue: DispatchQueue
    private var hasResolvedInitialPath = false

    init(
        monitor: PathMonitoring = SystemPathMonitor(),
        queue: DispatchQueue = DispatchQueue(label: "com.yourapp.networkmonitor")
    ) {
        self.monitor = monitor
        self.queue = queue
        startMonitoring()
    }

    private func startMonitoring() {
        monitor.pathUpdateHandler = { [weak self] status in
            guard let self else { return }

            let resolvedStatus = status == .satisfied

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }

                let wasConnected = self.isConnected
                let hadResolvedInitialPath = self.hasResolvedInitialPath

                self.hasResolvedInitialPath = true
                self.isConnected = resolvedStatus

                guard hadResolvedInitialPath, wasConnected != resolvedStatus else {
                    return
                }

                if resolvedStatus {
                    print("Network connection restored")
                } else {
                    print("Network connection lost")
                }
            }
        }

        monitor.start(queue: queue)
    }

    deinit {
        monitor.cancel()
    }
}
