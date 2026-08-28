//
//  ChatLLMTests+AppServices
//  ChatLLMTests
//
//  Split out of ChatLLMTests.swift; part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
import MLXLMCommon
import SwiftUI
import SwiftData
import WebKit
import UIKit
import FoundationModels
@testable import ChatLLM

extension ChatLLMTests {
    @Test func chatViewOnlyInterceptsWebURLsForInAppBrowser() throws {
        #expect(ChatView.shouldPresentInAppBrowser(for: try #require(URL(string: "https://example.com"))))
        #expect(!ChatView.shouldPresentInAppBrowser(for: try #require(URL(string: "mailto:test@example.com"))))
        #expect(!ChatView.shouldPresentInAppBrowser(for: try #require(URL(string: "tel:+123456"))))
    }

    @Test func networkMonitorStartsOfflineUntilFirstPathResolves() async {
        let monitor = FakePathMonitor()
        let networkMonitor = NetworkMonitor(
            monitor: monitor,
            queue: DispatchQueue(label: "test.network.monitor")
        )

        #expect(!networkMonitor.isConnected)

        monitor.emit(.satisfied)
        await Task.yield()
        try? await Task.sleep(for: .milliseconds(20))

        #expect(networkMonitor.isConnected)
    }

    @Test func composerReturnKeyBehaviorSendsOnlyWhenEnabledAndSendable() {
        #expect(
            ComposerReturnKeyBehavior.action(
                sendOnReturn: true,
                canSend: true,
                isGenerating: false,
                replacementText: "\n"
            ) == .send
        )
        #expect(
            ComposerReturnKeyBehavior.action(
                sendOnReturn: false,
                canSend: true,
                isGenerating: false,
                replacementText: "\n"
            ) == .insertNewline
        )
        #expect(
            ComposerReturnKeyBehavior.action(
                sendOnReturn: true,
                canSend: false,
                isGenerating: false,
                replacementText: "\n"
            ) == .insertNewline
        )
    }

    @Test func disabledHapticsSuppressFeedbackDispatch() throws {
        let suiteName = "haptics-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }

        defaults.enableHapticsPreference = false
        var invoked = false

        let didPerform = AppHaptics.performIfEnabled(defaults: defaults) {
            invoked = true
        }

        #expect(!didPerform)
        #expect(!invoked)
    }
}
