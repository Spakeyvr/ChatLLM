//
//  ChatLLMTests+AppServices
//  ChatLLMTests
//
//  Split out of ChatLLMTests.swift; part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
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

    @Test func composerOnlyWritesFocusWhenUIKitChangesIt() {
        let probe = FocusBindingProbe(isFocused: true)
        let composer = ComposerTextView(
            text: .constant(""),
            isFocused: probe.binding,
            measuredHeight: .constant(24),
            sendOnReturn: false,
            canSend: false,
            isGenerating: false,
            maxHeight: 120,
            onSend: {}
        )
        let coordinator = composer.makeCoordinator()
        let textView = UITextView()

        // Focusing from updateUIView reports straight back; re-writing the same
        // value modified state during the view update.
        coordinator.textViewDidBeginEditing(textView)
        #expect(probe.writes == 0)

        coordinator.textViewDidEndEditing(textView)
        #expect(probe.writes == 1)
        #expect(!probe.isFocused)
        coordinator.textViewDidEndEditing(textView)
        #expect(probe.writes == 1)
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

@MainActor
private final class FocusBindingProbe {
    var isFocused: Bool
    var writes = 0

    init(isFocused: Bool) {
        self.isFocused = isFocused
    }

    var binding: Binding<Bool> {
        Binding(
            get: { MainActor.assumeIsolated { self.isFocused } },
            set: { value in MainActor.assumeIsolated { self.isFocused = value; self.writes += 1 } }
        )
    }
}
