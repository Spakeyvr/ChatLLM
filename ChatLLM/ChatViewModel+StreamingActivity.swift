//
//  ChatViewModel+StreamingActivity.swift
//  ChatLLM
//
//  The shimmering status label shown while a reply streams: "Loading…" until the
//  first token, "Thinking…" afterwards, and "Searching…" while a web search runs.
//

import Foundation
import UIKit

enum StreamingActivity: Equatable {
    case loading
    case thinking
    case searching

    /// What the reply is doing right now, from the stream's progress.
    static func current(hasReceivedOutput: Bool, isSearching: Bool) -> StreamingActivity {
        if isSearching { return .searching }
        return hasReceivedOutput ? .thinking : .loading
    }

    func label(locale: Locale = .current) -> String {
        switch self {
        case .loading: String(localized: "Loading…", bundle: .appLocalized, locale: locale)
        case .thinking: String(localized: "Thinking…", bundle: .appLocalized, locale: locale)
        case .searching: String(localized: "Searching…", bundle: .appLocalized, locale: locale)
        }
    }
}

extension ChatViewModel {

    /// Switches the status label to `activity` once the glow currently sweeping across it
    /// has finished, so the text never changes mid-sweep. While a switch is waiting, only
    /// the most recent request is applied.
    func requestStreamingActivity(_ activity: StreamingActivity, now: Date = Date()) {
        guard activity != streamingActivity else {
            cancelPendingStreamingActivity()
            return
        }
        pendingStreamingActivity = activity
        // A switch is already waiting for the sweep to end; it will pick up the new target.
        guard streamingActivityTask == nil else { return }

        let delay = UIAccessibility.isReduceMotionEnabled
            ? 0
            : InlineThinkingView.remainingSweepDuration(at: now)
        guard delay > 0 else {
            applyPendingStreamingActivity()
            return
        }
        streamingActivityTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled, let self else { return }
            self.streamingActivityTask = nil
            self.applyPendingStreamingActivity()
        }
    }

    /// Starts a new reply at "Loading…" without waiting for a sweep.
    func resetStreamingActivity() {
        cancelPendingStreamingActivity()
        streamingActivity = .loading
    }

    private func applyPendingStreamingActivity() {
        guard let pending = pendingStreamingActivity else { return }
        pendingStreamingActivity = nil
        streamingActivity = pending
    }

    private func cancelPendingStreamingActivity() {
        streamingActivityTask?.cancel()
        streamingActivityTask = nil
        pendingStreamingActivity = nil
    }
}
