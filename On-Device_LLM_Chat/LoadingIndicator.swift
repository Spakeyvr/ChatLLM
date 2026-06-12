//
//  LoadingIndicator.swift
//  On-Device_LLM_Chat
//
//  Reusable loading views: an animated label whose trailing dots cycle
//  1 → 2 → 3 → 1 to signal liveness, plus a convenience wrapper that
//  pairs it with a spinner.
//

import Combine
import SwiftUI

/// Just the animated text label — pair with your own spinner when you
/// already render one (e.g. inside a toolbar capsule).
struct LoadingDotsLabel: View {
    var label: String = String(localized: "Loading")
    var font: Font = .subheadline
    var color: Color = .secondary

    @State private var dotCount: Int = 1
    private let timer = Timer.publish(every: 0.4, on: .main, in: .common).autoconnect()

    var body: some View {
        Text(label + String(repeating: ".", count: dotCount))
            .font(font)
            .foregroundStyle(color)
            .monospacedDigit()
            .accessibilityLabel(Text(label))
            .onReceive(timer) { _ in
                dotCount = dotCount % 3 + 1
            }
    }
}

/// Spinner + animated label. Use for full-screen / empty-state loading.
struct LoadingIndicator: View {
    var label: String = String(localized: "Loading")

    var body: some View {
        HStack(spacing: 8) {
            ProgressView()
            LoadingDotsLabel(label: label)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text(label))
    }
}
