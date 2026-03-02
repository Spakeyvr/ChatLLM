//
//  ConversationSidebarView.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI
import SwiftData
import UIKit

// MARK: - Conversation Row

struct ConversationRow: View {
    let conversation: Conversation

    // CRITICAL FIX: Eagerly resolve message preview to avoid SwiftData fault errors during deletion
    // This prevents "detached from context without resolving attribute faults" errors
    private var messagePreview: MessagePreview {
        // Eagerly access and snapshot properties before they might be deleted
        let nonSystemMessages = conversation.messages.compactMap { msg -> (order: Int, role: MessageRole, text: String)? in
            // Force fault resolution by accessing properties
            let role = msg.role
            let order = msg.order
            let text = msg.text

            guard role != .system else { return nil }
            return (order: order, role: role, text: text)
        }

        if let last = nonSystemMessages.max(by: { $0.order < $1.order }) {
            // Strip <sources>…</sources> blocks and <thinking>…</thinking> from the preview
            return .message(stripForPreview(last.text))
        } else if conversation.messages.isEmpty {
            return .empty
        } else {
            return .none
        }
    }

    // Removes <sources>…</sources>, stray <sources> tags, and <thinking>…</thinking> from preview text.
    private func stripForPreview(_ text: String) -> String {
        var output = text

        func replacing(_ pattern: String) {
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators, .caseInsensitive]) {
                let ns = output as NSString
                let range = NSRange(location: 0, length: ns.length)
                output = regex.stringByReplacingMatches(in: output, options: [], range: range, withTemplate: "")
            }
        }

        // Remove full blocks first
        replacing(#"<sources>.*?</sources>"#)
        replacing(#"<thinking>.*?</thinking>"#)
        // Remove any stray opening/closing tags that may remain
        replacing(#"</?sources>"#)
        replacing(#"</?thinking>"#)
        // Collapse excessive whitespace/newlines
        output = output
            .replacingOccurrences(of: "[ \\t]*\\n[ \\t]*\\n+", with: "\n", options: .regularExpression)
            .replacingOccurrences(of: "\\s{2,}", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return output
    }

    private enum MessagePreview {
        case message(String)
        case empty
        case none
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text(conversation.title.isEmpty ? String(localized: "Untitled") : conversation.title)
                    .font(.headline)
                    .lineLimit(1)
                    .accessibilityAddTraits(.isHeader)

                if conversation.reasoningMode {
                    Image(systemName: "brain.head.profile")
                        .font(.caption2)
                        .foregroundStyle(.blue)
                        .accessibilityLabel(String(localized: "Reasoning mode"))
                }

                Spacer(minLength: 0)
            }

            switch messagePreview {
            case .message(let text):
                // Use markdown Text initializer to render formatting like **bold** and *italic*
                Text(.init(text))
                    .font(.subheadline)
                    .lineLimit(2) // Allow 2 lines for better preview
                    .foregroundStyle(.secondary)
            case .empty:
                Text(String(localized: "No messages yet"))
                    .font(.subheadline)
                    .foregroundStyle(.tertiary)
                    .italic()
            case .none:
                EmptyView()
            }
        }
        .padding(.vertical, 0)
    }
}

// MARK: - Sidebar Header

struct SidebarHeader: View {
    var isAvailable: Bool
    var openSettings: () -> Void
    var hasChats: Bool

    var body: some View {
        ViewThatFits(in: .horizontal) {
            // Variant 1: Single row when everything fits comfortably
            singleRow

            // Variant 2: Two rows — full title on top
            twoRow
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .glassEffect(
            .regular.tint(isAvailable ? .blue.opacity(0.1) : .orange.opacity(0.1)),
            in: .rect(cornerRadius: 18)
        )
        .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .listRowSeparator(.hidden)
        .accessibilityElement(children: .contain)
    }

    private var titleBlock: some View {
        HStack(spacing: 12) {
            Image(systemName: "sparkles")
                .symbolRenderingMode(.hierarchical)
                .font(.system(size: 24, weight: .semibold))
                // Slightly darker purple than previous, still soft
                .foregroundStyle(Color(red: 0.65, green: 0.52, blue: 0.94).opacity(0.95))

            VStack(alignment: .leading, spacing: 1) {
                Text(String(localized: "Apple Intelligence"))
                    .font(.headline)
                    .fixedSize(horizontal: true, vertical: false)
                Text(isAvailable ? String(localized: "On‑device available") : String(localized: "On‑device unavailable"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var settingsButton: some View {
        Button(action: openSettings) {
            Image(systemName: "gear")
                .font(.system(size: 26, weight: .semibold))
                .foregroundStyle(.primary) // Stronger contrast
                .frame(width: 33, height: 42)
                .contentShape(Circle())
        }
        .buttonStyle(.glass)
        .accessibilityLabel(String(localized: "Settings"))
    }

    private var singleRow: some View {
        HStack(spacing: 12) {
            titleBlock

            Spacer(minLength: 8)

            settingsButton
        }
    }

    private var twoRow: some View {
        VStack(spacing: 8) {
            HStack(spacing: 12) {
                titleBlock
                Spacer()
                settingsButton
            }
        }
    }
}
