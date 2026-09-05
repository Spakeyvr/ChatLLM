//
//  ConversationSidebarView.swift
//  ChatLLM
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI
import SwiftData
import UIKit

// Pre-compiled regexes for stripForPreview (compiled once at app launch)
// swiftlint:disable force_try
private let _previewSourcesBlockRegex = try! NSRegularExpression(pattern: #"<sources>.*?</sources>"#,  options: [.dotMatchesLineSeparators, .caseInsensitive])
private let _previewThinkingBlockRegex = try! NSRegularExpression(pattern: #"<(?:think|thinking)>.*?</(?:think|thinking)>"#, options: [.dotMatchesLineSeparators, .caseInsensitive])
private let _previewSourcesTagRegex   = try! NSRegularExpression(pattern: #"</?sources>"#,              options: [.caseInsensitive])
private let _previewThinkingTagRegex  = try! NSRegularExpression(pattern: #"</?(?:think|thinking)>"#,   options: [.caseInsensitive])
// swiftlint:enable force_try

// MARK: - Conversation Row

struct ConversationRow: View {
    let conversation: Conversation

    private var messagePreview: MessagePreview {
        let nonSystemMessages = conversation.messages.lazy
            .filter { $0.role != .system }
            .map { (order: $0.order, role: $0.role, text: $0.displayText) }

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

        func applying(_ regex: NSRegularExpression) {
            let ns = output as NSString
            let range = NSRange(location: 0, length: ns.length)
            output = regex.stringByReplacingMatches(in: output, options: [], range: range, withTemplate: "")
        }

        // Remove full blocks first
        applying(_previewSourcesBlockRegex)
        applying(_previewThinkingBlockRegex)
        // Remove any stray opening/closing tags that may remain
        applying(_previewSourcesTagRegex)
        applying(_previewThinkingTagRegex)
        // Collapse excessive whitespace/newlines
        var ns = output as NSString
        output = SharedRegexes.multipleBlankLines.stringByReplacingMatches(
            in: output, options: [], range: NSRange(location: 0, length: ns.length), withTemplate: "\n")
        ns = output as NSString
        output = SharedRegexes.excessiveWhitespace.stringByReplacingMatches(
            in: output, options: [], range: NSRange(location: 0, length: ns.length), withTemplate: " ")
        output = output.trimmingCharacters(in: .whitespacesAndNewlines)

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
