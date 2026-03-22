//
//  MessageBubbleView.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI
import SwiftData
import UIKit
import SafariServices

// MARK: - Localization Constants (Performance Optimization)
enum Strings {
    static let thinking = String(localized: "Thinking…")
    static let loading = String(localized: "Loading…")
    static let reasoning = String(localized: "Reasoning")
    static let alwaysUseReasoning = String(localized: "Always Use Reasoning")
    static let smartReasoningMode = String(localized: "Smart Reasoning Mode")
    static let reasoningModeEnabled = String(localized: "Reasoning mode enabled")
    static let smartReasoningModeEnabled = String(localized: "Smart reasoning mode enabled")
}

// MARK: - Shared error callout

private struct ErrorCalloutView: View {
    let text: String
    var topPadding: CGFloat = 0

    var body: some View {
        Label(text, systemImage: "exclamationmark.triangle.fill")
            .font(.callout)
            .foregroundStyle(.red)
            .padding(10)
            .background(Color.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
            .padding(.top, topPadding)
    }
}

// swiftlint:disable:next force_try
private let _messageBubbleThinkTagRegex = try! NSRegularExpression(
    pattern: #"<(thinking|think)>([\s\S]*?)</\1>"#, options: [.caseInsensitive])

// MARK: - String helper to extract <sources> blocks

extension String {
    // Returns the visible text with all <sources>…</sources> blocks removed,
    // and a single combined sources string (if any) for UI presentation.
    func extractSourcesBlocks() -> (visible: String, sources: String?) {
        let regex = SharedRegexes.sourcesBlock
        let ns = self as NSString
        let fullRange = NSRange(location: 0, length: ns.length)

        var sourcesParts: [String] = []

        // Find all matches; collect content forward then remove tags backward
        let matches = regex.matches(in: self, options: [], range: fullRange)
        for match in matches where match.numberOfRanges >= 2 {
            let content = ns.substring(with: match.range(at: 1))
            if !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                sourcesParts.append(content)
            }
        }
        var mutable = self
        for match in matches.reversed() {
            if let swiftRange = Range(match.range(at: 0), in: mutable) {
                mutable.removeSubrange(swiftRange)
            }
        }

        let combinedSources = sourcesParts.isEmpty ? nil : sourcesParts.joined(separator: "\n\n")
        let visible = mutable.trimmingCharacters(in: .whitespacesAndNewlines)
        return (visible.isEmpty ? "" : visible, combinedSources)
    }
}

// MARK: - In-app browser (SFSafariViewController)

struct SafariView: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> SFSafariViewController {
        let config = SFSafariViewController.Configuration()
        config.entersReaderIfAvailable = true
        let vc = SFSafariViewController(url: url, configuration: config)
        // iOS 26+: avoid deprecated tint customization to preserve system background effects.
        return vc
    }

    func updateUIViewController(_ uiViewController: SFSafariViewController, context: Context) {}
}

// Allow URL to be used with .sheet(item:)
extension URL: @retroactive Identifiable {
    public var id: URL { self }
}

// MARK: - Performance-optimized message cell view

struct MessageCellView: View {
    let message: Message
    @ObservedObject var viewModel: ChatViewModel
    let onEdit: (Message) -> Void
    let onShare: (String) -> Void
    @AppStorage("developerModeEnabled") private var developerModeEnabled: Bool = false
    @State private var showDeveloperSheet = false

    private var isCurrentlyStreaming: Bool {
        viewModel.isGenerating && viewModel.streamingMessageID == message.id
    }

    private var hasContent: Bool {
        !message.text.isEmpty ||
        !(message.finalAnswer?.isEmpty ?? true) ||
        !(message.reasoning?.isEmpty ?? true) ||
        message.generationError != nil
    }

    private var canAct: Bool {
        !viewModel.isGenerating && !isCurrentlyStreaming && hasContent && message.isFinal
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Bubble or loading placeholder
            if message.role == .assistant, isCurrentlyStreaming, !hasContent {
                if message.isReasoningMode {
                    InlineThinkingView(onTap: nil)
                } else {
                    LoadingIndicatorView(isModelLoading: true)
                }
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    MessageBubble(message: message, viewModel: viewModel)
                        .contextMenu {
                            MessageContextMenuView(
                                message: message,
                                onEdit: onEdit,
                                onShare: onShare,
                                onCopy: { text in
                                    UIPasteboard.general.string = text
                                }
                            )
                        }
                        .swipeActions {
                            MessageSwipeActionsView(
                                message: message,
                                onEdit: onEdit,
                                onShare: onShare,
                                onCopy: { text in
                                    UIPasteboard.general.string = text
                                }
                            )
                        }

                }
            }

            // Actions bar for assistant messages - ONLY show when message is complete
            // CRITICAL FIX (Bug 8): Use animation and transition to prevent flicker during regeneration
            if message.role == .assistant && hasContent && message.isFinal && !isCurrentlyStreaming {
                AssistantActionsBar(
                    messageID: message.id,
                    isGenerating: viewModel.isGenerating,
                    isStreamingForThisMessage: isCurrentlyStreaming,
                    canAct: canAct,
                    isFinal: message.isFinal,
                    onStop: { viewModel.cancelGeneration() },
                    onTryAgain: {
                        viewModel.scheduleRegeneration(messageID: message.id, instruction: nil)
                    },
                    onConcise: {
                        viewModel.scheduleRegeneration(
                            messageID: message.id,
                            instruction: String(localized: "Please answer again, but be more concise.")
                        )
                    },
                    onFormal: {
                        viewModel.scheduleRegeneration(
                            messageID: message.id,
                            instruction: String(localized: "Please answer again using a formal tone.")
                        )
                    },
                    developerModeEnabled: developerModeEnabled && message.role == .assistant,
                    onDeveloper: {
                        showDeveloperSheet = true
                    }
                )
                .padding(.leading, 12)
                .transition(.opacity.combined(with: .move(edge: .bottom)))
                .animation(.easeInOut(duration: 0.2), value: message.isFinal)
            }
        }
        .sheet(isPresented: $showDeveloperSheet) {
            DeveloperMessageSheet(message: message)
        }
    }

    /// Check if the previous user message (that this assistant message is responding to) has an image
    private var userMessageHasImage: Bool {
        guard message.role == .assistant else { return false }

        // Find the user message immediately before this assistant message
        let userMessage = viewModel.conversation.messages
            .filter { $0.role == .user && $0.order < message.order }
            .sortedByOrder
            .last

        return userMessage?.attachments.contains(where: { $0.type == .image }) ?? false
    }
}

// MARK: - Loading indicator reusing the shimmer style

struct LoadingIndicatorView: View {
    var isModelLoading: Bool = false

    var body: some View {
        InlineThinkingView(text: isModelLoading ? Strings.loading : Strings.thinking, onTap: nil)
            .padding(.vertical, 4)
            .transition(.move(edge: .bottom).combined(with: .opacity))
    }
}

// MARK: - Basic message bubble with reasoning mode support

struct MessageBubble: View {
    let message: Message
    @ObservedObject var viewModel: ChatViewModel

    var isUser: Bool { message.role == .user }
    var isAssistant: Bool { message.role == .assistant }
    var isSystem: Bool { message.role == .system }

    var body: some View {
        // Show reasoning bubble if message is flagged as reasoning mode OR if it contains thinking tags
        if isAssistant && (message.isReasoningMode || message.text.contains("<thinking>")) {
            ReasoningMessageBubble(message: message, viewModel: viewModel)
        } else {
            StandardMessageBubble(message: message, viewModel: viewModel)
        }
    }
}

// MARK: - Standard message bubble for non-reasoning messages with cached markdown rendering

struct StandardMessageBubble: View {
    let message: Message
    @ObservedObject var viewModel: ChatViewModel
    @AppStorage("messageFontSize") private var messageFontSize: Double = 16.0

    // Cache for markdown rendering
    @State private var cachedAttributedString: AttributedString?
    @State private var lastRenderedText: String = ""
    @State private var lastRenderedFontSize: Double = 16.0

    // Sources UI state
    @State private var showSources = false

    var isUser: Bool { message.role == .user }
    var isAssistant: Bool { message.role == .assistant }
    var isSystem: Bool { message.role == .system }

    private var isStreaming: Bool {
        viewModel.isGenerating && viewModel.streamingMessageID == message.id
    }

    var body: some View {
        // During streaming, avoid expensive full-text processing on every repaint.
        // Use displayText (which strips hidden image context) even during streaming.
        let split: (visible: String, sources: String?) = isStreaming
            ? (message.displayText, nil)
            : message.displayText.extractSourcesBlocks()
        let visibleText = split.visible
        let sourcesText = split.sources

        HStack(alignment: .bottom, spacing: 8) {
            VStack(alignment: .leading, spacing: 8) {
                // Show image attachments for user messages
                if isUser && !message.attachments.isEmpty {
                    ForEach(message.attachments, id: \.id) { attachment in
                        if attachment.type == .image {
                            MessageImageAttachmentView(attachment: attachment)
                        }
                    }
                }

                // Show text content with cached markdown
                if !visibleText.isEmpty {
                    renderMarkdownOrPlain(visibleText, isSystem: isSystem)
                        .textSelection(.enabled)
                }

                // Error callout — shown below partial content (if any) or alone
                if let errorText = message.generationError {
                    ErrorCalloutView(text: errorText, topPadding: visibleText.isEmpty ? 0 : 6)
                }

                // Sources button
                if let sourcesText, !sourcesText.isEmpty, isAssistant {
                    SourcesButton(showSources: $showSources)
                        .accessibilityLabel("Show sources")
                        .sheet(isPresented: $showSources) {
                            SourcesSheetView(
                                sourcesText: sourcesText,
                                title: String(localized: "Sources"),
                                searchQuery: message.searchQuery
                            )
                        }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(alignment: .center) {
                if isUser {
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(Color.blue.opacity(0.15))
                } else {
                    RoundedRectangle(cornerRadius: isSystem ? 12 : 14, style: .continuous)
                        .fill(Color.clear)
                        .background(
                            .ultraThinMaterial,
                            in: RoundedRectangle(cornerRadius: isSystem ? 12 : 14, style: .continuous)
                        )
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: isSystem ? 12 : 14, style: .continuous)
                    .strokeBorder(Color.secondary.opacity(0.12))
            )
            .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)

        }
    }

    @ViewBuilder
    private func renderMarkdownOrPlain(_ text: String, isSystem: Bool) -> some View {
        if !isStreaming && !RichTextFeatureDetector.requiresAdvancedRendering(text) &&
            lastRenderedText == text && lastRenderedFontSize == messageFontSize && cachedAttributedString != nil {
            // Cache hit
            Text(cachedAttributedString!)
                .font(.system(size: messageFontSize))
                .foregroundStyle(isSystem ? .secondary : .primary)
        } else {
            if !RichTextFeatureDetector.requiresAdvancedRendering(text) {
                let processedText = LatexProcessor.process(text)
                Group {
                    if let attributed = try? AttributedString(markdown: processedText) {
                        Text(attributed)
                            .font(.system(size: messageFontSize))
                            .foregroundStyle(isSystem ? .secondary : .primary)
                            .onAppear {
                                if !isStreaming {
                                    cachedAttributedString = attributed
                                    lastRenderedText = text
                                    lastRenderedFontSize = messageFontSize
                                }
                            }
                    } else {
                        Text(processedText.isEmpty ? " " : processedText)
                            .font(.system(size: messageFontSize))
                            .foregroundStyle(isSystem ? .secondary : .primary)
                    }
                }
            } else {
                RichMarkdownView(
                    text: text,
                    fontSize: messageFontSize,
                    textTone: isSystem ? .secondary : .primary
                )
            }
        }
    }
}

// MARK: - Reasoning mode message bubble with modern design

struct ReasoningMessageBubble: View {
    let message: Message
    @AppStorage("messageFontSize") private var messageFontSize: Double = 16.0
    @ObservedObject var viewModel: ChatViewModel

    // Sources UI state
    @State private var showSources = false
    @State private var selectedSearchInvocation: SearchInvocation?
    @State private var showReasoningSheet = false

    private var isCurrentlyStreaming: Bool {
        viewModel.isGenerating && viewModel.streamingMessageID == message.id
    }

    // Parse reasoning content on-demand if not already parsed
    private var parsedContent: (reasoning: String?, finalAnswer: String?) {
        // If already parsed, return stored values
        if let reasoning = message.reasoning, !reasoning.isEmpty {
            return (reasoning, message.finalAnswer)
        }
        // If not parsed but contains thinking tags, parse on-demand
        let lowercased = message.text.lowercased()
        if lowercased.contains("<thinking>") || lowercased.contains("<think>") {
            return parseReasoningFromText(message.text)
        }
        return (nil, nil)
    }

    private var hasReasoningContent: Bool {
        return parsedContent.reasoning != nil && !parsedContent.reasoning!.isEmpty
    }

    private func parseReasoningFromText(_ text: String) -> (reasoning: String?, finalAnswer: String?) {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return (nil, nil) }
        let range = NSRange(trimmedText.startIndex..<trimmedText.endIndex, in: trimmedText)
        guard let match = _messageBubbleThinkTagRegex.firstMatch(in: trimmedText, options: [], range: range),
              match.numberOfRanges >= 3,
              let reasoningRange = Range(match.range(at: 2), in: trimmedText),
              let fullMatchRange = Range(match.range(at: 0), in: trimmedText) else {
            return (nil, text)
        }

        let reasoningContent = String(trimmedText[reasoningRange]).trimmingCharacters(in: .whitespacesAndNewlines)
        let afterThinking = String(trimmedText[fullMatchRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        let finalAnswer = afterThinking
            .replacingOccurrences(
                of: "^Final answer:\\s*",
                with: "",
                options: [.regularExpression, .caseInsensitive]
            )
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (
            reasoning: reasoningContent.isEmpty ? nil : reasoningContent,
            finalAnswer: finalAnswer.isEmpty ? nil : finalAnswer
        )
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            VStack(alignment: .leading, spacing: 8) {
                let rawFinal = parsedContent.finalAnswer ?? message.finalAnswer ?? message.displayText
                let split = rawFinal.extractSourcesBlocks()
                let finalText = split.visible
                let sourcesText = split.sources

                // Show "Thinking…" text during entire streaming phase; tappable once sheet has content
                if isCurrentlyStreaming {
                    InlineThinkingView(
                        onTap: hasReasoningContent ? { showReasoningSheet = true } : nil
                    )
                }

                // Final answer section — only shown when there is content to display
                let hasFinalSection = !finalText.isEmpty || message.generationError != nil ||
                    (message.searchInvocations?.isEmpty == false && message.isFinal) ||
                    (message.isFinal && sourcesText != nil) ||
                    (hasReasoningContent && message.isFinal)
                if hasFinalSection {
                    VStack(alignment: .leading, spacing: 8) {
                        if !finalText.isEmpty {
                            if isCurrentlyStreaming {
                                // Fast path: skip LaTeX + markdown parsing during streaming to avoid per-frame stutter
                                Text(finalText)
                                    .font(.system(size: messageFontSize))
                                    .foregroundStyle(.primary)
                                    .textSelection(.enabled)
                                    .lineLimit(nil)
                                    .fixedSize(horizontal: false, vertical: true)
                            } else {
                                RichMarkdownView(text: finalText, fontSize: messageFontSize)
                                    .textSelection(.enabled)
                                    .lineLimit(nil)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }

                        // Error callout for reasoning messages
                        if let errorText = message.generationError {
                            ErrorCalloutView(text: errorText, topPadding: finalText.isEmpty ? 0 : 6)
                        }

                        // Search cards inline (for reasoning messages with searches)
                        if let invocations = message.searchInvocations, !invocations.isEmpty, message.isFinal {
                            SearchInvocationsList(invocations: invocations) { invocation in
                                selectedSearchInvocation = invocation
                            }
                            .padding(.vertical, 4)
                        }

                        HStack(spacing: 12) {
                            // View Reasoning button — opens step-by-step sheet
                            if hasReasoningContent && message.isFinal {
                                Button {
                                    showReasoningSheet = true
                                } label: {
                                    HStack(spacing: 6) {
                                        Image(systemName: "brain.head.profile")
                                        Text("View Reasoning")
                                    }
                                    .font(.caption)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(.thinMaterial, in: Capsule())
                                }
                                .buttonStyle(.plain)
                            }

                            // Only show combined Sources button for non-search-invocation messages
                            if let sourcesText, !sourcesText.isEmpty, (message.searchInvocations ?? []).isEmpty {
                                SourcesButton(showSources: $showSources)
                                    .sheet(isPresented: $showSources) {
                                        SourcesSheetView(
                                            sourcesText: sourcesText,
                                            title: String(localized: "Sources"),
                                            searchQuery: message.searchQuery
                                        )
                                    }
                            }
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(Color.clear)
                            .background(
                                .ultraThinMaterial,
                                in: RoundedRectangle(cornerRadius: 14, style: .continuous)
                            )
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .strokeBorder(Color.secondary.opacity(0.12))
                    )
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Spacer(minLength: 0)
        }
        .sheet(isPresented: $showReasoningSheet) {
            StepByStepReasoningSheet(
                reasoning: parsedContent.reasoning ?? "",
                reasoningSteps: message.reasoningSteps,
                searchInvocations: message.searchInvocations
            )
        }
        .sheet(item: $selectedSearchInvocation) { invocation in
            SourcesSheetView(
                sourcesText: invocation.results,
                title: String(localized: "Search Results"),
                searchQuery: invocation.query
            )
        }
    }
}

// MARK: - Compact Sources button

struct SourcesButton: View {
    @Binding var showSources: Bool

    var body: some View {
        Button {
            showSources = true
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "list.bullet.rectangle.portrait")
                Text(String(localized: "Sources"))
            }
            .font(.caption)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.thinMaterial, in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Sheet that renders the <sources> content

struct SourcesSheetView: View {
    let sourcesText: String
    let title: String
    let searchQuery: String?  // The user's search query, if search was used
    @State private var activeURL: URL?   // Local in-sheet browser state

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Show search query at the top if available
                    if let query = searchQuery, !query.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Label {
                                Text("Showing results for:")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            } icon: {
                                Image(systemName: "magnifyingglass")
                                    .foregroundStyle(.primary)
                            }

                            Text(query)
                                .font(.body)
                                .fontWeight(.medium)
                                .foregroundStyle(.primary)
                                .padding(.leading, 28) // Align with label text
                        }
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .overlay {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .strokeBorder(.secondary.opacity(0.2), lineWidth: 1)
                        }
                    }

                    // Sources content
                    VStack(alignment: .leading, spacing: 12) {
                        RichMarkdownView(
                            text: sourcesText,
                            fontSize: UIFont.preferredFont(forTextStyle: .body).pointSize,
                            forceAdvancedRenderer: true
                        )
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding()
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
        // Intercept links inside this sheet and present SafariView immediately
        .environment(\.openURL, OpenURLAction { url in
            activeURL = url
            return .handled
        })
        .sheet(item: $activeURL) { url in
            SafariView(url: url)
                .ignoresSafeArea()
        }
    }
}

// MARK: - Developer diagnostics

private struct DeveloperMessageSheet: View {
    let message: Message

    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .medium
        return formatter
    }()

    private var rawText: String {
        let raw = message.developerRawText
        return raw.isEmpty ? String(localized: "No raw output captured for this message.") : raw
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(String(localized: "Stats"))
                            .font(.headline)

                        DeveloperStatRow(label: String(localized: "Backend"), value: message.generationBackend ?? "Unknown")

                        if let generationModelName = message.generationModelName, !generationModelName.isEmpty {
                            DeveloperStatRow(label: String(localized: "Model"), value: generationModelName)
                        }

                        if let startedAt = message.generationStartedAt {
                            DeveloperStatRow(
                                label: String(localized: "Started"),
                                value: Self.dateFormatter.string(from: startedAt)
                            )
                        }

                        if let completedAt = message.generationCompletedAt {
                            DeveloperStatRow(
                                label: String(localized: "Finished"),
                                value: Self.dateFormatter.string(from: completedAt)
                            )
                        }

                        if let duration = message.generationDuration {
                            DeveloperStatRow(label: String(localized: "Duration"), value: String(format: "%.2fs", duration))
                        }

                        if let estimatedOutputTokenCount = message.estimatedOutputTokenCount {
                            DeveloperStatRow(label: String(localized: "Est. output tokens"), value: "\(estimatedOutputTokenCount)")
                        }

                        if let estimatedTokensPerSecond = message.estimatedTokensPerSecond {
                            DeveloperStatRow(label: String(localized: "Est. tokens/sec"), value: String(format: "%.2f", estimatedTokensPerSecond))
                        }

                        DeveloperStatRow(label: String(localized: "Raw chars"), value: "\(message.developerRawText.count)")
                        DeveloperStatRow(label: String(localized: "Visible chars"), value: "\(message.displayText.count)")
                    }
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(.ultraThinMaterial)
                    )

                    VStack(alignment: .leading, spacing: 10) {
                        Text(String(localized: "Raw Output"))
                            .font(.headline)

                        Text(rawText)
                            .font(.system(.body, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                            .background(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .fill(Color.primary.opacity(0.06))
                            )
                    }
                }
                .padding()
            }
            .navigationTitle(String(localized: "Developer"))
            .navigationBarTitleDisplayMode(.inline)
        }
        .presentationDetents([.fraction(0.5), .large])
        .presentationDragIndicator(.visible)
    }
}

private struct DeveloperStatRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer(minLength: 12)
            Text(value)
                .multilineTextAlignment(.trailing)
                .font(.system(.body, design: .monospaced))
        }
    }
}

// MARK: - Message context menu

struct MessageContextMenuView: View {
    let message: Message
    let onEdit: (Message) -> Void
    let onShare: (String) -> Void
    let onCopy: (String) -> Void

    var body: some View {
        Button {
            deferMenuSafe { onCopy(message.displayText) }
        } label: {
            Label(String(localized: "Copy"), systemImage: "doc.on.doc")
        }

        Button {
            deferMenuSafe { onShare(message.displayText) }
        } label: {
            Label(String(localized: "Share"), systemImage: "square.and.arrow.up")
        }

        if message.role == .user {
            Button {
                deferMenuSafe { onEdit(message) }
            } label: {
                Label(String(localized: "Edit & Regenerate"), systemImage: "pencil.circle")
            }
        }
    }

    private func deferMenuSafe(_ action: @escaping () -> Void) {
        Task { @MainActor in
            await Task.yield()
            try? await Task.sleep(nanoseconds: 150_000_000) // ~150 ms
            action()
        }
    }
}

// MARK: - Message swipe actions

struct MessageSwipeActionsView: View {
    let message: Message
    let onEdit: (Message) -> Void
    let onShare: (String) -> Void
    let onCopy: (String) -> Void

    var body: some View {
        Button {
            // Small deferral improves reliability
            Task { @MainActor in
                await Task.yield()
                try? await Task.sleep(nanoseconds: 120_000_000)
                onCopy(message.displayText)
            }
        } label: {
            Label(String(localized: "Copy"), systemImage: "doc.on.doc")
        }
        .tint(.green)

        Button {
            // Small deferral improves reliability when share controller presents
            Task { @MainActor in
                await Task.yield()
                try? await Task.sleep(nanoseconds: 120_000_000)
                onShare(message.displayText)
            }
        } label: {
            Label(String(localized: "Share"), systemImage: "square.and.arrow.up")
        }
        .tint(.blue)

        if message.role == .user {
            Button {
                Task { @MainActor in
                    await Task.yield()
                    try? await Task.sleep(nanoseconds: 120_000_000)
                    onEdit(message)
                }
            } label: {
                Label(String(localized: "Edit"), systemImage: "pencil")
            }
            .tint(.orange)
        }
    }
}

// MARK: - Edit message sheet

struct EditMessageSheet: View {
    @Binding var editedText: String
    let isGenerating: Bool
    let onSave: () -> Void
    let onCancel: () -> Void

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    TextEditor(text: $editedText)
                        .font(.body)
                        .frame(minHeight: 200)
                        .textInputAutocapitalization(.sentences)
                        .autocorrectionDisabled(false)
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .fill(.ultraThinMaterial)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                                        .strokeBorder(Color.secondary.opacity(0.2))
                                )
                        )
                        .overlay(alignment: .topLeading) {
                            if editedText.isEmpty {
                                Text(String(localized: "Enter your message here…"))
                                    .foregroundStyle(.secondary)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 20)
                                    .allowsHitTesting(false)
                            }
                        }
                }
                .padding()
            }
            .navigationTitle(String(localized: "Edit Message"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(String(localized: "Cancel"), action: onCancel)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(String(localized: "Save & Regenerate")) {
                        onSave()
                    }
                    .disabled(editedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isGenerating)
                    .fontWeight(.semibold)
                }
            }
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }
}

// MARK: - Enhanced Assistant Actions Bar

struct AssistantActionsBar: View {
    let messageID: UUID
    var isGenerating: Bool
    var isStreamingForThisMessage: Bool
    var canAct: Bool
    var isFinal: Bool
    var onStop: () -> Void
    var onTryAgain: () -> Void
    var onConcise: () -> Void
    var onFormal: () -> Void
    var developerModeEnabled: Bool
    var onDeveloper: () -> Void

    // State for fade-in animation
    @State private var isVisible = false

    var body: some View {
        HStack(spacing: 16) {
            // Stop button removed from here; lives in the composer now.

            Group {
                Menu {
                    Button(action: onTryAgain) {
                        Label(String(localized: "Try again"), systemImage: "arrow.triangle.2.circlepath")
                    }
                    Button(action: onConcise) {
                        Label(String(localized: "More concise"), systemImage: "text.justify.leading")
                    }
                    Button(action: onFormal) {
                        Label(String(localized: "More formal"), systemImage: "textformat.abc")
                    }
                } label: {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .foregroundStyle(.primary)
                }
                // Removed .menuStyle(.borderlessButton) – flaky in scrollable cells
                .disabled(!canAct)
                .accessibilityLabel(String(localized: "Regenerate"))
            }

            if developerModeEnabled {
                Button(action: onDeveloper) {
                    Image(systemName: "hammer")
                        .foregroundStyle(.primary)
                }
                .buttonStyle(.plain)
                .disabled(!canAct)
                .accessibilityLabel(String(localized: "Developer"))
            }

            Spacer()
        }
        .font(.subheadline)
        .foregroundStyle(.secondary)
        .opacity(isVisible ? 1.0 : 0.0)
        .animation(.easeInOut(duration: 0.3), value: isVisible)
        .onAppear {
            // Slight delay before fade-in for a smoother feel
            Task {
                try? await Task.sleep(for: .milliseconds(50))
                isVisible = true
            }
        }
    }
}

// MARK: - Compact image preview in chat bubbles

struct MessageImageAttachmentView: View {
    let attachment: MessageAttachment
    @State private var image: UIImage?
    @State private var isLoading = true
    @State private var showFullscreen = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let image = image {
                Button {
                    showFullscreen = true
                } label: {
                    VStack(alignment: .leading, spacing: 0) {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(maxWidth: 240, maxHeight: 180)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                            )

                        if let detections = attachment.getDetectionResults(), !detections.isEmpty {
                            HStack(spacing: 6) {
                                Image(systemName: "viewfinder.circle.fill")
                                    .font(.caption2)
                                    .foregroundStyle(.blue)

                                Text("\(detections.count) object\(detections.count == 1 ? "" : "s") detected")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.blue.opacity(0.1))
                            .clipShape(Capsule())
                            .padding(.top, 4)
                        }
                    }
                }
                .buttonStyle(.plain)
            } else if isLoading {
                RoundedRectangle(cornerRadius: 10)
                    .fill(Color.secondary.opacity(0.1))
                    .frame(width: 240, height: 180)
                    .overlay {
                        ProgressView()
                    }
            } else {
                HStack(spacing: 6) {
                    Image(systemName: "photo.badge.exclamationmark")
                        .foregroundStyle(.secondary)
                    Text("Image unavailable")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .background(Color.secondary.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
        .task {
            await loadImage()
        }
        .sheet(isPresented: $showFullscreen) {
            if let image = image {
                FullscreenImageView(
                    image: image,
                    detections: attachment.getDetectionResults(),
                    isPresented: $showFullscreen
                )
            }
        }
    }

    private func loadImage() async {
        isLoading = true
        image = await attachment.loadImage()
        isLoading = false
    }
}

// MARK: - UIActivityViewController wrapper

struct ActivityView: UIViewControllerRepresentable {
    var activityItems: [Any]
    var applicationActivities: [UIActivity]? = nil

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: applicationActivities)
    }

    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
