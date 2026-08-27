//
//  MessageBubbleView.swift
//  ChatLLM
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
            .background(Color.red.opacity(0.08), in: RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.small))
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

    private var sourceInvocations: [SearchInvocation] {
        let stored = (message.searchInvocations ?? []).filter { !$0.displaySources.isEmpty }
        if !stored.isEmpty { return stored }

        guard let legacyText = extractSourcesFromMessage(), !legacyText.isEmpty else { return [] }
        return [SearchInvocation(
            query: message.searchQuery?.isEmpty == false ? message.searchQuery! : String(localized: "Saved web search"),
            results: legacyText
        )]
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
                        .accessibilityActions {
                            if message.role == .user {
                                Button(String(localized: "Edit")) {
                                    onEdit(message)
                                }
                            }
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
                    sourceInvocations: sourceInvocations,
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

    private func extractSourcesFromMessage() -> String? {
        let candidateText = message.isReasoningMode
            ? (message.finalAnswer ?? message.displayText)
            : message.displayText
        let extracted = candidateText.extractSourcesBlocks().sources.map {
            SearchInvocation.userVisibleResults(from: $0).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return (extracted?.isEmpty == false) ? extracted : nil
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

    var isUser: Bool { message.role == .user }
    var isAssistant: Bool { message.role == .assistant }
    var isSystem: Bool { message.role == .system }

    private var isStreaming: Bool {
        viewModel.isGenerating && viewModel.streamingMessageID == message.id
    }

    var body: some View {
        // During streaming, avoid expensive full-text processing on every repaint.
        // Use displayText (which strips hidden image context) even during streaming.
        let visibleText = isStreaming
            ? message.displayText
            : message.displayText.extractSourcesBlocks().visible

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

            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(alignment: .center) {
                if isUser {
                    RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.large, style: .continuous)
                        .fill(Color.blue.opacity(0.15))
                } else {
                    RoundedRectangle(cornerRadius: isSystem ? DesignTokens.CornerRadius.medium : DesignTokens.CornerRadius.large, style: .continuous)
                        .fill(Color.clear)
                        .background(
                            .ultraThinMaterial,
                            in: RoundedRectangle(cornerRadius: isSystem ? DesignTokens.CornerRadius.medium : DesignTokens.CornerRadius.large, style: .continuous)
                        )
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: isSystem ? DesignTokens.CornerRadius.medium : DesignTokens.CornerRadius.large, style: .continuous)
                    .strokeBorder(Color.secondary.opacity(0.12))
            )
            .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)

        }
    }

    @ViewBuilder
    private func renderMarkdownOrPlain(_ text: String, isSystem: Bool) -> some View {
        // Detection runs a dozen regexes over the whole message; evaluate it once.
        let requiresAdvancedRendering = RichTextFeatureDetector.requiresAdvancedRendering(text)

        if !isStreaming && !requiresAdvancedRendering &&
            lastRenderedText == text && lastRenderedFontSize == messageFontSize && cachedAttributedString != nil {
            // Cache hit
            Text(cachedAttributedString!)
                .font(.system(size: messageFontSize))
                .foregroundStyle(isSystem ? .secondary : .primary)
        } else {
            if isStreaming || !requiresAdvancedRendering {
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

    @State private var showReasoningSheet = false

    private var isCurrentlyStreaming: Bool {
        viewModel.isGenerating && viewModel.streamingMessageID == message.id
    }

    // Parse reasoning content on-demand if not already parsed.
    // Callers should bind this once per body pass — it rescans the full message text,
    // which runs on every streamed frame otherwise.
    private var parsedContent: (reasoning: String?, finalAnswer: String?) {
        // If already parsed, return stored values
        if let reasoning = message.reasoning?.trimmingCharacters(in: .whitespacesAndNewlines),
           !reasoning.isEmpty {
            return (reasoning, message.finalAnswer)
        }
        // If not parsed but contains thinking tags, parse on-demand.
        // Case-insensitive range lookups avoid copying the whole text just to lowercase it.
        let text = message.text
        if text.range(of: "<thinking>", options: .caseInsensitive) != nil ||
            text.range(of: "<think>", options: .caseInsensitive) != nil {
            return parseReasoningFromText(text)
        }
        return (nil, nil)
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
        // Parse once per body pass; every access rescans the full message text.
        let parsed = parsedContent
        let reasoningText = parsed.reasoning
        let hasReasoningContent = !(reasoningText?.isEmpty ?? true)
        let hasCompletedReasoning = hasReasoningContent && (
            message.isFinal ||
            message.reasoningCompletedAt != nil ||
            message.streamingReasoningPhase == .finalAnswer
        )
        let reasoningStatusText: String? = {
            if hasCompletedReasoning {
                if let duration = message.reasoningDuration ?? message.generationDuration {
                    return "Thought for \(ThoughtDurationFormatter.string(for: duration))"
                }
                return "Thought"
            }
            return isCurrentlyStreaming ? "Thinking…" : nil
        }()

        HStack(alignment: .bottom, spacing: 8) {
            VStack(alignment: .leading, spacing: 8) {
                let rawFinal = parsed.finalAnswer ?? message.finalAnswer ?? message.displayText
                let split = rawFinal.extractSourcesBlocks()
                let finalText = split.visible
                let inlineSourcesText = split.sources

                if let reasoningStatusText {
                    InlineThinkingView(
                        text: reasoningStatusText,
                        onTap: hasReasoningContent ? { showReasoningSheet = true } : nil
                    )
                    .accessibilityIdentifier("message.thought")
                }

                // Final answer section — only shown when there is content to display
                let hasFinalSection = !finalText.isEmpty || message.generationError != nil ||
                    (message.searchInvocations?.isEmpty == false && message.isFinal) ||
                    (message.isFinal && inlineSourcesText != nil)
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

                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.large, style: .continuous)
                            .fill(Color.clear)
                            .background(
                                .ultraThinMaterial,
                                in: RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.large, style: .continuous)
                            )
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.large, style: .continuous)
                            .strokeBorder(Color.secondary.opacity(0.12))
                    )
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Spacer(minLength: 0)
        }
        .sheet(isPresented: $showReasoningSheet) {
            StepByStepReasoningSheet(
                reasoning: reasoningText ?? "",
                searchInvocations: message.searchInvocations
            )
        }
    }
}

// MARK: - Compact Sources button

struct SourcesButton: View {
    @Binding var showSources: Bool
    let invocations: [SearchInvocation]

    @ScaledMetric(relativeTo: .subheadline) private var faviconSize: CGFloat = 20

    private var previewSources: [WebSearchSource] {
        var seenDomains: Set<String> = []
        var sources: [WebSearchSource] = []

        for source in invocations.flatMap(\.displaySources) {
            guard seenDomains.insert(source.domainName).inserted else { continue }
            sources.append(source)
            if sources.count == 3 { break }
        }

        return sources
    }

    private var totalSourceCount: Int {
        invocations.reduce(0) { $0 + $1.sourceCount }
    }

    var body: some View {
        Button {
            showSources = true
        } label: {
            HStack(spacing: 8) {
                if !previewSources.isEmpty {
                    faviconStack
                }

                Text("Sources")
                    .font(.subheadline.weight(.medium))
            }
            .foregroundStyle(.primary)
            .padding(.vertical, 6)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Sources")
        .accessibilityValue("\(totalSourceCount) sources")
        .accessibilityHint("Shows the sources used for this response")
        .accessibilityIdentifier("message.sources")
    }

    private var faviconStack: some View {
        let overlapOffset = faviconSize * 0.62

        return ZStack(alignment: .leading) {
            ForEach(Array(previewSources.enumerated()), id: \.element.id) { index, source in
                SourceButtonFavicon(source: source, size: faviconSize)
                    .offset(x: CGFloat(index) * overlapOffset)
                    .zIndex(Double(previewSources.count - index))
            }
        }
        .frame(
            width: faviconSize + CGFloat(max(0, previewSources.count - 1)) * overlapOffset,
            height: faviconSize
        )
        .accessibilityHidden(true)
    }
}

private struct SourceButtonFavicon: View {
    let source: WebSearchSource
    let size: CGFloat

    var body: some View {
        Group {
            if let faviconURL = source.faviconURL.flatMap(URL.init(string:)) {
                AsyncImage(url: faviconURL) { image in
                    image.resizable().scaledToFit()
                } placeholder: {
                    fallback
                }
            } else {
                fallback
            }
        }
        .frame(width: size, height: size)
        .background(.background, in: Circle())
        .overlay {
            Circle()
                .strokeBorder(Color.secondary.opacity(0.25), lineWidth: 1)
        }
        .compositingGroup()
        .clipShape(Circle())
    }

    private var fallback: some View {
        Text(String(source.domainName.prefix(1)).uppercased())
            .font(.system(size: size * 0.46, weight: .bold))
            .foregroundStyle(.secondary)
    }
}

// MARK: - Structured web sources

struct SourcesSheetView: View {
    let invocations: [SearchInvocation]
    let title: String

    @State private var selectedInvocationID: UUID?
    @State private var activeURL: URL?
    @State private var selectedDetent: PresentationDetent = .large

    private var availableInvocations: [SearchInvocation] {
        invocations.filter { !$0.displaySources.isEmpty }
    }

    private var selectedInvocation: SearchInvocation? {
        availableInvocations.first(where: { $0.id == selectedInvocationID }) ?? availableInvocations.first
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if availableInvocations.count > 1 {
                        queryPicker
                    }

                    if let selectedInvocation {
                        queryCard(selectedInvocation)
                        SourceListView(sources: selectedInvocation.displaySources) { source in
                            activeURL = source.resolvedURL
                        }
                    } else {
                        ContentUnavailableView(
                            "No Sources",
                            systemImage: "doc.text.magnifyingglass",
                            description: Text("This search did not return a readable source.")
                        )
                    }
                }
                .padding()
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
        }
        .foregroundStyle(.primary)
        .presentationDetents([.medium, .large], selection: $selectedDetent)
        .presentationDragIndicator(.visible)
        .onAppear { selectDefaults() }
        .sheet(item: $activeURL) { url in
            SafariView(url: url)
                .ignoresSafeArea()
        }
    }

    private var queryPicker: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 8) {
                ForEach(Array(availableInvocations.enumerated()), id: \.element.id) { index, invocation in
                    Button {
                        selectedInvocationID = invocation.id
                    } label: {
                        HStack(spacing: 7) {
                            Image(systemName: "magnifyingglass")
                            Text(invocation.query)
                                .lineLimit(1)
                                .truncationMode(.tail)
                                .frame(maxWidth: 190, alignment: .leading)
                        }
                        .font(.subheadline.weight(.medium))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(
                            selectedInvocationID == invocation.id
                                ? Color.accentColor.opacity(0.16)
                                : Color.secondary.opacity(0.08),
                            in: Capsule()
                        )
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(invocation.query)
                    .accessibilityAddTraits(selectedInvocationID == invocation.id ? .isSelected : [])
                    .accessibilityIdentifier("sources.query.\(index)")
                }
            }
        }
        .scrollIndicators(.hidden)
    }

    private func queryCard(_ invocation: SearchInvocation) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Label("Search query", systemImage: "text.magnifyingglass")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(invocation.query)
                .font(.body.weight(.medium))
                .textSelection(.enabled)
            Text("\(invocation.sourceCount) source\(invocation.sourceCount == 1 ? "" : "s")")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.medium))
    }

    private func selectDefaults() {
        guard selectedInvocationID == nil else { return }
        selectedInvocationID = availableInvocations.first?.id
    }
}

private struct SourceListView: View {
    let sources: [WebSearchSource]
    let onOpen: (WebSearchSource) -> Void

    private var lastSourceID: UUID? { sources.last?.id }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Sources")
                .font(.headline)

            LazyVStack(spacing: 0) {
                ForEach(sources) { source in
                    VStack(spacing: 0) {
                        SourceListRow(source: source) {
                            onOpen(source)
                        }

                        if source.id != lastSourceID {
                            Divider()
                                .padding(.leading, 52)
                        }
                    }
                }
            }
            .background(
                Color.secondary.opacity(0.06),
                in: RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.large, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.large, style: .continuous)
                    .strokeBorder(Color.secondary.opacity(0.1))
            }
        }
    }
}

private struct SourceListRow: View {
    let source: WebSearchSource
    let onOpen: () -> Void

    var body: some View {
        Button(action: onOpen) {
            HStack(spacing: 12) {
                SourceIcon(source: source, size: 30)
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 3) {
                    Text(source.title)
                        .font(.body.weight(.medium))
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)

                    Text(source.domainName)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                Spacer(minLength: 8)

                Image(systemName: "arrow.up.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.tertiary)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity, minHeight: 64, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(source.resolvedURL == nil)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(source.title), \(source.domainName)")
        .accessibilityHint("Opens the website")
    }
}

private struct SourceIcon: View {
    let source: WebSearchSource
    let size: CGFloat

    var body: some View {
        Group {
            if let faviconURL = source.faviconURL.flatMap(URL.init(string:)) {
                AsyncImage(url: faviconURL) { image in
                    image.resizable().scaledToFit()
                } placeholder: {
                    fallback
                }
            } else {
                fallback
            }
        }
        .frame(width: size, height: size)
        .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: size * 0.25))
        .compositingGroup()
        .clipShape(RoundedRectangle(cornerRadius: size * 0.25))
    }

    private var fallback: some View {
        Text(String(source.domainName.prefix(1)).uppercased())
            .font(.system(size: size * 0.48, weight: .bold))
            .foregroundStyle(.secondary)
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
                        RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.large, style: .continuous)
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
                                RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.large, style: .continuous)
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
            deferMenuSafe { onCopy(message.userVisibleText) }
        } label: {
            Label(String(localized: "Copy"), systemImage: "doc.on.doc")
        }

        Button {
            deferMenuSafe { onShare(message.userVisibleText) }
        } label: {
            Label(String(localized: "Share"), systemImage: "square.and.arrow.up")
        }

        if message.role == .user {
            Button {
                deferMenuSafe { onEdit(message) }
            } label: {
                Label(String(localized: "Edit"), systemImage: "pencil.circle")
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
                onCopy(message.userVisibleText)
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
                onShare(message.userVisibleText)
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

// MARK: - Enhanced Assistant Actions Bar

struct AssistantActionsBar: View {
    let messageID: UUID
    var isGenerating: Bool
    var isStreamingForThisMessage: Bool
    var canAct: Bool
    var isFinal: Bool
    var sourceInvocations: [SearchInvocation]
    var onStop: () -> Void
    var onTryAgain: () -> Void
    var onConcise: () -> Void
    var onFormal: () -> Void
    var developerModeEnabled: Bool
    var onDeveloper: () -> Void

    // State for fade-in animation
    @State private var isVisible = false
    @State private var showSources = false

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

            if !sourceInvocations.isEmpty {
                SourcesButton(showSources: $showSources, invocations: sourceInvocations)
                    .sheet(isPresented: $showSources) {
                        SourcesSheetView(
                            invocations: sourceInvocations,
                            title: String(localized: "Sources")
                        )
                    }
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
            FullscreenAttachmentImageView(
                attachment: attachment,
                isPresented: $showFullscreen
            )
        }
    }

    private func loadImage() async {
        isLoading = true
        image = await DiskBackedImageLoader.loadThumbnail(
            at: attachment.actualFileURL,
            maxPixelSize: 720
        )
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
