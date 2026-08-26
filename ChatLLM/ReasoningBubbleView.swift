//
//  ReasoningBubbleView.swift
//  ChatLLM
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI

// MARK: - Inline thinking indicator shown during streaming

struct InlineThinkingView: View {
    var text: String = "Thinking…"
    let onTap: (() -> Void)?
    @State private var shimmerOffset: CGFloat = -0.4

    var body: some View {
        Group {
            if let onTap {
                Button(action: onTap) {
                    thinkingLabel
                }
                .buttonStyle(.plain)
            } else {
                thinkingLabel
            }
        }
    }

    private var thinkingLabel: some View {
        Text(text)
            .font(.subheadline)
            .foregroundStyle(.gray)
            .overlay {
                LinearGradient(
                    stops: [
                        .init(color: .clear, location: 0),
                        .init(color: .white.opacity(0.85), location: 0.5),
                        .init(color: .clear, location: 1),
                    ],
                    startPoint: UnitPoint(x: shimmerOffset - 0.25, y: 0.5),
                    endPoint: UnitPoint(x: shimmerOffset + 0.25, y: 0.5)
                )
                .mask {
                    Text(text)
                        .font(.subheadline)
                }
                .blendMode(.plusLighter)
            }
            .onAppear {
                shimmerOffset = -0.4
                withAnimation(.linear(duration: 1.6).repeatForever(autoreverses: false)) {
                    shimmerOffset = 1.4
                }
            }
    }
}

// MARK: - Individual Chain of Thought step view

struct CoTStepView: View {
    let step: ReasoningStep
    let isLast: Bool

    @AppStorage("messageFontSize") private var messageFontSize: Double = 16.0

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Step indicator
            VStack(spacing: 4) {
                ZStack {
                    Circle()
                        .fill(.blue.opacity(0.15))
                        .frame(width: 28, height: 28)

                    Text("\(step.stepNumber)")
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundStyle(.blue)
                }

                if !isLast {
                    Rectangle()
                        .fill(.blue.opacity(0.2))
                        .frame(width: 2)
                        .frame(maxHeight: .infinity)
                }
            }
            .frame(width: 28)

            // Step content
            VStack(alignment: .leading, spacing: 6) {
                if let title = step.title, !title.isEmpty {
                    Text(title)
                        .font(.system(size: messageFontSize * 0.9))
                        .fontWeight(.semibold)
                        .foregroundStyle(.primary)
                }

                RichMarkdownView(
                    text: step.content,
                    fontSize: messageFontSize * 0.85,
                    textTone: .secondary
                )
                .textSelection(.enabled)
                .lineLimit(nil)
                .fixedSize(horizontal: false, vertical: true)
                .multilineTextAlignment(.leading)
            }
            .padding(.vertical, 8)
            .padding(.trailing, 16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.leading, 16)
        .background {
            RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.medium, style: .continuous)
                .fill(.regularMaterial.opacity(0.5))
        }
        .overlay {
            RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.medium, style: .continuous)
                .stroke(.blue.opacity(0.1), lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.medium, style: .continuous))
    }
}

// MARK: - Step-by-Step Reasoning Sheet

struct StepByStepReasoningSheet: View {
    let reasoning: String
    let reasoningSteps: [ReasoningStep]?
    let searchInvocations: [SearchInvocation]?
    @Environment(\.dismiss) private var dismiss
    @AppStorage("messageFontSize") private var messageFontSize: Double = 16.0
    @State private var selectedInvocation: SearchInvocation?

    private enum TimelineItem: Identifiable {
        case reasoningChunk(Int, String)
        case search(SearchInvocation)

        var id: String {
            switch self {
            case .reasoningChunk(let index, _):
                return "reasoning-\(index)"
            case .search(let invocation):
                return "search-\(invocation.id.uuidString)"
            }
        }
    }

    private var timelineItems: [TimelineItem] {
        let chunks = reasoningChunks
        let searches = (searchInvocations ?? []).sorted { lhs, rhs in
            if lhs.anchorStepNumber == rhs.anchorStepNumber {
                return lhs.timestamp < rhs.timestamp
            }
            return (lhs.anchorStepNumber ?? 0) < (rhs.anchorStepNumber ?? 0)
        }

        guard !chunks.isEmpty else {
            return searches.map(TimelineItem.search)
        }

        var items: [TimelineItem] = []

        let leadingSearches = searches.filter { ($0.anchorStepNumber ?? 0) <= 0 }
        items.append(contentsOf: leadingSearches.map(TimelineItem.search))

        for (index, chunk) in chunks.enumerated() {
            items.append(.reasoningChunk(chunk.index, chunk.text))
            let anchoredSearches = searches.filter { $0.anchorStepNumber == index + 1 }
            items.append(contentsOf: anchoredSearches.map(TimelineItem.search))
        }

        let trailingSearches = searches.filter { ($0.anchorStepNumber ?? 0) > chunks.count }
        items.append(contentsOf: trailingSearches.map(TimelineItem.search))

        return items
    }

    private var reasoningChunks: [(index: Int, text: String)] {
        let trimmed = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let chunks = trimmed
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        if chunks.isEmpty {
            return [(0, trimmed)]
        }

        return Array(chunks.enumerated()).map { ($0.offset, $0.element) }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    if !timelineItems.isEmpty {
                        VStack(alignment: .leading, spacing: 14) {
                            ForEach(Array(timelineItems.enumerated()), id: \.element.id) { index, item in
                                switch item {
                                case .reasoningChunk(_, let text):
                                    RawReasoningChunkView(
                                        text: text,
                                        isLast: index == timelineItems.count - 1
                                    )
                                case .search(let invocation):
                                    SearchStepCard(invocation: invocation) {
                                        selectedInvocation = invocation
                                    }
                                    .padding(.bottom, index == timelineItems.count - 1 ? 0 : 20)
                                }
                            }
                        }
                    } else if !reasoning.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            RichMarkdownView(
                                text: reasoning,
                                fontSize: UIFont.preferredFont(forTextStyle: .body).pointSize,
                                forceAdvancedRenderer: true
                            )
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .padding()
                        .background {
                            RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.medium, style: .continuous)
                                .fill(.regularMaterial)
                        }
                    } else {
                        Text("No reasoning content available.")
                            .font(.body)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding()
                    }
                }
                .padding()
            }
            .navigationTitle("Step-by-Step Reasoning")
            .navigationBarTitleDisplayMode(.inline)
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
        .sheet(item: $selectedInvocation) { invocation in
            SourcesSheetView(
                sourcesText: invocation.userVisibleResults,
                title: String(localized: "Search Results"),
                searchQuery: invocation.query
            )
        }
    }
}

struct RawReasoningChunkView: View {
    let text: String
    let isLast: Bool
    @AppStorage("messageFontSize") private var messageFontSize: Double = 16.0

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            RichMarkdownView(
                text: text,
                fontSize: messageFontSize,
                forceAdvancedRenderer: true
            )
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.bottom, isLast ? 0 : 20)
    }
}

// MARK: - Individual step card for the sheet view

struct StepCard: View {
    let step: ReasoningStep
    let isLast: Bool
    let isFirst: Bool
    @AppStorage("messageFontSize") private var messageFontSize: Double = 16.0

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Left side indicator - overlay approach for perfect alignment
            ZStack(alignment: .top) {
                // The continuous connecting line
                VStack(spacing: 0) {
                    // Top segment (above dot center)
                    if isFirst {
                        Color.clear
                            .frame(width: 1, height: 10)
                    } else {
                        Rectangle()
                            .fill(Color.gray.opacity(0.3))
                            .frame(width: 1, height: 10)
                    }

                    // Bottom segment (below dot center)
                    if isLast {
                        Color.clear
                            .frame(width: 1)
                    } else {
                        Rectangle()
                            .fill(Color.gray.opacity(0.3))
                            .frame(width: 1)
                    }
                }

                // The dot
                Circle()
                    .fill(Color.gray)
                    .frame(width: 8, height: 8)
                    .padding(.top, 6)
            }
            .frame(width: 8)

            // Content
            HStack(alignment: .top, spacing: 8) {
                Text("\(step.stepNumber).")
                    .font(.system(size: messageFontSize))
                    .fontWeight(.medium)
                    .foregroundStyle(.primary)
                    .fixedSize()

                VStack(alignment: .leading, spacing: 0) {
                    RichMarkdownView(text: step.content, fontSize: messageFontSize)
                        .textSelection(.enabled)
                        .lineLimit(nil)
                        .fixedSize(horizontal: false, vertical: true)
                        .multilineTextAlignment(.leading)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.bottom, isLast ? 0 : 20)
        }
    }
}

// MARK: - Search step card for reasoning flow (display only — parent owns the sheet)

struct SearchStepCard: View {
    let invocation: SearchInvocation
    let onTap: () -> Void

    private var resultSnippet: String {
        let visibleResults = invocation.userVisibleResults
        let lines = visibleResults.components(separatedBy: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        let count = lines.filter {
            $0.hasPrefix("Title:") ||
            $0.hasPrefix("- ") ||
            $0.range(of: #"^\d+\.\s"#, options: .regularExpression) != nil
        }.count
        if count > 0 {
            return "\(count) result\(count == 1 ? "" : "s") found"
        }
        let preview = visibleResults.prefix(60)
        return preview.count < visibleResults.count ? "\(preview)…" : String(preview)
    }

    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(width: 18, height: 18)
                    .padding(.top, 2)

                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("Searched the web")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundStyle(.secondary)

                        Spacer(minLength: 0)

                        Text("Open")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }

                    Text(invocation.query)
                        .font(.subheadline)
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)

                    Text(resultSnippet)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background {
                RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.medium, style: .continuous)
                    .fill(.thinMaterial)
            }
            .overlay {
                RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.medium, style: .continuous)
                    .stroke(Color.secondary.opacity(0.12), lineWidth: 1)
            }
        }
        .buttonStyle(.plain)
        .clipShape(RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.medium, style: .continuous))
        .contentShape(Rectangle())
    }
}

// MARK: - Container for a list of search cards (sheet ownership delegated to caller)

struct SearchInvocationsList: View {
    let invocations: [SearchInvocation]
    let onSelect: (SearchInvocation) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(invocations) { invocation in
                SearchStepCard(invocation: invocation) {
                    onSelect(invocation)
                }
            }
        }
    }
}
