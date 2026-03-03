//
//  ReasoningBubbleView.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI
import UIKit

// MARK: - Modern Chain of Thought bubble with structured steps

struct ThinkingBubbleView: View {
    let hasContent: Bool
    let reasoning: String?
    let reasoningSteps: [ReasoningStep]?
    let searchInvocations: [SearchInvocation]?
    @Binding var isExpanded: Bool
    let isGenerating: Bool

    @State private var animateThinking: Bool = false
    @State private var showCopiedConfirmation = false

    private var searchCount: Int {
        return searchInvocations?.count ?? 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header button
            Button(action: {
                if hasContent {
                    withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                        isExpanded.toggle()
                    }
                }
            }) {
                HStack(spacing: 12) {
                    ZStack {
                        if hasContent {
                            Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .transition(.opacity.combined(with: .scale))
                        } else {
                            HStack(spacing: 4) {
                                ForEach(0..<3) { index in
                                    Circle()
                                        .frame(width: 4, height: 4)
                                        .foregroundStyle(.blue)
                                        .opacity(animateThinking ? 0.3 : 1.0)
                                        .animation(
                                            .easeInOut(duration: 0.8)
                                            .repeatForever()
                                            .delay(Double(index) * 0.2),
                                            value: animateThinking
                                        )
                                }
                            }
                            .onAppear { animateThinking = true }
                        }
                    }
                    .frame(width: 20, height: 16)

                    Text("Thinking")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundStyle(.primary)

                    if hasContent && searchCount > 0 {
                        Text("• \(searchCount) search\(searchCount == 1 ? "" : "es")")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else if isGenerating && !hasContent {
                        Text("AI is reasoning...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    // Copy button when expanded
                    if hasContent && isExpanded {
                        Button {
                            copyAllSteps()
                        } label: {
                            Image(systemName: showCopiedConfirmation ? "checkmark" : "doc.on.doc")
                                .font(.caption)
                                .foregroundStyle(showCopiedConfirmation ? .green : .blue)
                        }
                        .buttonStyle(.plain)
                        .transition(.scale.combined(with: .opacity))
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            }
            .buttonStyle(.plain)
            .disabled(!hasContent)
            .background {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(.thinMaterial)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(.blue.opacity(0.3), lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .zIndex(1)

            // Expanded content
            if isExpanded && hasContent {
                VStack(alignment: .leading, spacing: 8) {
                    if let rawReasoning = reasoning, !rawReasoning.isEmpty {
                        let processed = LatexProcessor.process(rawReasoning)
                        VStack(alignment: .leading, spacing: 8) {
                            if let attributed = try? AttributedString(markdown: processed,
                                    options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
                                Text(attributed)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                                    .lineLimit(nil)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .multilineTextAlignment(.leading)
                            } else {
                                Text(processed)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                                    .lineLimit(nil)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .multilineTextAlignment(.leading)
                            }
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background { RoundedRectangle(cornerRadius: 12, style: .continuous).fill(.regularMaterial.opacity(0.8)) }
                        .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(.blue.opacity(0.15), lineWidth: 1) }
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    }

                    // Search cards after reasoning steps
                    if let invocations = searchInvocations, !invocations.isEmpty {
                        SearchInvocationsList(invocations: invocations) { _ in }
                    }
                }
                .padding(.top, 4)
                .transition(.asymmetric(
                    insertion: .move(edge: .top)
                        .combined(with: .opacity)
                        .combined(with: .scale(scale: 0.95, anchor: .top)),
                    removal: .move(edge: .top)
                        .combined(with: .opacity)
                        .combined(with: .scale(scale: 0.95, anchor: .top))
                ))
                .animation(.spring(response: 0.5, dampingFraction: 0.8), value: isExpanded)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func copyAllSteps() {
        UIPasteboard.general.string = reasoning ?? ""
        withAnimation { showCopiedConfirmation = true }
        Task {
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            withAnimation { showCopiedConfirmation = false }
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

                if let attributed = try? AttributedString(markdown: step.content, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
                    Text(attributed)
                        .font(.system(size: messageFontSize * 0.85))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(nil)
                        .fixedSize(horizontal: false, vertical: true)
                        .multilineTextAlignment(.leading)
                } else {
                    Text(step.content)
                        .font(.system(size: messageFontSize * 0.85))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(nil)
                        .fixedSize(horizontal: false, vertical: true)
                        .multilineTextAlignment(.leading)
                }
            }
            .padding(.vertical, 8)
            .padding(.trailing, 16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.leading, 16)
        .background {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(.regularMaterial.opacity(0.5))
        }
        .overlay {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(.blue.opacity(0.1), lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
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

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    // Search cards at the top — clearly separated from reasoning
                    if let invocations = searchInvocations, !invocations.isEmpty {
                        VStack(alignment: .leading, spacing: 10) {
                            Label {
                                Text("\(invocations.count) Web Search\(invocations.count == 1 ? "" : "es") Performed")
                                    .font(.subheadline)
                                    .fontWeight(.semibold)
                            } icon: {
                                Image(systemName: "globe.americas.fill")
                                    .foregroundStyle(.teal)
                            }

                            SearchInvocationsList(invocations: invocations) { invocation in
                                selectedInvocation = invocation
                            }
                        }
                        .padding(.bottom, 20)
                    }

                    if !reasoning.isEmpty {
                        let processed = LatexProcessor.process(reasoning)
                        VStack(alignment: .leading, spacing: 12) {
                            if let attributed = try? AttributedString(markdown: processed,
                                    options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
                                Text(attributed)
                                    .font(.body)
                                    .foregroundStyle(.primary)
                                    .textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            } else {
                                Text(processed)
                                    .font(.body)
                                    .foregroundStyle(.primary)
                                    .textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                        .padding()
                        .background {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
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
                sourcesText: invocation.results,
                title: String(localized: "Search Results"),
                searchQuery: invocation.query
            )
        }
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
                    if let attributed = try? AttributedString(markdown: step.content, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
                        Text(attributed)
                            .font(.system(size: messageFontSize))
                            .foregroundStyle(.primary)
                            .textSelection(.enabled)
                            .lineLimit(nil)
                            .fixedSize(horizontal: false, vertical: true)
                            .multilineTextAlignment(.leading)
                    } else {
                        Text(step.content)
                            .font(.system(size: messageFontSize))
                            .foregroundStyle(.primary)
                            .textSelection(.enabled)
                            .lineLimit(nil)
                            .fixedSize(horizontal: false, vertical: true)
                            .multilineTextAlignment(.leading)
                    }
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
        let lines = invocation.results.components(separatedBy: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        let count = lines.filter { $0.hasPrefix("Title:") || $0.hasPrefix("- ") }.count
        if count > 0 {
            return "\(count) result\(count == 1 ? "" : "s") found"
        }
        let preview = invocation.results.prefix(60)
        return preview.count < invocation.results.count ? "\(preview)…" : String(preview)
    }

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(.teal.opacity(0.25))
                        .frame(width: 36, height: 36)

                    Image(systemName: "globe.americas.fill")
                        .font(.system(size: 17))
                        .foregroundStyle(.teal)
                }

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 5) {
                        Text("Web Search")
                            .font(.caption2)
                            .fontWeight(.bold)
                            .foregroundStyle(.teal)
                            .textCase(.uppercase)
                            .tracking(0.4)

                        Spacer()

                        Text("View results")
                            .font(.caption2)
                            .foregroundStyle(.teal)

                        Image(systemName: "arrow.up.right")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(.teal)
                    }

                    Text(invocation.query)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)

                    Text(resultSnippet)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            .padding(12)
            .background {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(.teal.opacity(0.08))
            }
            .overlay {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(.teal.opacity(0.5), lineWidth: 1.5)
            }
        }
        .buttonStyle(.plain)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
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
