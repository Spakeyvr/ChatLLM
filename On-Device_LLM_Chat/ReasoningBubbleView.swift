//
//  ReasoningBubbleView.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI
import Combine

// MARK: - Inline thinking indicator shown during streaming

struct InlineThinkingView: View {
    let onTap: (() -> Void)?
    @State private var phase: Int = 0
    @State private var shimmerOffset: CGFloat = -0.4

    private let ticker = Timer.publish(every: 0.4, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 6) {
            HStack(spacing: 3) {
            }
            .onReceive(ticker) { _ in
                phase = (phase + 1) % 3
            }

            Text("Thinking…")
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
                        Text("Thinking…")
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
        .contentShape(Rectangle())
        .onTapGesture { onTap?() }
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
