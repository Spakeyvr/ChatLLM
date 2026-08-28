//
//  MarkdownBlocksView.swift
//  ChatLLM
//
//  Native SwiftUI rendering for parsed Markdown blocks.
//
//  This is the path almost every assistant message takes. It lays out with the
//  rest of the chat, so there is no height round-trip, no reload when the
//  transcript recycles a bubble off screen, and no per-token IPC while a model
//  is streaming. Only math falls back to the bundled KaTeX WebView.
//

import SwiftUI

struct MarkdownBlocksView: View {
    let blocks: [MarkdownBlock]
    let fontSize: Double
    var tone: RichMarkdownView.TextTone = .primary

    /// Plain prose hugs its text, the way a short chat bubble always has.
    /// Structured content — headings, lists, quotes, code, tables, rules — fills
    /// the bubble instead, matching how those blocks used to lay out.
    private var wantsFullWidth: Bool {
        blocks.contains { block in
            if case .paragraph = block { return false }
            return true
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { offset, block in
                MarkdownBlockView(block: block, fontSize: fontSize, tone: tone, depth: 0)
                    .padding(.top, offset == 0 ? 0 : topSpacing(before: block))
            }
        }
        .frame(maxWidth: wantsFullWidth ? .infinity : nil, alignment: .leading)
    }

    private func topSpacing(before block: MarkdownBlock) -> CGFloat {
        switch block {
        case .heading:
            return fontSize * 1.05
        case .thematicBreak:
            return fontSize * 0.9
        default:
            return fontSize * 0.75
        }
    }
}

// MARK: - Single block

private struct MarkdownBlockView: View {
    let block: MarkdownBlock
    let fontSize: Double
    let tone: RichMarkdownView.TextTone
    let depth: Int

    var body: some View {
        switch block {
        case let .heading(level, text):
            MarkdownInlineText(markdown: text, fontSize: fontSize * Self.headingScale(level), tone: tone)
                .fontWeight(.bold)
                .fixedSize(horizontal: false, vertical: true)

        case let .paragraph(text):
            MarkdownInlineText(markdown: text, fontSize: fontSize, tone: tone)
                .fixedSize(horizontal: false, vertical: true)

        case let .codeBlock(_, code):
            MarkdownCodeBlockView(code: code, fontSize: fontSize)

        case let .blockQuote(inner):
            MarkdownBlockQuoteView(blocks: inner, fontSize: fontSize, tone: tone, depth: depth)

        case let .list(list):
            MarkdownListView(list: list, fontSize: fontSize, tone: tone, depth: depth)

        case let .table(table):
            MarkdownTableView(table: table, fontSize: fontSize, tone: tone)

        case .thematicBreak:
            Rectangle()
                .fill(MarkdownPalette.separator)
                .frame(height: 1)
                .frame(maxWidth: .infinity)
        }
    }

    private static func headingScale(_ level: Int) -> Double {
        switch level {
        case 1: return 1.65
        case 2: return 1.45
        case 3: return 1.25
        case 4: return 1.12
        default: return 1.0
        }
    }
}

// MARK: - Code

private struct MarkdownCodeBlockView: View {
    let code: String
    let fontSize: Double

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Text(code)
                .font(.system(size: fontSize * 0.9, design: .monospaced))
                .foregroundStyle(.primary)
                .textSelection(.enabled)
                .padding(.horizontal, fontSize * 0.9)
                .padding(.vertical, fontSize * 0.8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: true, vertical: false)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(MarkdownPalette.codeBackground)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(MarkdownPalette.codeBorder, lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

// MARK: - Block quote

private struct MarkdownBlockQuoteView: View {
    let blocks: [MarkdownBlock]
    let fontSize: Double
    let tone: RichMarkdownView.TextTone
    let depth: Int

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            Rectangle()
                .fill(MarkdownPalette.quoteBorder)
                .frame(width: 3)

            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(blocks.enumerated()), id: \.offset) { offset, block in
                    MarkdownBlockView(block: block, fontSize: fontSize, tone: tone, depth: depth + 1)
                        .padding(.top, offset == 0 ? 0 : fontSize * 0.7)
                }
            }
            .padding(.horizontal, fontSize * 0.9)
            .padding(.vertical, fontSize * 0.75)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background {
            UnevenRoundedRectangle(
                topLeadingRadius: 0,
                bottomLeadingRadius: 0,
                bottomTrailingRadius: 10,
                topTrailingRadius: 10,
                style: .continuous
            )
            .fill(MarkdownPalette.quoteBackground)
        }
    }
}

// MARK: - Lists

private struct MarkdownListView: View {
    let list: MarkdownList
    let fontSize: Double
    let tone: RichMarkdownView.TextTone
    let depth: Int

    private var itemSpacing: CGFloat { list.isLoose ? fontSize * 0.7 : fontSize * 0.3 }

    var body: some View {
        VStack(alignment: .leading, spacing: itemSpacing) {
            ForEach(Array(list.items.enumerated()), id: \.offset) { offset, item in
                HStack(alignment: .firstTextBaseline, spacing: fontSize * 0.4) {
                    Text(marker(at: offset))
                        .font(.system(size: fontSize))
                        .foregroundStyle(tone == .secondary ? .secondary : .primary)
                        .monospacedDigit()
                        .frame(minWidth: markerWidth, alignment: .trailing)

                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(Array(item.blocks.enumerated()), id: \.offset) { innerOffset, block in
                            MarkdownBlockView(block: block, fontSize: fontSize, tone: tone, depth: depth + 1)
                                .padding(.top, innerOffset == 0 ? 0 : fontSize * 0.5)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .padding(.leading, depth == 0 ? 0 : fontSize * 0.5)
    }

    private var markerWidth: CGFloat {
        list.isOrdered ? fontSize * 1.35 : fontSize * 0.55
    }

    private func marker(at offset: Int) -> String {
        guard list.isOrdered else {
            switch depth % 3 {
            case 0: return "•"
            case 1: return "◦"
            default: return "▪"
            }
        }
        return "\(list.start + offset)."
    }
}

// MARK: - Tables

private struct MarkdownTableView: View {
    let table: MarkdownTable
    let fontSize: Double
    let tone: RichMarkdownView.TextTone

    private var cellFontSize: Double { fontSize * 0.96 }

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Grid(alignment: .topLeading, horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(Array(table.headers.enumerated()), id: \.offset) { offset, header in
                        cell(header, column: offset, isHeader: true)
                    }
                }
                .background(MarkdownPalette.codeBackground)

                ForEach(Array(table.rows.enumerated()), id: \.offset) { rowOffset, row in
                    GridRow {
                        ForEach(Array(row.enumerated()), id: \.offset) { offset, value in
                            cell(value, column: offset, isHeader: false)
                        }
                    }
                    .background(rowOffset.isMultiple(of: 2) ? Color.clear : MarkdownPalette.tableStripe)
                }
            }
            .overlay {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(MarkdownPalette.separator, lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .padding(1)
        }
    }

    @ViewBuilder
    private func cell(_ value: String, column: Int, isHeader: Bool) -> some View {
        let alignment = table.alignments.indices.contains(column) ? table.alignments[column] : .leading
        MarkdownInlineText(markdown: value, fontSize: cellFontSize, tone: tone)
            .fontWeight(isHeader ? .semibold : .regular)
            .multilineTextAlignment(textAlignment(alignment))
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, cellFontSize * 0.7)
            .padding(.vertical, cellFontSize * 0.55)
            .frame(minWidth: cellFontSize * 3, alignment: frameAlignment(alignment))
            .overlay(alignment: .trailing) {
                Rectangle().fill(MarkdownPalette.separator).frame(width: 1)
            }
            .overlay(alignment: .bottom) {
                Rectangle().fill(MarkdownPalette.separator).frame(height: 1)
            }
    }

    private func textAlignment(_ alignment: MarkdownTable.ColumnAlignment) -> TextAlignment {
        switch alignment {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }

    private func frameAlignment(_ alignment: MarkdownTable.ColumnAlignment) -> Alignment {
        switch alignment {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }
}

// MARK: - Inline text

/// Renders one block's inline Markdown. Images are reduced to their alt text and
/// non-web links are stripped before the string ever reaches SwiftUI, so model
/// output can never trigger a network load or open a custom scheme.
struct MarkdownInlineText: View {
    let markdown: String
    let fontSize: Double
    var tone: RichMarkdownView.TextTone = .primary

    var body: some View {
        Text(MarkdownInlineRenderer.attributed(markdown, fontSize: fontSize))
            .font(.system(size: fontSize))
            .foregroundStyle(tone == .secondary ? .secondary : .primary)
    }
}

enum MarkdownInlineRenderer {

    static func attributed(_ markdown: String, fontSize: Double) -> AttributedString {
        let source = sanitizedSource(markdown)
        guard !source.isEmpty else { return AttributedString() }
        guard var parsed = NativeMarkdownParser.attributedString(from: source) else {
            return AttributedString(source)
        }
        stripUnsafeLinks(&parsed)
        styleInlineCode(&parsed, fontSize: fontSize)
        return parsed
    }

    /// Removes image syntax and link reference definitions, which the inline-only
    /// parser would otherwise surface as literal text.
    static func sanitizedSource(_ markdown: String) -> String {
        var text = markdown
        if text.contains("![") {
            text = SharedRegexes.markdownInlineImage.stringByReplacingMatches(
                in: text,
                range: NSRange(text.startIndex..., in: text),
                withTemplate: "$1"
            )
            text = SharedRegexes.markdownReferenceImage.stringByReplacingMatches(
                in: text,
                range: NSRange(text.startIndex..., in: text),
                withTemplate: "$1"
            )
        }
        guard text.contains("]:") else { return text }
        let kept = text.components(separatedBy: "\n").filter { line in
            SharedRegexes.markdownLinkReferenceDefinition.firstMatch(
                in: line,
                range: NSRange(line.startIndex..., in: line)
            ) == nil
        }
        return kept.joined(separator: "\n")
    }

    private static func stripUnsafeLinks(_ text: inout AttributedString) {
        for run in text.runs where run.link != nil {
            let scheme = run.link?.scheme?.lowercased()
            if scheme != "http" && scheme != "https" {
                text[run.range].link = nil
            }
        }
    }

    private static func styleInlineCode(_ text: inout AttributedString, fontSize: Double) {
        for run in text.runs where run.inlinePresentationIntent?.contains(.code) == true {
            text[run.range].font = .system(size: fontSize * 0.92, design: .monospaced)
            text[run.range].foregroundColor = Color(uiColor: .secondaryLabel)
        }
    }
}

// MARK: - Palette

enum MarkdownPalette {
    static let codeBackground = Color(uiColor: .secondarySystemBackground)
    static let codeBorder = Color(uiColor: .separator).opacity(0.45)
    static let quoteBorder = Color(uiColor: .systemBlue).opacity(0.5)
    static let quoteBackground = Color(uiColor: .tertiarySystemBackground).opacity(0.7)
    static let separator = Color(uiColor: .separator).opacity(0.35)
    static let tableStripe = Color(uiColor: .secondarySystemBackground).opacity(0.45)
}
