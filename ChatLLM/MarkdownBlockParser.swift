//
//  MarkdownBlockParser.swift
//  ChatLLM
//
//  A dependency-free, streaming-tolerant Markdown block parser.
//
//  The chat transcript re-parses an assistant message on every streamed token,
//  so this parser is written to be allocation-light and to degrade gracefully on
//  partial input: an unterminated fence, table, or list still produces the block
//  the author is halfway through writing rather than collapsing to a paragraph.
//
//  Inline spans (emphasis, links, code) are deliberately *not* handled here.
//  They stay in `AttributedString(markdown:)` so streamed prefixes keep the
//  whitespace behaviour the chat relies on.
//

import Foundation

// MARK: - Model

nonisolated enum MarkdownBlock: Equatable, Sendable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case codeBlock(language: String?, code: String)
    case blockQuote([MarkdownBlock])
    case list(MarkdownList)
    case table(MarkdownTable)
    case thematicBreak
}

nonisolated struct MarkdownList: Equatable, Sendable {
    var isOrdered: Bool
    var start: Int
    /// A list is "loose" when its items are separated by blank lines. CommonMark
    /// renders those items as paragraphs; we use it to pick vertical spacing.
    var isLoose: Bool
    var items: [MarkdownListItem]
}

nonisolated struct MarkdownListItem: Equatable, Sendable {
    var blocks: [MarkdownBlock]
}

nonisolated struct MarkdownTable: Equatable, Sendable {
    enum ColumnAlignment: Equatable {
        case leading
        case center
        case trailing
    }

    var headers: [String]
    var alignments: [ColumnAlignment]
    var rows: [[String]]
}

// MARK: - Parser

nonisolated enum MarkdownBlockParser {

    /// Maximum nesting depth for quotes and lists. Deeply indented model output
    /// would otherwise recurse once per level.
    private static let maximumDepth = 6

    static func parse(_ source: String) -> [MarkdownBlock] {
        guard !source.isEmpty else { return [] }
        let normalized = source
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        var lines = normalized.components(separatedBy: "\n")
        // `components` yields a trailing empty element for text ending in a
        // newline. Dropping it keeps a streamed "para\n" from opening a block.
        if lines.last?.isEmpty == true { lines.removeLast() }
        var index = 0
        return parseBlocks(lines, &index, depth: 0)
    }

    // MARK: Block loop

    private static func parseBlocks(_ lines: [String], _ index: inout Int, depth: Int) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []

        while index < lines.count {
            let line = lines[index]

            if isBlank(line) {
                index += 1
                continue
            }

            if let fence = fenceOpening(line) {
                blocks.append(parseFencedCode(lines, &index, fence: fence))
                continue
            }

            if let heading = atxHeading(line) {
                blocks.append(.heading(level: heading.level, text: heading.text))
                index += 1
                continue
            }

            if isThematicBreak(line) {
                blocks.append(.thematicBreak)
                index += 1
                continue
            }

            if depth < maximumDepth, blockQuoteContent(line) != nil {
                blocks.append(parseBlockQuote(lines, &index, depth: depth))
                continue
            }

            if depth < maximumDepth, let marker = listMarker(line) {
                blocks.append(parseList(lines, &index, firstMarker: marker, depth: depth))
                continue
            }

            if let table = parseTable(lines, &index) {
                blocks.append(.table(table))
                continue
            }

            if indentWidth(line) >= 4 {
                blocks.append(parseIndentedCode(lines, &index))
                continue
            }

            blocks.append(contentsOf: parseParagraph(lines, &index, depth: depth))
        }

        return blocks
    }

    // MARK: Paragraphs & setext headings

    /// Returns one block in the common case, or a heading plus a trailing
    /// paragraph when a setext underline splits the run.
    private static func parseParagraph(_ lines: [String], _ index: inout Int, depth: Int) -> [MarkdownBlock] {
        var collected: [String] = []

        while index < lines.count {
            let line = lines[index]

            if isBlank(line) { break }

            // A setext underline only applies when it follows paragraph text.
            if !collected.isEmpty, let level = setextUnderlineLevel(line) {
                index += 1
                let text = collected.joined(separator: "\n").trimmingCharacters(in: .whitespaces)
                return [.heading(level: level, text: text)]
            }

            if !collected.isEmpty, interruptsParagraph(line, depth: depth) { break }

            collected.append(line.trimmingTrailingWhitespace())
            index += 1
        }

        guard !collected.isEmpty else { return [] }
        return [.paragraph(collected.joined(separator: "\n"))]
    }

    /// CommonMark lets some constructs interrupt a paragraph without a blank
    /// line. Bullet lists may; ordered lists may only when they start at 1.
    private static func interruptsParagraph(_ line: String, depth: Int) -> Bool {
        if fenceOpening(line) != nil { return true }
        if atxHeading(line) != nil { return true }
        if isThematicBreak(line) { return true }
        if depth < maximumDepth, blockQuoteContent(line) != nil { return true }
        if depth < maximumDepth, let marker = listMarker(line) {
            if !marker.isOrdered { return !marker.content.isEmpty }
            return marker.number == 1
        }
        return false
    }

    // MARK: Fenced code

    private struct Fence {
        let character: Character
        let length: Int
        let indent: Int
        let language: String?
    }

    private static func fenceOpening(_ line: String) -> Fence? {
        let indent = indentWidth(line)
        guard indent < 4 else { return nil }
        let trimmed = line.drop(while: { $0 == " " || $0 == "\t" })
        guard let first = trimmed.first, first == "`" || first == "~" else { return nil }
        let run = trimmed.prefix(while: { $0 == first })
        guard run.count >= 3 else { return nil }
        let info = trimmed.dropFirst(run.count).trimmingCharacters(in: .whitespaces)
        // A backtick fence's info string may not itself contain a backtick.
        if first == "`", info.contains("`") { return nil }
        let language = info.isEmpty ? nil : String(info.prefix(while: { !$0.isWhitespace }))
        return Fence(character: first, length: run.count, indent: indent, language: language)
    }

    private static func isFenceClosing(_ line: String, fence: Fence) -> Bool {
        guard indentWidth(line) < 4 else { return false }
        let trimmed = line.drop(while: { $0 == " " || $0 == "\t" })
        guard let first = trimmed.first, first == fence.character else { return false }
        let run = trimmed.prefix(while: { $0 == first })
        guard run.count >= fence.length else { return false }
        return trimmed.dropFirst(run.count).allSatisfy { $0 == " " || $0 == "\t" }
    }

    private static func parseFencedCode(_ lines: [String], _ index: inout Int, fence: Fence) -> MarkdownBlock {
        index += 1 // consume the opening fence
        var body: [String] = []

        while index < lines.count {
            let line = lines[index]
            if isFenceClosing(line, fence: fence) {
                index += 1
                return .codeBlock(language: fence.language, code: body.joined(separator: "\n"))
            }
            // The opening fence's indentation is stripped from each content line.
            body.append(line.removingLeadingSpaces(upTo: fence.indent))
            index += 1
        }

        // Unterminated fence — the model is still streaming the block.
        return .codeBlock(language: fence.language, code: body.joined(separator: "\n"))
    }

    private static func parseIndentedCode(_ lines: [String], _ index: inout Int) -> MarkdownBlock {
        var body: [String] = []
        var pendingBlanks: [String] = []

        while index < lines.count {
            let line = lines[index]
            if isBlank(line) {
                pendingBlanks.append("")
                index += 1
                continue
            }
            guard indentWidth(line) >= 4 else { break }
            body.append(contentsOf: pendingBlanks)
            pendingBlanks.removeAll()
            body.append(line.removingLeadingSpaces(upTo: 4))
            index += 1
        }

        return .codeBlock(language: nil, code: body.joined(separator: "\n"))
    }

    // MARK: Block quotes

    private static func blockQuoteContent(_ line: String) -> String? {
        guard indentWidth(line) < 4 else { return nil }
        let trimmed = line.drop(while: { $0 == " " || $0 == "\t" })
        guard trimmed.first == ">" else { return nil }
        var rest = trimmed.dropFirst()
        if rest.first == " " { rest = rest.dropFirst() }
        return String(rest)
    }

    private static func parseBlockQuote(_ lines: [String], _ index: inout Int, depth: Int) -> MarkdownBlock {
        var inner: [String] = []

        while index < lines.count {
            let line = lines[index]
            if let content = blockQuoteContent(line) {
                inner.append(content)
                index += 1
                continue
            }
            // Lazy continuation: a plain line extends the quote's paragraph.
            if !isBlank(line), !inner.isEmpty, !isBlank(inner[inner.count - 1]),
               !interruptsParagraph(line, depth: depth) {
                inner.append(line)
                index += 1
                continue
            }
            break
        }

        var innerIndex = 0
        return .blockQuote(parseBlocks(inner, &innerIndex, depth: depth + 1))
    }

    // MARK: Lists

    private struct ListMarker {
        let isOrdered: Bool
        let number: Int
        let indent: Int
        /// Column at which the item's content starts, used to dedent continuations.
        let contentIndent: Int
        let content: String
    }

    private static func listMarker(_ line: String) -> ListMarker? {
        let indent = indentWidth(line)
        guard indent < 4 else { return nil }
        var cursor = line.startIndex
        var column = 0
        while cursor < line.endIndex, line[cursor] == " " || line[cursor] == "\t" {
            column += line[cursor] == "\t" ? 4 : 1
            cursor = line.index(after: cursor)
        }
        guard cursor < line.endIndex else { return nil }

        var isOrdered = false
        var number = 1
        var markerWidth = 0

        if line[cursor] == "-" || line[cursor] == "*" || line[cursor] == "+" {
            markerWidth = 1
            cursor = line.index(after: cursor)
        } else if line[cursor].isNumber {
            var digits = ""
            var probe = cursor
            while probe < line.endIndex, line[probe].isNumber, digits.count < 9 {
                digits.append(line[probe])
                probe = line.index(after: probe)
            }
            guard probe < line.endIndex, line[probe] == "." || line[probe] == ")" else { return nil }
            isOrdered = true
            number = Int(digits) ?? 1
            markerWidth = digits.count + 1
            cursor = line.index(after: probe)
        } else {
            return nil
        }

        // The marker must be followed by whitespace, or end the line.
        var spaces = 0
        while cursor < line.endIndex, line[cursor] == " " || line[cursor] == "\t" {
            spaces += line[cursor] == "\t" ? 4 : 1
            cursor = line.index(after: cursor)
        }
        if cursor < line.endIndex, spaces == 0 { return nil }

        // An empty item, or one padded far past its marker, indents by one space.
        let effectiveSpaces = (spaces == 0 || spaces > 4) ? 1 : spaces
        return ListMarker(
            isOrdered: isOrdered,
            number: number,
            indent: column,
            contentIndent: column + markerWidth + effectiveSpaces,
            content: String(line[cursor...])
        )
    }

    private static func parseList(
        _ lines: [String],
        _ index: inout Int,
        firstMarker: ListMarker,
        depth: Int
    ) -> MarkdownBlock {
        var items: [MarkdownListItem] = []
        var isLoose = false
        var sawTrailingBlank = false

        while index < lines.count {
            let line = lines[index]

            if isBlank(line) {
                // A blank line ends the list unless another item follows.
                sawTrailingBlank = true
                index += 1
                continue
            }

            // Anything that is not a sibling marker ends the list. Content that
            // belongs to the preceding item was already taken by
            // `collectItemLines`, so there is nothing left to absorb.
            guard let marker = listMarker(line),
                  marker.isOrdered == firstMarker.isOrdered,
                  marker.indent < firstMarker.contentIndent else { break }

            if sawTrailingBlank, !items.isEmpty { isLoose = true }
            sawTrailingBlank = false
            index += 1

            let itemLines = collectItemLines(
                lines,
                &index,
                contentIndent: marker.contentIndent,
                firstLine: marker.content
            )

            var innerIndex = 0
            let blocks = parseBlocks(itemLines, &innerIndex, depth: depth + 1)
            items.append(MarkdownListItem(blocks: blocks))
        }

        return .list(MarkdownList(
            isOrdered: firstMarker.isOrdered,
            start: firstMarker.number,
            isLoose: isLoose,
            items: items
        ))
    }

    /// Consumes the lines belonging to the item whose content starts at
    /// `contentIndent`, dedenting them so they can be parsed on their own.
    /// `firstLine` is the text that followed the marker; it is included so a
    /// wrapped, unindented continuation line still attaches to this item.
    private static func collectItemLines(
        _ lines: [String],
        _ index: inout Int,
        contentIndent: Int,
        firstLine: String
    ) -> [String] {
        var collected: [String] = [firstLine]
        var pendingBlanks = 0

        while index < lines.count {
            let line = lines[index]

            if isBlank(line) {
                pendingBlanks += 1
                index += 1
                continue
            }

            if indentWidth(line) >= contentIndent {
                for _ in 0..<pendingBlanks { collected.append("") }
                pendingBlanks = 0
                collected.append(line.removingLeadingSpaces(upTo: contentIndent))
                index += 1
                continue
            }

            // Lazy continuation of the item's paragraph: only when no blank line
            // intervened and the line does not start a new block of its own.
            if pendingBlanks == 0, let previous = collected.last, !isBlank(previous),
               listMarker(line) == nil, blockQuoteContent(line) == nil,
               fenceOpening(line) == nil, atxHeading(line) == nil, !isThematicBreak(line) {
                collected.append(line.trimmingTrailingWhitespace())
                index += 1
                continue
            }

            break
        }

        // Blank lines that trailed the item belong to the list, not the item.
        index -= pendingBlanks
        return collected
    }

    // MARK: Tables

    private static func parseTable(_ lines: [String], _ index: inout Int) -> MarkdownTable? {
        guard index + 1 < lines.count else { return nil }
        let headerLine = lines[index]
        guard headerLine.contains("|") else { return nil }
        guard let alignments = tableDelimiterAlignments(lines[index + 1]) else { return nil }

        let headers = splitTableRow(headerLine)
        guard headers.count == alignments.count, !headers.isEmpty else { return nil }

        index += 2
        var rows: [[String]] = []

        while index < lines.count {
            let line = lines[index]
            if isBlank(line) || !line.contains("|") { break }
            if tableDelimiterAlignments(line) != nil { break }
            var cells = splitTableRow(line)
            // Ragged rows are padded or truncated to the header's width.
            if cells.count < headers.count {
                cells.append(contentsOf: Array(repeating: "", count: headers.count - cells.count))
            } else if cells.count > headers.count {
                cells = Array(cells.prefix(headers.count))
            }
            rows.append(cells)
            index += 1
        }

        return MarkdownTable(headers: headers, alignments: alignments, rows: rows)
    }

    private static func tableDelimiterAlignments(_ line: String) -> [MarkdownTable.ColumnAlignment]? {
        guard line.contains("-") else { return nil }
        let cells = splitTableRow(line)
        guard !cells.isEmpty else { return nil }

        var alignments: [MarkdownTable.ColumnAlignment] = []
        for cell in cells {
            let trimmed = cell.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { return nil }
            let leading = trimmed.hasPrefix(":")
            let trailing = trimmed.hasSuffix(":")
            let dashes = trimmed.dropFirst(leading ? 1 : 0).dropLast(trailing && trimmed.count > 1 ? 1 : 0)
            guard !dashes.isEmpty, dashes.allSatisfy({ $0 == "-" }) else { return nil }
            switch (leading, trailing) {
            case (true, true): alignments.append(.center)
            case (false, true): alignments.append(.trailing)
            default: alignments.append(.leading)
            }
        }
        return alignments
    }

    /// Splits a pipe row, honouring `\|` escapes and dropping the optional
    /// leading and trailing pipes.
    private static func splitTableRow(_ line: String) -> [String] {
        var cells: [String] = []
        var current = ""
        var isEscaped = false

        for character in line.trimmingCharacters(in: .whitespaces) {
            if isEscaped {
                // Keep the pipe, drop the backslash that protected it.
                if character != "|" { current.append("\\") }
                current.append(character)
                isEscaped = false
                continue
            }
            if character == "\\" {
                isEscaped = true
                continue
            }
            if character == "|" {
                cells.append(current)
                current = ""
                continue
            }
            current.append(character)
        }
        if isEscaped { current.append("\\") }
        cells.append(current)

        if cells.first?.trimmingCharacters(in: .whitespaces).isEmpty == true { cells.removeFirst() }
        if cells.last?.trimmingCharacters(in: .whitespaces).isEmpty == true { cells.removeLast() }
        return cells.map { $0.trimmingCharacters(in: .whitespaces) }
    }

    // MARK: Line classification

    private static func atxHeading(_ line: String) -> (level: Int, text: String)? {
        guard indentWidth(line) < 4 else { return nil }
        let trimmed = line.drop(while: { $0 == " " || $0 == "\t" })
        let hashes = trimmed.prefix(while: { $0 == "#" })
        guard (1...6).contains(hashes.count) else { return nil }
        let rest = trimmed.dropFirst(hashes.count)
        guard rest.isEmpty || rest.first == " " || rest.first == "\t" else { return nil }
        // A closing sequence of hashes is decorative and is stripped.
        var text = rest.trimmingCharacters(in: .whitespaces)
        while text.hasSuffix("#") { text.removeLast() }
        return (hashes.count, text.trimmingCharacters(in: .whitespaces))
    }

    private static func isThematicBreak(_ line: String) -> Bool {
        guard indentWidth(line) < 4 else { return false }
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard let first = trimmed.first, first == "-" || first == "*" || first == "_" else { return false }
        var count = 0
        for character in trimmed {
            if character == first { count += 1 } else if character != " " && character != "\t" { return false }
        }
        return count >= 3
    }

    private static func setextUnderlineLevel(_ line: String) -> Int? {
        guard indentWidth(line) < 4 else { return nil }
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard let first = trimmed.first, first == "=" || first == "-" else { return nil }
        guard trimmed.allSatisfy({ $0 == first }) else { return nil }
        return first == "=" ? 1 : 2
    }

    private static func isBlank(_ line: String) -> Bool {
        line.allSatisfy { $0 == " " || $0 == "\t" }
    }

    private static func indentWidth(_ line: String) -> Int {
        var width = 0
        for character in line {
            if character == " " { width += 1 } else if character == "\t" { width += 4 } else { break }
        }
        return width
    }
}

// MARK: - String helpers

private extension String {
    nonisolated func removingLeadingSpaces(upTo limit: Int) -> String {
        var removed = 0
        var cursor = startIndex
        while cursor < endIndex, removed < limit {
            if self[cursor] == " " {
                removed += 1
            } else if self[cursor] == "\t" {
                removed += 4
            } else {
                break
            }
            cursor = index(after: cursor)
        }
        return String(self[cursor...])
    }

    nonisolated func trimmingTrailingWhitespace() -> String {
        var copy = self
        while let last = copy.last, last == " " || last == "\t" { copy.removeLast() }
        return copy
    }
}
