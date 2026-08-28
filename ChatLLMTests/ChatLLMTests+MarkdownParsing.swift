//
//  ChatLLMTests+MarkdownParsing
//  ChatLLMTests
//
//  Split out of ChatLLMTests.swift; part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
import MLXLMCommon
import SwiftUI
import SwiftData
import WebKit
import UIKit
import FoundationModels
@testable import ChatLLM

extension ChatLLMTests {
    @Test func parserReadsHeadingsAndParagraphs() {
        #expect(MarkdownBlockParser.parse("## Heading") == [.heading(level: 2, text: "Heading")])
        #expect(MarkdownBlockParser.parse("### Heading ###") == [.heading(level: 3, text: "Heading")])
        #expect(MarkdownBlockParser.parse("####### Too deep") == [.paragraph("####### Too deep")])
        #expect(MarkdownBlockParser.parse("#NoSpace") == [.paragraph("#NoSpace")])
        #expect(MarkdownBlockParser.parse("Heading\n===") == [.heading(level: 1, text: "Heading")])
        #expect(MarkdownBlockParser.parse("Heading\n---") == [.heading(level: 2, text: "Heading")])
    }

    @Test func parserPreservesSoftLineBreaksInsideParagraphs() {
        // The chat renders a single newline as a line break, matching how the
        // bundled document was configured. Paragraph splits stay separate blocks.
        #expect(MarkdownBlockParser.parse("First line.\nNext line.") == [.paragraph("First line.\nNext line.")])
        #expect(MarkdownBlockParser.parse("First.\n\nSecond.") == [.paragraph("First."), .paragraph("Second.")])
    }

    @Test func parserReadsFencedCodeIncludingUnterminatedStreams() {
        #expect(MarkdownBlockParser.parse("```swift\nlet first_name = 1\n```")
                == [.codeBlock(language: "swift", code: "let first_name = 1")])
        #expect(MarkdownBlockParser.parse("```\nplain\n```")
                == [.codeBlock(language: nil, code: "plain")])
        // A half-streamed fence must already render as code, not as a paragraph.
        #expect(MarkdownBlockParser.parse("```swift\nlet first_name = 1")
                == [.codeBlock(language: "swift", code: "let first_name = 1")])
        #expect(MarkdownBlockParser.parse("    indented = 1")
                == [.codeBlock(language: nil, code: "indented = 1")])
        // Blank lines inside a fence are content, not block separators.
        #expect(MarkdownBlockParser.parse("```\na\n\nb\n```")
                == [.codeBlock(language: nil, code: "a\n\nb")])
    }

    @Test func parserReadsListsWithNestingAndLooseness() {
        let tight = MarkdownBlockParser.parse("- First\n- Second")
        #expect(tight == [.list(MarkdownList(isOrdered: false, start: 1, isLoose: false, items: [
            MarkdownListItem(blocks: [.paragraph("First")]),
            MarkdownListItem(blocks: [.paragraph("Second")])
        ]))])

        let loose = MarkdownBlockParser.parse("- First\n\n- Second")
        #expect(loose == [.list(MarkdownList(isOrdered: false, start: 1, isLoose: true, items: [
            MarkdownListItem(blocks: [.paragraph("First")]),
            MarkdownListItem(blocks: [.paragraph("Second")])
        ]))])

        let ordered = MarkdownBlockParser.parse("3. Third\n4. Fourth")
        #expect(ordered == [.list(MarkdownList(isOrdered: true, start: 3, isLoose: false, items: [
            MarkdownListItem(blocks: [.paragraph("Third")]),
            MarkdownListItem(blocks: [.paragraph("Fourth")])
        ]))])

        let nested = MarkdownBlockParser.parse("- First\n    - Nested\n- Second")
        #expect(nested == [.list(MarkdownList(isOrdered: false, start: 1, isLoose: false, items: [
            MarkdownListItem(blocks: [
                .paragraph("First"),
                .list(MarkdownList(isOrdered: false, start: 1, isLoose: false, items: [
                    MarkdownListItem(blocks: [.paragraph("Nested")])
                ]))
            ]),
            MarkdownListItem(blocks: [.paragraph("Second")])
        ]))])
    }

    @Test func parserAbsorbsWrappedListItemLines() {
        // Models routinely wrap a long bullet onto the next line without
        // indenting it. That line belongs to the item, not to a new paragraph.
        #expect(MarkdownBlockParser.parse("- First line\ncontinued here\n- Second")
                == [.list(MarkdownList(isOrdered: false, start: 1, isLoose: false, items: [
                    MarkdownListItem(blocks: [.paragraph("First line\ncontinued here")]),
                    MarkdownListItem(blocks: [.paragraph("Second")])
                ]))])

        // An empty item cannot swallow the following line.
        #expect(MarkdownBlockParser.parse("-\nSeparate paragraph")
                == [.list(MarkdownList(isOrdered: false, start: 1, isLoose: false, items: [
                    MarkdownListItem(blocks: [])
                ])), .paragraph("Separate paragraph")])
    }

    @Test func parserKeepsMultiBlockListItemsWithTheirMarker() {
        let parsed = MarkdownBlockParser.parse("""
        1. Install it:

           ```sh
           brew install thing
           ```

        2. Run it.
        """)

        #expect(parsed == [.list(MarkdownList(isOrdered: true, start: 1, isLoose: true, items: [
            MarkdownListItem(blocks: [
                .paragraph("Install it:"),
                .codeBlock(language: "sh", code: "brew install thing")
            ]),
            MarkdownListItem(blocks: [.paragraph("Run it.")])
        ]))])
    }

    @Test func parserReadsBlockQuotesAndThematicBreaks() {
        #expect(MarkdownBlockParser.parse("> Quoted text") == [.blockQuote([.paragraph("Quoted text")])])
        #expect(MarkdownBlockParser.parse(">Quote") == [.blockQuote([.paragraph("Quote")])])
        #expect(MarkdownBlockParser.parse("> One\n> - Item")
                == [.blockQuote([
                    .paragraph("One"),
                    .list(MarkdownList(isOrdered: false, start: 1, isLoose: false, items: [
                        MarkdownListItem(blocks: [.paragraph("Item")])
                    ]))
                ])])
        #expect(MarkdownBlockParser.parse("***") == [.thematicBreak])
        #expect(MarkdownBlockParser.parse("Above\n\n---\n\nBelow")
                == [.paragraph("Above"), .thematicBreak, .paragraph("Below")])
    }

    @Test func parserReadsTablesWithAlignmentAndRaggedRows() {
        let parsed = MarkdownBlockParser.parse("""
        | Name | Count | Note |
        |:-----|------:|:----:|
        | a | 1 | ok |
        | b | 2 |
        """)

        #expect(parsed == [.table(MarkdownTable(
            headers: ["Name", "Count", "Note"],
            alignments: [.leading, .trailing, .center],
            rows: [["a", "1", "ok"], ["b", "2", ""]]
        ))])
    }

    @Test func parserHonorsEscapedPipesInTableCells() {
        let parsed = MarkdownBlockParser.parse("| A | B |\n|---|---|\n| a \\| b | c |")

        #expect(parsed == [.table(MarkdownTable(
            headers: ["A", "B"],
            alignments: [.leading, .leading],
            rows: [["a | b", "c"]]
        ))])
    }

    @Test func parserRejectsPipeTextThatIsNotATable() {
        #expect(MarkdownBlockParser.parse("a | b | c") == [.paragraph("a | b | c")])
    }

    @Test func parserToleratesEveryStreamedPrefix() {
        // Bubbles re-parse on each token, so no prefix of a realistic answer may
        // trap the parser or silently drop the text the model has produced.
        let document = """
        # Title

        Intro **text** with `code`.

        - First
            - Nested
        - Second

        > Note

        | A | B |
        |---|---|
        | 1 | 2 |

        ```swift
        let first_name = 1
        ```

        Done.
        """

        for prefixLength in 1...document.count {
            let prefix = String(document.prefix(prefixLength))
            let blocks = MarkdownBlockParser.parse(prefix)
            #expect(!blocks.isEmpty)
        }

        #expect(MarkdownBlockParser.parse(document).count == 7)
    }

    // MARK: - Markdown inline rendering
}
