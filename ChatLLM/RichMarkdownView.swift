//
//  RichMarkdownView.swift
//  ChatLLM
//

import SwiftUI
import WebKit

/// Renders assistant Markdown.
///
/// There are two rendering paths, and the split is math-based:
///
/// - **Native** (`MarkdownBlocksView` + `MarkdownBlockParser`): headings, lists,
///   quotes, fenced code, tables, hr, and inline emphasis. Used for every
///   message that contains no math. No WebView, no height round-trip, no reload
///   when the transcript recycles a bubble off screen.
/// - **Bundled WebView document** (`RichMarkdownWebViewRepresentable`): the only
///   remaining user of `RenderingAssets/` (`markdown-it.min.js`, `katex.min.js`,
///   `katex.min.css`). It renders the *whole* message with markdown-it, whose math
///   rules hand each formula to KaTeX. Chosen only when
///   `RichTextFeatureDetector.requiresAdvancedRendering` finds math outside code
///   spans, when the WebView fails to load, and while a math document is still
///   measuring its height.
///
/// Do not delete the `RenderingAssets/` files: the native path cannot render
/// KaTeX, and the WebView document inlines all three of them (see `makeHTML()`).
/// All parser assets and WOFF2 fonts are required; a missing asset triggers the
/// existing native fallback. Fonts are embedded as data URIs under the local CSP.
/// `$…$`, `$$…$$`, `\(…\)`, `\[…\]` and bare KaTeX environments (align, cases,
/// pmatrix, …) share one delimiter list for detection and Markdown tokenizing.
/// Display math that starts a line may span blank lines.
///
/// Known limitations:
/// - Images are disabled in both paths: the native renderer reduces
///   `![alt](url)` to its alt text, while markdown-it's `image` rule is disabled
///   so the raw markup shows. Keep `RichMarkdownRenderingPolicy
///   .disabledMarkdownRules` in sync with `MarkdownInlineRenderer
///   .sanitizedSource` if that policy ever changes.
///
/// If you add a rendered feature, decide explicitly whether it belongs in
/// `MarkdownBlockParser` (native) or in the WebView document's JS, and keep the
/// detector in sync so math stays the only trigger for the WebView.
struct RichMarkdownView: View {
    let text: String
    let fontSize: Double
    var textTone: TextTone = .primary

    @Environment(\.colorScheme) private var colorScheme
    /// Body text size under the current Dynamic Type setting (17 at the default size).
    @ScaledMetric(relativeTo: .body) private var bodyPointSize: CGFloat = 17

    enum TextTone {
        case primary
        case secondary
    }

    /// The message size setting applies at the default text size; Dynamic Type scales
    /// it from there, so larger accessibility sizes enlarge messages like the rest of the app.
    static func scaledFontSize(_ fontSize: Double, bodyPointSize: CGFloat) -> Double {
        fontSize * Double(bodyPointSize) / 17
    }

    var body: some View {
        let fontSize = Self.scaledFontSize(fontSize, bodyPointSize: bodyPointSize)
        if RichTextFeatureDetector.requiresAdvancedRendering(text) {
            RichMarkdownWebView(
                text: text,
                fontSize: fontSize,
                palette: RichTextPalette(
                    colorScheme: colorScheme,
                    tone: textTone
                ),
                fallbackTone: textTone
            )
        } else {
            MarkdownBlocksView(
                blocks: MarkdownBlockCache.blocks(for: text),
                fontSize: fontSize,
                tone: textTone
            )
        }
    }
}

/// Used while the math document loads, and if it fails outright. Parsing is
/// shared with the primary path so the fallback keeps the same block structure.
private struct NativeMarkdownText: View {
    let text: String
    let fontSize: Double
    let textTone: RichMarkdownView.TextTone

    var body: some View {
        MarkdownBlocksView(
            blocks: MarkdownBlockCache.blocks(for: text),
            fontSize: fontSize,
            tone: textTone
        )
    }
}

enum NativeMarkdownParser {
    static func attributedString(from markdown: String) -> AttributedString? {
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace
        )
        return try? AttributedString(markdown: markdown, options: options)
    }
}

/// Bounded memo for parsed block trees.
///
/// A streaming message is re-parsed once per token, which is cheap and expected.
/// This only exists so that redundant `body` calls — an appearance change, a
/// parent re-render, a bubble scrolling back into view — do not repeat the work.
nonisolated enum MarkdownBlockCache {
    private static let limit = 32
    private static let lock = NSLock()
    nonisolated(unsafe) private static var entries: [String: [MarkdownBlock]] = [:]
    nonisolated(unsafe) private static var order: [String] = []

    static func blocks(for text: String) -> [MarkdownBlock] {
        lock.lock()
        if let cached = entries[text] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let parsed = MarkdownBlockParser.parse(text)

        lock.lock()
        if entries[text] == nil {
            entries[text] = parsed
            order.append(text)
            while order.count > limit {
                entries.removeValue(forKey: order.removeFirst())
            }
        }
        lock.unlock()
        return parsed
    }

    static func removeAll() {
        lock.lock()
        entries.removeAll()
        order.removeAll()
        lock.unlock()
    }
}

/// Remembers the measured height of a rendered math document.
///
/// The transcript is a `LazyVStack`, so scrolling a math bubble off screen and
/// back tears down its WebView and reloads the document. Without a cached height
/// the bubble collapses to a single point and everything below it jumps.
nonisolated enum MarkdownHeightCache {
    nonisolated struct Key: Hashable, Sendable {
        let text: String
        let fontSize: Double
    }

    private static let limit = 64
    private static let lock = NSLock()
    nonisolated(unsafe) private static var entries: [Key: CGFloat] = [:]
    nonisolated(unsafe) private static var order: [Key] = []

    static func height(for key: Key) -> CGFloat? {
        lock.lock()
        defer { lock.unlock() }
        return entries[key]
    }

    static func store(_ height: CGFloat, for key: Key) {
        guard height > 1 else { return }
        lock.lock()
        defer { lock.unlock() }
        if entries.updateValue(height, forKey: key) == nil {
            order.append(key)
            while order.count > limit {
                entries.removeValue(forKey: order.removeFirst())
            }
        }
    }

    static func removeAll() {
        lock.lock()
        entries.removeAll()
        order.removeAll()
        lock.unlock()
    }
}

/// KaTeX's display environments, which models also write without `$$`. Keep
/// parsing and routing aligned. `$…$` is not listed: it has stricter rules so
/// prices stay text (see `RichTextFeatureDetector.inlineDollarMath`).
private enum RichMathDelimiters {
    static let environments = [
        "equation", "equation*", "align", "align*", "aligned", "alignat", "alignat*", "alignedat",
        "gather", "gather*", "gathered", "split", "CD",
        "cases", "dcases", "rcases", "drcases",
        "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
        "matrix*", "pmatrix*", "bmatrix*", "Bmatrix*", "vmatrix*", "Vmatrix*",
        "smallmatrix", "array", "darray", "subarray"
    ]
    static let json: String = {
        let delimiters: [[String: Any]] = [
            ["left": "$$", "right": "$$", "display": true],
            ["left": "\\[", "right": "\\]", "display": true],
            ["left": "\\(", "right": "\\)", "display": false]
        ] + environments.map {
            ["left": "\\begin{\($0)}", "right": "\\end{\($0)}", "display": true]
        }
        return String(data: try! JSONSerialization.data(withJSONObject: delimiters), encoding: .utf8)!
    }()
}

/// Decides whether a message needs the bundled KaTeX document.
///
/// Only math does. Headings, lists, quotes, code, tables and inline emphasis are
/// all rendered natively, so they no longer force a WebView.
enum RichTextFeatureDetector {
    private static let mathPatterns: [NSRegularExpression] = {
        let patterns = [
            #"(?s)\\\((.+?)\\\)"#,
            #"(?s)\\\[(.+?)\\\]"#,
            #"(?s)\$\$(.+?)\$\$"#
        ]
        return patterns.compactMap { try? NSRegularExpression(pattern: $0, options: []) }
    }()

    /// `$x$` on one line. The opener needs a non-space after it, and the next `$`
    /// closes only with a non-space before it and no digit after it, so
    /// "$5 and $10" stays text. Mirrors the `$` branch of the document's math rule.
    private static let inlineDollarMath = try? NSRegularExpression(
        pattern: #"(?<![\\$])\$(?![\s$])[^$\n]*(?<![\s\\])\$(?![$\d])"#,
        options: []
    )

    private static let inlineCodeSpan = try? NSRegularExpression(pattern: #"`+[^`]*`+"#, options: [])

    static func requiresAdvancedRendering(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        // Swift interpolation (`\(name)`) and TeX-looking strings inside code are
        // not math. Masking code first keeps a snippet from spawning a WebView.
        let searchable = textOutsideCode(text)
        guard !searchable.isEmpty else { return false }

        if RichMathDelimiters.environments.contains(where: { searchable.contains("\\begin{\($0)}") }) {
            return true
        }

        let range = NSRange(searchable.startIndex..., in: searchable)
        return (mathPatterns + [inlineDollarMath].compactMap { $0 })
            .contains { $0.firstMatch(in: searchable, options: [], range: range) != nil }
    }

    /// Returns `text` with fenced blocks and inline code spans removed.
    static func textOutsideCode(_ text: String) -> String {
        guard text.contains("`") || text.contains("~~~") else { return text }

        var kept: [String] = []
        var fenceCharacter: Character?
        var fenceLength = 0

        for line in text.components(separatedBy: "\n") {
            let trimmed = line.drop(while: { $0 == " " || $0 == "\t" })
            let first = trimmed.first
            let isFenceLine = (first == "`" || first == "~") && trimmed.prefix(while: { $0 == first }).count >= 3

            if let open = fenceCharacter {
                if isFenceLine, first == open, trimmed.prefix(while: { $0 == open }).count >= fenceLength {
                    fenceCharacter = nil
                    fenceLength = 0
                }
                continue
            }

            if isFenceLine, let first {
                fenceCharacter = first
                fenceLength = trimmed.prefix(while: { $0 == first }).count
                continue
            }

            kept.append(line)
        }

        let withoutFences = kept.joined(separator: "\n")
        guard withoutFences.contains("`"), let inlineCodeSpan else { return withoutFences }
        return inlineCodeSpan.stringByReplacingMatches(
            in: withoutFences,
            range: NSRange(withoutFences.startIndex..., in: withoutFences),
            withTemplate: " "
        )
    }
}

enum RichMarkdownRenderingPolicy {
    static let disabledMarkdownRules = ["image"]
    static let contentSecurityPolicy = "default-src 'none'; img-src data:; font-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; object-src 'none'; base-uri 'none'"

    static var disabledMarkdownRulesJSON: String {
        guard let data = try? JSONEncoder().encode(disabledMarkdownRules),
              let json = String(data: data, encoding: .utf8) else {
            return #"["image"]"#
        }
        return json
    }
}

struct RichTextPalette: Hashable {
    let text: String
    let link: String
    let codeBackground: String
    let codeBorder: String
    let quoteBorder: String
    let quoteBackground: String
    let separator: String
    let tableStripe: String
    let inlineCode: String

    var cssVariables: [String: String] {
        ["text": text, "link": link, "code-background": codeBackground,
         "code-border": codeBorder, "quote-border": quoteBorder,
         "quote-background": quoteBackground, "separator": separator,
         "table-stripe": tableStripe, "inline-code": inlineCode]
    }

    init(colorScheme: ColorScheme, tone: RichMarkdownView.TextTone) {
        let trait = UITraitCollection(userInterfaceStyle: colorScheme == .dark ? .dark : .light)
        let textColor = (tone == .secondary ? UIColor.secondaryLabel : UIColor.label).resolvedColor(with: trait)
        self.text = textColor.cssRGBA
        self.link = UIColor.systemBlue.resolvedColor(with: trait).cssRGBA
        self.codeBackground = UIColor.secondarySystemBackground.resolvedColor(with: trait).cssRGBA
        self.codeBorder = UIColor.separator.resolvedColor(with: trait).withAlphaComponent(0.45).cssRGBA
        self.quoteBorder = UIColor.systemBlue.resolvedColor(with: trait).withAlphaComponent(0.5).cssRGBA
        self.quoteBackground = UIColor.tertiarySystemBackground.resolvedColor(with: trait).withAlphaComponent(0.7).cssRGBA
        self.separator = UIColor.separator.resolvedColor(with: trait).withAlphaComponent(0.35).cssRGBA
        self.tableStripe = UIColor.secondarySystemBackground.resolvedColor(with: trait).withAlphaComponent(0.45).cssRGBA
        self.inlineCode = UIColor.secondaryLabel.resolvedColor(with: trait).cssRGBA
    }
}

private extension UIColor {
    var cssRGBA: String {
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        return String(
            format: "rgba(%d,%d,%d,%.3f)",
            Int(round(red * 255)),
            Int(round(green * 255)),
            Int(round(blue * 255)),
            alpha
        )
    }
}

private struct RichMarkdownWebView: View {
    let text: String
    let fontSize: Double
    let palette: RichTextPalette
    let fallbackTone: RichMarkdownView.TextTone

    @State private var height: CGFloat
    @State private var hasMeasuredHeight = false
    @State private var failedToLoad = false
    @Environment(\.openURL) private var openURL

    init(text: String, fontSize: Double, palette: RichTextPalette, fallbackTone: RichMarkdownView.TextTone) {
        self.text = text
        self.fontSize = fontSize
        self.palette = palette
        self.fallbackTone = fallbackTone
        // Seeding the last known height keeps the transcript from jumping when a
        // bubble is recycled back into view and its document has to reload.
        // `hasMeasuredHeight` stays false so the native fallback still covers the
        // gap until KaTeX has actually painted.
        let cached = MarkdownHeightCache.height(for: .init(text: text, fontSize: fontSize))
        _height = State(initialValue: cached ?? 1)
    }

    var body: some View {
        Group {
            if failedToLoad {
                NativeMarkdownText(text: text, fontSize: fontSize, textTone: fallbackTone)
            } else {
                ZStack(alignment: .topLeading) {
                    if !hasMeasuredHeight {
                        NativeMarkdownText(text: text, fontSize: fontSize, textTone: fallbackTone)
                    }

                    RichMarkdownWebViewRepresentable(
                        text: text,
                        fontSize: fontSize,
                        palette: palette,
                        dynamicHeight: $height,
                        hasMeasuredHeight: $hasMeasuredHeight,
                        failedToLoad: $failedToLoad,
                        openURL: openURL
                    )
                    .frame(height: max(height, 1))
                    .opacity(hasMeasuredHeight ? 1 : 0)
                }
            }
        }
    }
}

struct RichMarkdownWebViewRepresentable: UIViewRepresentable {
    let text: String
    let fontSize: Double
    let palette: RichTextPalette
    @Binding var dynamicHeight: CGFloat
    @Binding var hasMeasuredHeight: Bool
    @Binding var failedToLoad: Bool
    let openURL: OpenURLAction

    func makeCoordinator() -> Coordinator {
        Coordinator(
            dynamicHeight: $dynamicHeight,
            hasMeasuredHeight: $hasMeasuredHeight,
            failedToLoad: $failedToLoad,
            openURL: openURL
        )
    }

    func makeUIView(context: Context) -> WKWebView {
        context.coordinator.makeWebView()
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        context.coordinator.update(webView, text: text, fontSize: fontSize, palette: palette)
    }

    static func dismantleUIView(_ webView: WKWebView, coordinator: Coordinator) {
        coordinator.dismantle(webView)
    }

    private static let markdownItJS: String? = bundledTextResource(named: "markdown-it.min", extension: "js")?
        .escapedInlineScript
    private static let katexCSS: String? = bundledMathCSS()?.escapedInlineStyle
    private static let katexJS: String? = bundledTextResource(named: "katex.min", extension: "js")?
        .escapedInlineScript

    // A single document per WebView. Message text only crosses the argument bridge,
    // so streamed tokens never reload the parser or become executable HTML/scripts.
    static let rendererHTML: String? = makeHTML()

    private static func makeHTML() -> String? {
        let disabledMarkdownRules = RichMarkdownRenderingPolicy.disabledMarkdownRulesJSON
        guard let markdownItJS, let katexCSS, let katexJS else { return nil }

        return """
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
          <meta http-equiv="Content-Security-Policy" content="\(RichMarkdownRenderingPolicy.contentSecurityPolicy)">
          <style>
          \(katexCSS)
          </style>
          <style>
            :root {
              color-scheme: light dark;
            }
            html, body {
              margin: 0;
              padding: 0;
              background: transparent;
              overflow: hidden;
            }
            body {
              color: var(--text);
              font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
              font-size: var(--font-size, 16px);
              line-height: 1.45;
              -webkit-font-smoothing: antialiased;
              word-wrap: break-word;
              overflow-wrap: anywhere;
              user-select: text;
              -webkit-user-select: text;
            }
            #content { display: flow-root; }
            #content > :first-child { margin-top: 0; }
            #content > :last-child { margin-bottom: 0; }
            p, ul, ol, blockquote, pre, table, hr {
              margin: 0 0 0.85em;
            }
            h1, h2, h3, h4, h5, h6 {
              margin: 0 0 0.55em;
              line-height: 1.2;
              font-weight: 700;
            }
            h1 { font-size: 1.65em; }
            h2 { font-size: 1.45em; }
            h3 { font-size: 1.25em; }
            h4 { font-size: 1.12em; }
            h5, h6 { font-size: 1em; }
            ul, ol {
              padding-left: 1.35em;
            }
            li + li {
              margin-top: 0.25em;
            }
            a {
              color: var(--link);
              text-decoration: none;
            }
            pre, code {
              font-family: "SF Mono", "Menlo", "Consolas", monospace;
            }
            code {
              color: var(--inline-code);
              background: var(--code-background);
              border: 1px solid var(--code-border);
              border-radius: 6px;
              padding: 0.12em 0.35em;
              font-size: 0.92em;
            }
            pre {
              background: var(--code-background);
              border: 1px solid var(--code-border);
              border-radius: 10px;
              padding: 0.8em 0.9em;
              overflow-x: auto;
            }
            pre code {
              background: transparent;
              border: 0;
              color: inherit;
              padding: 0;
              font-size: 0.9em;
              white-space: pre;
              overflow-wrap: normal;
            }
            blockquote {
              border-left: 3px solid var(--quote-border);
              background: var(--quote-background);
              padding: 0.75em 0.9em;
              border-radius: 0 10px 10px 0;
            }
            hr {
              border: 0;
              border-top: 1px solid var(--separator);
            }
            table {
              width: 100%;
              border-collapse: collapse;
              font-size: 0.96em;
            }
            th, td {
              border: 1px solid var(--separator);
              padding: 0.55em 0.7em;
              text-align: left;
              vertical-align: top;
            }
            thead th {
              background: var(--code-background);
            }
            tbody tr:nth-child(even) {
              background: var(--table-stripe);
            }
            img {
              max-width: 100%;
              height: auto;
              border-radius: 10px;
            }
            .katex-display {
              overflow-x: auto;
              overflow-y: hidden;
              padding: 0.25em 0;
            }
          </style>
        </head>
        <body>
          <div id="content"></div>
          <script>
          \(markdownItJS)
          </script>
          <script>
          \(katexJS)
          </script>
          <script>
            const root = document.getElementById('content');
            const md = window.markdownit({
              html: false,
              linkify: true,
              typographer: false,
              breaks: true
            });
            md.disable(\(disabledMarkdownRules));
            // Math is tokenized here and rendered by KaTeX directly, so Markdown's
            // escape/emphasis rules never see it (they would strip \\( and \\[),
            // and code spans or fences are never mistaken for math.
            const mathDelimiters = \(RichMathDelimiters.json);
            const isSpace = (character) => character === undefined || /\\s/.test(character);
            // Environments go to KaTeX whole; other delimiters only wrap the TeX.
            const mathToken = (state, type, source, delimiter) => {
              const token = state.push(type, '', 0);
              token.content = delimiter.left.startsWith('\\\\begin')
                ? source
                : source.slice(delimiter.left.length, source.length - delimiter.right.length);
              token.meta = {display: delimiter.display, source};
              return token;
            };
            md.inline.ruler.before('escape', 'math', (state, silent) => {
              const src = state.src;
              const pos = state.pos;
              for (const delimiter of mathDelimiters) {
                if (!src.startsWith(delimiter.left, pos)) continue;
                const end = src.indexOf(delimiter.right, pos + delimiter.left.length);
                if (end < 0) return false;
                const next = end + delimiter.right.length;
                if (!silent) mathToken(state, 'math', src.slice(pos, next), delimiter);
                state.pos = next;
                return true;
              }
              // $x$: same rules as RichTextFeatureDetector.inlineDollarMath.
              if (src[pos] !== '$' || src[pos + 1] === '$' || isSpace(src[pos + 1])) return false;
              const end = src.indexOf('$', pos + 1);
              if (end < 0 || isSpace(src[end - 1]) || src[end - 1] === '\\\\'
                  || /[\\d$]/.test(src[end + 1] ?? '') || src.slice(pos, end).includes('\\n')) return false;
              if (!silent) mathToken(state, 'math', src.slice(pos, end + 1), {left: '$', right: '$', display: false});
              state.pos = end + 1;
              return true;
            });
            // A display block that starts a line may contain blank lines, which
            // would otherwise split it into separate paragraphs.
            md.block.ruler.before('paragraph', 'math_block', (state, startLine, endLine, silent) => {
              if (state.sCount[startLine] - state.blkIndent >= 4) return false;
              const lineText = (line) => state.src.slice(state.bMarks[line] + state.tShift[line], state.eMarks[line]);
              const first = lineText(startLine);
              const delimiter = mathDelimiters.find(({left, display}) => display && first.startsWith(left));
              if (!delimiter) return false;
              let source = first;
              let searchFrom = delimiter.left.length;
              for (let line = startLine; line < endLine; line++) {
                if (line > startLine) {
                  const text = lineText(line);
                  if (text && state.sCount[line] < state.blkIndent) return false;
                  searchFrom = source.length;
                  source += '\\n' + text;
                }
                const end = source.indexOf(delimiter.right, searchFrom);
                if (end < 0) continue;
                const next = end + delimiter.right.length;
                if (source.slice(next).trim()) return false;
                if (!silent) {
                  mathToken(state, 'math_block', source.slice(0, next), delimiter).map = [startLine, line + 1];
                }
                state.line = line + 1;
                return true;
              }
              return false;
            }, {alt: ['paragraph', 'reference', 'blockquote', 'list']});
            const renderMath = (token) => {
              try {
                return katex.renderToString(token.content, {
                  displayMode: token.meta.display,
                  throwOnError: false,
                  strict: 'ignore'
                });
              } catch {
                return md.utils.escapeHtml(token.meta.source);
              }
            };
            md.renderer.rules.math = (tokens, index) => renderMath(tokens[index]);
            md.renderer.rules.math_block = (tokens, index) => '<p>' + renderMath(tokens[index]) + '</p>\\n';
            let revision = 0;

            const postHeight = () => {
              if (!revision) return;
              // Document scrollHeight is at least the viewport height, which
              // prevents a previously tall bubble from ever shrinking.
              const height = Math.max(1, Math.ceil(root.getBoundingClientRect().height));
              window.webkit?.messageHandlers?.\(Coordinator.heightHandlerName)?.postMessage({height, revision});
            };

            window.updateMarkdown = (source, fontSize, palette, nextRevision) => {
              revision = nextRevision;
              document.documentElement.style.setProperty('--font-size', fontSize + 'px');
              for (const [key, value] of Object.entries(palette)) {
                document.documentElement.style.setProperty('--' + key, value);
              }
              root.innerHTML = md.render(source);
              postHeight();
              requestAnimationFrame(postHeight);
            };

            if (window.ResizeObserver) {
              const observer = new ResizeObserver(postHeight);
              observer.observe(root);
            }
            window.addEventListener('resize', postHeight);
            if (document.fonts && document.fonts.ready) {
              document.fonts.ready.then(postHeight);
            }
          </script>
        </body>
        </html>
        """
    }

    /// Use only the bundled WOFF2 source; other formats would leave dangling
    /// relative URLs in a document loaded from an HTML string.
    private static func bundledMathCSS() -> String? {
        guard var css = bundledTextResource(named: "katex.min", extension: "css"),
              let sources = try? NSRegularExpression(pattern: #"src:url\(fonts/(KaTeX_[^)]+)\.woff2\)[^;}]+"#) else { return nil }
        let matches = sources.matches(in: css, range: NSRange(css.startIndex..., in: css))
        guard !matches.isEmpty else { return nil }
        for match in matches.reversed() {
            guard let nameRange = Range(match.range(at: 1), in: css),
                  let sourceRange = Range(match.range, in: css) else { return nil }
            let name = String(css[nameRange])
            let url = Bundle.main.url(forResource: name, withExtension: "woff2")
                ?? Bundle.main.url(forResource: name, withExtension: "woff2", subdirectory: "RenderingAssets/Fonts")
                ?? Bundle.main.url(forResource: name, withExtension: "woff2", subdirectory: "Fonts")
            guard let url, let data = try? Data(contentsOf: url) else { return nil }
            css.replaceSubrange(sourceRange, with: "src:url(data:font/woff2;base64,\(data.base64EncodedString())) format(\"woff2\")")
        }
        return css
    }

    private static func bundledTextResource(named name: String, extension fileExtension: String) -> String? {
        let candidateURLs = [
            Bundle.main.url(forResource: name, withExtension: fileExtension),
            Bundle.main.url(forResource: name, withExtension: fileExtension, subdirectory: "RenderingAssets")
        ]

        for url in candidateURLs.compactMap({ $0 }) {
            if let contents = try? String(contentsOf: url, encoding: .utf8) {
                return contents
            }
        }

        return nil
    }

    final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        static let heightHandlerName = "richTextHeight"

        @Binding var dynamicHeight: CGFloat
        @Binding var hasMeasuredHeight: Bool
        @Binding var failedToLoad: Bool
        private struct Payload: Equatable {
            let text: String
            let fontSize: Double
            let palette: RichTextPalette
        }

        private var latestPayload: Payload?
        private var appliedPayload: Payload?
        private var isLoadingDocument = false
        private var isReady = false
        private var isRendering = false
        private var isDismantled = false
        private var revision = 0
        private var measuredRevision = 0
        private let openURL: OpenURLAction

        init(
            dynamicHeight: Binding<CGFloat>,
            hasMeasuredHeight: Binding<Bool>,
            failedToLoad: Binding<Bool>,
            openURL: OpenURLAction
        ) {
            self._dynamicHeight = dynamicHeight
            self._hasMeasuredHeight = hasMeasuredHeight
            self._failedToLoad = failedToLoad
            self.openURL = openURL
        }

        func makeWebView() -> WKWebView {
            // No explicit process pool: WebKit has shared content processes
            // between web views on its own since iOS 15, and `WKProcessPool` is
            // deprecated precisely because setting one no longer does anything.
            let config = WKWebViewConfiguration()
            config.defaultWebpagePreferences.allowsContentJavaScript = true
            config.preferences.javaScriptCanOpenWindowsAutomatically = false
            config.userContentController.add(self, name: Self.heightHandlerName)

            let webView = WKWebView(frame: .zero, configuration: config)
            webView.isOpaque = false
            webView.backgroundColor = .clear
            webView.scrollView.backgroundColor = .clear
            webView.scrollView.isScrollEnabled = false
            webView.scrollView.showsVerticalScrollIndicator = false
            webView.scrollView.showsHorizontalScrollIndicator = false
            webView.navigationDelegate = self
            webView.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
            webView.setContentHuggingPriority(.defaultLow, for: .horizontal)
            return webView
        }

        func update(_ webView: WKWebView, text: String, fontSize: Double, palette: RichTextPalette) {
            guard !isDismantled else { return }
            latestPayload = Payload(text: text, fontSize: fontSize, palette: palette)
            if isReady {
                renderLatest(in: webView)
            } else if !isLoadingDocument {
                guard let html = RichMarkdownWebViewRepresentable.rendererHTML else {
                    setFailedToLoad(true)
                    return
                }
                isLoadingDocument = true
                webView.loadHTMLString(html, baseURL: nil)
            }
        }

        private func renderLatest(in webView: WKWebView) {
            guard !isDismantled, isReady, !isRendering,
                  let payload = latestPayload, payload != appliedPayload else { return }
            isRendering = true
            revision += 1
            webView.callAsyncJavaScript(
                "window.updateMarkdown(text, fontSize, palette, revision); return true;",
                arguments: ["text": payload.text, "fontSize": payload.fontSize,
                            "palette": payload.palette.cssVariables, "revision": revision],
                in: nil,
                in: .page
            ) { [weak self, weak webView] result in
                guard let self, !self.isDismantled else { return }
                self.isRendering = false
                switch result {
                case .success:
                    self.appliedPayload = payload
                    // Coalesce tokens that arrived while JS was running, including
                    // the final update. Never debounce until generation stops.
                    if let webView { self.renderLatest(in: webView) }
                case .failure:
                    self.setFailedToLoad(true)
                }
            }
        }

        func dismantle(_ webView: WKWebView) {
            isDismantled = true
            webView.stopLoading()
            webView.configuration.userContentController.removeScriptMessageHandler(forName: Self.heightHandlerName)
            webView.navigationDelegate = nil
        }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            guard !isDismantled, message.name == Self.heightHandlerName,
                  let body = message.body as? [String: Any],
                  let number = body["height"] as? NSNumber,
                  let nextRevision = body["revision"] as? Int,
                  nextRevision > 0, nextRevision <= revision else { return }
            let nextHeight = CGFloat(truncating: number)
            guard nextHeight.isFinite, nextHeight > 0 else { return }
            DispatchQueue.main.async { [weak self] in
                guard let self, !self.isDismantled, nextRevision >= self.measuredRevision else { return }
                // Accept completed renders while a newer token is queued. Requiring
                // the latest requested revision can starve height updates mid-stream.
                self.measuredRevision = nextRevision
                if !self.hasMeasuredHeight { self.hasMeasuredHeight = true }
                if abs(self.dynamicHeight - nextHeight) >= 1 {
                    self.dynamicHeight = nextHeight
                }
                if let payload = self.appliedPayload ?? self.latestPayload {
                    MarkdownHeightCache.store(
                        nextHeight,
                        for: .init(text: payload.text, fontSize: payload.fontSize)
                    )
                }
            }
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            isLoadingDocument = false
            isReady = true
            renderLatest(in: webView)
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            setFailedToLoad(true)
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            setFailedToLoad(true)
        }

        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            setFailedToLoad(true)
        }

        func setFailedToLoad(_ nextValue: Bool) {
            DispatchQueue.main.async { [weak self] in
                guard let self, !self.isDismantled, self.failedToLoad != nextValue else { return }
                self.failedToLoad = nextValue
            }
        }

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            guard let url = navigationAction.request.url else {
                decisionHandler(.allow)
                return
            }

            let isMainFrameInitialLoad = navigationAction.targetFrame?.isMainFrame == true && url.scheme == "about"
            guard !isMainFrameInitialLoad else {
                decisionHandler(.allow)
                return
            }

            if ["http", "https"].contains(url.scheme?.lowercased() ?? "") {
                openURL(url)
                decisionHandler(.cancel)
                return
            }

            decisionHandler(.cancel)
        }
    }
}

private extension String {
    var escapedInlineScript: String {
        replacingOccurrences(of: "</script", with: "<\\/script")
    }

    var escapedInlineStyle: String {
        replacingOccurrences(of: "</style", with: "<\\/style")
    }

}
