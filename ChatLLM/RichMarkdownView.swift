//
//  RichMarkdownView.swift
//  ChatLLM
//

import SwiftUI
import WebKit

struct RichMarkdownView: View {
    let text: String
    let fontSize: Double
    var textTone: TextTone = .primary
    var forceAdvancedRenderer: Bool = false

    @Environment(\.colorScheme) private var colorScheme

    enum TextTone {
        case primary
        case secondary
    }

    private var shouldUseAdvancedRenderer: Bool {
        forceAdvancedRenderer || RichTextFeatureDetector.requiresAdvancedRendering(text)
    }

    var body: some View {
        if shouldUseAdvancedRenderer {
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
            NativeMarkdownText(
                text: text,
                fontSize: fontSize,
                textTone: textTone
            )
        }
    }
}

private struct NativeMarkdownText: View {
    let text: String
    let fontSize: Double
    let textTone: RichMarkdownView.TextTone

    var body: some View {
        Group {
            if let attributed = NativeMarkdownParser.attributedString(from: text) {
                Text(attributed)
                    .font(.system(size: fontSize))
                    .foregroundStyle(textTone == .secondary ? .secondary : .primary)
            } else {
                Text(verbatim: text.isEmpty ? " " : text)
                    .font(.system(size: fontSize))
                    .foregroundStyle(textTone == .secondary ? .secondary : .primary)
            }
        }
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

enum RichTextFeatureDetector {
    // Pre-compiled patterns (compiled once at app launch, avoiding per-call regex recompilation)
    private static let compiledPatterns: [NSRegularExpression] = {
        let patterns = [
            // Block patterns require the bundled renderer. SwiftUI Text can
            // display inline AttributedString styles, but it does not preserve
            // the visual hierarchy and layout of these block structures.
            #"(?m)^\s{0,3}#{1,6}\s+\S"#,
            #"(?m)^[ \t]{0,3}(?:[-*+]\s+\S|\d+[.)]\s+\S)"#,
            #"(?m)^[ \t]{0,3}>\s*\S"#,
            #"(?m)^\S[^\n]*\n[ \t]{0,3}(?:=+|-+)[ \t]*$"#,
            #"(?m)^(?: {4}|\t)\S"#,
            #"(?m)^\s{0,3}```"#,
            #"(?m)^\s{0,3}~~~"#,
            #"(?m)^\s{0,3}(?:\|.*\|)$"#,
            #"(?m)^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$"#,
            // Route every inline or reference-style image through the hardened
            // WebView policy instead of leaving it to the native fallback.
            #"!\["#,
            // Math patterns
            #"(?s)\\\((.+?)\\\)"#,
            #"(?s)\\\[(.+?)\\\]"#,
            #"(?s)\$\$(.+?)\$\$"#
        ]
        return patterns.compactMap { try? NSRegularExpression(pattern: $0, options: [.caseInsensitive]) }
    }()

    static func requiresAdvancedRendering(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)

        if text.contains("\\begin{") || text.contains("\\end{") {
            return true
        }
        for regex in compiledPatterns {
            if regex.firstMatch(in: text, options: [], range: range) != nil {
                return true
            }
        }
        return false
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

    @State private var height: CGFloat = 1
    @State private var hasMeasuredHeight = false
    @State private var failedToLoad = false
    @Environment(\.openURL) private var openURL

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
    private static let katexCSS: String = (bundledTextResource(named: "katex.min", extension: "css") ?? "")
        .escapedInlineStyle
    private static let katexJS: String = (bundledTextResource(named: "katex.min", extension: "js") ?? "")
        .escapedInlineScript
    private static let katexAutoRenderJS: String = (bundledTextResource(named: "katex-auto-render.min", extension: "js") ?? "")
        .escapedInlineScript

    // A single document per WebView. Message text only crosses the argument bridge,
    // so streamed tokens never reload the parser or become executable HTML/scripts.
    static let rendererHTML: String? = makeHTML()

    private static func makeHTML() -> String? {
        let disabledMarkdownRules = RichMarkdownRenderingPolicy.disabledMarkdownRulesJSON
        guard let markdownItJS else { return nil }

        let katexCSS = Self.katexCSS
        let katexJS = Self.katexJS
        let katexAutoRenderJS = Self.katexAutoRenderJS

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
          \(katexAutoRenderJS)
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
            // Protect math from Markdown's escape/emphasis rules before KaTeX
            // sees it. In particular, Markdown would strip \\( and \\[ delimiters.
            md.inline.ruler.before('escape', 'math', (state, silent) => {
              const delimiters = [['$$', '$$'], ['\\\\(', '\\\\)'], ['\\\\[', '\\\\]']];
              for (const [left, right] of delimiters) {
                if (!state.src.startsWith(left, state.pos)) continue;
                const end = state.src.indexOf(right, state.pos + left.length);
                if (end < 0) return false;
                if (!silent) {
                  const token = state.push('math', '', 0);
                  token.content = state.src.slice(state.pos, end + right.length);
                }
                state.pos = end + right.length;
                return true;
              }
              return false;
            });
            md.renderer.rules.math = (tokens, index) => md.utils.escapeHtml(tokens[index].content);
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
              if (window.renderMathInElement) {
                window.renderMathInElement(root, {
                  throwOnError: false,
                  strict: 'ignore',
                  ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                  delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '\\\\[', right: '\\\\]', display: true },
                    { left: '\\\\(', right: '\\\\)', display: false }
                  ]
                });
              }
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
