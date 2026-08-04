//
//  RichMarkdownView.swift
//  On-Device_LLM_Chat
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
        let processedText = LatexProcessor.process(text)

        Group {
            if let attributed = try? AttributedString(markdown: processedText) {
                Text(attributed)
                    .font(.system(size: fontSize))
                    .foregroundStyle(textTone == .secondary ? .secondary : .primary)
            } else {
                Text(processedText.isEmpty ? " " : processedText)
                    .font(.system(size: fontSize))
                    .foregroundStyle(textTone == .secondary ? .secondary : .primary)
            }
        }
    }
}

enum RichTextFeatureDetector {
    // Pre-compiled patterns (compiled once at app launch, avoiding per-call regex recompilation)
    private static let compiledPatterns: [NSRegularExpression] = {
        let patterns = [
            // Block patterns
            #"(?m)^\s{0,3}#{1,6}\s+\S"#,
            #"(?m)^\s{0,3}(?:[-*+]\s+\S|\d+\.\s+\S)"#,
            #"(?m)^\s{0,3}>\s+\S"#,
            #"(?m)^\s{0,3}```"#,
            #"(?m)^\s{0,3}~~~"#,
            #"(?m)^\s{0,3}(?:\|.*\|)$"#,
            #"(?m)^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$"#,
            // Math patterns
            #"\\\((.+?)\\\)"#,
            #"\\\[(.+?)\\\]"#,
            #"\$\$(.+?)\$\$"#,
            #"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)"#
        ]
        return patterns.compactMap { try? NSRegularExpression(pattern: $0, options: [.caseInsensitive]) }
    }()

    private static let paragraphBreakRegex: NSRegularExpression? =
        try? NSRegularExpression(pattern: #"\n\s*\n"#, options: [])

    static func requiresAdvancedRendering(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)

        if let re = paragraphBreakRegex, re.firstMatch(in: text, options: [], range: range) != nil {
            return true
        }
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

private struct RichTextPalette: Hashable {
    let text: String
    let link: String
    let codeBackground: String
    let codeBorder: String
    let quoteBorder: String
    let quoteBackground: String
    let separator: String
    let tableStripe: String
    let inlineCode: String

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

private struct RichMarkdownWebViewRepresentable: UIViewRepresentable {
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
        let config = WKWebViewConfiguration()
        config.defaultWebpagePreferences.allowsContentJavaScript = true
        config.preferences.javaScriptCanOpenWindowsAutomatically = false
        config.userContentController.add(context.coordinator, name: Coordinator.heightHandlerName)

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.isOpaque = false
        webView.backgroundColor = .clear
        webView.scrollView.backgroundColor = .clear
        webView.scrollView.isScrollEnabled = false
        webView.scrollView.showsVerticalScrollIndicator = false
        webView.scrollView.showsHorizontalScrollIndicator = false
        webView.navigationDelegate = context.coordinator
        webView.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        webView.setContentHuggingPriority(.defaultLow, for: .horizontal)
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        // Hash the inputs, not the assembled document: the document embeds ~430 KB of
        // bundled KaTeX/markdown-it sources, and building + hashing it on every SwiftUI
        // update was dominating the cost of showing a message bubble.
        var hasher = Hasher()
        hasher.combine(text)
        hasher.combine(fontSize)
        hasher.combine(palette)
        let payloadHash = hasher.finalize()

        guard context.coordinator.lastPayloadHash != payloadHash else { return }

        guard let html = Self.makeHTML(text: text, fontSize: fontSize, palette: palette) else {
            context.coordinator.setFailedToLoad(true)
            return
        }

        context.coordinator.lastPayloadHash = payloadHash
        context.coordinator.setHasMeasuredHeight(false)
        context.coordinator.setDynamicHeight(1)
        context.coordinator.setFailedToLoad(false)

        webView.loadHTMLString(html, baseURL: nil)
    }

    static func dismantleUIView(_ webView: WKWebView, coordinator: Coordinator) {
        webView.configuration.userContentController.removeScriptMessageHandler(forName: Coordinator.heightHandlerName)
        webView.navigationDelegate = nil
    }

    // Bundled renderer sources are ~430 KB in total and never change at runtime.
    // Read and escape them once instead of on every updateUIView call.
    private static let markdownItJS: String? = bundledTextResource(named: "markdown-it.min", extension: "js")?
        .escapedInlineScript
    private static let katexCSS: String = (bundledTextResource(named: "katex.min", extension: "css") ?? "")
        .escapedInlineStyle
    private static let katexJS: String = (bundledTextResource(named: "katex.min", extension: "js") ?? "")
        .escapedInlineScript
    private static let katexAutoRenderJS: String = (bundledTextResource(named: "katex-auto-render.min", extension: "js") ?? "")
        .escapedInlineScript

    private static func makeHTML(text: String, fontSize: Double, palette: RichTextPalette) -> String? {
        let encodedText = text.jsLiteral
        guard let markdownItJS else {
            return nil
        }

        let katexCSS = Self.katexCSS
        let katexJS = Self.katexJS
        let katexAutoRenderJS = Self.katexAutoRenderJS

        return """
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
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
              color: \(palette.text);
              font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
              font-size: \(fontSize)px;
              line-height: 1.45;
              -webkit-font-smoothing: antialiased;
              word-wrap: break-word;
              overflow-wrap: anywhere;
              user-select: text;
              -webkit-user-select: text;
            }
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
              color: \(palette.link);
              text-decoration: none;
            }
            pre, code {
              font-family: "SF Mono", "Menlo", "Consolas", monospace;
            }
            code {
              color: \(palette.inlineCode);
              background: \(palette.codeBackground);
              border: 1px solid \(palette.codeBorder);
              border-radius: 6px;
              padding: 0.12em 0.35em;
              font-size: 0.92em;
            }
            pre {
              background: \(palette.codeBackground);
              border: 1px solid \(palette.codeBorder);
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
              border-left: 3px solid \(palette.quoteBorder);
              background: \(palette.quoteBackground);
              padding: 0.75em 0.9em;
              border-radius: 0 10px 10px 0;
            }
            hr {
              border: 0;
              border-top: 1px solid \(palette.separator);
            }
            table {
              width: 100%;
              border-collapse: collapse;
              font-size: 0.96em;
            }
            th, td {
              border: 1px solid \(palette.separator);
              padding: 0.55em 0.7em;
              text-align: left;
              vertical-align: top;
            }
            thead th {
              background: \(palette.codeBackground);
            }
            tbody tr:nth-child(even) {
              background: \(palette.tableStripe);
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
            const source = \(encodedText);
            const root = document.getElementById('content');
            const md = window.markdownit({
              html: false,
              linkify: true,
              typographer: true,
              breaks: true
            });
            root.innerHTML = md.render(source);
            if (window.renderMathInElement) {
              window.renderMathInElement(root, {
                throwOnError: false,
                strict: "ignore",
                ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
                delimiters: [
                  { left: "$$", right: "$$", display: true },
                  { left: "\\\\[", right: "\\\\]", display: true },
                  { left: "\\\\(", right: "\\\\)", display: false },
                  { left: "$", right: "$", display: false }
                ]
              });
            }

            const postHeight = () => {
              const height = Math.max(
                document.body.scrollHeight,
                document.documentElement.scrollHeight,
                root.scrollHeight
              );
              window.webkit?.messageHandlers?.\(Coordinator.heightHandlerName)?.postMessage(height);
            };

            if (window.ResizeObserver) {
              const observer = new ResizeObserver(() => postHeight());
              observer.observe(document.body);
              observer.observe(root);
            }

            window.addEventListener('load', postHeight);
            if (document.fonts && document.fonts.ready) {
              document.fonts.ready.then(postHeight);
            }
            setTimeout(postHeight, 0);
            setTimeout(postHeight, 60);
            setTimeout(postHeight, 180);
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
        var lastPayloadHash: Int?
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

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            guard message.name == Self.heightHandlerName else { return }

            let nextHeight: CGFloat
            if let number = message.body as? NSNumber {
                nextHeight = CGFloat(truncating: number)
            } else {
                return
            }

            guard nextHeight.isFinite, nextHeight > 0 else { return }
            DispatchQueue.main.async {
                self.hasMeasuredHeight = true
                if abs(self.dynamicHeight - nextHeight) > 1 {
                    self.dynamicHeight = nextHeight
                }
            }
        }

        func setDynamicHeight(_ nextValue: CGFloat) {
            guard dynamicHeight != nextValue else { return }
            DispatchQueue.main.async {
                self.dynamicHeight = nextValue
            }
        }

        func setHasMeasuredHeight(_ nextValue: Bool) {
            guard hasMeasuredHeight != nextValue else { return }
            DispatchQueue.main.async {
                self.hasMeasuredHeight = nextValue
            }
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            setFailedToLoad(false)
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            setFailedToLoad(true)
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            setFailedToLoad(true)
        }

        func setFailedToLoad(_ nextValue: Bool) {
            guard failedToLoad != nextValue else { return }
            DispatchQueue.main.async {
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

    var jsLiteral: String {
        let encoder = JSONEncoder()
        guard
            let data = try? encoder.encode(self),
            let encoded = String(data: data, encoding: .utf8)
        else {
            return "\"\""
        }
        return encoded
    }
}
