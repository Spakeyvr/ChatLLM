# Local math assets

The WebView inlines these resources; it makes no remote font or script requests.
`Fonts/*.woff2` and `KaTeX-LICENSE.txt` come from the official KaTeX **0.16.28**
npm package, matching the version in `katex.min.js`:
https://registry.npmjs.org/katex/-/katex-0.16.28.tgz

`RichMarkdownWebViewRepresentable.bundledMathCSS()` replaces each CSS font source
with its bundled WOFF2 data URI. Missing fonts or parser files cause the existing
native fallback instead of silently displaying incomplete math.

`RichMathDelimiters` is the shared list for routing, Markdown protection and KaTeX
auto-render. Keep the version, fonts and license together when updating KaTeX.
