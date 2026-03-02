//
//  LatexProcessor.swift
//  On-Device_LLM_Chat
//
//  Converts LLM LaTeX math output to readable Unicode so expressions like
//  \cdot, \div, \frac{a}{b} and \[...\] display as ·, ÷, a/b and styled math.
//

import Foundation

enum LatexProcessor {

    // MARK: - Public entry point

    /// Convert LaTeX math notation in `text` to Unicode/readable equivalents.
    static func process(_ text: String) -> String {
        var result = text
        result = processDisplayMath(result)   // \[...\] and $$...$$
        result = processInlineMath(result)    // \(...\) and $...$
        result = convertLatexCommands(result) // bare \cdot etc. outside delimiters
        return result
    }

    // MARK: - Display math blocks

    private static func processDisplayMath(_ text: String) -> String {
        var result = text

        // \[...\]
        result = mapCaptures(in: result, pattern: #"\\\[(.*?)\\\]"#, dotAll: true) { content in
            let inner = convertLatexCommands(content.trimmingCharacters(in: .whitespacesAndNewlines))
            return "\n`\(inner)`\n"
        }

        // $$...$$
        result = mapCaptures(in: result, pattern: #"\$\$(.*?)\$\$"#, dotAll: true) { content in
            let inner = convertLatexCommands(content.trimmingCharacters(in: .whitespacesAndNewlines))
            return "\n`\(inner)`\n"
        }

        return result
    }

    // MARK: - Inline math

    private static func processInlineMath(_ text: String) -> String {
        var result = text

        // \(...\)
        result = mapCaptures(in: result, pattern: #"\\\((.*?)\\\)"#, dotAll: false) { content in
            convertLatexCommands(content)
        }

        // $...$ — avoid $$ already consumed above
        result = mapCaptures(in: result, pattern: #"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)"#, dotAll: false) { content in
            convertLatexCommands(content)
        }

        return result
    }

    // MARK: - Symbol & structure conversion

    /// Convert LaTeX commands within a string that has already had delimiters removed.
    static func convertLatexCommands(_ text: String) -> String {
        var s = text

        // Multi-argument structures first
        s = processFrac(s)
        s = processSqrt(s)

        // Super/subscripts
        s = processSuperscript(s)
        s = processSubscript(s)

        // Symbol table (longer patterns before shorter to avoid partial matches)
        for (latex, unicode) in symbolTable {
            s = s.replacingOccurrences(of: latex, with: unicode)
        }

        return s
    }

    // MARK: - \frac{num}{den}

    private static func processFrac(_ text: String) -> String {
        guard text.contains("\\frac") else { return text }
        guard let regex = try? NSRegularExpression(pattern: #"\\frac\{([^{}]*)\}\{([^{}]*)\}"#) else { return text }
        let ns = text as NSString
        var result = text
        for match in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).reversed() {
            guard match.numberOfRanges >= 3 else { continue }
            let num = convertLatexCommands(ns.substring(with: match.range(at: 1)))
            let den = convertLatexCommands(ns.substring(with: match.range(at: 2)))
            let rep: String
            if let uf = unicodeFraction(num: num, den: den) {
                rep = uf
            } else if den.count == 1 || (den.count <= 4 && !den.contains(" ")) {
                rep = "\(num)/\(den)"
            } else {
                rep = "(\(num))/(\(den))"
            }
            if let r = Range(match.range(at: 0), in: result) { result.replaceSubrange(r, with: rep) }
        }
        return result
    }

    private static func unicodeFraction(num: String, den: String) -> String? {
        let table: [String: [String: String]] = [
            "1": ["2": "½", "3": "⅓", "4": "¼", "5": "⅕", "6": "⅙", "7": "⅐", "8": "⅛", "9": "⅑", "10": "⅒"],
            "2": ["3": "⅔", "5": "⅖"],
            "3": ["4": "¾", "5": "⅗", "8": "⅜"],
            "4": ["5": "⅘"],
            "5": ["6": "⅚", "8": "⅝"],
            "7": ["8": "⅞"],
        ]
        return table[num.trimmingCharacters(in: .whitespaces)]?[den.trimmingCharacters(in: .whitespaces)]
    }

    // MARK: - \sqrt{x} / \sqrt[n]{x}

    private static func processSqrt(_ text: String) -> String {
        guard text.contains("\\sqrt") else { return text }
        var result = text

        // \sqrt[n]{x}
        result = mapCaptures(in: result, pattern: #"\\sqrt\[([^\]]*)\]\{([^{}]*)\}"#, dotAll: false,
                             captureCount: 2) { captures in
            let n = toSuperscript(convertLatexCommands(captures[0]))
            let x = convertLatexCommands(captures[1])
            return "\(n)√\(x)"
        }

        // \sqrt{x}
        result = mapCaptures(in: result, pattern: #"\\sqrt\{([^{}]*)\}"#, dotAll: false) { content in
            let x = convertLatexCommands(content)
            // Omit parens for short single-token content
            return x.count <= 3 ? "√\(x)" : "√(\(x))"
        }

        return result
    }

    // MARK: - Superscripts  ^{...} or ^x

    private static func processSuperscript(_ text: String) -> String {
        var result = text
        // ^{content}
        result = mapCaptures(in: result, pattern: #"\^\{([^{}]+)\}"#, dotAll: false) { toSuperscript($0) }
        // ^single-char
        result = mapCaptures(in: result, pattern: #"\^([0-9a-zA-Z+\-])"#, dotAll: false) { toSuperscript($0) }
        return result
    }

    private static func toSuperscript(_ s: String) -> String {
        let converted = convertLatexCommands(s)
        let mapped = converted.map { superMap[$0] }
        if mapped.allSatisfy({ $0 != nil }) { return mapped.compactMap { $0 }.joined() }
        return converted.count == 1 ? "^\(converted)" : "^(\(converted))"
    }

    private static let superMap: [Character: String] = [
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
        "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
        "n": "ⁿ", "i": "ⁱ",
    ]

    // MARK: - Subscripts  _{...} or _x

    private static func processSubscript(_ text: String) -> String {
        var result = text
        // _{content}
        result = mapCaptures(in: result, pattern: #"_\{([^{}]+)\}"#, dotAll: false) { toSubscript($0) }
        // _single-char (digit or letter)
        result = mapCaptures(in: result, pattern: #"_([0-9a-zA-Z])"#, dotAll: false) { toSubscript($0) }
        return result
    }

    private static func toSubscript(_ s: String) -> String {
        let converted = convertLatexCommands(s)
        let mapped = converted.map { subMap[$0] }
        if mapped.allSatisfy({ $0 != nil }) { return mapped.compactMap { $0 }.joined() }
        return converted.count == 1 ? "_\(converted)" : "_(\(converted))"
    }

    private static let subMap: [Character: String] = [
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
        "n": "ₙ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ", "m": "ₘ",
    ]

    // MARK: - Symbol table  (longer commands first)

    private static let symbolTable: [(String, String)] = [
        // Dots
        ("\\cdots", "⋯"), ("\\ldots", "…"), ("\\vdots", "⋮"), ("\\ddots", "⋱"),
        // Comparison (long forms first)
        ("\\leqslant", "≤"), ("\\geqslant", "≥"),
        ("\\leq", "≤"), ("\\le", "≤"), ("\\geq", "≥"), ("\\ge", "≥"),
        ("\\neq", "≠"), ("\\ne", "≠"), ("\\approx", "≈"), ("\\equiv", "≡"),
        ("\\sim", "∼"), ("\\simeq", "≃"), ("\\cong", "≅"), ("\\propto", "∝"),
        ("\\ll", "≪"), ("\\gg", "≫"),
        // Arrows
        ("\\leftarrow", "←"), ("\\rightarrow", "→"), ("\\leftrightarrow", "↔"),
        ("\\Leftarrow", "⇐"), ("\\Rightarrow", "⇒"), ("\\Leftrightarrow", "⇔"),
        ("\\uparrow", "↑"), ("\\downarrow", "↓"), ("\\updownarrow", "↕"),
        ("\\Uparrow", "⇑"), ("\\Downarrow", "⇓"), ("\\to", "→"),
        ("\\mapsto", "↦"), ("\\hookrightarrow", "↪"), ("\\nearrow", "↗"),
        // Delimiters
        ("\\langle", "⟨"), ("\\rangle", "⟩"),
        ("\\lfloor", "⌊"), ("\\rfloor", "⌋"),
        ("\\lceil", "⌈"), ("\\rceil", "⌉"),
        ("\\{", "{"), ("\\}", "}"),
        // Set / logic
        ("\\subseteq", "⊆"), ("\\supseteq", "⊇"), ("\\subset", "⊂"), ("\\supset", "⊃"),
        ("\\bigcup", "⋃"), ("\\bigcap", "⋂"), ("\\cup", "∪"), ("\\cap", "∩"),
        ("\\bigvee", "⋁"), ("\\bigwedge", "⋀"),
        ("\\notin", "∉"), ("\\in", "∈"),
        ("\\emptyset", "∅"), ("\\varnothing", "∅"),
        ("\\forall", "∀"), ("\\exists", "∃"), ("\\nexists", "∄"),
        ("\\wedge", "∧"), ("\\land", "∧"), ("\\vee", "∨"), ("\\lor", "∨"),
        ("\\lnot", "¬"), ("\\neg", "¬"),
        // Operators
        ("\\oplus", "⊕"), ("\\ominus", "⊖"), ("\\otimes", "⊗"), ("\\odot", "⊙"),
        ("\\cdot", "·"), ("\\times", "×"), ("\\div", "÷"),
        ("\\pm", "±"), ("\\mp", "∓"),
        ("\\oint", "∮"), ("\\int", "∫"), ("\\sum", "∑"), ("\\prod", "∏"),
        ("\\partial", "∂"), ("\\nabla", "∇"), ("\\infty", "∞"),
        ("\\perp", "⊥"), ("\\parallel", "∥"),
        ("\\angle", "∠"), ("\\triangle", "△"),
        // Misc
        ("\\ell", "ℓ"), ("\\hbar", "ℏ"), ("\\Re", "ℜ"), ("\\Im", "ℑ"),
        ("\\sqrt", "√"),  // bare \sqrt without braces
        // Greek lowercase
        ("\\alpha", "α"), ("\\beta", "β"), ("\\gamma", "γ"), ("\\delta", "δ"),
        ("\\varepsilon", "ε"), ("\\epsilon", "ε"), ("\\zeta", "ζ"), ("\\eta", "η"),
        ("\\vartheta", "ϑ"), ("\\theta", "θ"), ("\\iota", "ι"), ("\\kappa", "κ"),
        ("\\lambda", "λ"), ("\\mu", "μ"), ("\\nu", "ν"), ("\\xi", "ξ"),
        ("\\varpi", "ϖ"), ("\\pi", "π"), ("\\varrho", "ϱ"), ("\\rho", "ρ"),
        ("\\varsigma", "ς"), ("\\sigma", "σ"), ("\\tau", "τ"),
        ("\\upsilon", "υ"), ("\\varphi", "ϕ"), ("\\phi", "φ"), ("\\chi", "χ"),
        ("\\psi", "ψ"), ("\\omega", "ω"),
        // Greek uppercase
        ("\\Gamma", "Γ"), ("\\Delta", "Δ"), ("\\Theta", "Θ"), ("\\Lambda", "Λ"),
        ("\\Xi", "Ξ"), ("\\Pi", "Π"), ("\\Sigma", "Σ"), ("\\Upsilon", "Υ"),
        ("\\Phi", "Φ"), ("\\Psi", "Ψ"), ("\\Omega", "Ω"),
    ]

    // MARK: - Regex helper

    /// Replace all non-overlapping matches of `pattern`, transforming capture group 1 via `transform`.
    private static func mapCaptures(
        in text: String,
        pattern: String,
        dotAll: Bool,
        transform: (String) -> String
    ) -> String {
        mapCaptures(in: text, pattern: pattern, dotAll: dotAll, captureCount: 1) { transform($0[0]) }
    }

    /// Multi-capture-group variant. `captureCount` must equal the number of groups.
    private static func mapCaptures(
        in text: String,
        pattern: String,
        dotAll: Bool,
        captureCount: Int,
        transform: ([String]) -> String
    ) -> String {
        var opts: NSRegularExpression.Options = []
        if dotAll { opts.insert(.dotMatchesLineSeparators) }
        guard let regex = try? NSRegularExpression(pattern: pattern, options: opts) else { return text }
        let ns = text as NSString
        var result = text
        for match in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).reversed() {
            guard match.numberOfRanges >= captureCount + 1 else { continue }
            var captures: [String] = []
            for i in 1...captureCount {
                let r = match.range(at: i)
                captures.append(r.location != NSNotFound ? ns.substring(with: r) : "")
            }
            let rep = transform(captures)
            if let r = Range(match.range(at: 0), in: result) { result.replaceSubrange(r, with: rep) }
        }
        return result
    }
}
