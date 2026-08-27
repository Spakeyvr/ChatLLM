//
//  Message.swift
//  ChatLLM
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import SwiftData

// Pre-compiled regexes used in Message display and parsing (compiled once at app launch)
// swiftlint:disable force_try
private let _thinkingTagRegex = try! NSRegularExpression(
    pattern: #"<(?:think|thinking)>[\s\S]*?</(?:think|thinking)>\s*"#, options: [.caseInsensitive])
private let _answerTagDisplayRegex = try! NSRegularExpression(
    pattern: #"<answer>([\s\S]*?)</answer>"#, options: [.caseInsensitive])
private let _hiddenImageContextRegex = try! NSRegularExpression(
    pattern: #"\[HIDDEN_IMAGE_CONTEXT\].*?\[/HIDDEN_IMAGE_CONTEXT\]"#, options: [.dotMatchesLineSeparators])
private let _imageAnalysisBlockRegex = try! NSRegularExpression(
    pattern: #"--- Image Analysis Data ---.*?(?:--- End Image Analysis ---\s*User's question:\s*|--- User Question ---\n?)"#,
    options: [.dotMatchesLineSeparators])
private let _numberedStepRegex = try! NSRegularExpression(
    pattern: #"(?:^|\n)(?:(?:Step\s*)?(\d+)[:.\)]\s*([^\n]*)\n?)"#, options: [.caseInsensitive])
private let _headingStepRegex = try! NSRegularExpression(
    pattern: #"(?:^|\n)(#{1,3})\s+([^\n]+)\n"#, options: [])
// swiftlint:enable force_try

// MARK: - Reasoning Step Model

/// Represents a single step in the Chain of Thought reasoning process
struct ReasoningStep: Codable, Sendable, Identifiable {
    var id: UUID
    var stepNumber: Int
    var title: String?
    var content: String
    var timestamp: Date
    
    init(id: UUID = UUID(), stepNumber: Int, title: String? = nil, content: String, timestamp: Date = Date()) {
        self.id = id
        self.stepNumber = stepNumber
        self.title = title
        self.content = content
        self.timestamp = timestamp
    }
}

enum ReasoningStreamPhase: String, Sendable {
    case initialThinking
    case postToolReasoning
    case finalAnswer
}

enum MessageRole: String, Codable, CaseIterable, Sendable {
    case system
    case user
    case assistant
    
    // Performance optimization: Pre-computed localized display names
    var displayName: String {
        switch self {
        case .system: return String(localized: "System")
        case .user: return String(localized: "User")
        case .assistant: return String(localized: "Assistant")
        }
    }
    
    // Performance optimization: Quick role checks without string comparisons
    var isUser: Bool { self == .user }
    var isAssistant: Bool { self == .assistant }
    var isSystem: Bool { self == .system }
}

@Model
final class Message {
    // Performance: Order properties by frequency of access and memory alignment
    @Attribute(.unique) var id: UUID
    var order: Int
    var role: MessageRole
    var isFinal: Bool
    var isReasoningMode: Bool
    var createdAt: Date
    
    // String properties grouped together for better memory locality
    var text: String
    var reasoning: String?
    var finalAnswer: String?
    var promptSnapshot: String?
    var searchQuery: String?  // Stores the user's query when web search was used
    var requiresWebSearch: Bool?
    var rawText: String?
    var generationBackend: String?
    var generationModelName: String?
    var generationStartedAt: Date?
    var generationCompletedAt: Date?
    var reasoningStartedAt: Date?
    var reasoningCompletedAt: Date?
    @Transient var generationError: String?  // ephemeral; not persisted
    @Transient var streamingReasoningPhase: ReasoningStreamPhase?
    @Transient var postToolReasoningStartCloseTagCount: Int?

    // All web search invocations stored as JSON for SwiftData compatibility
    var searchInvocationsJSON: String?

    // Chain of Thought steps stored as JSON string for SwiftData compatibility
    private var reasoningStepsJSON: String?
    
    // Computed property to access reasoning steps
    @Transient
    var reasoningSteps: [ReasoningStep]? {
        get {
            guard let json = reasoningStepsJSON else { return nil }
            if json == _cachedReasoningStepsJSON { return _cachedReasoningSteps }
            _cachedReasoningStepsJSON = json
            _cachedReasoningSteps = try? JSONDecoder().decode([ReasoningStep].self, from: Data(json.utf8))
            return _cachedReasoningSteps
        }
        set {
            if let steps = newValue,
               let data = try? JSONEncoder().encode(steps),
               let json = String(data: data, encoding: .utf8) {
                reasoningStepsJSON = json
            } else {
                reasoningStepsJSON = nil
            }
            _cachedReasoningStepsJSON = nil
            _cachedReasoningSteps = nil
        }
    }
    
    // Computed property to access search invocations
    @Transient
    var searchInvocations: [SearchInvocation]? {
        get {
            guard let json = searchInvocationsJSON else { return nil }
            if json == _cachedSearchInvocationsJSON { return _cachedSearchInvocations }
            _cachedSearchInvocationsJSON = json
            _cachedSearchInvocations = try? JSONDecoder().decode([SearchInvocation].self, from: Data(json.utf8))
            return _cachedSearchInvocations
        }
        set {
            if let invocations = newValue, !invocations.isEmpty,
               let data = try? JSONEncoder().encode(invocations),
               let json = String(data: data, encoding: .utf8) {
                searchInvocationsJSON = json
                // Backward compat: set searchQuery to first query
                searchQuery = invocations.first?.query
            } else {
                searchInvocationsJSON = nil
                searchQuery = nil
            }
            _cachedSearchInvocationsJSON = nil
            _cachedSearchInvocations = nil
        }
    }

    // Relationships last to minimize impact on property access
    @Relationship var conversation: Conversation?
    @Relationship(deleteRule: .cascade, inverse: \MessageAttachment.message)
    var attachments: [MessageAttachment]

    // MARK: - Performance-optimized computed properties
    
    // Cached values for expensive computations
    @Transient private var _cachedDisplayText: String?
    @Transient private var _cachedContentLength: Int?
    @Transient private var _lastDisplayTextHash: Int?
    @Transient private var _lastContentLengthHash: Int?

    // Cached decoded values for JSON-backed computed properties (not persisted)
    @Transient private var _cachedReasoningSteps: [ReasoningStep]?
    @Transient private var _cachedReasoningStepsJSON: String?
    @Transient private var _cachedSearchInvocations: [SearchInvocation]?
    @Transient private var _cachedSearchInvocationsJSON: String?
    
    /// Quick role checks without enum comparison overhead
    var isUser: Bool { role.isUser }
    var isAssistant: Bool { role.isAssistant }
    var isSystem: Bool { role.isSystem }
    
    /// Optimized text content access for display with caching
    var displayText: String {
        // CRITICAL FIX (Bug #3): Include ALL relevant fields in hash, not just text and finalAnswer
        let textHash = text.hashValue
        let reasoningHash = reasoning?.hashValue ?? 0
        let finalAnswerHash = finalAnswer?.hashValue ?? 0
        // Use a more collision-resistant combination than XOR
        let combinedHash = textHash &+ (reasoningHash &* 31) &+ (finalAnswerHash &* 17)
        
        if let cached = _cachedDisplayText, _lastDisplayTextHash == combinedHash {
            return cached
        }
        
        var result: String
        if isReasoningMode && isAssistant {
            // In reasoning mode, always prefer finalAnswer if it exists, even if it only contains sources
            // The UI will properly extract and display sources from finalAnswer
            result = finalAnswer ?? text
        } else {
            result = text
        }
        
        // CRITICAL FIX (Bug #6): Always strip raw <thinking> tags from display
        let lowercasedResult = result.lowercased()
        if lowercasedResult.contains("<thinking>") || lowercasedResult.contains("<think>") {
            let range = NSRange(result.startIndex..<result.endIndex, in: result)
            result = _thinkingTagRegex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if result.lowercased().hasPrefix("final answer:") {
                result = String(result.dropFirst("final answer:".count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        // Strip <answer> tags from display (Qwen3 sometimes wraps responses in these)
        if result.contains("<answer>") || result.contains("<Answer>") || result.contains("<ANSWER>") {
            let range = NSRange(result.startIndex..<result.endIndex, in: result)
            result = _answerTagDisplayRegex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "$1")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Strip hidden image analysis context from display
        if result.contains("[HIDDEN_IMAGE_CONTEXT]") {
            let range = NSRange(result.startIndex..<result.endIndex, in: result)
            result = _hiddenImageContextRegex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Strip "--- Image Analysis Data ---" block from display
        if result.contains("--- Image Analysis Data ---") {
            let range = NSRange(result.startIndex..<result.endIndex, in: result)
            result = _imageAnalysisBlockRegex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if result.hasSuffix("Please respond based on the image analysis data above.") {
                result = result.replacingOccurrences(of: "\n\nPlease respond based on the image analysis data above.", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        
        _cachedDisplayText = result
        _lastDisplayTextHash = combinedHash
        return result
    }

    var userVisibleText: String {
        let text = displayText
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        return SharedRegexes.sourcesBlock.stringByReplacingMatches(
            in: text,
            options: [],
            range: range,
            withTemplate: ""
        )
        .trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    /// Optimized content length calculation with caching
    /// CRITICAL FIX (Bug 3 & 6): Use consistent hash and return display-visible length
    var contentLength: Int {
        // Use the SAME hash calculation as displayText for consistency
        let textHash = text.hashValue
        let reasoningHash = reasoning?.hashValue ?? 0
        let finalAnswerHash = finalAnswer?.hashValue ?? 0
        // Match displayText's hash calculation exactly
        let combinedHash = textHash &+ (reasoningHash &* 31) &+ (finalAnswerHash &* 17)
        
        if let cached = _cachedContentLength, _lastContentLengthHash == combinedHash {
            return cached
        }
        
        // CRITICAL FIX (Bug 6): Return display-visible length, not raw storage length
        // This ensures UI sizing is based on what the user actually sees
        let result = displayText.count
        _cachedContentLength = result
        _lastContentLengthHash = combinedHash
        return result
    }
    
    /// Fast isEmpty check without string operations
    var hasContent: Bool {
        !text.isEmpty || !(reasoning?.isEmpty ?? true) || !(finalAnswer?.isEmpty ?? true)
    }
    
    /// Invalidate caches when content changes
    private func invalidateCache() {
        _cachedDisplayText = nil
        _cachedContentLength = nil
        _lastDisplayTextHash = nil
        _lastContentLengthHash = nil
    }
    
    // MARK: - Optimized Initializers
    
    /// Primary initializer with performance optimizations
    init(
        id: UUID = UUID(),
        role: MessageRole,
        text: String,
        createdAt: Date = Date(),
        order: Int,
        conversation: Conversation? = nil,
        attachments: [MessageAttachment] = [],
        promptSnapshot: String? = nil,
        isFinal: Bool = false,
        reasoning: String? = nil,
        finalAnswer: String? = nil,
        isReasoningMode: Bool = false,
        reasoningSteps: [ReasoningStep]? = nil,
        searchQuery: String? = nil,
        requiresWebSearch: Bool? = nil,
        rawText: String? = nil,
        generationBackend: String? = nil,
        generationModelName: String? = nil,
        generationStartedAt: Date? = nil,
        generationCompletedAt: Date? = nil,
        reasoningStartedAt: Date? = nil,
        reasoningCompletedAt: Date? = nil
    ) {
        self.id = id
        self.order = order
        self.role = role
        self.isFinal = isFinal
        self.isReasoningMode = isReasoningMode
        self.createdAt = createdAt
        self.text = text
        self.reasoning = reasoning
        self.finalAnswer = finalAnswer
        self.promptSnapshot = promptSnapshot
        self.searchQuery = searchQuery
        self.requiresWebSearch = requiresWebSearch
        self.rawText = rawText
        self.generationBackend = generationBackend
        self.generationModelName = generationModelName
        self.generationStartedAt = generationStartedAt
        self.generationCompletedAt = generationCompletedAt
        self.reasoningStartedAt = reasoningStartedAt
        self.reasoningCompletedAt = reasoningCompletedAt
        self.conversation = conversation
        self.attachments = attachments
        self.reasoningSteps = reasoningSteps
    }
    
    /// Fast initializer for user messages (most common case)
    convenience init(userText: String, order: Int, conversation: Conversation) {
        self.init(
            role: .user,
            text: userText,
            order: order,
            conversation: conversation,
            isFinal: true
        )
    }
    
    /// Fast initializer for assistant messages
    convenience init(assistantText: String = "", order: Int, conversation: Conversation, isReasoningMode: Bool = false) {
        self.init(
            role: .assistant,
            text: assistantText,
            order: order,
            conversation: conversation,
            isFinal: false,
            isReasoningMode: isReasoningMode
        )
    }
    
    /// Fast initializer for system messages
    convenience init(systemPrompt: String, conversation: Conversation) {
        self.init(
            role: .system,
            text: systemPrompt,
            order: 0,
            conversation: conversation,
            isFinal: true
        )
    }
}

// MARK: - Performance Extensions

extension Message {
    nonisolated static let maximumCapturedRawTextCharacters = 32_768

    var developerRawText: String {
        if let rawText, !rawText.isEmpty {
            return rawText
        }
        if isReasoningMode {
            return finalAnswer ?? text
        }
        return text
    }

    var generationDuration: TimeInterval? {
        guard let generationStartedAt, let generationCompletedAt else { return nil }
        let duration = generationCompletedAt.timeIntervalSince(generationStartedAt)
        return duration > 0 ? duration : nil
    }

    var reasoningDuration: TimeInterval? {
        guard let reasoningStartedAt, let reasoningCompletedAt else { return nil }
        let duration = reasoningCompletedAt.timeIntervalSince(reasoningStartedAt)
        return duration > 0 ? duration : nil
    }

    var estimatedOutputTokenCount: Int? {
        let raw = developerRawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return nil }
        // A lightweight fallback for developer diagnostics when the backend does not expose true token counts.
        return max(1, Int(ceil(Double(raw.count) / 4.0)))
    }

    var estimatedTokensPerSecond: Double? {
        guard let duration = generationDuration,
              let estimatedOutputTokenCount,
              duration > 0 else { return nil }
        return Double(estimatedOutputTokenCount) / duration
    }

    func beginGenerationCapture(backend: String, modelName: String?, startedAt: Date = Date()) {
        rawText = nil
        generationBackend = backend
        generationModelName = modelName
        generationStartedAt = startedAt
        generationCompletedAt = nil
        reasoningStartedAt = isReasoningMode ? startedAt : nil
        reasoningCompletedAt = nil
        streamingReasoningPhase = isReasoningMode ? .initialThinking : nil
        postToolReasoningStartCloseTagCount = nil
    }

    func completeReasoningCapture(completedAt: Date = Date()) {
        guard isReasoningMode,
              reasoningStartedAt != nil,
              reasoningCompletedAt == nil else { return }
        reasoningCompletedAt = completedAt
    }

    func completeGenerationCapture(
        rawText: String,
        captureRawText: Bool,
        completedAt: Date = Date()
    ) {
        self.rawText = captureRawText
            ? String(rawText.prefix(Self.maximumCapturedRawTextCharacters))
            : nil
        generationCompletedAt = completedAt
        if !(reasoning?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true) {
            completeReasoningCapture(completedAt: completedAt)
        }
    }

    /// Efficient bulk text update for streaming with conditional invalidation
    func updateStreamingContent(_ newText: String, reasoning: String? = nil, finalAnswer: String? = nil) {
        // Only invalidate cache if content actually changed
        let contentChanged = self.text != newText ||
                           self.reasoning != reasoning ||
                           self.finalAnswer != finalAnswer
        
        if contentChanged {
            invalidateCache()
            
            // Batch property updates to minimize SwiftData notifications
            self.text = newText
            if let reasoning = reasoning {
                self.reasoning = reasoning
            }
            if let finalAnswer = finalAnswer {
                self.finalAnswer = finalAnswer
            }
        }
    }
    
    /// Mark as complete and finalize
    func markAsComplete() {
        guard !self.isFinal else { return } // Skip if already final
        self.isFinal = true
        invalidateCache()
    }
    
    /// Reset message state before regeneration so the next run starts clean.
    func resetForRegeneration() {
        let preservedRequiresWebSearch = requiresWebSearch
        let preservedForcedSearchQuery = requiresWebSearch == true ? searchQuery : nil

        // Only clear if needed
        let needsReset = !self.text.isEmpty ||
                        self.reasoning != nil ||
            self.finalAnswer != nil ||
            self.isFinal ||
            self.generationError != nil ||
            self.rawText != nil ||
            self.generationBackend != nil ||
            self.generationModelName != nil ||
            self.generationStartedAt != nil ||
            self.generationCompletedAt != nil ||
            self.reasoningStartedAt != nil ||
            self.reasoningCompletedAt != nil ||
            self.searchInvocations != nil ||
            self.searchQuery != nil

        if needsReset {
            invalidateCache()
            self.text = ""
            self.reasoning = nil
            self.finalAnswer = nil
            self.isFinal = false
            self.generationError = nil
            self.rawText = nil
            self.generationBackend = nil
            self.generationModelName = nil
            self.generationStartedAt = nil
            self.generationCompletedAt = nil
            self.reasoningStartedAt = nil
            self.reasoningCompletedAt = nil
            self.streamingReasoningPhase = nil
            self.postToolReasoningStartCloseTagCount = nil
            self.searchInvocations = nil
            self.requiresWebSearch = preservedRequiresWebSearch
            self.searchQuery = preservedForcedSearchQuery
        }
    }
}

enum ThoughtDurationFormatter {
    nonisolated static func string(for duration: TimeInterval) -> String {
        let totalSeconds = max(1, Int(duration.rounded()))
        let hours = totalSeconds / 3_600
        let minutes = (totalSeconds % 3_600) / 60
        let seconds = totalSeconds % 60

        if hours > 0 {
            let hourText = unit(hours, singular: "hour", plural: "hours")
            if minutes > 0 {
                return "\(hourText) and \(unit(minutes, singular: "minute", plural: "minutes"))"
            }
            guard seconds > 0 else { return hourText }
            return "\(hourText) and \(unit(seconds, singular: "second", plural: "seconds"))"
        }

        if minutes > 0 {
            let minuteText = unit(minutes, singular: "minute", plural: "minutes")
            guard seconds > 0 else { return minuteText }
            return "\(minuteText) and \(unit(seconds, singular: "second", plural: "seconds"))"
        }

        return unit(seconds, singular: "second", plural: "seconds")
    }

    nonisolated private static func unit(_ value: Int, singular: String, plural: String) -> String {
        "\(value) \(value == 1 ? singular : plural)"
    }
}

enum ReasoningTextSanitizer {
    nonisolated static func string(from text: String) -> String {
        let stripped = text
            .replacingOccurrences(of: "<think>", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "</think>", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "<thinking>", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "</thinking>", with: "", options: .caseInsensitive)
        return stripped
            .replacingOccurrences(
                of: #"\n[ \t]*\n(?:[ \t]*\n)+"#,
                with: "\n\n",
                options: .regularExpression
            )
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}



// MARK: - Identity-based Equality (SwiftData handles Hashable automatically)

// Note: SwiftData's @Model macro automatically provides Sendable conformance

// MARK: - Extensions

extension Message {
    /// Debug description for reasoning state
    var reasoningDebugDescription: String {
        var parts: [String] = []
        parts.append("isReasoningMode: \(isReasoningMode)")
        parts.append("hasReasoning: \(reasoning != nil && !reasoning!.isEmpty)")
        parts.append("hasFinalAnswer: \(finalAnswer != nil && !finalAnswer!.isEmpty)")
        parts.append("text.count: \(text.count)")
        return "[\(parts.joined(separator: ", "))]"
    }
}

// MARK: - Collection helpers for Message

extension Array where Element == Message {
    /// Returns messages sorted by `order`, breaking ties with `createdAt`.
    var sortedByOrder: [Message] {
        self.sorted {
            if $0.order == $1.order {
                return $0.createdAt < $1.createdAt
            }
            return $0.order < $1.order
        }
    }
}

// MARK: - Chain of Thought Parsing

extension Message {
    /// Parse reasoning text into structured CoT steps
    /// Supports multiple formats:
    /// - Numbered lists (1., 2., Step 1:, etc.)
    /// - Titled sections (## Step, ### Analysis, etc.)
    /// - Paragraph-based reasoning (split by double newlines)
    func parseReasoningIntoSteps(from reasoningText: String) -> [ReasoningStep] {
        let trimmed = reasoningText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        
        var steps: [ReasoningStep] = []
        var currentStepNumber = 1
        
        // Strategy 1: Try to detect numbered steps (1., 2., Step 1:, etc.)
        let regex = _numberedStepRegex
        do {
            let nsString = trimmed as NSString
            let matches = regex.matches(in: trimmed, options: [], range: NSRange(location: 0, length: nsString.length))
            
            if !matches.isEmpty {
                for (index, match) in matches.enumerated() {
                    let fullRange = match.range
                    let titleRange = match.numberOfRanges > 2 ? match.range(at: 2) : NSRange(location: 0, length: 0)
                    
                    // Extract content between this match and the next (or end of string)
                    let contentStart = fullRange.location + fullRange.length
                    let contentEnd = index < matches.count - 1 ? matches[index + 1].range.location : nsString.length
                    
                    var content = ""
                    if contentStart < contentEnd {
                        content = nsString.substring(with: NSRange(location: contentStart, length: contentEnd - contentStart))
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                    
                    // Get title if present
                    let title = titleRange.length > 0 ? nsString.substring(with: titleRange).trimmingCharacters(in: .whitespacesAndNewlines) : nil
                    
                    // If content is empty, the title might actually be the content
                    if content.isEmpty && !(title?.isEmpty ?? true) {
                        content = title ?? ""
                    }
                    
                    if !content.isEmpty {
                        steps.append(ReasoningStep(
                            stepNumber: currentStepNumber,
                            title: title?.isEmpty == false ? title : nil,
                            content: content
                        ))
                        currentStepNumber += 1
                    }
                }
                
                if !steps.isEmpty {
                    return steps
                }
            }
        }
        
        // Strategy 2: Try markdown headings (## Step, ### Analysis)
        do {
            let regex = _headingStepRegex
            let nsString = trimmed as NSString
            let matches = regex.matches(in: trimmed, options: [], range: NSRange(location: 0, length: nsString.length))
            
            if !matches.isEmpty {
                for (index, match) in matches.enumerated() {
                    let fullRange = match.range
                    let titleRange = match.range(at: 2)
                    
                    let title = nsString.substring(with: titleRange)
                    
                    // Extract content until next heading
                    let contentStart = fullRange.location + fullRange.length
                    let contentEnd = index < matches.count - 1 ? matches[index + 1].range.location : nsString.length
                    
                    var content = ""
                    if contentStart < contentEnd {
                        content = nsString.substring(with: NSRange(location: contentStart, length: contentEnd - contentStart))
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                    
                    if !content.isEmpty {
                        steps.append(ReasoningStep(
                            stepNumber: currentStepNumber,
                            title: title,
                            content: content
                        ))
                        currentStepNumber += 1
                    }
                }
                
                if !steps.isEmpty {
                    return steps
                }
            }
        }
        
        // Strategy 3: Split by double newlines (paragraphs)
        let paragraphs = trimmed.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        
        if paragraphs.count > 1 {
            for paragraph in paragraphs {
                // Try to extract a title from the first line if it looks like one
                let lines = paragraph.components(separatedBy: "\n")
                let firstLine = lines.first ?? ""
                
                // Check if first line is short and looks like a title (ends with : or is < 60 chars and one line)
                let isTitle = firstLine.count < 60 && (firstLine.hasSuffix(":") || lines.count > 1)
                
                let title = isTitle ? firstLine.trimmingCharacters(in: CharacterSet(charactersIn: ":")) : nil
                let content = isTitle && lines.count > 1 ? lines.dropFirst().joined(separator: "\n") : paragraph
                
                steps.append(ReasoningStep(
                    stepNumber: currentStepNumber,
                    title: title,
                    content: content
                ))
                currentStepNumber += 1
            }
            
            if !steps.isEmpty {
                return steps
            }
        }
        
        // Fallback: Single step with all content
        return [ReasoningStep(stepNumber: 1, title: nil, content: trimmed)]
    }
    
    /// Update or create reasoning steps from current reasoning text
    func updateReasoningSteps() {
        guard let reasoningText = self.reasoning, !reasoningText.isEmpty else {
            self.reasoningSteps = nil
            return
        }
        
        self.reasoningSteps = parseReasoningIntoSteps(from: reasoningText)
    }
}
