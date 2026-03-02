//
//  Message.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import SwiftData

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

    // All web search invocations stored as JSON for SwiftData compatibility
    var searchInvocationsJSON: String?

    // Chain of Thought steps stored as JSON string for SwiftData compatibility
    private var reasoningStepsJSON: String?
    
    // Computed property to access reasoning steps
    @Transient
    var reasoningSteps: [ReasoningStep]? {
        get {
            guard let json = reasoningStepsJSON,
                  let data = json.data(using: .utf8) else {
                return nil
            }
            return try? JSONDecoder().decode([ReasoningStep].self, from: data)
        }
        set {
            if let steps = newValue,
               let data = try? JSONEncoder().encode(steps),
               let json = String(data: data, encoding: .utf8) {
                reasoningStepsJSON = json
            } else {
                reasoningStepsJSON = nil
            }
        }
    }
    
    // Computed property to access search invocations
    @Transient
    var searchInvocations: [SearchInvocation]? {
        get {
            guard let json = searchInvocationsJSON,
                  let data = json.data(using: .utf8) else {
                return nil
            }
            return try? JSONDecoder().decode([SearchInvocation].self, from: data)
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
            }
        }
    }

    // Relationships last to minimize impact on property access
    @Relationship var conversation: Conversation?
    @Relationship(deleteRule: .cascade, inverse: \MessageAttachment.message)
    var attachments: [MessageAttachment]

    // MARK: - Performance-optimized computed properties
    
    // Cached values for expensive computations
    private var _cachedDisplayText: String?
    private var _cachedContentLength: Int?
    private var _lastTextHash: Int?
    
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
        
        if let cached = _cachedDisplayText, _lastTextHash == combinedHash {
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
        // This handles cases where isReasoningMode is false but text contains thinking tags
        if result.contains("<thinking>") || result.contains("<THINKING>") {
            // Pattern to match thinking tags and their content (case insensitive)
            let thinkingPattern = #"<thinking>[\s\S]*?</thinking>\s*"#
            if let re = try? NSRegularExpression(pattern: thinkingPattern, options: [.caseInsensitive]) {
                let range = NSRange(result.startIndex..<result.endIndex, in: result)
                result = re.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
            // Also remove "Final answer:" prefix if present after stripping thinking
            if result.lowercased().hasPrefix("final answer:") {
                result = String(result.dropFirst("final answer:".count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        
        // Strip <answer> tags from display (Qwen3 sometimes wraps responses in these)
        if result.contains("<answer>") || result.contains("<Answer>") || result.contains("<ANSWER>") {
            let answerPattern = #"<answer>([\s\S]*?)</answer>"#
            if let re = try? NSRegularExpression(pattern: answerPattern, options: [.caseInsensitive]) {
                let range = NSRange(result.startIndex..<result.endIndex, in: result)
                result = re.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "$1")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        // Strip hidden image analysis context from display
        // This keeps it in the stored text for model prompts but hides it from user view
        if result.contains("[HIDDEN_IMAGE_CONTEXT]") {
            let pattern = #"\[HIDDEN_IMAGE_CONTEXT\].*?\[/HIDDEN_IMAGE_CONTEXT\]"#
            if let re = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) {
                let range = NSRange(result.startIndex..<result.endIndex, in: result)
                result = re.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        
        // Also strip the newer "--- Image Analysis Data ---" format from display
        // This format provides context to the LLM but should be hidden from users
        if result.contains("--- Image Analysis Data ---") {
            // Pattern matches the entire image analysis block including the user question prompt
            let pattern = #"--- Image Analysis Data ---.*?(?:--- End Image Analysis ---\s*User's question:\s*|--- User Question ---\n?)"#
            if let re = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) {
                let range = NSRange(result.startIndex..<result.endIndex, in: result)
                result = re.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
            // Also remove any trailing instruction about responding
            if result.hasSuffix("Please respond based on the image analysis data above.") {
                result = result.replacingOccurrences(of: "\n\nPlease respond based on the image analysis data above.", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        
        _cachedDisplayText = result
        _lastTextHash = combinedHash
        return result
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
        
        if let cached = _cachedContentLength, _lastTextHash == combinedHash {
            return cached
        }
        
        // CRITICAL FIX (Bug 6): Return display-visible length, not raw storage length
        // This ensures UI sizing is based on what the user actually sees
        let result = displayText.count
        _cachedContentLength = result
        // Note: We don't set _lastTextHash here since displayText already sets it
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
        _lastTextHash = nil
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
        searchQuery: String? = nil
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
    
    /// Reset for regeneration
    /// - Parameter preserveSources: If true, extracts and preserves any <sources> block for reinjection
    /// - Returns: The preserved sources content (if any) when preserveSources is true
    @discardableResult
    func resetForRegeneration(preserveSources: Bool = false) -> String? {
        // CRITICAL FIX (Bug 4): Optionally preserve sources from previous web search
        var savedSources: String? = nil
        if preserveSources {
            // Extract sources from wherever they might be stored
            let carrier = isReasoningMode ? (finalAnswer ?? text) : text
            let pattern = #"<sources>(.*?)</sources>"#
            if let re = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators, .caseInsensitive]) {
                let ns = carrier as NSString
                let range = NSRange(location: 0, length: ns.length)
                if let match = re.firstMatch(in: carrier, options: [], range: range), match.numberOfRanges >= 2 {
                    let inner = match.range(at: 1)
                    if inner.location != NSNotFound {
                        savedSources = ns.substring(with: inner).trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                }
            }
        }
        
        // Only clear if needed
        let needsReset = !self.text.isEmpty ||
                        self.reasoning != nil ||
                        self.finalAnswer != nil ||
                        self.isFinal
        
        if needsReset {
            invalidateCache()
            self.text = ""
            self.reasoning = nil
            self.finalAnswer = nil
            self.isFinal = false
        }
        
        return savedSources
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
        let numberedPattern = #"(?:^|\n)(?:(?:Step\s*)?(\d+)[:.\)]\s*([^\n]*)\n?)"#
        if let regex = try? NSRegularExpression(pattern: numberedPattern, options: [.caseInsensitive]) {
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
        let headingPattern = #"(?:^|\n)(#{1,3})\s+([^\n]+)\n"#
        if let regex = try? NSRegularExpression(pattern: headingPattern, options: []) {
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
