//
//  WrappedXMLToolCallStreamFilter.swift
//  On-Device_LLM_Chat
//
//  Filters wrapped XML tool-call markup out of streamed MLX output so the UI
//  only sees user-visible assistant text while tool invocations are dispatched
//  separately.
//

import Foundation

actor WrappedXMLToolCallStreamFilter {
    private static let toolMarkupMarkers = [
        "<tool_call>",
        "<function=",
        "<parameter=",
        "</parameter>",
        "</function>",
        "</tool_call>"
    ]

    private var pendingText = ""
    private var suppressingToolMarkup = false

    func consume(_ chunk: String) -> String? {
        guard !chunk.isEmpty else { return nil }

        if suppressingToolMarkup {
            return nil
        }

        pendingText += chunk

        if let markerRange = Self.firstMarkerRange(in: pendingText) {
            let visiblePrefix = String(pendingText[..<markerRange.lowerBound])
            pendingText = ""
            suppressingToolMarkup = true
            return visiblePrefix.isEmpty ? nil : visiblePrefix
        }

        let safePrefixLength = Self.safePrefixLength(in: pendingText)
        guard safePrefixLength > 0 else {
            return nil
        }

        let safeEnd = pendingText.index(pendingText.startIndex, offsetBy: safePrefixLength)
        let safeChunk = String(pendingText[..<safeEnd])
        pendingText = String(pendingText[safeEnd...])
        return safeChunk.isEmpty ? nil : safeChunk
    }

    func didDispatchToolCall() {
        pendingText = ""
        suppressingToolMarkup = false
    }

    func finish() -> String? {
        defer {
            pendingText = ""
            suppressingToolMarkup = false
        }

        guard !suppressingToolMarkup, !pendingText.isEmpty else {
            return nil
        }

        return pendingText
    }

    private static func firstMarkerRange(in text: String) -> Range<String.Index>? {
        toolMarkupMarkers
            .compactMap { marker in text.range(of: marker) }
            .min { $0.lowerBound < $1.lowerBound }
    }

    private static func safePrefixLength(in text: String) -> Int {
        guard text.contains("<") else {
            return text.count
        }

        let suffixLength = longestMarkerPrefixSuffix(in: text)
        return text.count - suffixLength
    }

    private static func longestMarkerPrefixSuffix(in text: String) -> Int {
        let maxSuffixLength = min(
            text.count,
            toolMarkupMarkers.map(\.count).max() ?? 0
        )
        guard maxSuffixLength > 0 else {
            return 0
        }

        for suffixLength in stride(from: maxSuffixLength, through: 1, by: -1) {
            let suffixStart = text.index(text.endIndex, offsetBy: -suffixLength)
            let suffix = String(text[suffixStart...])
            if toolMarkupMarkers.contains(where: { $0.hasPrefix(suffix) }) {
                return suffixLength
            }
        }

        return 0
    }
}
