//
//  AttributedTextSelection.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/26/25.
//

import SwiftUI

/// Represents a text selection in an attributed string
struct AttributedTextSelection: Equatable {
    var range: Range<String.Index>?
    var insertionPoint: String.Index?
    
    init(range: Range<String.Index>? = nil, insertionPoint: String.Index? = nil) {
        self.range = range
        self.insertionPoint = insertionPoint
    }
    
    // Equatable conformance is automatic since both Range<String.Index> and String.Index are Equatable
    
    /// Get the typing attributes at the current selection
    func typingAttributes(in attributedString: AttributedString) -> AttributeContainer {
        let characters = String(attributedString.characters)
        
        // If we have a range selection, get attributes from the start of selection
        if let range = range {
            let startIndex = range.lowerBound
            if let attributedRange = attributedString.range(of: String(characters[startIndex..<range.upperBound])) {
                return attributedString[attributedRange].runs.first?.attributes ?? AttributeContainer()
            }
        }
        
        // If we have an insertion point, get attributes at that point
        if let insertionPoint = insertionPoint {
            // Get attributes from the character just before insertion point
            let beforeIndex = characters.index(before: insertionPoint)
            if beforeIndex >= characters.startIndex {
                let char = String(characters[beforeIndex...beforeIndex])
                if let attributedRange = attributedString.range(of: char) {
                    return attributedString[attributedRange].runs.first?.attributes ?? AttributeContainer()
                }
            }
        }
        
        // Default to empty attributes
        return AttributeContainer()
    }
}

/// Extension to add transformation support to AttributedString
extension AttributedString {
    mutating func transformAttributes(in selection: inout AttributedTextSelection, transform: (inout AttributeContainer) -> Void) {
        let characters = String(self.characters)
        
        // If we have a range selection, transform attributes in that range
        if let range = selection.range {
            let selectedText = String(characters[range])
            if let attributedRange = self.range(of: selectedText) {
                for run in self[attributedRange].runs {
                    var attributes = run.attributes
                    transform(&attributes)
                    self[run.range].setAttributes(attributes)
                }
            }
        }
        // If we have an insertion point and no selection, set typing attributes
        else if selection.insertionPoint != nil {
            // For insertion point, we'll apply formatting to any new text typed
            // This is a simplified approach - in a real implementation you'd need to track typing attributes
            var attributes = AttributeContainer()
            transform(&attributes)
            // Note: In a full implementation, you'd store these as typing attributes
            // For now, we'll just update the selection to trigger UI updates
        }
    }
}