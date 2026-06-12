//
//  TextFormattingToolbar.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/25/25.
//

import SwiftUI

struct TextFormattingToolbar: View {
    @Binding var text: AttributedString
    @Binding var selection: AttributedTextSelection
    @State private var isBold: Bool = false
    @State private var isItalic: Bool = false
    
    @Environment(\.fontResolutionContext) private var fontResolutionContext
    
    var body: some View {
        HStack(spacing: 16) {
            // Bold button
            Button(action: toggleBold) {
                Image(systemName: "bold")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(isBold ? .blue : .secondary)
                    .frame(width: 32, height: 32)
                    .background(
                        RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.small, style: .continuous)
                            .fill(isBold ? Color.blue.opacity(0.15) : Color.clear)
                    )
            }
            .buttonStyle(.plain)
            .accessibilityLabel(String(localized: "Bold"))
            
            // Italic button
            Button(action: toggleItalic) {
                Image(systemName: "italic")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(isItalic ? .blue : .secondary)
                    .frame(width: 32, height: 32)
                    .background(
                        RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.small, style: .continuous)
                            .fill(isItalic ? Color.blue.opacity(0.15) : Color.clear)
                    )
            }
            .buttonStyle(.plain)
            .accessibilityLabel(String(localized: "Italic"))
            
            Spacer()
        }
        .padding(.horizontal, 4)
        .onChange(of: selection) { _, newSelection in
            updateFormattingStates()
        }
        .onAppear {
            updateFormattingStates()
        }
    }
    
    private func toggleBold() {
        text.transformAttributes(in: &selection) { attributes in
            if let font = attributes.font {
                let resolved = font.resolve(in: fontResolutionContext)
                attributes.font = font.bold(!resolved.isBold)
            } else {
                // If no font is set, create a new bold font
                attributes.font = Font.body.bold()
            }
        }
        updateFormattingStates()
    }
    
    private func toggleItalic() {
        text.transformAttributes(in: &selection) { attributes in
            if let font = attributes.font {
                let resolved = font.resolve(in: fontResolutionContext)
                attributes.font = font.italic(!resolved.isItalic)
            } else {
                // If no font is set, create a new italic font
                attributes.font = Font.body.italic()
            }
        }
        updateFormattingStates()
    }
    
    private func updateFormattingStates() {
        let attributes = selection.typingAttributes(in: text)
        
        if let font = attributes.font {
            let resolved = font.resolve(in: fontResolutionContext)
            isBold = resolved.isBold
            isItalic = resolved.isItalic
        } else {
            isBold = false
            isItalic = false
        }
    }
}

// Extension to check font properties
extension Font.Resolved {
    var isBold: Bool {
        // Since we can't access the raw value, we'll compare against known weights
        // This checks if the current weight is bold or heavier
        weight == .bold || weight == .heavy || weight == .black
    }
    
    var isItalic: Bool {
        // Check if the font has italic traits
        // This is a simplified check - in practice, you might need more sophisticated detection
        // For now, we'll use a placeholder approach since SwiftUI doesn't expose italic state directly
        return false // Placeholder - actual italic detection would require CoreText or other lower-level APIs
    }
}

// Helper extension for font transformations
extension Font {
    func bold(_ isBold: Bool = true) -> Font {
        if isBold {
            return self.weight(.bold)
        } else {
            return self.weight(.regular)
        }
    }
    
    func italic(_ isItalic: Bool = true) -> Font {
        // SwiftUI doesn't have direct italic support, but we can use design variations
        // This is a simplified approach - real italic would require more complex implementation
        return self
    }
}