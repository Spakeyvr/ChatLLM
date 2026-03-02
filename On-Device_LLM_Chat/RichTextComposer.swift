//
//  RichTextComposer.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/25/25.
//

import SwiftUI
import UIKit

struct RichTextComposer: View {
    @Binding var attributedText: AttributedString
    @State private var selection = AttributedTextSelection()
    @State private var showFormatting: Bool = false
    @State private var textEditorHeight: CGFloat = 36
    @FocusState private var isTextFieldFocused: Bool
    
    var placeholder: String
    var onSend: () -> Void
    var onClear: () -> Void
    var canSend: Bool
    
    var body: some View {
        VStack(spacing: 0) {
            // Formatting toolbar (shown when text field is focused and has content)
            if showFormatting && !attributedText.characters.isEmpty {
                VStack(spacing: 0) {
                    Divider()
                    TextFormattingToolbar(text: $attributedText, selection: $selection)
                        .padding(.vertical, 8)
                        .background(.ultraThinMaterial)
                }
                .transition(.move(edge: .top).combined(with: .opacity))
            }
            
            // Main composer
            HStack(spacing: 10) {
                // Add menu (keeping existing functionality)
                Menu {
                    Button {
                        // Add photo action - you can implement this
                    } label: { Label(String(localized: "Photos"), systemImage: "photo.on.rectangle") }
                    Button {
                        // Add file action - you can implement this  
                    } label: { Label(String(localized: "Files"), systemImage: "folder") }
                } label: {
                    Image(systemName: "plus.circle")
                        .foregroundStyle(.secondary)
                }
                .accessibilityLabel(String(localized: "Add"))
                
                // Rich text editor
                ZStack(alignment: .topLeading) {
                    if attributedText.characters.isEmpty {
                        Text(placeholder)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 8)
                    }
                    
                    DynamicTextEditor(
                        text: $attributedText,
                        selection: $selection,
                        minHeight: 36,
                        maxHeight: showFormatting ? 120 : 100,
                        currentHeight: $textEditorHeight
                    )
                    .textInputAutocapitalization(.sentences)
                    .disableAutocorrection(false)
                    .font(.body)
                    .focused($isTextFieldFocused)
                    .background(Color.clear)
                }
                .frame(height: max(36, textEditorHeight))
                .animation(.easeInOut(duration: 0.15), value: textEditorHeight)
                .onSubmit(onSend)
                
                // Format button
                Button(action: {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showFormatting.toggle()
                        if showFormatting {
                            isTextFieldFocused = true
                        }
                    }
                }) {
                    Image(systemName: "textformat")
                        .foregroundStyle(showFormatting ? .blue : .secondary)
                }
                .accessibilityLabel(String(localized: "Text formatting"))
                
                // Clear button (if text exists)
                if !attributedText.characters.isEmpty {
                    Button(action: {
                        onClear()
                        withAnimation(.easeInOut(duration: 0.15)) {
                            textEditorHeight = 36
                        }
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .accessibilityLabel(String(localized: "Clear message"))
                }
                
                // Send button
                Button(action: onSend) {
                    Image(systemName: "paperplane.fill")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(canSend ? Color.blue : Color.secondary)
                }
                .disabled(!canSend)
                .accessibilityLabel(String(localized: "Send"))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .frame(minHeight: 44, alignment: .center)
            .background(.ultraThinMaterial, in: Capsule(style: .continuous))
            .overlay(Capsule(style: .continuous).strokeBorder(Color.secondary.opacity(0.12)))
            .shadow(color: Color.black.opacity(0.08), radius: 12, x: 0, y: 6)
            .animation(.easeInOut(duration: 0.2), value: textEditorHeight)
        }
        .onChange(of: isTextFieldFocused) { _, focused in
            if !focused {
                withAnimation(.easeInOut(duration: 0.15)) {
                    showFormatting = false
                }
            }
        }
        .onChange(of: attributedText.characters.isEmpty) { _, isEmpty in
            if isEmpty {
                withAnimation(.easeInOut(duration: 0.15)) {
                    textEditorHeight = 36
                }
            }
        }
    }
}

// Helper extension to check if AttributedString is empty
extension AttributedString {
    var isEmpty: Bool {
        characters.isEmpty
    }
    
    var trimmedIsEmpty: Bool {
        String(characters).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

// Dynamic TextEditor that adjusts height based on content and supports AttributedString
private struct DynamicTextEditor: UIViewRepresentable {
    @Binding var text: AttributedString
    @Binding var selection: AttributedTextSelection
    let minHeight: CGFloat
    let maxHeight: CGFloat
    @Binding var currentHeight: CGFloat
    
    func makeUIView(context: Context) -> UITextView {
        let textView = UITextView()
        textView.delegate = context.coordinator
        textView.isScrollEnabled = false
        textView.backgroundColor = UIColor.clear
        textView.textContainer.lineFragmentPadding = 0
        textView.textContainerInset = UIEdgeInsets(top: 8, left: 8, bottom: 8, right: 8)
        textView.textContainer.widthTracksTextView = true
        textView.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        textView.setContentHuggingPriority(.defaultHigh, for: .vertical)
        textView.font = UIFont.preferredFont(forTextStyle: .body)
        textView.textColor = UIColor.label
        textView.returnKeyType = .default
        textView.isEditable = true
        textView.isUserInteractionEnabled = true
        
        // Set initial text
        textView.attributedText = NSAttributedString(text)
        
        // Initial height
        DispatchQueue.main.async {
            self.updateHeight(for: textView)
            self.ensureCaretVisible(textView)
        }
        
        return textView
    }
    
    func updateUIView(_ uiView: UITextView, context: Context) {
        // Update text if it has changed (preserve cursor)
        let currentAttributedString = NSAttributedString(text)
        if !uiView.attributedText.isEqual(to: currentAttributedString) {
            context.coordinator.isUpdatingFromSwiftUI = true
            let cursorPosition = uiView.selectedRange
            uiView.attributedText = currentAttributedString
            
            // Safely restore cursor position with proper bounds checking
            let maxPosition = currentAttributedString.length
            let safeLocation = min(max(0, cursorPosition.location), maxPosition)
            let safeLength = max(0, min(cursorPosition.length, maxPosition - safeLocation))
            
            // Only restore if the position is valid
            if safeLocation <= maxPosition {
                uiView.selectedRange = NSRange(location: safeLocation, length: safeLength)
            }
            
            context.coordinator.isUpdatingFromSwiftUI = false
        }
        // Do NOT update SwiftUI state here (no updateHeight/currentHeight changes).
        ensureCaretVisible(uiView)
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    private func measuredWidth(for textView: UITextView) -> CGFloat {
        if textView.bounds.width > 0 {
            return textView.bounds.width
        }
        if let superW = textView.superview?.bounds.width, superW > 0 {
            return superW
        }
        if let windowW = textView.window?.bounds.width, windowW > 0 {
            return windowW
        }
        return 600 // safe fallback
    }
    
    private func updateHeight(for textView: UITextView) {
        // Ensure layout is up-to-date
        textView.setNeedsLayout()
        textView.layoutIfNeeded()
        
        let width = measuredWidth(for: textView)
        let fittingSize = CGSize(width: width, height: .greatestFiniteMagnitude)
        let size = textView.sizeThatFits(fittingSize)
        let clamped = min(max(size.height, minHeight), maxHeight)
        
        // Update height binding without feedback loops
        if abs(currentHeight - clamped) > 1.0 {
            currentHeight = clamped
        }
        
        // Toggle scrolling when content exceeds max
        let shouldScroll = size.height > maxHeight + 1.0
        if textView.isScrollEnabled != shouldScroll {
            textView.isScrollEnabled = shouldScroll
        }
    }
    
    private func ensureCaretVisible(_ textView: UITextView) {
        // If scrolling is enabled or we’re at the max height, make sure caret is visible.
        guard textView.isScrollEnabled || abs(currentHeight - maxHeight) < 0.5 else { return }
        let range = textView.selectedRange.length == 0
            ? NSRange(location: max(0, textView.text.count - 1), length: 0)
            : textView.selectedRange
        textView.scrollRangeToVisible(range)
    }
    
    class Coordinator: NSObject, UITextViewDelegate {
        var parent: DynamicTextEditor
        var isUpdatingFromSwiftUI: Bool = false
        
        init(_ parent: DynamicTextEditor) {
            self.parent = parent
        }
        
        func textViewDidChange(_ textView: UITextView) {
            guard !isUpdatingFromSwiftUI else { return }
            
            // Update the attributed text binding immediately
            if let attributedText = textView.attributedText {
                parent.text = AttributedString(attributedText)
            }
            
            // Update height and keep caret visible
            parent.updateHeight(for: textView)
            parent.ensureCaretVisible(textView)
        }
        
        func textViewDidChangeSelection(_ textView: UITextView) {
            guard !isUpdatingFromSwiftUI else { return }
            
            // Update selection - create new instance to trigger updates
            parent.selection = AttributedTextSelection()
            
            // Keep caret visible when the user moves it
            parent.ensureCaretVisible(textView)
        }
    }
}
