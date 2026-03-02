//
//  SimpleTextComposer.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/26/25.
//

import SwiftUI

struct SimpleTextComposer: View {
    @Binding var text: String
    var placeholder: String
    var onSend: () -> Void
    var onStop: () -> Void
    var onClear: () -> Void
    var canSend: Bool
    var isGenerating: Bool
    var onPhotosPicker: (() -> Void)?
    var onFileImporter: (() -> Void)?
    @Binding var forceSearch: Bool
    var searchAvailable: Bool = true

    @FocusState private var isTextFieldFocused: Bool

    // Layout constants (aligned sizes so the circle doesn't stick out)
    private let circleSize: CGFloat = 44            // match pill height
    private let pillMinHeight: CGFloat = 44         // match circle height
    private let pillHorizontalPadding: CGFloat = 10
    private var reservedTrailingForSend: CGFloat { circleSize + 6 } // space for the overlaid circle

    // Live measurement of the pill's height (used to adapt the corner radius)
    @State private var measuredPillHeight: CGFloat = 44

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {

            // LEFT: Attachment menu button with Liquid Glass - OUTSIDE GlassEffectContainer to preserve morphing
            Menu {
                Button(action: { onPhotosPicker?() }) {
                    Label("Photos", systemImage: "photo.on.rectangle")
                }
                Button(action: { onFileImporter?() }) {
                    Label("Files", systemImage: "folder")
                }
            } label: {
                Image(systemName: "paperclip")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(width: 21, height: 30)
            }
            .buttonStyle(.glass)
            .contentShape(.circle)
            .accessibilityLabel("Add attachments")

            // Web search toggle
            Button {
                forceSearch.toggle()
                UIImpactFeedbackGenerator(style: .light).impactOccurred()
            } label: {
                Image(systemName: forceSearch && searchAvailable ? "globe.badge.chevron.backward" : "globe")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(width: 21, height: 30)
                    .foregroundStyle(forceSearch && searchAvailable ? Color.blue : Color.primary)
            }
            .buttonStyle(.glass)
            .contentShape(.circle)
            .disabled(!searchAvailable)
            .accessibilityLabel(forceSearch ? "Disable web search" : "Enable web search")

           
            // MIDDLE: Text pill with send button overlaid on its right edge - IN GlassEffectContainer
            GlassEffectContainer(spacing: 10.0) {
                HStack(spacing: 8) {
                    DynamicHeightTextEditor(
                        text: $text,
                        height: .constant(0),
                        placeholder: placeholder
                    )
                    .focused($isTextFieldFocused)
                    .onSubmit {
                        if !isGenerating && canSend {
                            onSend()
                        }
                    }
                }
                .padding(.horizontal, pillHorizontalPadding)
                .padding(.vertical, 4)
                .padding(.trailing, reservedTrailingForSend)
                .frame(minHeight: pillMinHeight, alignment: .center)
                .tint(.blue) // caret/selection color
                // Adaptive corner radius based on content height
                .glassEffect(
                    .regular.interactive(),
                    in: .rect(cornerRadius: adaptiveCornerRadius(for: measuredPillHeight))
                )
                .clipShape(
                    RoundedRectangle(
                        cornerRadius: adaptiveCornerRadius(for: measuredPillHeight),
                        style: .continuous
                    )
                )
                // Measure the pill's rendered height to drive the corner radius
                .overlay(alignment: .topLeading) {
                    GeometryReader { geo in
                        Color.clear
                            .preference(key: PillHeightKey.self, value: geo.size.height)
                    }
                }
                .onPreferenceChange(PillHeightKey.self) { newHeight in
                    // Debounce tiny fluctuations
                    if abs(measuredPillHeight - newHeight) > 0.5 {
                        measuredPillHeight = newHeight
                    }
                }
                .overlay(alignment: .trailing) {
                    // RIGHT: Send/Stop button sitting "on" the pill
                    Button {
                        // Immediate haptic for snappy feel
                        let h = UIImpactFeedbackGenerator(style: .light)
                        h.impactOccurred()
                       
                        if isGenerating {
                            onStop()
                        } else if canSend {
                            onSend()
                        }
                    } label: {
                        Group {
                            if isGenerating {
                                Image(systemName: "stop.fill")
                                    .font(.system(size: 16, weight: .semibold))
                                    .foregroundStyle(.red)
                            } else {
                                Image(systemName: "arrow.up")
                                    .font(.system(size: 18, weight: .semibold))
                                    .foregroundStyle(.primary)
                            }
                        }
                        .frame(width: circleSize, height: circleSize)
                        .contentShape(Circle())
                    }
                    .disabled(!isGenerating && !canSend)
                    .buttonStyle(.plain)
                    .buttonRepeatBehavior(.enabled)
                    .tint(canSend ? .blue : .secondary)
                    .glassEffect(.regular.interactive(), in: .circle)
                    .offset(x: -6) // tuck the circle slightly into the pill
                    .zIndex(10) // ensure it wins hit-testing over the text field
                    .shadow(color: Color.black.opacity(0.08), radius: 10, x: 0, y: 5)
                    .accessibilityLabel(isGenerating ? String(localized: "Stop") : String(localized: "Send"))
                    .accessibilityAddTraits(.isButton)
                }
                .frame(maxWidth: .infinity, minHeight: pillMinHeight)
            }
        }
    }
    
    // Corner radius eases from a capsule (22) toward a rounded rectangle (~14)
    // as the pill grows taller when the text wraps to more lines.
    private func adaptiveCornerRadius(for height: CGFloat) -> CGFloat {
        let maxRadius = pillMinHeight / 2           // 22 when pillMinHeight is 44 (capsule look)
        let minRadius: CGFloat = 14                 // less rounded for multi-line
        let ratio = max(0.0, min(1.0, pillMinHeight / max(height, 1)))
        return minRadius + (maxRadius - minRadius) * ratio
    }
}

// MARK: - PreferenceKey to read the pill height

private struct PillHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 44
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

// MARK: - Dynamic Height Text Editor

private struct DynamicHeightTextEditor: View {
    @Binding var text: String
    @Binding var height: CGFloat
    let placeholder: String
    
    var body: some View {
        TextField(placeholder, text: $text, axis: .vertical)
            .font(.body)
            .lineLimit(1...4)
            .textInputAutocapitalization(.sentences)
            .disableAutocorrection(false)
            .tint(.blue) // explicit caret color
    }
}
