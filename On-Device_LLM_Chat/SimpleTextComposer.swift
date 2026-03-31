//
//  SimpleTextComposer.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/26/25.
//

import SwiftUI
import UIKit

enum ComposerReturnKeyBehaviorAction: Equatable {
    case send
    case insertNewline
}

enum ComposerReturnKeyBehavior {
    static func action(
        sendOnReturn: Bool,
        canSend: Bool,
        isGenerating: Bool,
        replacementText: String
    ) -> ComposerReturnKeyBehaviorAction {
        guard replacementText == "\n" else {
            return .insertNewline
        }

        guard sendOnReturn, canSend, !isGenerating else {
            return .insertNewline
        }

        return .send
    }
}

struct SimpleTextComposer: View {
    @Binding var text: String
    var placeholder: String
    var onSend: () -> Void
    var onStop: () -> Void
    var onClear: () -> Void
    var canSend: Bool
    var isGenerating: Bool
    var onCamera: (() -> Void)?
    var onPhotosPicker: (() -> Void)?
    var onFileImporter: (() -> Void)?
    @Binding var forceSearch: Bool
    var searchAvailable: Bool = true
    @Binding var disableToolCalls: Bool
    var toolCallsLockedDisabled: Bool = false
    @Binding var isReasoningEnabled: Bool
    @Binding var isSmartReasoningEnabled: Bool
    var reasoningAvailable: Bool = false

    @AppStorage(AppSettingsKeys.sendOnReturn) private var sendOnReturn = false

    @State private var isTextViewFocused = false
    @State private var showAddSheet = false
    @State private var composerHeight: CGFloat = 24

    private let circleSize: CGFloat = 44
    private let pillMinHeight: CGFloat = 44
    private let pillHorizontalPadding: CGFloat = 10
    private let maxTextViewHeight: CGFloat = 112
    private var reservedTrailingForSend: CGFloat { circleSize + 6 }

    @State private var measuredPillHeight: CGFloat = 44

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            Button {
                showAddSheet = true
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(width: 21, height: 30)
                    .foregroundStyle(forceSearch && searchAvailable ? Color.blue : Color.primary)
            }
            .buttonStyle(.glass)
            .contentShape(.circle)
            .accessibilityLabel("Add attachment or web search")
            .sheet(isPresented: $showAddSheet) {
                AddOptionsSheet(
                    onCamera: { showAddSheet = false; onCamera?() },
                    onPhotoLibrary: { showAddSheet = false; onPhotosPicker?() },
                    onFile: { showAddSheet = false; onFileImporter?() },
                    forceSearch: $forceSearch,
                    searchAvailable: searchAvailable,
                    disableToolCalls: $disableToolCalls,
                    toolCallsLockedDisabled: toolCallsLockedDisabled,
                    isReasoningEnabled: $isReasoningEnabled,
                    isSmartReasoningEnabled: $isSmartReasoningEnabled,
                    reasoningAvailable: reasoningAvailable
                )
                .presentationDetents([.medium])
                .presentationDragIndicator(.visible)
            }

            GlassEffectContainer(spacing: 10.0) {
                HStack(spacing: 8) {
                    ZStack(alignment: .topLeading) {
                        ComposerTextView(
                            text: $text,
                            isFocused: $isTextViewFocused,
                            measuredHeight: $composerHeight,
                            sendOnReturn: sendOnReturn,
                            canSend: canSend,
                            isGenerating: isGenerating,
                            maxHeight: maxTextViewHeight,
                            onSend: onSend
                        )
                        .frame(height: max(24, composerHeight))

                        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Text(placeholder)
                                .font(.body)
                                .foregroundStyle(.secondary)
                                .padding(.top, 2)
                                .allowsHitTesting(false)
                        }
                    }
                }
                .padding(.horizontal, pillHorizontalPadding)
                .padding(.vertical, 8)
                .padding(.trailing, reservedTrailingForSend)
                .frame(minHeight: pillMinHeight, alignment: .center)
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
                .overlay(alignment: .topLeading) {
                    GeometryReader { geo in
                        Color.clear
                            .preference(key: PillHeightKey.self, value: geo.size.height)
                    }
                }
                .onPreferenceChange(PillHeightKey.self) { newHeight in
                    if abs(measuredPillHeight - newHeight) > 0.5 {
                        measuredPillHeight = newHeight
                    }
                }
                .overlay(alignment: .trailing) {
                    Button {
                        AppHaptics.impact(.light)

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
                    .offset(x: -6)
                    .zIndex(10)
                    .shadow(color: Color.black.opacity(0.08), radius: 10, x: 0, y: 5)
                    .accessibilityLabel(isGenerating ? String(localized: "Stop") : String(localized: "Send"))
                    .accessibilityAddTraits(.isButton)
                }
                .frame(maxWidth: .infinity, minHeight: pillMinHeight)
            }
        }
    }

    private func adaptiveCornerRadius(for height: CGFloat) -> CGFloat {
        let maxRadius = pillMinHeight / 2
        let minRadius: CGFloat = 14
        let ratio = max(0.0, min(1.0, pillMinHeight / max(height, 1)))
        return minRadius + (maxRadius - minRadius) * ratio
    }
}

private struct PillHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 44
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

private struct AddOptionsSheet: View {
    var onCamera: () -> Void
    var onPhotoLibrary: () -> Void
    var onFile: () -> Void
    @Binding var forceSearch: Bool
    var searchAvailable: Bool
    @Binding var disableToolCalls: Bool
    var toolCallsLockedDisabled: Bool
    @Binding var isReasoningEnabled: Bool
    @Binding var isSmartReasoningEnabled: Bool
    var reasoningAvailable: Bool

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 12) {
                    if UIImagePickerController.isSourceTypeAvailable(.camera) {
                        AttachTile(icon: "camera", label: "Camera", action: onCamera)
                    }
                    AttachTile(icon: "photo.on.rectangle", label: "Photos", action: onPhotoLibrary)
                    AttachTile(icon: "doc.badge.arrow.up", label: "Files", action: onFile)
                }
                .padding(.horizontal, 20)
                .padding(.top, 8)
                .padding(.bottom, 24)

                Divider()
                    .padding(.horizontal, 20)

                VStack(spacing: 0) {
                    Toggle(isOn: $disableToolCalls) {
                        Label {
                            Text("Disable Tool Calls")
                                .font(.body)
                        } icon: {
                            Image(systemName: "wrench.and.screwdriver")
                                .font(.body)
                                .frame(width: 28)
                        }
                    }
                    .tint(.orange)
                    .disabled(toolCallsLockedDisabled)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 8)
                    .onChange(of: disableToolCalls) { _, isDisabled in
                        if isDisabled {
                            forceSearch = false
                        }
                    }

                    Toggle(isOn: $forceSearch) {
                        Label {
                            Text("Web Search")
                                .font(.body)
                        } icon: {
                            Image(systemName: "globe")
                                .font(.body)
                                .frame(width: 28)
                        }
                    }
                    .disabled(!searchAvailable || disableToolCalls)
                    .tint(.blue)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 8)

                    Divider()
                        .padding(.horizontal, 20)

                    Toggle(isOn: Binding(
                        get: { isReasoningEnabled },
                        set: { newValue in
                            if newValue { isSmartReasoningEnabled = false }
                            isReasoningEnabled = newValue
                        }
                    )) {
                        Label {
                            Text("Always Reason")
                                .font(.body)
                        } icon: {
                            Image(systemName: "brain.head.profile")
                                .font(.body)
                                .frame(width: 28)
                                .foregroundStyle(.blue)
                        }
                    }
                    .disabled(!reasoningAvailable || isSmartReasoningEnabled)
                    .tint(.blue)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 8)

                    Toggle(isOn: Binding(
                        get: { isSmartReasoningEnabled },
                        set: { newValue in
                            if newValue { isReasoningEnabled = false }
                            isSmartReasoningEnabled = newValue
                        }
                    )) {
                        Label {
                            Text("Smart Reasoning")
                                .font(.body)
                        } icon: {
                            Image(systemName: "sparkles")
                                .font(.body)
                                .frame(width: 28)
                                .foregroundStyle(.purple)
                        }
                    }
                    .disabled(!reasoningAvailable || isReasoningEnabled)
                    .tint(.purple)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 8)

                    Divider()
                        .padding(.horizontal, 20)
                }

                Spacer()
            }
            .navigationTitle("Add to Chat")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

private struct AttachTile: View {
    let icon: String
    let label: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.system(size: 26, weight: .regular))
                Text(label)
                    .font(.subheadline)
                    .fontWeight(.medium)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 100)
            .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
        .buttonStyle(.plain)
        .foregroundStyle(.primary)
    }
}

private struct ComposerTextView: UIViewRepresentable {
    @Binding var text: String
    @Binding var isFocused: Bool
    @Binding var measuredHeight: CGFloat

    let sendOnReturn: Bool
    let canSend: Bool
    let isGenerating: Bool
    let maxHeight: CGFloat
    let onSend: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    func makeUIView(context: Context) -> UITextView {
        let textView = UITextView()
        textView.backgroundColor = .clear
        textView.delegate = context.coordinator
        textView.font = UIFont.preferredFont(forTextStyle: .body)
        textView.textColor = .label
        textView.tintColor = .systemBlue
        textView.isScrollEnabled = false
        textView.textContainerInset = .zero
        textView.textContainer.lineFragmentPadding = 0
        textView.autocapitalizationType = .sentences
        textView.autocorrectionType = .default
        textView.keyboardDismissMode = .interactive
        textView.returnKeyType = .default
        textView.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        textView.setContentHuggingPriority(.defaultLow, for: .horizontal)
        textView.text = text
        context.coordinator.updateHeight(for: textView)
        return textView
    }

    func updateUIView(_ uiView: UITextView, context: Context) {
        context.coordinator.parent = self

        if uiView.text != text {
            uiView.text = text
        }

        if isFocused, !uiView.isFirstResponder {
            uiView.becomeFirstResponder()
        } else if !isFocused, uiView.isFirstResponder {
            uiView.resignFirstResponder()
        }

        context.coordinator.updateHeight(for: uiView)
    }

    @MainActor
    final class Coordinator: NSObject, UITextViewDelegate {
        var parent: ComposerTextView

        init(parent: ComposerTextView) {
            self.parent = parent
        }

        func textViewDidBeginEditing(_ textView: UITextView) {
            parent.isFocused = true
        }

        func textViewDidEndEditing(_ textView: UITextView) {
            parent.isFocused = false
        }

        func textViewDidChange(_ textView: UITextView) {
            parent.text = textView.text
            updateHeight(for: textView)
        }

        func textView(
            _ textView: UITextView,
            shouldChangeTextIn range: NSRange,
            replacementText text: String
        ) -> Bool {
            switch ComposerReturnKeyBehavior.action(
                sendOnReturn: parent.sendOnReturn,
                canSend: parent.canSend,
                isGenerating: parent.isGenerating,
                replacementText: text
            ) {
            case .send:
                parent.onSend()
                return false
            case .insertNewline:
                return true
            }
        }

        func updateHeight(for textView: UITextView) {
            let fittingSize = CGSize(
                width: max(textView.bounds.width, 1),
                height: .greatestFiniteMagnitude
            )
            let targetHeight = max(24, ceil(textView.sizeThatFits(fittingSize).height))
            let clampedHeight = min(targetHeight, parent.maxHeight)

            if textView.isScrollEnabled != (targetHeight > parent.maxHeight) {
                textView.isScrollEnabled = targetHeight > parent.maxHeight
            }

            guard abs(parent.measuredHeight - clampedHeight) > 0.5 else {
                return
            }

            DispatchQueue.main.async {
                self.parent.measuredHeight = clampedHeight
            }
        }
    }
}
