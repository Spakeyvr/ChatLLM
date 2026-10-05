//
//  SidebarDrawer.swift
//  ChatLLM
//
//  Compact-width navigation: the chat fills the screen and slides aside to
//  reveal the conversation sidebar underneath, staying visible as a card.
//

import SwiftUI
import UIKit
import UIKit.UIGestureRecognizerSubclass

struct SidebarDrawer<Sidebar: View, Content: View>: View {
    @Binding var isOpen: Bool
    @ViewBuilder var sidebar: Sidebar
    @ViewBuilder var content: Content

    /// Live horizontal drag translation while the user is swiping; nil when idle.
    @State private var dragTranslation: CGFloat?

    private static var settleAnimation: Animation { .spring(response: 0.38, dampingFraction: 0.86) }

    var body: some View {
        GeometryReader { proxy in
            let sidebarWidth = Self.sidebarWidth(for: proxy.size.width)
            let offset = currentOffset(sidebarWidth: sidebarWidth)
            let progress = sidebarWidth > 0 ? offset / sidebarWidth : 0

            ZStack(alignment: .leading) {
                sidebar
                    .frame(width: sidebarWidth)
                    .frame(maxHeight: .infinity)
                    // Slight parallax so the sidebar appears to slide out from under the chat.
                    .offset(x: -(1 - progress) * sidebarWidth * 0.25)
                    .accessibilityHidden(!isOpen)

                contentCard(progress: progress, sidebarWidth: sidebarWidth)
                    .offset(x: offset)
            }
            .background(Color(uiColor: .secondarySystemBackground).ignoresSafeArea())
        }
        .onChange(of: isOpen) {
            // Opening hides the chat composer and closing hides the sidebar's search field,
            // so either way the focused field is leaving the screen.
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        }
    }

    private func contentCard(progress: CGFloat, sidebarWidth: CGFloat) -> some View {
        let shape = RoundedRectangle(cornerRadius: 44 * progress, style: .continuous)

        return content
            // Keep the peeking chat still while the sidebar's search keyboard is up.
            .ignoresSafeArea(.keyboard, edges: isOpen ? .bottom : [])
            .accessibilityHidden(isOpen)
            .overlay {
                if isOpen || dragTranslation != nil {
                    // Captures taps on the peeking chat so it closes the drawer instead of
                    // interacting with the chat underneath.
                    Color.black.opacity(0.12 * progress)
                        .contentShape(Rectangle())
                        .onTapGesture { setOpen(false) }
                        .accessibilityElement()
                        .accessibilityLabel("Close sidebar")
                        .accessibilityAddTraits(.isButton)
                        .accessibilityIdentifier("sidebar.dismiss")
                        .accessibilityAction { setOpen(false) }
                }
            }
            .clipShape(shape)
            .overlay {
                shape
                    .strokeBorder(Color(uiColor: .separator).opacity(progress), lineWidth: 1)
                    .allowsHitTesting(false)
            }
            .ignoresSafeArea()
            .shadow(color: .black.opacity(0.18 * progress), radius: 24, x: -4, y: 0)
            .gesture(
                DrawerPanGesture(
                    isOpen: isOpen,
                    onChanged: { dragTranslation = $0 },
                    onEnded: { translation, velocity in
                        finishDrag(translation: translation, velocity: velocity, sidebarWidth: sidebarWidth)
                    }
                )
            )
    }

    private func currentOffset(sidebarWidth: CGFloat) -> CGFloat {
        let base = isOpen ? sidebarWidth : 0
        guard let dragTranslation else { return base }
        return min(max(base + dragTranslation, 0), sidebarWidth)
    }

    private func finishDrag(translation: CGFloat, velocity: CGFloat, sidebarWidth: CGFloat) {
        let base = isOpen ? sidebarWidth : 0
        let projected = base + translation + velocity * 0.15
        let shouldOpen = projected > sidebarWidth / 2
        withAnimation(Self.settleAnimation) {
            dragTranslation = nil
            isOpen = shouldOpen
        }
        if shouldOpen != (base > 0) {
            AppHaptics.selectionChanged()
        }
    }

    private func setOpen(_ open: Bool) {
        withAnimation(Self.settleAnimation) {
            isOpen = open
        }
    }

    static func sidebarWidth(for containerWidth: CGFloat) -> CGFloat {
        min(containerWidth * 0.72, 340)
    }
}

// MARK: - Pan Gesture

/// Horizontal pan that only begins in the direction that changes the drawer state
/// (rightward to open, leftward to close), so vertical scrolling keeps working.
private struct DrawerPanGesture: UIGestureRecognizerRepresentable {
    let isOpen: Bool
    let onChanged: (CGFloat) -> Void
    let onEnded: (_ translation: CGFloat, _ velocity: CGFloat) -> Void

    func makeUIGestureRecognizer(context: Context) -> DirectionalPanGestureRecognizer {
        DirectionalPanGestureRecognizer()
    }

    func updateUIGestureRecognizer(_ recognizer: DirectionalPanGestureRecognizer, context: Context) {
        recognizer.allowsRightward = !isOpen
    }

    func handleUIGestureRecognizerAction(_ recognizer: DirectionalPanGestureRecognizer, context: Context) {
        let translation = recognizer.translation(in: recognizer.view).x
        switch recognizer.state {
        case .began, .changed:
            onChanged(translation)
        case .ended, .cancelled, .failed:
            onEnded(translation, recognizer.velocity(in: recognizer.view).x)
        default:
            break
        }
    }
}

final class DirectionalPanGestureRecognizer: UIPanGestureRecognizer {
    /// True to only begin on rightward swipes, false to only begin on leftward swipes.
    var allowsRightward = true
    private var startLocation: CGPoint?

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent) {
        super.touchesBegan(touches, with: event)
        if startLocation == nil {
            startLocation = touches.first?.location(in: view)
        }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent) {
        if state == .possible,
           let startLocation,
           let location = touches.first?.location(in: view) {
            let dx = location.x - startLocation.x
            let dy = location.y - startLocation.y
            if hypot(dx, dy) >= 6 {
                let isHorizontal = abs(dx) > abs(dy) * 1.2
                let isAllowedDirection = allowsRightward ? dx > 0 : dx < 0
                if !isHorizontal || !isAllowedDirection {
                    state = .failed
                    return
                }
            }
        }
        super.touchesMoved(touches, with: event)
    }

    override func reset() {
        super.reset()
        startLocation = nil
    }
}
