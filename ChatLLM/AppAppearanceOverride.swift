import SwiftUI
import UIKit

// SwiftUI's preferredColorScheme(nil) can retain the previous override on a
// presented cover. A single window-level override also updates open sheets and
// restores live system appearance with .unspecified, without rebuilding views.
struct AppAppearanceOverride: UIViewRepresentable {
    let appearance: String

    func makeUIView(context: Context) -> AppearanceView {
        let view = AppearanceView()
        view.isUserInteractionEnabled = false
        return view
    }

    func updateUIView(_ view: AppearanceView, context: Context) {
        switch appearance {
        case "light": view.style = .light
        case "dark": view.style = .dark
        default: view.style = .unspecified
        }
    }

    final class AppearanceView: UIView {
        var style: UIUserInterfaceStyle = .unspecified {
            didSet { applyStyle() }
        }

        override func didMoveToWindow() {
            super.didMoveToWindow()
            applyStyle()
        }

        private func applyStyle() {
            guard let window, window.overrideUserInterfaceStyle != style else { return }
            window.overrideUserInterfaceStyle = style
        }
    }
}
