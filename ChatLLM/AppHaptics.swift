import UIKit

@MainActor
enum AppHaptics {
    @discardableResult
    static func performIfEnabled(
        defaults: UserDefaults = .standard,
        action: () -> Void
    ) -> Bool {
        guard defaults.enableHapticsPreference else {
            return false
        }

        action()
        return true
    }

    static func selectionChanged() {
        performIfEnabled {
            let generator = UISelectionFeedbackGenerator()
            generator.selectionChanged()
        }
    }

    static func impact(
        _ style: UIImpactFeedbackGenerator.FeedbackStyle,
        intensity: CGFloat? = nil
    ) {
        performIfEnabled {
            let generator = UIImpactFeedbackGenerator(style: style)
            if let intensity {
                generator.prepare()
                generator.impactOccurred(intensity: intensity)
            } else {
                generator.impactOccurred()
            }
        }
    }

    static func notification(_ type: UINotificationFeedbackGenerator.FeedbackType) {
        performIfEnabled {
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(type)
        }
    }

    static func generationCompleted() async {
        guard UserDefaults.standard.enableHapticsPreference else { return }
        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.prepare()
        generator.impactOccurred(intensity: 0.7)
        try? await Task.sleep(for: .milliseconds(80))
        generator.impactOccurred(intensity: 1.0)
    }
}
