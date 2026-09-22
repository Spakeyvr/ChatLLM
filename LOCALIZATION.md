# Interface translations

ChatLLM supports English (`en`), German (`de`), and Spanish (`es`). Choose a language in **Settings → Appearance → Language**. The choice takes effect immediately and is saved between launches. English remains the default and fallback.

`ChatLLM/Localizable.xcstrings` contains interface text and plural forms. `ChatLLM/InfoPlist.xcstrings` contains the camera permission explanation; iOS chooses the language of system permission dialogs using its own app-language setting.

When adding interface text:

- Pass literals directly to SwiftUI controls, or give custom view parameters the `LocalizedStringKey` type.
- For Foundation/UIKit strings, use `String(localized: "…", bundle: .appLocalized)`. In views that resolve strings themselves, also read `@Environment(\.locale)` and pass `locale: locale` so language changes invalidate the view.
- Keep translations computed instead of storing a resolved translation in a `static let` or a long-lived model. Interpolate whole sentences and use catalog plural variations for counts.
- Keep conversation content, model prompts, identifiers, product names, and logs separate from interface translations.
- Build in Xcode to discover new keys, then translate them into German and Spanish. Preserve formatting placeholders such as `%@` and `%lld`.

Run `python3 Scripts/validate_localizations.py` to check coverage and placeholders. The simulator test `testInterfaceLanguageSwitchesImmediatelyAndPersists` checks switching through all three languages and relaunching. `ChatLLMTests+Localization.swift` covers bundled translations, plural forms, fallback, and draft-title recognition.

Apple reference: [Localizing and varying text with a string catalog](https://developer.apple.com/documentation/xcode/localizing-and-varying-text-with-a-string-catalog).
