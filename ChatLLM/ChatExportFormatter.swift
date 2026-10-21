import Foundation

/// Plain-text export of saved chats, written in the app's interface language so the
/// file reads like the rest of the app. Message content is exported as written.
enum ChatExportFormatter {
    static func text(
        for conversations: [Conversation],
        language: AppLanguage = .current,
        exportedAt: Date = Date()
    ) -> String {
        let bundle = language.bundle
        let locale = language.locale
        func formatted(_ date: Date) -> String {
            date.formatted(Date.FormatStyle(date: .abbreviated, time: .shortened).locale(locale))
        }

        let separator = String(repeating: "=", count: 60)
        let chatSeparator = String(repeating: "-", count: 60)
        let exportedDate = formatted(exportedAt)
        let count = conversations.count
        var text = "# \(String(localized: "Chat Export", bundle: bundle, locale: locale))\n"
        text += String(localized: "Exported: \(exportedDate)", bundle: bundle, locale: locale) + "\n"
        text += String(localized: "Conversations: \(count)", bundle: bundle, locale: locale) + "\n\n"
        text += separator + "\n\n"

        for conversation in conversations.sorted(by: { $0.lastUpdated > $1.lastUpdated }) {
            let lastUpdated = formatted(conversation.lastUpdated)
            text += "## \(conversation.title)\n"
            text += String(localized: "Last updated: \(lastUpdated)", bundle: bundle, locale: locale) + "\n"
            if conversation.reasoningMode {
                text += String(localized: "Reasoning mode", bundle: bundle, locale: locale) + "\n"
            }
            text += "\n"

            let messages = conversation.messages.sorted { $0.order < $1.order }
            for message in messages where message.role != .system {
                let role = message.role == .user
                    ? String(localized: "User", bundle: bundle, locale: locale)
                    : String(localized: "Assistant", bundle: bundle, locale: locale)
                text += "**\(role):**\n"
                let visibleText = message.userVisibleText
                if !visibleText.isEmpty {
                    text += visibleText + "\n"
                }
                if let generationError = message.generationError {
                    text += "[\(String(localized: "Error: \(generationError)", bundle: bundle, locale: locale))]\n"
                }
                for attachment in message.attachments where attachment.type == .image {
                    let fileName = attachment.fileName
                    text += "[\(String(localized: "Image: \(fileName)", bundle: bundle, locale: locale))]\n"
                }
                text += "\n"
            }

            text += chatSeparator + "\n\n"
        }
        return text
    }

    /// A readable, sortable name for Files and AirDrop instead of a raw timestamp.
    static func fileName(exportedAt: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd-HHmmss"
        return "ChatLLM-Export-\(formatter.string(from: exportedAt)).txt"
    }
}
