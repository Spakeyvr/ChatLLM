//
//  ChatLLMTests+Sidebar
//  ChatLLMTests
//
//  Part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
import SwiftUI
@testable import ChatLLM

extension ChatLLMTests {
    @Test func sidebarGroupsConversationsByRecencyInDisplayOrder() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try #require(TimeZone(identifier: "Europe/Vienna"))
        let now = try #require(calendar.date(from: DateComponents(year: 2026, month: 10, day: 5, hour: 12)))
        func daysAgo(_ days: Int) throws -> Date {
            try #require(calendar.date(byAdding: .day, value: -days, to: now))
        }

        let today = Conversation(title: "Today", lastUpdated: now)
        let yesterday = Conversation(title: "Yesterday", lastUpdated: try daysAgo(1))
        let lastWeek = Conversation(title: "Last week", lastUpdated: try daysAgo(4))
        let lastMonth = Conversation(title: "Last month", lastUpdated: try daysAgo(20))
        let older = Conversation(title: "Older", lastUpdated: try daysAgo(45))

        let sections = ConversationRecencySection.grouping(
            [today, yesterday, lastWeek, lastMonth, older],
            now: now,
            calendar: calendar
        )

        #expect(sections.map(\.id) == ["today", "yesterday", "last7Days", "last30Days", "older"])
        #expect(sections.map { $0.conversations.map(\.title) } == [
            ["Today"], ["Yesterday"], ["Last week"], ["Last month"], ["Older"]
        ])
    }

    @Test func sidebarOmitsEmptyRecencySectionsAndKeepsOrderWithinSection() throws {
        let calendar = Calendar(identifier: .gregorian)
        let now = Date()
        let newer = Conversation(title: "Newer", lastUpdated: now.addingTimeInterval(-60 * 60 * 24 * 10))
        let older = Conversation(title: "Older", lastUpdated: now.addingTimeInterval(-60 * 60 * 24 * 12))

        let sections = ConversationRecencySection.grouping([newer, older], now: now, calendar: calendar)

        #expect(sections.map(\.id) == ["last30Days"])
        #expect(sections.first?.conversations.map(\.title) == ["Newer", "Older"])
        #expect(ConversationRecencySection.grouping([], now: now, calendar: calendar).isEmpty)
    }

    @Test func sidebarRendersAgainOnlyWhenItsListChanges() {
        let now = Date()
        let first = Conversation(title: "First", lastUpdated: now)
        let second = Conversation(title: "Second", lastUpdated: now.addingTimeInterval(-60))
        func sidebar(_ conversations: [Conversation]) -> ConversationSidebar {
            ConversationSidebar(
                sections: ConversationRecencySection.grouping(conversations, now: now),
                hasConversations: !conversations.isEmpty,
                selection: .constant(nil),
                searchText: .constant(""),
                isSearchPresented: .constant(false),
                onSubmitSearch: {}, onSelect: {}, onNewChat: {}, onShowSettings: {},
                onRename: { _ in }, onDelete: { _ in }
            )
        }

        // A streaming reply changes the chats, not the list, so the sidebar is skipped.
        let before = sidebar([first, second])
        first.title = "Renamed"
        #expect(before == sidebar([first, second]))

        #expect(before != sidebar([second, first]))
        #expect(before != sidebar([first]))
        second.lastUpdated = now.addingTimeInterval(-60 * 60 * 24 * 3)
        #expect(before != sidebar([first, second]))
    }

    @Test func sidebarDrawerLeavesChatVisibleBesideSidebar() {
        let phoneWidth: CGFloat = 402
        let sidebarWidth = SidebarDrawer<EmptyView, EmptyView>.sidebarWidth(for: phoneWidth)
        #expect(sidebarWidth < phoneWidth)
        #expect(phoneWidth - sidebarWidth >= 80)
        #expect(SidebarDrawer<EmptyView, EmptyView>.sidebarWidth(for: 1_000) == 340)
    }

    @Test func composerDraftKeepsPreEditTextWhileEditingAMessage() {
        var draft = ComposerDraft()
        draft.recordText("Half-written question", preEditText: nil)
        #expect(draft.text == "Half-written question")

        // Editing an earlier message fills the composer with that message, which isn't a draft.
        draft.recordText("Earlier message being edited", preEditText: "Half-written question")
        #expect(draft.text == "Half-written question")

        draft.recordText("", preEditText: nil)
        #expect(draft.text.isEmpty)
    }
}

extension ChatLLMTests {
    @Test func reasoningBadgeMarksOnlyChatsWhoseModelReasons() {
        func chat(reasoning: Bool, backend: ModelBackendBridge.Backend?, model: String? = nil) -> Conversation {
            Conversation(
                title: "Chat",
                reasoningMode: reasoning,
                preferredBackendRawValue: backend?.rawValue,
                preferredModelID: model
            )
        }

        #expect(!ConversationRow.showsReasoningBadge(for: chat(reasoning: true, backend: .foundationModels)))
        #expect(ConversationRow.showsReasoningBadge(for: chat(reasoning: true, backend: .mlx, model: "qwen3.5-2b-4bit")))
        #expect(!ConversationRow.showsReasoningBadge(for: chat(reasoning: false, backend: .mlx, model: "qwen3.5-2b-4bit")))
        #expect(ConversationRow.showsReasoningBadge(for: chat(reasoning: false, backend: .mlx, model: "lfm2.5-2.6b-4bit")))
        // Older chats that never recorded a backend keep their saved setting.
        #expect(ConversationRow.showsReasoningBadge(for: chat(reasoning: true, backend: nil)))
    }
}
