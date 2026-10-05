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
