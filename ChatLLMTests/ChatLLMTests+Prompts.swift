//
//  ChatLLMTests+Prompts
//  ChatLLMTests
//
//  Split out of ChatLLMTests.swift; part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
import MLXLMCommon
import SwiftUI
import SwiftData
import WebKit
import UIKit
import FoundationModels
@testable import ChatLLM

extension ChatLLMTests {
    @Test func searchInvocationMergePreservesAnchorsAcrossFinalization() {
        let preservedID = UUID()
        let newID = UUID()
        let liveInvocations = [
            SearchInvocation(
                id: preservedID,
                query: "swift release date",
                results: "Result A"
            ),
            SearchInvocation(
                id: newID,
                query: "swift evolution",
                results: "Result B"
            )
        ]
        let existingInvocations = [
            SearchInvocation(
                id: preservedID,
                query: "swift release date",
                results: "Result A",
                anchorStepNumber: 2
            )
        ]

        let merged = ChatViewModel.mergeSearchInvocations(
            liveInvocations,
            preservingAnchorsFrom: existingInvocations,
            currentChunkCount: 3
        )

        #expect(merged.count == 2)
        #expect(merged.first(where: { $0.id == preservedID })?.anchorStepNumber == 2)
        #expect(merged.first(where: { $0.id == newID })?.anchorStepNumber == 3)
    }

    @Test func clearingSearchInvocationsAlsoClearsLegacySearchQuery() {
        let conversation = Conversation(title: "Test")
        let message = ChatLLM.Message(
            role: .assistant,
            text: "Answer",
            order: 1,
            conversation: conversation
        )

        message.searchInvocations = [
            SearchInvocation(
                query: "latest swift release",
                results: "Swift 6.2 Released"
            )
        ]
        #expect(message.searchQuery == "latest swift release")

        message.searchInvocations = nil

        #expect(message.searchInvocations == nil)
        #expect(message.searchQuery == nil)
    }

    @Test func assistantPlaceholderTracksForcedWebSearchSeparatelyFromSearchHistory() throws {
        let viewModel = try makeViewModel()

        let forced = viewModel.appendAssistantPlaceholder(
            isReasoningMode: false,
            searchQuery: "latest swift release",
            requiresWebSearch: true
        )
        let autonomous = viewModel.appendAssistantPlaceholder(isReasoningMode: false)
        autonomous.searchInvocations = [
            SearchInvocation(
                query: "latest swift release",
                results: "Swift 6.2 Released"
            )
        ]

        #expect(forced.requiresWebSearch == true)
        #expect(forced.searchQuery == "latest swift release")
        #expect(autonomous.requiresWebSearch != true)
        #expect(autonomous.searchQuery == "latest swift release")
    }

    @Test func currentDateTimeContextIncludesExactDateAndYearHint() {
        let referenceDate = Date(timeIntervalSince1970: 1_772_884_800) // 2026-03-07 12:00:00 UTC
        let utc = TimeZone(secondsFromGMT: 0)!

        let context = ChatViewModel.currentDateTimeContext(
            referenceDate: referenceDate,
            timeZone: utc
        )

        #expect(context.contains("Saturday, March 7, 2026 at 12:00 UTC"))
        #expect(context.contains("current year"))
        #expect(context.contains("exact date"))
    }

    @Test func mlxSessionReusePolicyAllowsKeyedSessionReuseAcrossModes() {
        #expect(MLXModelManager.shouldPersistSessionAcrossTurns(enableThinking: false, hasTools: false, hasMedia: false))
        #expect(MLXModelManager.shouldPersistSessionAcrossTurns(enableThinking: true, hasTools: false, hasMedia: false))
        #expect(MLXModelManager.shouldPersistSessionAcrossTurns(enableThinking: false, hasTools: true, hasMedia: false))
        #expect(MLXModelManager.shouldPersistSessionAcrossTurns(enableThinking: false, hasTools: false, hasMedia: true))
    }

    @Test func mlxTimeContextGateSkipsOrdinaryTurns() {
        let shouldInject = ChatViewModel.shouldInjectMLXCurrentDateTimeContext(
            latestUserText: "Explain Swift actors simply.",
            forceWebSearch: false,
            webSearchEnabled: true,
            referenceDate: Date(timeIntervalSince1970: 1_772_884_800)
        )

        #expect(!shouldInject)
    }

    @Test func mlxTimeContextGateIncludesTimeSensitiveTurns() {
        let shouldInject = ChatViewModel.shouldInjectMLXCurrentDateTimeContext(
            latestUserText: "What is the latest Swift release today?",
            forceWebSearch: false,
            webSearchEnabled: false,
            referenceDate: Date(timeIntervalSince1970: 1_772_884_800)
        )

        #expect(shouldInject)
    }

    @Test func mlxTimeContextGateIncludesForcedWebSearchTurns() {
        let shouldInject = ChatViewModel.shouldInjectMLXCurrentDateTimeContext(
            latestUserText: "Compare the newest Xcode beta to stable.",
            forceWebSearch: true,
            webSearchEnabled: true,
            referenceDate: Date(timeIntervalSince1970: 1_772_884_800)
        )

        #expect(shouldInject)
    }

    @Test func mlxTimeContextGateDoesNotTreatEmbeddedSubstringsAsTimeSensitive() {
        let shouldInject = ChatViewModel.shouldInjectMLXCurrentDateTimeContext(
            latestUserText: "Explain how snow forms in clouds.",
            forceWebSearch: false,
            webSearchEnabled: true,
            referenceDate: Date(timeIntervalSince1970: 1_772_884_800)
        )

        #expect(!shouldInject)
    }
}
