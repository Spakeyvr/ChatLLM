//
//  ChatLLMTests+Factories.swift
//  ChatLLMTests
//
//  Shared factory methods used by multiple ChatLLMTests topic files.
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
    // MARK: - Test helpers
    /// A persisted conversation seeded with finished turns, for editing tests.
    func makeChat(_ turns: [(MessageRole, String)]) throws -> ChatViewModel {
        let schema = Schema([Conversation.self, ChatLLM.Message.self, MessageAttachment.self])
        let container = try ModelContainer(
            for: schema,
            configurations: [ModelConfiguration(isStoredInMemoryOnly: true)]
        )
        let context = ModelContext(container)
        let conversation = Conversation(title: "Test")
        context.insert(conversation)
        conversation.messages = turns.enumerated().map { index, turn in
            ChatLLM.Message(
                role: turn.0,
                text: turn.1,
                order: index,
                conversation: conversation,
                isFinal: true
            )
        }
        try context.save()
        return ChatViewModel(generator: TestLLMGenerator(), context: context, conversation: conversation)
    }


    /// An isolated `UserDefaults` domain so settings tests never touch the
    /// simulator's real preferences or each other's.
    func makeScratchDefaults(_ label: String) throws -> UserDefaults {
        let suite = "ChatLLMTests." + label.replacingOccurrences(of: "()", with: "")
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    func makeTestImage(width: Int, height: Int) -> UIImage {
        let size = CGSize(width: width, height: height)
        return UIGraphicsImageRenderer(size: size).image { context in
            UIColor.systemTeal.setFill()
            context.fill(CGRect(origin: .zero, size: size))
        }
    }

    func makeSearchBridge() throws -> AppWebSearchToolBridge {
        MockTavilyURLProtocol.responseStatusCode = 200
        MockTavilyURLProtocol.responseData = """
        {
          "query": "latest swift release",
          "answer": "Swift 6.2 is the latest stable release.",
          "auto_parameters": {
            "topic": "news",
            "search_depth": "basic",
            "time_range": "week"
          },
          "response_time": "1.67",
          "results": [
            {
              "title": "Swift 6.2 Released",
              "url": "https://example.com/swift",
              "content": "Swift 6.2 adds more concurrency fixes.",
              "score": 0.99,
              "published_date": "2026-02-28"
            }
          ]
        }
        """.data(using: .utf8)!

        let service = try makeSearchService()
        return AppWebSearchToolBridge(searchService: service)
    }

    func makeSearchService() throws -> TavilySearchService {
        MockTavilyURLProtocol.lastRequest = nil
        MockTavilyURLProtocol.lastRequestBody = nil
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockTavilyURLProtocol.self]
        let session = URLSession(configuration: configuration)
        return try TavilySearchService(apiKey: "test-key", session: session)
    }

    func makeViewModel(generator: LLMGenerator = TestLLMGenerator()) throws -> ChatViewModel {
        let schema = Schema([Conversation.self, ChatLLM.Message.self, MessageAttachment.self])
        let container = try ModelContainer(
            for: schema,
            configurations: [ModelConfiguration(isStoredInMemoryOnly: true)]
        )
        let context = ModelContext(container)
        let conversation = Conversation(title: "Test")
        return ChatViewModel(
            generator: generator,
            context: context,
            conversation: conversation
        )
    }

}
