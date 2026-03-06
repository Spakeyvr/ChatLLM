//
//  On_Device_LLM_ChatTests.swift
//  On-Device_LLM_ChatTests
//
//  Created by Nevio on 10/24/25.
//

import Testing
import Foundation
import MLXLMCommon
@testable import On_Device_LLM_Chat

@MainActor
struct On_Device_LLM_ChatTests {

    @Test func webSearchBridgeStoresInvocationFromDirectExecution() async throws {
        let bridge = try makeSearchBridge()

        let result = try await bridge.executeSearch(query: "latest swift release")

        #expect(result.contains("Swift 6.2 Released"))
        #expect(result.contains("https://example.com/swift"))
        #expect(bridge.allInvocations.count == 1)
        #expect(bridge.allInvocations.first?.query == "latest swift release")
    }

    @Test func webSearchBridgeExecutesMLXToolCall() async throws {
        let bridge = try makeSearchBridge()
        let toolName = AppWebSearchToolBridge.toolName
        let toolCall = MLXToolCall(function: .init(name: toolName, arguments: [
            "query": "qwen tool calling"
        ]))

        let result = try await bridge.dispatchMLXToolCall(toolCall)

        #expect(result.contains("Swift 6.2 Released"))
        #expect(bridge.allInvocations.count == 1)
        #expect(bridge.allInvocations.first?.query == "qwen tool calling")
        let function = bridge.mlxToolSpec["function"] as? [String: any Sendable]
        #expect(function?["name"] as? String == toolName)
    }

    @Test func webSearchBridgeEnforcesInvocationLimit() async throws {
        let bridge = try makeSearchBridge()

        for index in 0..<4 {
            let result = try await bridge.executeSearch(query: "query \(index)")
            #expect(result.contains("Swift 6.2 Released"))
        }

        let limited = try await bridge.executeSearch(query: "query overflow")

        #expect(bridge.allInvocations.count == 4)
        #expect(limited.contains("Search limit reached"))
    }

    private func makeSearchBridge() throws -> AppWebSearchToolBridge {
        MockTavilyURLProtocol.responseData = """
        {
          "query": "latest swift release",
          "results": [
            {
              "title": "Swift 6.2 Released",
              "url": "https://example.com/swift",
              "content": "Swift 6.2 adds more concurrency fixes.",
              "score": 0.99
            }
          ]
        }
        """.data(using: .utf8)!

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockTavilyURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let service = try TavilySearchService(apiKey: "test-key", session: session)
        return AppWebSearchToolBridge(searchService: service)
    }

}

private final class MockTavilyURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var responseData = Data()

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "api.tavily.com"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url ?? URL(string: "https://api.tavily.com/search")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.responseData)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
