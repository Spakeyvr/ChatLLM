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
@Suite(.serialized)
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

    @Test func tavilySearchServiceUsesAuthorizationHeader() async throws {
        let service = try makeSearchService()

        _ = try await service.search(query: " latest swift release ")

        let request = try #require(MockTavilyURLProtocol.lastRequest)
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer test-key")
        #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
        let body = try #require(MockTavilyURLProtocol.lastRequestBody)
        let payload = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(payload["api_key"] == nil)
        #expect(payload["query"] as? String == "latest swift release")
    }

    @Test func tavilySearchServiceMapsUnauthorizedResponsesToInvalidAPIKey() async throws {
        MockTavilyURLProtocol.responseStatusCode = 401
        MockTavilyURLProtocol.responseData = """
        {
          "detail": "Invalid API key"
        }
        """.data(using: .utf8)!

        let service = try makeSearchService()
        var receivedInvalidAPIKey = false

        do {
            _ = try await service.search(query: "latest swift release")
        } catch let error as TavilySearchError {
            if case .invalidAPIKey = error {
                receivedInvalidAPIKey = true
            }
        }

        #expect(receivedInvalidAPIKey)
    }

    @Test func webSearchBridgeEnforcesInvocationLimit() async throws {
        let bridge = try makeSearchBridge()

        for index in 0..<AppWebSearchToolBridge.maxInvocations {
            let result = try await bridge.executeSearch(query: "query \(index)")
            #expect(result.contains("Swift 6.2 Released"))
        }

        let limited = try await bridge.executeSearch(query: "query overflow")

        #expect(bridge.allInvocations.count == AppWebSearchToolBridge.maxInvocations)
        #expect(bridge.searchLimitReached)
        #expect(AppWebSearchToolBridge.isSearchLimitToolResponse(limited))
    }

    @Test func qwenToolTemplateInspectionPrefersXMLFunctionFormat() {
        let packageContents = """
        <tool_call>
        <function=webSearch>
        <parameter=query>
        latest OpenAI model
        </parameter>
        </function>
        </tool_call>
        """

        let inferredFormat = MLXModelManager.inferToolCallFormat(
            packageContents: packageContents,
            modelType: "qwen3_5"
        )

        #expect(inferredFormat == .xmlFunction)
        #expect(MLXModelManager.usesWrappedXMLToolCallTemplate(packageContents: packageContents))
    }

    @Test func wrappedXMLToolCallStreamFilterSuppressesToolMarkup() async {
        let filter = WrappedXMLToolCallStreamFilter()

        let firstChunk = await filter.consume("I will check that.\n<tool")
        let secondChunk = await filter.consume("_call>\n<function=webSearch>")
        let thirdChunk = await filter.consume("\n<parameter=query>\nlatest OpenAI model\n</parameter>")
        await filter.didDispatchToolCall()
        let trailingChunk = await filter.finish()

        #expect(firstChunk == "I will check that.\n")
        #expect(secondChunk == nil)
        #expect(thirdChunk == nil)
        #expect(trailingChunk == nil)
    }

    @Test func qwenToolArgumentNormalizationUnwrapsJSONValueStrings() {
        let normalized = MLXModelManager.normalizedToolArguments([
            "query": JSONValue.string("OpenAI latest model")
        ])

        #expect(normalized["query"] as? String == "OpenAI latest model")
    }

    @Test func mlxToolLoopDisablesToolsOnlyAfterSearchLimitResponse() {
        #expect(
            MLXModelManager.shouldDisableTools(
                after: AppWebSearchToolBridge.searchLimitToolResponse(limit: AppWebSearchToolBridge.maxInvocations)
            )
        )
        #expect(
            !MLXModelManager.shouldDisableTools(
                after: AppWebSearchToolBridge.invalidArgumentsToolResponse()
            )
        )
    }

    @Test func mlxToolLoopHasHardStopResponse() {
        let response = MLXModelManager.excessiveToolCallToolResponse(
            maximum: MLXModelManager.maxToolInvocationsPerResponse
        )

        #expect(response.contains("limit reached"))
        #expect(response.contains("\(MLXModelManager.maxToolInvocationsPerResponse)"))
    }

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

    @Test func currentDateTimeContextIncludesExactDateAndYearHint() {
        let referenceDate = Date(timeIntervalSince1970: 1_762_845_600) // 2026-03-07 12:00:00 UTC
        let utc = TimeZone(secondsFromGMT: 0)!

        let context = ChatViewModel.currentDateTimeContext(
            referenceDate: referenceDate,
            timeZone: utc
        )

        #expect(context.contains("Saturday, March 7, 2026 at 12:00 UTC"))
        #expect(context.contains("current year"))
        #expect(context.contains("exact date"))
    }

    @Test func searchSessionParserKeepsBulletAnswerAfterThinkingCloses() {
        let parsed = ChatViewModel.parseReasoningResponseForSearchSessionText(
            """
            I should verify the release notes.
            </think>

            Based on the search, Swift 6.2 is the latest stable release.

            - Adds more concurrency fixes
            - Available now
            """
        )

        #expect(parsed.reasoning == "I should verify the release notes.")
        #expect(parsed.finalAnswer == """
        Based on the search, Swift 6.2 is the latest stable release.

        - Adds more concurrency fixes
        - Available now
        """)
    }

    @Test func webSearchBridgeReturnsRecoverableErrorForMissingQueryArgument() async throws {
        let bridge = try makeSearchBridge()
        let toolCall = MLXToolCall(function: .init(
            name: AppWebSearchToolBridge.toolName,
            arguments: [:]
        ))

        let result = try await bridge.dispatchMLXToolCall(toolCall)

        #expect(AppWebSearchToolBridge.isInternalToolErrorResponse(result))
        #expect(bridge.allInvocations.isEmpty)
    }

    @Test func taskBackedAsyncThrowingStreamCancelsProducerWhenConsumerStops() async {
        let probe = CancellationProbe()
        let stream: AsyncThrowingStream<String, Error> = TaskBackedAsyncThrowingStream.make { continuation in
            Task {
                do {
                    try await Task.sleep(for: .seconds(30))
                    continuation.finish()
                } catch is CancellationError {
                    await probe.markCancelled()
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }

        let consumer = Task {
            var iterator = stream.makeAsyncIterator()
            _ = try? await iterator.next()
        }

        try? await Task.sleep(for: .milliseconds(50))
        consumer.cancel()
        _ = await consumer.result

        #expect(await waitForCancellation(probe))
    }

    private func makeSearchBridge() throws -> AppWebSearchToolBridge {
        MockTavilyURLProtocol.responseStatusCode = 200
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

        let service = try makeSearchService()
        return AppWebSearchToolBridge(searchService: service)
    }

    private func makeSearchService() throws -> TavilySearchService {
        MockTavilyURLProtocol.lastRequest = nil
        MockTavilyURLProtocol.lastRequestBody = nil
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockTavilyURLProtocol.self]
        let session = URLSession(configuration: configuration)
        return try TavilySearchService(apiKey: "test-key", session: session)
    }

}

private final class MockTavilyURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var responseData = Data()
    nonisolated(unsafe) static var responseStatusCode = 200
    nonisolated(unsafe) static var lastRequest: URLRequest?
    nonisolated(unsafe) static var lastRequestBody: Data?

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "api.tavily.com"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        Self.lastRequest = request
        Self.lastRequestBody = Self.requestBody(from: request)
        let response = HTTPURLResponse(
            url: request.url ?? URL(string: "https://api.tavily.com/search")!,
            statusCode: Self.responseStatusCode,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.responseData)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func requestBody(from request: URLRequest) -> Data? {
        if let body = request.httpBody {
            return body
        }

        guard let stream = request.httpBodyStream else {
            return nil
        }

        stream.open()
        defer { stream.close() }

        let bufferSize = 1024
        var data = Data()
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }

        while stream.hasBytesAvailable {
            let read = stream.read(buffer, maxLength: bufferSize)
            if read < 0 {
                return data.isEmpty ? nil : data
            }
            if read == 0 {
                break
            }
            data.append(buffer, count: read)
        }

        return data.isEmpty ? nil : data
    }
}

private actor CancellationProbe {
    private var cancelled = false

    func markCancelled() {
        cancelled = true
    }

    func isCancelled() -> Bool {
        cancelled
    }
}

private func waitForCancellation(_ probe: CancellationProbe) async -> Bool {
    for _ in 0..<20 {
        if await probe.isCancelled() {
            return true
        }
        try? await Task.sleep(for: .milliseconds(25))
    }
    return await probe.isCancelled()
}
