//
//  ChatLLMTestHelpers.swift
//  ChatLLMTests
//
//  Shared mocks, fakes, and harnesses for the ChatLLMTests suite.
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

internal enum StaleTavilyValidationError: Error {
    case rejected
}

internal actor TavilyValidationGate {
    private var startedKeys: Set<String> = []
    private var continuations: [String: CheckedContinuation<Void, Never>] = [:]

    func suspend(_ key: String) async {
        startedKeys.insert(key)
        await withCheckedContinuation { continuation in
            continuations[key] = continuation
        }
    }

    func waitUntilStarted(_ key: String) async {
        while !startedKeys.contains(key) {
            await Task.yield()
        }
    }

    func resume(_ key: String) {
        continuations.removeValue(forKey: key)?.resume()
    }
}

internal final class MockTavilyURLProtocol: URLProtocol, @unchecked Sendable {
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

internal actor CancellationProbe {
    private var cancelled = false

    func markCancelled() {
        cancelled = true
    }

    func isCancelled() -> Bool {
        cancelled
    }
}

internal final class FakePathMonitor: PathMonitoring {
    nonisolated(unsafe) var pathUpdateHandler: ((NetworkPathStatus) -> Void)?

    func start(queue: DispatchQueue) {
        _ = queue
    }

    func cancel() {}

    func emit(_ status: NetworkPathStatus) {
        pathUpdateHandler?(status)
    }
}

func waitForCancellation(_ probe: CancellationProbe) async -> Bool {
    for _ in 0..<20 {
        if await probe.isCancelled() {
            return true
        }
        try? await Task.sleep(for: .milliseconds(25))
    }
    return await probe.isCancelled()
}

internal struct TestLLMGenerator: LLMGenerator {
    func isAvailable() -> Bool { true }

    func respond(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> String {
        ""
    }

    func streamResponse(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }
}

internal struct BlockingLLMGenerator: LLMGenerator {
    func isAvailable() -> Bool { true }

    func respond(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> String {
        ""
    }

    func streamResponse(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    try await Task.sleep(for: .seconds(30))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                producer.cancel()
            }
        }
    }
}

internal struct PartialReasoningLLMGenerator: LLMGenerator {
    func isAvailable() -> Bool { true }

    func respond(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> String {
        "<thinking>\nAnalyze the request carefully.\n</thinking>"
    }

    func streamResponse(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield("<thinking>\nAnalyze the request carefully.\n</thinking>")
            continuation.finish()
        }
    }
}

@MainActor
internal final class MarkdownWebTestHarness {
    var height: CGFloat = 1
    var hasMeasuredHeight = false
    var failedToLoad = false

    lazy var coordinator = RichMarkdownWebViewRepresentable.Coordinator(
        dynamicHeight: Binding(get: { [weak self] in MainActor.assumeIsolated { self?.height ?? 1 } }, set: { [weak self] value in MainActor.assumeIsolated { self?.height = value } }),
        hasMeasuredHeight: Binding(get: { [weak self] in MainActor.assumeIsolated { self?.hasMeasuredHeight ?? false } }, set: { [weak self] value in MainActor.assumeIsolated { self?.hasMeasuredHeight = value } }),
        failedToLoad: Binding(get: { [weak self] in MainActor.assumeIsolated { self?.failedToLoad ?? false } }, set: { [weak self] value in MainActor.assumeIsolated { self?.failedToLoad = value } }),
        openURL: OpenURLAction { _ in .handled }
    )
    lazy var webView: WKWebView = {
        let view = coordinator.makeWebView()
        view.frame = CGRect(x: 0, y: 0, width: 320, height: 400)
        return view
    }()

    func update(_ text: String, fontSize: Double = 16, colorScheme: ColorScheme = .light) {
        coordinator.update(webView, text: text, fontSize: fontSize,
                           palette: RichTextPalette(colorScheme: colorScheme, tone: .primary))
    }

    func waitFor(_ condition: String) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(15))
        while ContinuousClock.now < deadline {
            if hasMeasuredHeight,
               (try? await webView.evaluateJavaScript(condition)) as? Bool == true {
                // Allow the height binding dispatched from the script message to settle.
                try await Task.sleep(for: .milliseconds(50))
                return
            }
            if failedToLoad { break }
            try await Task.sleep(for: .milliseconds(25))
        }
        throw NSError(domain: "MarkdownWebTest", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Renderer did not satisfy: \(condition); failed: \(failedToLoad)"])
    }

    func close() {
        coordinator.dismantle(webView)
    }
}

internal struct ControlledMarkdownGenerator: LLMGenerator {
    let stream: AsyncThrowingStream<String, Error>
    func isAvailable() -> Bool { true }
    func respond(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> String { "" }
    func streamResponse(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error> { stream }
}

@MainActor
internal final class CapturingFoundationGenerator: LLMGenerator {
    var request: LLMRequest?
    func isAvailable() -> Bool { true }
    func respond(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> String {
        self.request = request
        return "A plain answer."
    }
    func streamResponse(to request: LLMRequest, tools: [any FoundationModelTool]) async throws -> AsyncThrowingStream<String, Error> {
        self.request = request
        return AsyncThrowingStream { continuation in
            continuation.yield("A plain answer.")
            continuation.finish()
        }
    }
}
