//
//  ChatLLMTests+Runtime
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

    @Test func modelDownloaderRejectsRemoteTraversalPaths() {
        var rejected = false

        do {
            _ = try ModelDownloader.destinationURLs(
                for: "../escape.txt",
                inside: URL(fileURLWithPath: "/tmp/models", isDirectory: true)
            )
        } catch let error as ModelDownloader.DownloadError {
            if case .invalidRemotePath(let file) = error {
                rejected = file == "../escape.txt"
            }
        } catch {}

        #expect(rejected)
    }

    @Test func modelDownloaderRequestsHuggingFaceBlobMetadata() throws {
        let url = try #require(ModelDownloader.metadataURL(repoId: "owner/model"))
        let components = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))

        #expect(components.path == "/api/models/owner/model")
        #expect(components.queryItems == [URLQueryItem(name: "blobs", value: "true")])
    }

    @Test func backgroundModelDownloadDescriptorRoundTrips() throws {
        let encoded = try #require(BackgroundModelDownloadSession.encodeDescriptor(
            id: "owner/model/model.safetensors",
            destinationPath: "/tmp/model.safetensors.download"
        ))
        let decoded = try #require(
            BackgroundModelDownloadSession.decodeDescriptorForTesting(encoded)
        )

        #expect(decoded.id == "owner/model/model.safetensors")
        #expect(decoded.destinationPath == "/tmp/model.safetensors.download")
    }

    @Test func modelDownloaderKeepsNestedFilesInsideTargetDirectory() throws {
        let targetDir = URL(fileURLWithPath: "/tmp/models", isDirectory: true)
        let urls = try ModelDownloader.destinationURLs(
            for: "nested/config.json",
            inside: targetDir
        )

        #expect(urls.destination.standardizedFileURL.path == "/tmp/models/nested/config.json")
        #expect(urls.temporary.standardizedFileURL.path == "/tmp/models/nested/config.json.download")
    }

    // MARK: - Message display pipeline
}
