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

    @Test func mlxGenerationCommitRequiresMatchingLoadAndRevision() {
        let loadID = UUID()

        #expect(MLXInferenceWorker.generationStateIsCurrent(
            expectedLoadID: loadID,
            currentLoadID: loadID,
            expectedStateRevision: 4,
            currentStateRevision: 4
        ))
        #expect(!MLXInferenceWorker.generationStateIsCurrent(
            expectedLoadID: loadID,
            currentLoadID: UUID(),
            expectedStateRevision: 4,
            currentStateRevision: 4
        ))
        #expect(!MLXInferenceWorker.generationStateIsCurrent(
            expectedLoadID: loadID,
            currentLoadID: loadID,
            expectedStateRevision: 4,
            currentStateRevision: 5
        ))
    }

    @Test func promptPrefixCacheServesTextPromptsThatOpenWithSystemTurn() {
        #expect(MLXPromptPrefixCache.isEligible(messages: [.system("System"), .user("Hi")]))
        #expect(MLXPromptPrefixCache.isEligible(
            messages: [.system("System"), .user("Hi"), .assistant("Hello"), .user("Again")]
        ))
        #expect(!MLXPromptPrefixCache.isEligible(messages: [.user("Hi"), .assistant("Hello")]))
        #expect(!MLXPromptPrefixCache.isEligible(messages: [.system("System")]))
    }

    @Test func promptPrefixStopsBeforeConversationDependentTokens() {
        let shared = (0 ..< Int32(MLXPromptPrefixCache.minimumPrefixTokens)).map { $0 + 1_000 }
        let full = shared + [7, 8, 9]

        // The full prompt may share its first message token with one probe;
        // the shortest match keeps that token out of the cached prefix.
        #expect(MLXPromptPrefixCache.prefixTokenCount(full: full, probes: [shared + [7, 1], shared + [2, 1]]) == shared.count)
        // A prefix too short to be worth a file is skipped.
        #expect(MLXPromptPrefixCache.prefixTokenCount(full: full, probes: [Array(shared.prefix(3)) + [5]]) == nil)
        // Something must remain to prefill so the model produces logits.
        #expect(MLXPromptPrefixCache.prefixTokenCount(full: shared, probes: [shared + [1]]) == nil)
    }

    @Test func promptPrefixKeyChangesWithModelTokensAndCacheLayout() {
        let tokens: [Int32] = [1, 2, 3]
        let base = MLXPromptPrefixCache.entryKey(
            modelID: "qwen", prefixTokens: tokens[...], parameters: GenerateParameters(maxKVSize: 2_048))

        #expect(base == MLXPromptPrefixCache.entryKey(
            modelID: "qwen", prefixTokens: tokens[...], parameters: GenerateParameters(maxKVSize: 2_048)))
        #expect(base != MLXPromptPrefixCache.entryKey(
            modelID: "lfm", prefixTokens: tokens[...], parameters: GenerateParameters(maxKVSize: 2_048)))
        #expect(base != MLXPromptPrefixCache.entryKey(
            modelID: "qwen", prefixTokens: [1, 2, 4][...], parameters: GenerateParameters(maxKVSize: 2_048)))
        #expect(base != MLXPromptPrefixCache.entryKey(
            modelID: "qwen", prefixTokens: tokens[...], parameters: GenerateParameters(maxKVSize: 4_096)))
        #expect(base != MLXPromptPrefixCache.entryKey(
            modelID: "qwen", prefixTokens: tokens[...],
            parameters: GenerateParameters(cacheCompression: .rotorQuant(RotorQuantConfiguration()))))
    }

    @Test func mlxVisibleMessageSignatureStoresOnlyFixedSizeContentIdentity() {
        let first = MLXVisibleMessageSignature(
            message: Chat.Message(role: .user, content: "A long conversation message")
        )
        let same = MLXVisibleMessageSignature(
            message: Chat.Message(role: .user, content: "A long conversation message")
        )
        let changed = MLXVisibleMessageSignature(
            message: Chat.Message(role: .user, content: "A different conversation message")
        )

        #expect(first == same)
        #expect(first != changed)
        #expect(first.contentFingerprint.count == 32)
    }
}
