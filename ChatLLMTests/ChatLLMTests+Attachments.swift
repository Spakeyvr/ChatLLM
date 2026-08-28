//
//  ChatLLMTests+Attachments.swift
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
    @Test func attachmentStoresAContainerAgnosticFileName() {
        let attachment = MessageAttachment(
            type: .image,
            fileURL: URL(fileURLWithPath: "/var/mobile/Containers/Data/Application/OLD/Documents/Attachments/pic.jpg"),
            fileName: "pic.jpg"
        )

        // Only the file name survives; the old container path must not.
        #expect(!attachment.fileURL.path.contains("OLD"))
        #expect(attachment.fileURL.lastPathComponent == "pic.jpg")
    }

    @Test func attachmentResolvesItsFileInsideTheCurrentContainer() throws {
        let attachment = MessageAttachment(
            type: .image,
            fileURL: URL(fileURLWithPath: "/an/old/container/pic.jpg"),
            fileName: "pic.jpg"
        )

        let documents = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false
        )
        let expected = documents.appendingPathComponent("Attachments").appendingPathComponent("pic.jpg")

        #expect(attachment.actualFileURL.standardizedFileURL == expected.standardizedFileURL)
    }

    @Test func attachmentAnalysisResultRoundTripsWithASummary() {
        let attachment = MessageAttachment(
            type: .image,
            fileURL: URL(fileURLWithPath: "pic.jpg"),
            fileName: "pic.jpg"
        )
        let result = VisionAnalysisResult(
            objects: [DetectedObject(label: "cat", confidence: 0.9, boundingBox: .zero)],
            sceneLabels: ["indoor"]
        )

        attachment.storeAnalysisResult(result)

        #expect(attachment.detectionProcessed)
        #expect(attachment.detectionSummary?.isEmpty == false)
        #expect(attachment.getAnalysisResult()?.objects.map(\.label) == ["cat"])
        #expect(attachment.getDetectionResults()?.count == 1)
    }

    @Test func attachmentDecodesLegacyDetectionPayloads() throws {
        let attachment = MessageAttachment(
            type: .image,
            fileURL: URL(fileURLWithPath: "pic.jpg"),
            fileName: "pic.jpg"
        )
        // Chats saved before the payload became a VisionAnalysisResult still hold
        // a bare object array; those attachments must keep opening.
        attachment.analysisResultsData = try JSONEncoder().encode(
            [DetectedObject(label: "dog", confidence: 0.5, boundingBox: .zero)]
        )

        #expect(attachment.getAnalysisResult()?.objects.map(\.label) == ["dog"])
    }

    @Test func attachmentWithoutAnalysisReturnsNothing() {
        let attachment = MessageAttachment(
            type: .image,
            fileURL: URL(fileURLWithPath: "pic.jpg"),
            fileName: "pic.jpg"
        )

        #expect(attachment.getAnalysisResult() == nil)
        #expect(attachment.getDetectionResults() == nil)
        #expect(!attachment.detectionProcessed)
    }

    // MARK: - Image store

    @Test func imageStoreSavesAReadableDownscaledImage() async throws {
        let store = ImageStore.shared
        let url = try await store.save(image: makeTestImage(width: 2_000, height: 1_000), maxDimension: 200)
        defer { Task { try? await store.delete(url: url) } }

        let metrics = await store.imageMetrics(at: url)
        let unwrapped = try #require(metrics)
        #expect(max(unwrapped.width, unwrapped.height) <= 200)
        #expect(unwrapped.byteSize > 0)
        #expect(FileManager.default.fileExists(atPath: url.path))
    }

    @Test func recentlySavedImagesSurviveOrphanCleanup() async throws {
        // A just-attached image is not yet referenced by a saved message. The
        // protection window is what stops cleanup from deleting it underneath.
        let store = ImageStore.shared
        let url = try await store.save(image: makeTestImage(width: 60, height: 40), maxDimension: 60)
        defer { Task { try? await store.delete(url: url) } }

        try await store.cleanupOrphanedFiles(referencedURLs: [])

        #expect(FileManager.default.fileExists(atPath: url.path))
    }

    @Test func imageStoreRejectsDegenerateImages() async {
        let store = ImageStore.shared
        await #expect(throws: (any Error).self) {
            _ = try await store.save(image: UIImage())
        }
    }

    // MARK: - Conversation editing guards

}
