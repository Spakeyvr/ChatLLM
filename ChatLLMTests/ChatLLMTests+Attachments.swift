//
//  ChatLLMTests+Attachments.swift
//  ChatLLMTests
//
//  Split out of ChatLLMTests.swift; part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
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
