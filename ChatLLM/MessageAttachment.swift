//
//  Attachment.swift
//  ChatLLM
//
//  Created by Nevio on 10/24/25.
//
//  CRITICAL: Image Persistence Strategy
//  =====================================
//  This model stores image attachments with RELATIVE PATHS instead of absolute URLs.
//
//  Why? iOS app containers can move between launches, especially during:
//  - App updates
//  - Device restores
//  - iOS updates
//
//  If we stored absolute URLs (e.g., /var/mobile/.../Documents/Attachments/image.jpg),
//  they would become invalid when the container path changes, resulting in
//  "Image unavailable" errors after app restarts.
//
//  Solution: Store only the filename in `relativePath`, then reconstruct the full
//  path dynamically using FileManager when accessing `fileURL`.
//

import Foundation
import SwiftData
import UIKit
import OSLog

nonisolated private let attachmentLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "ChatLLM", category: "MessageAttachment")

enum AttachmentType: String, Codable, Sendable {
    case image
    case file
}

@Model
final class MessageAttachment {
    @Attribute(.unique) var id: UUID
    var type: AttachmentType
    
    // Store the file URL - but we'll normalize it to be relative-path-based
    var fileURL: URL
    var createdAt: Date
    var fileName: String
    
    @Relationship var message: Message?
    
    init(
        id: UUID = UUID(),
        type: AttachmentType,
        fileURL: URL,
        fileName: String,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.type = type
        // Store a relative URL with just the filename so it stays container-agnostic.
        self.fileURL = URL(string: fileURL.lastPathComponent) ?? URL(fileURLWithPath: fileURL.lastPathComponent)
        self.fileName = fileName
        self.createdAt = createdAt
    }
    
    /// Get the actual file URL in the current app container
    var actualFileURL: URL {
        // If the stored URL is already absolute and exists, use it (backward compatibility)
        if fileURL.path.starts(with: "/") && FileManager.default.fileExists(atPath: fileURL.path) {
            return fileURL
        }
        
        // Otherwise, reconstruct from Documents directory
        do {
            let documentsURL = try FileManager.default.url(
                for: .documentDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: false
            )
            let attachmentsDir = documentsURL.appendingPathComponent("Attachments", isDirectory: true)
            return attachmentsDir.appendingPathComponent(fileURL.lastPathComponent)
        } catch {
            attachmentLogger.warning("Error constructing attachment file URL: \((error as NSError).localizedDescription, privacy: .public)")
            return fileURL // Fallback to stored URL
        }
    }
    
    /// Load the image from disk
    func loadImage() async -> UIImage? {
        guard type == .image else { return nil }
        
        // Use actualFileURL which handles both old absolute paths and new relative paths
        let imageURL = actualFileURL

        if let image = await DiskBackedImageLoader.loadImage(at: imageURL) {
            return image
        }

        attachmentLogger.warning("Failed to load attachment image \(imageURL.lastPathComponent, privacy: .public); file exists: \(FileManager.default.fileExists(atPath: imageURL.path), privacy: .public)")
        return nil
    }
}

extension MessageAttachment {
    /// Files still used by saved chats. Fetches attachment rows directly, so cleanup
    /// doesn't fault in every message of every chat just to reach their attachments.
    static func referencedFileURLs(
        in context: ModelContext,
        excludingConversationIDs excludedConversationIDs: Set<UUID> = []
    ) throws -> Set<URL> {
        let attachments = try context.fetch(FetchDescriptor<MessageAttachment>())
        return Set(attachments.compactMap { attachment in
            guard let conversationID = attachment.message?.conversation?.id,
                  !excludedConversationIDs.contains(conversationID) else { return nil }
            return attachment.actualFileURL
        })
    }
}
