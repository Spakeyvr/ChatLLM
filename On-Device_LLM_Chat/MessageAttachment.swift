//
//  Attachment.swift
//  On-Device_LLM_Chat
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
    
    // --- UPDATED: Store full VisionAnalysisResult ---
    var analysisResultsData: Data? // JSON-encoded VisionAnalysisResult
    var detectionProcessed: Bool = false
    var detectionSummary: String? // Human-readable summary
    // --- END UPDATE ---
    
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
        // Store a relative URL using file:// scheme with just the filename
        // This makes it container-agnostic
        self.fileURL = URL(fileURLWithPath: fileURL.lastPathComponent)
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
            print("⚠️ Error constructing file URL: \(error)")
            return fileURL // Fallback to stored URL
        }
    }
    
    // MARK: - Detection Results Management
    
    /// --- NEW: Store comprehensive VisionAnalysisResult ---
    @MainActor
    func storeAnalysisResult(_ result: VisionAnalysisResult) {
        let encoder = JSONEncoder()
        if let data = try? encoder.encode(result) {
            self.analysisResultsData = data
            self.detectionProcessed = true
            self.detectionSummary = result.generateSummary() // Use the smart summary
        } else {
            print("❌ Failed to encode VisionAnalysisResult")
        }
    }
    
    /// --- NEW: Retrieve stored VisionAnalysisResult ---
    @MainActor
    func getAnalysisResult() -> VisionAnalysisResult? {
        guard let data = analysisResultsData else { return nil }
        let decoder = JSONDecoder()
        do {
            return try decoder.decode(VisionAnalysisResult.self, from: data)
        } catch {
            print("❌ Failed to decode VisionAnalysisResult: \(error)")
            // Fallback for old [DetectedObject] data
            if let oldDetections = try? decoder.decode([DetectedObject].self, from: data) {
                print("⚠️ Decoded legacy [DetectedObject] data.")
                // For legacy data, create the result
                return VisionAnalysisResult(objects: oldDetections)
            }
            return nil
        }
    }
    
    /// --- LEGACY (for compatibility if needed, but prefer getAnalysisResult) ---
    @MainActor
    func getDetectionResults() -> [DetectedObject]? {
        return getAnalysisResult()?.objects
    }
    
    /// Format detection results as a human-readable summary
    private func formatDetectionSummary(_ detections: [DetectedObject]) -> String {
        // This is now just a fallback, as storeAnalysisResult sets detectionSummary
        guard !detections.isEmpty else {
            return "No objects detected"
        }
        
        if detections.count == 1 {
            return "Detected: \(detections[0].description)"
        } else if detections.count <= 3 {
            let descriptions = detections.map { $0.label }.joined(separator: ", ")
            return "Detected: \(descriptions)"
        } else {
            let topThree = detections.prefix(3).map { $0.label }.joined(separator: ", ")
            return "Detected: \(topThree), and \(detections.count - 3) more"
        }
    }
    
    /// Load the image from disk
    func loadImage() async -> UIImage? {
        guard type == .image else { return nil }
        
        // Use actualFileURL which handles both old absolute paths and new relative paths
        let imageURL = actualFileURL
        
        // Perform file I/O on a background task
        return await Task.detached(priority: .userInitiated) {
            // Try to load from the reconstructed path
            if let data = try? Data(contentsOf: imageURL),
               let image = UIImage(data: data) {
                return image
            } else {
                print("⚠️ Failed to load image from: \(imageURL.path)")
                print("   File exists: \(FileManager.default.fileExists(atPath: imageURL.path))")
                return nil
            }
        }.value
    }
    
    /// Get detailed detection information for display
    @MainActor
    var detectionInfo: String? {
        guard detectionProcessed else { return nil }
        
        // --- UPDATED to use new summary ---
        if let summary = self.detectionSummary, !summary.isEmpty {
            return summary
        }
        
        // Fallback
        if let detections = getDetectionResults(), !detections.isEmpty {
            return formatDetectionSummary(detections)
        } else {
            return "No objects detected"
        }
    }
}

// MARK: - Migration Utilities

extension MessageAttachment {
    /// Diagnose why an image might not be loading
    func diagnoseImageIssue() -> String {
        var issues: [String] = []
        
        // Check stored URL
        issues.append("Stored URL: \(fileURL.path)")
        
        // Check reconstructed URL
        let url = actualFileURL
        issues.append("Actual URL: \(url.path)")
        
        // Check if file exists
        let fileExists = FileManager.default.fileExists(atPath: url.path)
        issues.append("File exists: \(fileExists)")
        
        // Check if we can get file size
        if fileExists {
            if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
               let size = attrs[.size] as? Int {
                issues.append("File size: \(size) bytes")
            }
        }
        
        // Check Attachments directory
        if let docsURL = try? FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false
        ) {
            let attachmentsDir = docsURL.appendingPathComponent("Attachments", isDirectory: true)
            let dirExists = FileManager.default.fileExists(atPath: attachmentsDir.path)
            issues.append("Attachments directory exists: \(dirExists)")
            
            if dirExists {
                if let files = try? FileManager.default.contentsOfDirectory(atPath: attachmentsDir.path) {
                    issues.append("Files in Attachments: \(files.count)")
                }
            }
        }
        
        return issues.joined(separator: "\n")
    }
}

// Provide a compatibility alias in case any code refers to `Attachment`
typealias Attachment = MessageAttachment
