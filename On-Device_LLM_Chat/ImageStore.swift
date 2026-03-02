// ImageStore.swift
//
//  ImageStore.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/26/25.
//

import UIKit
import os.log

actor ImageStore {
    static let shared = ImageStore()

    private let logger = Logger(subsystem: "com.yourapp.chatllm", category: "ImageStore")

    // Save a downscaled JPEG to Documents (or Caches). Returns the file URL.
    func save(image: UIImage, maxDimension: CGFloat = 2000, jpegQuality: CGFloat = 0.75) async throws -> URL {
        // Validate input dimensions and scale
        guard image.size.width > 0 && image.size.height > 0 && image.scale > 0 else {
            logger.error("❌ Cannot save image with invalid dimensions or scale: \(image.size.width)x\(image.size.height) @ \(image.scale)x")
            throw NSError(domain: "ImageStore", code: -2, userInfo: [
                NSLocalizedDescriptionKey: "Image has invalid dimensions or scale"
            ])
        }
        
        // Validate that image has actual bitmap data
        guard image.cgImage != nil else {
            logger.error("❌ Image has no CGImage backing")
            throw NSError(domain: "ImageStore", code: -3, userInfo: [
                NSLocalizedDescriptionKey: "Image has no bitmap data"
            ])
        }
        
        let scaled = await downscale(image: image, maxDimension: maxDimension)
        guard let data = scaled.jpegData(compressionQuality: jpegQuality) else {
            logger.error("❌ JPEG encoding failed for image")
            throw NSError(domain: "ImageStore", code: -1, userInfo: [NSLocalizedDescriptionKey: "JPEG encoding failed"])
        }
        
        let dir = try directory()
        let url = dir.appendingPathComponent(UUID().uuidString).appendingPathExtension("jpg")
        
        do {
            try data.write(to: url, options: [.atomic, .completeFileProtection])
            logger.info("✅ Image saved: \(url.lastPathComponent) (\(data.count) bytes)")
            return url
        } catch {
            logger.error("❌ Failed to write image to disk: \(error.localizedDescription)")
            throw error
        }
    }

    func delete(url: URL) throws {
        do {
            try FileManager.default.removeItem(at: url)
            logger.debug("✅ Deleted image: \(url.lastPathComponent)")
        } catch {
            logger.error("❌ Failed to delete image \(url.lastPathComponent): \(error.localizedDescription)")
            throw error
        }
    }
    
    /// Clean up orphaned image files that are no longer referenced
    func cleanupOrphanedFiles(referencedURLs: Set<URL>) async throws {
        let dir = try directory()
        let fileManager = FileManager.default
        
        guard let contents = try? fileManager.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey]) else {
            return
        }
        
        var deletedCount = 0
        var freedBytes: Int64 = 0
        
        for fileURL in contents where fileURL.pathExtension == "jpg" {
            if !referencedURLs.contains(fileURL) {
                // Orphaned file found
                if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    freedBytes += Int64(size)
                }
                try? fileManager.removeItem(at: fileURL)
                deletedCount += 1
            }
        }
        
        if deletedCount > 0 {
            logger.info("🧹 Cleaned up \(deletedCount) orphaned images (freed \(ByteCountFormatter.string(fromByteCount: freedBytes, countStyle: .file)))")
        }
    }

    private func directory() throws -> URL {
        let dir = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("Attachments", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: nil)
            logger.debug("✅ Created attachments directory")
        }
        return dir
    }

    private func downscale(image: UIImage, maxDimension: CGFloat) async -> UIImage {
        let size = image.size
        let maxSide = max(size.width, size.height)
        guard maxSide > maxDimension else { 
            logger.debug("Image already within max dimension (\(maxSide) <= \(maxDimension)), skipping downscale")
            return image 
        }

        let scale = maxDimension / maxSide
        let newSize = CGSize(width: size.width * scale, height: size.height * scale)
        
        logger.debug("Downscaling image from \(Int(size.width))x\(Int(size.height)) to \(Int(newSize.width))x\(Int(newSize.height))")

        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let renderer = UIGraphicsImageRenderer(size: newSize)
                let img = renderer.image { _ in
                    image.draw(in: CGRect(origin: .zero, size: newSize))
                }
                continuation.resume(returning: img)
            }
        }
    }
}
