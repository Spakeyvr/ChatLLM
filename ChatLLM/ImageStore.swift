// ImageStore.swift
//
//  ImageStore.swift
//  ChatLLM
//
//  Created by Nevio on 10/26/25.
//

import UIKit
import os.log
import ImageIO

actor ImageStore {
    static let shared = ImageStore()

    private let logger = Logger(subsystem: "com.yourapp.chatllm", category: "ImageStore")
    private let nativeInferenceMaxDimension: CGFloat = 336
    private let cleanupProtectionWindow: TimeInterval = 15

    struct ImageMetrics: Sendable {
        let width: Int
        let height: Int
        let byteSize: Int64
    }

    // Save the canonical attachment image used by UI/history/Vision fallback.
    func save(image: UIImage, maxDimension: CGFloat = 1200, jpegQuality: CGFloat = 0.75) async throws -> URL {
        // Validate input dimensions and scale
        guard image.size.width > 0 && image.size.height > 0 && image.scale > 0 else {
            logger.error("Error: Cannot save image with invalid dimensions or scale: \(image.size.width)x\(image.size.height) @ \(image.scale)x")
            throw NSError(domain: "ImageStore", code: -2, userInfo: [
                NSLocalizedDescriptionKey: "Image has invalid dimensions or scale"
            ])
        }
        
        // Validate that image has actual bitmap data
        guard image.cgImage != nil else {
            logger.error("Error: Image has no CGImage backing")
            throw NSError(domain: "ImageStore", code: -3, userInfo: [
                NSLocalizedDescriptionKey: "Image has no bitmap data"
            ])
        }

        let scaled = await downscale(image: image, maxDimension: maxDimension)
        guard let data = autoreleasepool(invoking: {
            scaled.jpegData(compressionQuality: jpegQuality)
        }) else {
            logger.error("Error: JPEG encoding failed for image")
            throw NSError(domain: "ImageStore", code: -1, userInfo: [NSLocalizedDescriptionKey: "JPEG encoding failed"])
        }
        
        let dir = try directory()
        let url = dir.appendingPathComponent(UUID().uuidString).appendingPathExtension("jpg")
        
        do {
            try data.write(to: url, options: [.atomic, .completeFileProtection])
            logger.info("Image saved: \(url.lastPathComponent) (\(data.count) bytes)")
            return url
        } catch {
            logger.error("Error: Failed to write image to disk: \(error.localizedDescription)")
            throw error
        }
    }

    /// Create (or reuse) the Qwen native-inference image derivative.
    /// The canonical attachment remains unchanged; this writes into Attachments/Inference/.
    func saveInferenceVariant(
        from sourceURL: URL,
        maxDimension: CGFloat = 336,
        jpegQuality: CGFloat = 0.90
    ) async throws -> URL {
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw NSError(domain: "ImageStore", code: -4, userInfo: [
                NSLocalizedDescriptionKey: "Source image not found for inference variant"
            ])
        }

        let constrainedDimension = max(64, min(maxDimension, nativeInferenceMaxDimension))
        let outputURL = try inferenceVariantURL(for: sourceURL, maxDimension: constrainedDimension)

        if FileManager.default.fileExists(atPath: outputURL.path),
           let existing = metricsSync(at: outputURL),
           max(existing.width, existing.height) <= Int(constrainedDimension.rounded(.up)) {
            return outputURL
        }

        let rendered = try autoreleasepool {
            try renderInferenceVariant(
                from: sourceURL,
                maxDimension: constrainedDimension,
                jpegQuality: jpegQuality
            )
        }

        try rendered.data.write(to: outputURL, options: [.atomic, .completeFileProtection])
        logger.info("Inference image ready \(outputURL.lastPathComponent, privacy: .public) [\(rendered.sourceWidth)x\(rendered.sourceHeight) -> \(rendered.outputWidth)x\(rendered.outputHeight)] (\(rendered.data.count) bytes)")
        return outputURL
    }

    func delete(url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            try FileManager.default.removeItem(at: url)
            logger.debug("Deleted image: \(url.lastPathComponent)")
        } catch {
            logger.error("Error: Failed to delete image \(url.lastPathComponent): \(error.localizedDescription)")
            throw error
        }
    }

    func deleteInferenceVariant(for sourceURL: URL, maxDimension: CGFloat = 512) async {
        do {
            let url = try inferenceVariantURL(for: sourceURL, maxDimension: max(64, min(maxDimension, nativeInferenceMaxDimension)))
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
                logger.debug("Deleted inference variant: \(url.lastPathComponent)")
            }
        } catch {
            logger.error("Error: Failed to delete inference variant for \(sourceURL.lastPathComponent): \(error.localizedDescription)")
        }
    }

    func imageMetrics(at url: URL) async -> ImageMetrics? {
        metricsSync(at: url)
    }

    /// Clean up orphaned image files that are no longer referenced
    func cleanupOrphanedFiles(referencedURLs: Set<URL>) async throws {
        let canonicalDir = try directory()
        let inferenceDir = try inferenceDirectory()
        let fileManager = FileManager.default

        guard let canonicalContents = try? fileManager.contentsOfDirectory(at: canonicalDir, includingPropertiesForKeys: [.fileSizeKey]) else {
            return
        }

        var deletedCount = 0
        var freedBytes: Int64 = 0

        for fileURL in canonicalContents where fileURL.pathExtension == "jpg" {
            if !referencedURLs.contains(fileURL) {
                guard shouldDeleteDuringCleanup(fileURL) else { continue }
                // Orphaned file found
                if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    freedBytes += Int64(size)
                }
                try? fileManager.removeItem(at: fileURL)
                deletedCount += 1
            }
        }

        let expectedInferenceNames = Set(
            referencedURLs.map {
                inferenceFileName(for: $0, maxDimension: nativeInferenceMaxDimension)
            }
        )

        if let inferenceContents = try? fileManager.contentsOfDirectory(at: inferenceDir, includingPropertiesForKeys: [.fileSizeKey]) {
            for fileURL in inferenceContents where fileURL.pathExtension == "jpg" {
                if !expectedInferenceNames.contains(fileURL.lastPathComponent) {
                    guard shouldDeleteDuringCleanup(fileURL) else { continue }
                    if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                        freedBytes += Int64(size)
                    }
                    try? fileManager.removeItem(at: fileURL)
                    deletedCount += 1
                }
            }
        }

        if deletedCount > 0 {
            logger.info("Cleaned up \(deletedCount) orphaned images (freed \(ByteCountFormatter.string(fromByteCount: freedBytes, countStyle: .file)))")
        }
    }

    private func shouldDeleteDuringCleanup(_ fileURL: URL) -> Bool {
        let resourceValues = try? fileURL.resourceValues(forKeys: [.creationDateKey, .contentModificationDateKey])
        let freshestDate = [resourceValues?.contentModificationDate, resourceValues?.creationDate]
            .compactMap { $0 }
            .max()

        guard let freshestDate else {
            return true
        }

        return Date().timeIntervalSince(freshestDate) >= cleanupProtectionWindow
    }

    private func directory() throws -> URL {
        let dir = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("Attachments", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: nil)
        return dir
    }

    private func inferenceDirectory() throws -> URL {
        let dir = try directory().appendingPathComponent("Inference", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: nil)
        return dir
    }

    private func inferenceVariantURL(for sourceURL: URL, maxDimension: CGFloat) throws -> URL {
        let dir = try inferenceDirectory()
        let fileName = inferenceFileName(for: sourceURL, maxDimension: maxDimension)
        return dir.appendingPathComponent(fileName)
    }

    private func inferenceFileName(for sourceURL: URL, maxDimension: CGFloat) -> String {
        let seed = sourceURL.deletingPathExtension().lastPathComponent
        let hash = stableHash(seed)
        let dimension = Int(maxDimension.rounded())
        return "qwen_\(hash)_\(dimension).jpg"
    }

    private func stableHash(_ input: String) -> String {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in input.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return String(hash, radix: 16)
    }

    private struct RenderedVariant {
        let data: Data
        let sourceWidth: Int
        let sourceHeight: Int
        let outputWidth: Int
        let outputHeight: Int
    }

    private func renderInferenceVariant(
        from sourceURL: URL,
        maxDimension: CGFloat,
        jpegQuality: CGFloat
    ) throws -> RenderedVariant {
        let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, sourceOptions) else {
            throw NSError(domain: "ImageStore", code: -5, userInfo: [
                NSLocalizedDescriptionKey: "Unable to read image source for inference variant"
            ])
        }

        let sourceProps = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let srcW = (sourceProps?[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue ?? 0
        let srcH = (sourceProps?[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue ?? 0
        let maxPixel = max(64, Int(maxDimension.rounded(.up)))

        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel,
            kCGImageSourceShouldCache: false,
            kCGImageSourceShouldCacheImmediately: false
        ]

        guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw NSError(domain: "ImageStore", code: -6, userInfo: [
                NSLocalizedDescriptionKey: "Unable to create thumbnail for inference variant"
            ])
        }

        let renderedImage = UIImage(cgImage: thumbnail)
        guard let data = renderedImage.jpegData(compressionQuality: jpegQuality) else {
            throw NSError(domain: "ImageStore", code: -7, userInfo: [
                NSLocalizedDescriptionKey: "Unable to encode inference variant"
            ])
        }

        return RenderedVariant(
            data: data,
            sourceWidth: srcW,
            sourceHeight: srcH,
            outputWidth: thumbnail.width,
            outputHeight: thumbnail.height
        )
    }

    private func metricsSync(at url: URL) -> ImageMetrics? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }

        let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithURL(url as CFURL, sourceOptions),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] else {
            return nil
        }

        let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue ?? 0
        let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue ?? 0
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? 0
        return ImageMetrics(width: width, height: height, byteSize: size)
    }

    private func downscale(image: UIImage, maxDimension: CGFloat) async -> UIImage {
        let pixelWidth = image.size.width * image.scale
        let pixelHeight = image.size.height * image.scale
        let maxSide = max(pixelWidth, pixelHeight)
        guard maxSide > maxDimension else { 
            logger.debug("Image already within max dimension (\(maxSide) <= \(maxDimension)), skipping downscale")
            return image 
        }

        let scale = maxDimension / maxSide
        let newSize = CGSize(
            width: (pixelWidth * scale).rounded(),
            height: (pixelHeight * scale).rounded()
        )
        
        logger.debug("Downscaling image from \(Int(pixelWidth))x\(Int(pixelHeight)) to \(Int(newSize.width))x\(Int(newSize.height))")

        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let img = autoreleasepool(invoking: {
                    let format = UIGraphicsImageRendererFormat()
                    format.scale = 1.0
                    let renderer = UIGraphicsImageRenderer(size: newSize, format: format)
                    return renderer.image { _ in
                        image.draw(in: CGRect(origin: .zero, size: newSize))
                    }
                })
                continuation.resume(returning: img)
            }
        }
    }
}

enum DiskBackedImageLoader {
    private static let thumbnailCache: NSCache<NSString, UIImage> = {
        let cache = NSCache<NSString, UIImage>()
        cache.countLimit = 24
        cache.totalCostLimit = 24 * 1_048_576
        return cache
    }()

    static func loadThumbnail(at url: URL, maxPixelSize: Int) async -> UIImage? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }

        let constrainedPixelSize = max(64, maxPixelSize)
        let cacheKey = "\(url.standardizedFileURL.path)#\(constrainedPixelSize)" as NSString
        if let cached = thumbnailCache.object(forKey: cacheKey) {
            return cached
        }

        let thumbnail = await Task.detached(priority: .userInitiated) {
            autoreleasepool { () -> UIImage? in
                let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
                guard let source = CGImageSourceCreateWithURL(url as CFURL, sourceOptions) else {
                    return nil
                }
                let thumbnailOptions: [CFString: Any] = [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceThumbnailMaxPixelSize: constrainedPixelSize,
                    kCGImageSourceShouldCacheImmediately: true
                ]
                guard let cgImage = CGImageSourceCreateThumbnailAtIndex(
                    source,
                    0,
                    thumbnailOptions as CFDictionary
                ) else {
                    return nil
                }
                return UIImage(cgImage: cgImage)
            }
        }.value

        if let thumbnail, let cgImage = thumbnail.cgImage {
            thumbnailCache.setObject(
                thumbnail,
                forKey: cacheKey,
                cost: cgImage.bytesPerRow * cgImage.height
            )
        }
        return thumbnail
    }

    static func loadImage(at url: URL) async -> UIImage? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }

        return await Task.detached(priority: .userInitiated) {
            autoreleasepool {
                guard let data = try? Data(contentsOf: url) else { return nil }
                return UIImage(data: data)
            }
        }.value
    }
}
