//
//  VisionObjectDetector.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 11/16/25.
//  Pure Apple Vision Framework object detector - no external models required
//

import Foundation
import Vision
import UIKit
import Combine

/// Pure Vision Framework object detector
/// Uses multiple Apple Vision APIs to detect objects without external ML models
final class VisionObjectDetector: ObservableObject {
    @MainActor @Published var isProcessing = false
    @MainActor @Published var lastError: String?
    
    // Detection thresholds
    private var confidenceThreshold: Float {
        let value = UserDefaults.standard.object(forKey: "visionConfidenceThreshold") as? Double ?? 0.5
        return Float(value)
    }
    
    private let maxDetections: Int = 50
    
    init() {
        print("✅ VisionObjectDetector initialized - using pure Apple Vision Framework")
    }
    
    // MARK: - Main Detection Method
    
    /// Detect objects in an image using Vision Framework APIs
    func detectObjects(in image: UIImage) async throws -> [DetectedObject] {
        print("🔍 Starting Vision Framework object detection...")
        
        guard let cgImage = image.cgImage else {
            print("❌ Could not get CGImage from UIImage")
            throw VisionDetectorError.invalidImage
        }
        
        await MainActor.run { isProcessing = true }
        defer { Task { @MainActor in isProcessing = false } }
        
        // Run Vision detection methods
        let animals = await detectAnimals(cgImage: cgImage)
        
        // Combine all detections
        var allDetections: [DetectedObject] = []
        allDetections.append(contentsOf: animals)
        
        // Apply NMS to remove duplicates
        let filteredDetections = applyNMS(to: allDetections)
        
        // Sort by confidence and limit results
        let sortedDetections = filteredDetections.sorted { $0.confidence > $1.confidence }
        let finalDetections = Array(sortedDetections.prefix(maxDetections))
        
        print("✅ Vision Framework detected \(finalDetections.count) objects")
        return finalDetections
    }
    
    // MARK: - Individual Detection Methods
    
    /// Detect animals (cats, dogs) using VNRecognizeAnimalsRequest
    private func detectAnimals(cgImage: CGImage) async -> [DetectedObject] {
        await withCheckedContinuation { continuation in
            print("🐾 Running animal detection...")
            
            let request = VNRecognizeAnimalsRequest { request, error in
                if let error = error {
                    print("❌ Animal detection failed: \(error.localizedDescription)")
                    continuation.resume(returning: [])
                    return
                }
                
                guard let observations = request.results as? [VNRecognizedObjectObservation] else {
                    print("⚠️ No animal observations")
                    continuation.resume(returning: [])
                    return
                }
                
                let detections = observations.compactMap { observation -> DetectedObject? in
                    guard let topLabel = observation.labels.first,
                          topLabel.confidence >= self.confidenceThreshold else {
                        return nil
                    }
                    
                    return DetectedObject(
                        label: topLabel.identifier.capitalized,
                        confidence: topLabel.confidence,
                        boundingBox: observation.boundingBox
                    )
                }
                
                print("✅ Detected \(detections.count) animal(s)")
                continuation.resume(returning: detections)
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("❌ Animal detection request failed: \(error.localizedDescription)")
                continuation.resume(returning: [])
            }
        }
    }
    
    // MARK: - Non-Maximum Suppression
    
    private func applyNMS(to detections: [DetectedObject]) -> [DetectedObject] {
        var detectionsByClass: [String: [DetectedObject]] = [:]
        for detection in detections {
            detectionsByClass[detection.label, default: []].append(detection)
        }
        
        var finalDetections: [DetectedObject] = []
        
        for (_, classDetections) in detectionsByClass {
            let filtered = nmsForClass(classDetections)
            finalDetections.append(contentsOf: filtered)
        }
        
        return finalDetections
    }
    
    private func nmsForClass(_ detections: [DetectedObject]) -> [DetectedObject] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var keep: [DetectedObject] = []
        var suppressed = Set<UUID>()
        
        for detection in sorted {
            if suppressed.contains(detection.id) { continue }
            
            keep.append(detection)
            
            for other in sorted {
                if other.id == detection.id { continue }
                if suppressed.contains(other.id) { continue }
                
                let iou = calculateIoU(detection.boundingBox, other.boundingBox)
                if iou > 0.5 {
                    suppressed.insert(other.id)
                }
            }
        }
        
        return keep
    }
    
    private func calculateIoU(_ box1: CGRect, _ box2: CGRect) -> Float {
        let intersection = box1.intersection(box2)
        guard !intersection.isNull else { return 0.0 }
        
        let intersectionArea = intersection.width * intersection.height
        let box1Area = box1.width * box1.height
        let box2Area = box2.width * box2.height
        let unionArea = box1Area + box2Area - intersectionArea
        
        guard unionArea > 0 else { return 0.0 }
        return Float(intersectionArea / unionArea)
    }
    
    // MARK: - Utility
    
    var isModelLoaded: Bool {
        // Vision Framework is always available - no external model needed
        true
    }
    
    @MainActor
    var statusMessage: String {
        if let error = lastError {
            return error
        } else if isProcessing {
            return "Analyzing image..."
        } else {
            return "Vision Framework ready"
        }
    }
}

// MARK: - Error Types

enum VisionDetectorError: LocalizedError {
    case invalidImage
    case processingFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "The provided image is invalid or cannot be processed."
        case .processingFailed(let message):
            return "Object detection failed: \(message)"
        }
    }
}
