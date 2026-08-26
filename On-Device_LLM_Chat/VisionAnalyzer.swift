//
//  VisionAnalyzer.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/27/25.
//  Unified Vision Framework analyzer - handles OCR, face detection, body pose,
//  barcodes, hand tracking, photo quality assessment, SCENE, and SALIENCY
//

import Foundation
import Vision
import UIKit
import Combine
import OSLog

// MARK: - Analysis Results

/// Represents a detected object in the image
nonisolated struct DetectedObject: Identifiable, Codable, Sendable {
    let id: UUID
    let label: String
    let confidence: Float
    let boundingBox: CGRect // Normalized 0-1 coordinates
    
    init(id: UUID = UUID(), label: String, confidence: Float, boundingBox: CGRect) {
        self.id = id
        self.label = label
        self.confidence = confidence
        self.boundingBox = boundingBox
    }
    
    var confidencePercentage: String {
        String(format: "%.1f%%", confidence * 100)
    }
    
    var description: String {
        "\(label) (\(confidencePercentage))"
    }
}

/// Text recognized by OCR
nonisolated struct RecognizedText: Identifiable, Codable, Sendable {
    let id: UUID
    let text: String
    let confidence: Float
    let boundingBox: CGRect // Normalized 0-1
    
    init(id: UUID = UUID(), text: String, confidence: Float, boundingBox: CGRect) {
        self.id = id
        self.text = text
        self.confidence = confidence
        self.boundingBox = boundingBox
    }
    
    var confidencePercentage: String {
        String(format: "%.1f%%", confidence * 100)
    }
}

/// Face detection with quality assessment
nonisolated struct DetectedFace: Codable, Sendable {
    let id: UUID
    let boundingBox: CGRect // Normalized 0-1
    let confidence: Float
    let landmarks: FaceLandmarks?
    let quality: Float?
    
    init(id: UUID = UUID(), boundingBox: CGRect, confidence: Float, landmarks: FaceLandmarks? = nil, quality: Float? = nil) {
        self.id = id
        self.boundingBox = boundingBox
        self.confidence = confidence
        self.landmarks = landmarks
        self.quality = quality
    }
    
    var qualityPercentage: String {
        guard let q = quality else { return "N/A" }
        return String(format: "%.0f%%", q * 100)
    }
}

nonisolated struct FaceLandmarks: Codable, Sendable {
    let leftEye: CGPoint?
    let rightEye: CGPoint?
    let nose: CGPoint?
    let mouth: CGPoint?
}

/// Body pose detection
nonisolated struct BodyPose: Codable, Sendable {
    let id: UUID
    let confidence: Float
    let joints: [String: CGPoint] // Normalized 0-1
    
    init(id: UUID = UUID(), confidence: Float, joints: [String: CGPoint]) {
        self.id = id
        self.confidence = confidence
        self.joints = joints
    }
    
    var jointNames: [String] {
        Array(joints.keys).sorted()
    }
}

/// Barcode/QR code
nonisolated struct Barcode: Codable, Sendable {
    let id: UUID
    let type: String
    let payload: String
    let boundingBox: CGRect // Normalized 0-1
    
    init(id: UUID = UUID(), type: String, payload: String, boundingBox: CGRect) {
        self.id = id
        self.type = type
        self.payload = payload
        self.boundingBox = boundingBox
    }
}

/// Hand pose detection
nonisolated struct HandPose: Codable, Sendable {
    let id: UUID
    let confidence: Float
    let chirality: String // "left" or "right"
    let fingerPoints: [String: CGPoint] // Normalized 0-1
    
    init(id: UUID = UUID(), confidence: Float, chirality: String, fingerPoints: [String: CGPoint]) {
        self.id = id
        self.confidence = confidence
        self.chirality = chirality
        self.fingerPoints = fingerPoints
    }
}

/// Photo quality assessment
nonisolated struct PhotoQuality: Codable, Sendable {
    let overallQuality: Float
    let faceCount: Int
    let issues: [String]
    
    var qualityLevel: String {
        if overallQuality > 0.7 {
            return "Good"
        } else if overallQuality > 0.4 {
            return "Fair"
        } else {
            return "Poor"
        }
    }
    
    var description: String {
        let percent = String(format: "%.0f%%", overallQuality * 100)
        var desc = "\(percent) - \(qualityLevel)"
        if !issues.isEmpty {
            desc += " (\(issues.joined(separator: ", ")))"
        }
        return desc
    }
}

// MARK: - Complete Analysis Result

nonisolated struct VisionAnalysisResult: Codable, Sendable {
    let objects: [DetectedObject]
    let textBlocks: [RecognizedText]
    let faces: [DetectedFace]?
    let bodyPoses: [BodyPose]?
    let barcodes: [Barcode]?
    let handPoses: [HandPose]?
    let photoQuality: PhotoQuality?
    
    // --- NEWLY ADDED ---
    let sceneLabels: [String]? // The "vibe check"
    let saliencyRect: CGRect? // The "main character" rect
    // --- END NEW ---
    
    let timestamp: Date
    
    init(
        objects: [DetectedObject] = [],
        textBlocks: [RecognizedText] = [],
        faces: [DetectedFace]? = nil,
        bodyPoses: [BodyPose]? = nil,
        barcodes: [Barcode]? = nil,
        handPoses: [HandPose]? = nil,
        photoQuality: PhotoQuality? = nil,
        // --- NEWLY ADDED ---
        sceneLabels: [String]? = nil,
        saliencyRect: CGRect? = nil,
        // --- END NEW ---
        timestamp: Date = Date()
    ) {
        self.objects = objects
        self.textBlocks = textBlocks
        self.faces = faces
        self.bodyPoses = bodyPoses
        self.barcodes = barcodes
        self.handPoses = handPoses
        self.photoQuality = photoQuality
        // --- NEWLY ADDED ---
        self.sceneLabels = sceneLabels
        self.saliencyRect = saliencyRect
        // --- END NEW ---
        self.timestamp = timestamp
    }
    
    var isEmpty: Bool {
        objects.isEmpty &&
        textBlocks.isEmpty &&
        (faces?.isEmpty ?? true) &&
        (bodyPoses?.isEmpty ?? true) &&
        (barcodes?.isEmpty ?? true) &&
        (handPoses?.isEmpty ?? true) &&
        (sceneLabels?.isEmpty ?? true)
    }
    
    var fullText: String {
        textBlocks.map { $0.text }.joined(separator: " ")
    }
    
    /// Generate a natural language summary for UI
    func generateSummary() -> String {
        var summary = ""
        
        // --- NEW: Lead with the scene "vibe" ---
        if let scene = sceneLabels?.first {
            summary += "Scene: \(scene.capitalized).\n"
        }
        
        if !objects.isEmpty {
            summary += "Detected \(objects.count) object\(objects.count == 1 ? "" : "s"): "
            summary += objects.prefix(5).map { $0.label }.joined(separator: ", ")
            if objects.count > 5 {
                summary += ", and \(objects.count - 5) more"
            }
            summary += ".\n"
        }
        
        if !textBlocks.isEmpty {
            summary += "Found \(textBlocks.count) text block\(textBlocks.count == 1 ? "" : "s").\n"
        }
        
        if let faces = faces, !faces.isEmpty {
            summary += "\(faces.count) face\(faces.count == 1 ? "" : "s") detected.\n"
        }
        
        if let poses = bodyPoses, !poses.isEmpty {
            summary += "\(poses.count) person\(poses.count == 1 ? "" : "s") with body pose.\n"
        }
        
        if let quality = photoQuality {
            summary += "Photo quality: \(quality.qualityLevel).\n"
        }
        
        return summary.isEmpty ? "No analysis results." : summary
    }
    
    // --- NEW: Spatial Reasoning ---
    /// Generates a simple spatial summary
    func generateSpatialSummary() -> String {
        var spatialContext = ""
        guard objects.count > 1 else { return "" }
        
        for (i, obj1) in objects.enumerated() {
            // Only check high-confidence objects for spatial relations
            guard obj1.confidence > 0.6 else { continue }
            
            for (j, obj2) in objects.enumerated() where i != j {
                guard obj2.confidence > 0.6 else { continue }
                
                let rect1 = obj1.boundingBox
                let rect2 = obj2.boundingBox
                
                // 1. obj1 is inside obj2
                if rect2.contains(rect1) {
                    spatialContext += "• The \(obj1.label) is inside the \(obj2.label).\n"
                }
                // 2. obj1 is (mostly) on top of obj2
                else if abs(rect1.maxY - rect2.minY) < 0.05 && // obj1 top edge near obj2 bottom edge
                         abs(rect1.midX - rect2.midX) < 0.1 { // aligned vertically
                    spatialContext += "• The \(obj1.label) is on top of the \(obj2.label).\n"
                }
                // 3. obj1 is next to obj2 (horizontal)
                else if abs(rect1.midY - rect2.midY) < 0.1 { // aligned horizontally
                    if abs(rect1.maxX - rect2.minX) < 0.05 { // obj1 left of obj2
                        spatialContext += "• The \(obj1.label) is next to the \(obj2.label).\n"
                    }
                }
            }
        }
        
        // De-duplicate
        let uniqueContexts = Set(spatialContext.components(separatedBy: "\n").filter { !$0.isEmpty })
        return uniqueContexts.isEmpty ? "" : "SPATIAL CONTEXT:\n" + uniqueContexts.joined(separator: "\n") + "\n"
    }

    /// --- REWRITTEN: Smart summary for LLM context (avoids prompt bloat) ---
    func generateDetailedDescription() -> String {
        var desc = "[IMAGE ANALYSIS]\n\n"
        
        // 1. Scene "Vibe Check"
        if let scenes = sceneLabels, !scenes.isEmpty {
            desc += "SCENE TYPE:\n  Looks like: \(scenes.joined(separator: ", ").capitalized)\n\n"
        }
        
        // 2. Saliency "Main Character"
        let mainObjects: [DetectedObject]
        if let saliencyRect = self.saliencyRect {
            // Filter objects that are in the "hot zone"
            mainObjects = objects.filter { $0.boundingBox.intersects(saliencyRect) }
        } else {
            // Fallback: just take high-confidence objects
            mainObjects = objects.filter { $0.confidence > 0.7 }
        }
        
        // 3. Primary Objects
        if !mainObjects.isEmpty {
            desc += "MAIN OBJECTS:\n"
            let counts = Dictionary(grouping: mainObjects, by: { $0.label })
                .map { "  • \($0.value.count)x \($0.key)" }
                .joined(separator: "\n")
            desc += "\(counts)\n\n"
        } else if !objects.isEmpty {
            // Fallback if saliency check found nothing but we *do* have objects
            desc += "DETECTED OBJECTS:\n"
            let counts = Dictionary(grouping: objects, by: { $0.label })
                .compactMap { key, value -> String? in
                    guard let first = value.first else { return nil }
                    return "  • \(value.count)x \(key) (\(first.confidencePercentage))"
                }
                .prefix(5) // Limit to top 5 to avoid bloat
                .joined(separator: "\n")
            desc += "\(counts)\n\n"
        }
        
        // 4. Text (Consolidated) - SANITIZED to avoid safety guardrails
        if !textBlocks.isEmpty {
            desc += "TEXT CONTENT:\n"
            // Consolidate text, replacing newlines with spaces for LLM
            var consolidatedText = fullText.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespaces)
            // Sanitize text to avoid triggering Apple's safety guardrails
            consolidatedText = Self.sanitizeTextForSafety(consolidatedText)
            desc += "  \"\(consolidatedText)\"\n\n"
        }
        
        // 5. Faces / People
        if let faces = faces, !faces.isEmpty {
            desc += "FACES & PEOPLE:\n"
            desc += "  • \(faces.count) face(s) detected.\n"
            if let quality = photoQuality {
                desc += "  • Face Quality: \(quality.qualityLevel)\n"
            }
        }
        if let poses = bodyPoses, !poses.isEmpty {
            desc += "  • \(poses.count) person(s) detected.\n"
        }
        if (faces?.isEmpty == false || bodyPoses?.isEmpty == false) {
             desc += "\n"
        }
        
        // 6. Spatial Context
        let spatial = generateSpatialSummary()
        if !spatial.isEmpty {
            desc += spatial + "\n"
        }
        
        // 7. Barcodes
        if let barcodes = barcodes, !barcodes.isEmpty {
            desc += "BARCODES:\n"
            for (_, barcode) in barcodes.enumerated() {
                desc += "  • [\(barcode.type)] \(barcode.payload)\n"
            }
            desc += "\n"
        }

        if isEmpty {
            desc = "No objects, text, or features detected in the image.\n"
        }
        
        return desc
    }
    
    /// Sanitize text to avoid triggering Apple's Foundation Models safety guardrails
    /// This replaces common profanity and offensive terms with safe alternatives while preserving meaning
    private static func sanitizeTextForSafety(_ text: String) -> String {
        var result = text
        for (regex, replacement) in _sanitizeRegexes {
            let range = NSRange(result.startIndex..<result.endIndex, in: result)
            result = regex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: replacement)
        }
        return result
    }
}

// MARK: - Analysis Options

nonisolated struct AnalysisOptions: Sendable {
    var detectObjects: Bool = true
    var recognizeText: Bool = true
    var detectFaces: Bool = true
    var detectBodyPose: Bool = true
    var detectBarcodes: Bool = true
    var detectHands: Bool = true
    var assessQuality: Bool = true
    
    // --- NEWLY ADDED ---
    var classifyScene: Bool = true
    var getSaliency: Bool = true
    // --- END NEW ---
    
    // OCR settings
    var ocrLanguages: [String] = []
    var useAccurateOCR: Bool = true
    var minimumTextConfidence: Float = 0.5
    
    // Presets
    static var all: AnalysisOptions { AnalysisOptions() }
    static var objectsOnly: AnalysisOptions {
        AnalysisOptions(
            detectObjects: true,
            recognizeText: false,
            detectFaces: false,
            detectBodyPose: false,
            detectBarcodes: false,
            detectHands: false,
            assessQuality: false,
            classifyScene: false, // --- NEW
            getSaliency: false    // --- NEW
        )
    }
    static var textOnly: AnalysisOptions {
        AnalysisOptions(
            detectObjects: false,
            recognizeText: true,
            detectFaces: false,
            detectBodyPose: false,
            detectBarcodes: false,
            detectHands: false,
            assessQuality: false,
            classifyScene: false, // --- NEW
            getSaliency: false    // --- NEW
        )
    }
    static var objectsAndText: AnalysisOptions {
        AnalysisOptions(
            detectObjects: true,
            recognizeText: true,
            detectFaces: false,
            detectBodyPose: false,
            detectBarcodes: false,
            detectHands: false,
            assessQuality: false,
            classifyScene: true,  // --- NEW
            getSaliency: true     // --- NEW
        )
    }
}

// Pre-compiled regexes for sanitizeTextForSafety (compiled once at app launch)
// swiftlint:disable force_try
nonisolated private let _sanitizeRegexes: [(NSRegularExpression, String)] = [
    (try! NSRegularExpression(pattern: #"\bhuge ass\b"#, options: .caseInsensitive), "huge"),
    (try! NSRegularExpression(pattern: #"\bbig ass\b"#,  options: .caseInsensitive), "big"),
    (try! NSRegularExpression(pattern: #"\basses\b"#,    options: .caseInsensitive), "butts"),
    (try! NSRegularExpression(pattern: #"\bass\b"#,      options: .caseInsensitive), "butt"),
    (try! NSRegularExpression(pattern: #"\bdamned\b"#,   options: .caseInsensitive), "darned"),
    (try! NSRegularExpression(pattern: #"\bdamn\b"#,     options: .caseInsensitive), "darn"),
    (try! NSRegularExpression(pattern: #"\bhell\b"#,     options: .caseInsensitive), "heck"),
    (try! NSRegularExpression(pattern: #"\bshitty\b"#,   options: .caseInsensitive), "bad"),
    (try! NSRegularExpression(pattern: #"\bshit\b"#,     options: .caseInsensitive), "stuff"),
    (try! NSRegularExpression(pattern: #"\bcrappy\b"#,   options: .caseInsensitive), "bad"),
    (try! NSRegularExpression(pattern: #"\bcrap\b"#,     options: .caseInsensitive), "stuff"),
    (try! NSRegularExpression(pattern: #"\bfucking\b"#,  options: .caseInsensitive), "very"),
    (try! NSRegularExpression(pattern: #"\bfucked\b"#,   options: .caseInsensitive), "messed"),
    (try! NSRegularExpression(pattern: #"\bfuck\b"#,     options: .caseInsensitive), "[expletive]"),
    (try! NSRegularExpression(pattern: #"\bbitches\b"#,  options: .caseInsensitive), "[expletives]"),
    (try! NSRegularExpression(pattern: #"\bbitch\b"#,    options: .caseInsensitive), "[expletive]"),
    (try! NSRegularExpression(pattern: #"\bbastard\b"#,  options: .caseInsensitive), "[expletive]"),
    (try! NSRegularExpression(pattern: #"\bpissed\b"#,   options: .caseInsensitive), "angry"),
    (try! NSRegularExpression(pattern: #"\bpiss\b"#,     options: .caseInsensitive), "[expletive]"),
    (try! NSRegularExpression(pattern: #"\bwtf\b"#,      options: .caseInsensitive), "what the heck"),
    (try! NSRegularExpression(pattern: #"\bomg\b"#,      options: .caseInsensitive), "oh my gosh"),
]
// swiftlint:enable force_try

// MARK: - Vision Analyzer

final class VisionAnalyzer: ObservableObject {
    nonisolated private static let logger = Logger(
        subsystem: "Nevio.On-Device-LLM-Chat",
        category: "VisionAnalyzer"
    )

    @MainActor @Published var isProcessing = false
    @MainActor @Published var lastError: String?
    @MainActor @Published var lastResult: VisionAnalysisResult?
    
    private let objectDetector = VisionObjectDetector()
    
    // MARK: - Main Analysis Method
    
    /// Comprehensive image analysis using Vision framework
    func analyze(image: UIImage, options: AnalysisOptions? = nil) async throws -> VisionAnalysisResult {
        let options = options ?? .all
        Self.logger.debug("Starting Vision analysis")
        
        await MainActor.run { isProcessing = true }
        defer { Task { @MainActor in isProcessing = false } }
        
        guard let cgImage = image.cgImage else {
            throw VisionError.invalidImage
        }
        
        let objectDetector = self.objectDetector
        let result = await Task.detached(priority: .userInitiated) {
            await Self.performSequentialAnalysis(
                cgImage: cgImage,
                options: options,
                objectDetector: objectDetector
            )
        }.value
        
        await MainActor.run {
            lastResult = result
            lastError = nil
        }
        
        Self.logger.debug(
            "Vision analysis complete: objects=\(result.objects.count, privacy: .public) text_blocks=\(result.textBlocks.count, privacy: .public) faces=\(result.faces?.count ?? 0, privacy: .public)"
        )
        
        return result
    }

    nonisolated private static func performSequentialAnalysis(
        cgImage: CGImage,
        options: AnalysisOptions,
        objectDetector: VisionObjectDetector
    ) async -> VisionAnalysisResult {
        let objects = options.detectObjects
            ? await detectObjects(cgImage: cgImage, objectDetector: objectDetector)
            : []
        let textBlocks = options.recognizeText
            ? await recognizeText(cgImage: cgImage, options: options)
            : []
        let faces = options.detectFaces ? await detectFaces(cgImage: cgImage) : nil
        let bodyPoses = options.detectBodyPose ? await detectBodyPoses(cgImage: cgImage) : nil
        let barcodes = options.detectBarcodes ? await detectBarcodes(cgImage: cgImage) : nil
        let handPoses = options.detectHands ? await detectHands(cgImage: cgImage) : nil
        let sceneLabels = options.classifyScene ? await classifyScene(cgImage: cgImage) : nil
        let saliencyRect = options.getSaliency ? await getSaliencyRect(cgImage: cgImage) : nil
        let photoQuality = options.assessQuality ? assessPhotoQuality(faces: faces) : nil

        return VisionAnalysisResult(
            objects: objects,
            textBlocks: textBlocks,
            faces: faces,
            bodyPoses: bodyPoses,
            barcodes: barcodes,
            handPoses: handPoses,
            photoQuality: photoQuality,
            sceneLabels: sceneLabels,
            saliencyRect: saliencyRect
        )
    }
    
    // MARK: - Individual Detection Methods
    
    nonisolated private static func detectObjects(
        cgImage: CGImage,
        objectDetector: VisionObjectDetector
    ) async -> [DetectedObject] {
        do {
            return try await objectDetector.detectObjects(in: UIImage(cgImage: cgImage))
        } catch {
            Self.logger.error("Vision object detection failed: \((error as NSError).localizedDescription, privacy: .public)")
            return []
        }
    }
    
    nonisolated private static func recognizeText(cgImage: CGImage, options: AnalysisOptions) async -> [RecognizedText] {
        await withCheckedContinuation { continuation in
            let request = VNRecognizeTextRequest { request, error in
                if let error = error {
                    Self.logger.error("Vision OCR failed: \((error as NSError).localizedDescription, privacy: .public)")
                    continuation.resume(returning: [])
                    return
                }
                
                guard let observations = request.results as? [VNRecognizedTextObservation] else {
                    continuation.resume(returning: [])
                    return
                }
                
                var recognizedTexts: [RecognizedText] = []
                
                for observation in observations {
                    guard let topCandidate = observation.topCandidates(1).first else { continue }
                    
                    // Filter by confidence
                    guard topCandidate.confidence >= options.minimumTextConfidence else { continue }
                    
                    let recognizedText = RecognizedText(
                        text: topCandidate.string,
                        confidence: topCandidate.confidence,
                        boundingBox: observation.boundingBox
                    )
                    
                    recognizedTexts.append(recognizedText)
                }
                
                // Sort by position (top to bottom, left to right)
                // NOTE: Vision coordinates have origin at BOTTOM-LEFT, so we need to sort by maxY descending for top-to-bottom
                recognizedTexts.sort { first, second in
                    // Compare Y position (maxY for top-to-bottom since origin is bottom-left)
                    // Allow for a small tolerance to group lines on the same row
                    let yTolerance: CGFloat = 0.02
                    let firstY = first.boundingBox.maxY
                    let secondY = second.boundingBox.maxY
                    
                    if abs(firstY - secondY) > yTolerance {
                        return firstY > secondY // Higher Y = closer to top
                    }
                    // If Y is similar (same line), sort by X position (left to right)
                    return first.boundingBox.minX < second.boundingBox.minX
                }
                
                continuation.resume(returning: recognizedTexts)
            }
            
            // Configure OCR request
            request.recognitionLevel = options.useAccurateOCR ? .accurate : .fast
            request.usesLanguageCorrection = true
            request.minimumTextHeight = 0.0 // Detect all text sizes
            request.automaticallyDetectsLanguage = true
            
            if !options.ocrLanguages.isEmpty {
                request.recognitionLanguages = options.ocrLanguages
            } else {
                request.recognitionLanguages = Locale.preferredLanguages
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                Self.logger.error("Vision OCR request failed: \((error as NSError).localizedDescription, privacy: .public)")
                continuation.resume(returning: [])
            }
        }
    }
    
    nonisolated private static func detectFaces(cgImage: CGImage) async -> [DetectedFace]? {
        await withCheckedContinuation { continuation in
            let request = VNDetectFaceCaptureQualityRequest { request, error in
                if let error = error {
                    Self.logger.error("Vision face detection failed: \((error as NSError).localizedDescription, privacy: .public)")
                    continuation.resume(returning: nil)
                    return
                }
                
                guard let observations = request.results as? [VNFaceObservation], !observations.isEmpty else {
                    continuation.resume(returning: nil)
                    return
                }
                
                let faces = observations.map { observation -> DetectedFace in
                    var landmarks: FaceLandmarks? = nil
                    
                    if let faceLandmarks = observation.landmarks {
                        landmarks = FaceLandmarks(
                            leftEye: faceLandmarks.leftEye?.normalizedPoints.first,
                            rightEye: faceLandmarks.rightEye?.normalizedPoints.first,
                            nose: faceLandmarks.nose?.normalizedPoints.first,
                            mouth: faceLandmarks.innerLips?.normalizedPoints.first
                        )
                    }
                    
                    return DetectedFace(
                        boundingBox: observation.boundingBox,
                        confidence: observation.confidence,
                        landmarks: landmarks,
                        quality: observation.faceCaptureQuality
                    )
                }
                
                continuation.resume(returning: faces)
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                Self.logger.error("Vision face request failed: \((error as NSError).localizedDescription, privacy: .public)")
                continuation.resume(returning: nil)
            }
        }
    }
    
    nonisolated private static func detectBodyPoses(cgImage: CGImage) async -> [BodyPose]? {
        await withCheckedContinuation { continuation in
            let request = VNDetectHumanBodyPoseRequest { [continuation] request, error in
                if let error = error {
                    Self.logger.error("Vision body-pose detection failed: \((error as NSError).localizedDescription, privacy: .public)")
                    continuation.resume(returning: nil)
                    return
                }

                guard let observations = request.results as? [VNHumanBodyPoseObservation], !observations.isEmpty else {
                    continuation.resume(returning: nil)
                    return
                }

                let poses = observations.compactMap { observation -> BodyPose? in
                    guard let recognizedPoints = try? observation.recognizedPoints(.all) else {
                        return nil
                    }

                    var joints: [String: CGPoint] = [:]
                    for (jointName, point) in recognizedPoints where point.confidence > 0.5 {
                        joints[jointName.rawValue.rawValue] = point.location
                    }

                    guard !joints.isEmpty else { return nil }

                    return BodyPose(
                        confidence: observation.confidence,
                        joints: joints
                    )
                }

                continuation.resume(returning: poses.isEmpty ? nil : poses)
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                Self.logger.error("Vision body-pose request failed: \((error as NSError).localizedDescription, privacy: .public)")
                continuation.resume(returning: nil)
            }
        }
    }
    
    nonisolated private static func detectBarcodes(cgImage: CGImage) async -> [Barcode]? {
        await withCheckedContinuation { continuation in
            let request = VNDetectBarcodesRequest { request, error in
                if let error = error {
                    Self.logger.error("Vision barcode detection failed: \((error as NSError).localizedDescription, privacy: .public)")
                    continuation.resume(returning: nil)
                    return
                }
                
                guard let observations = request.results as? [VNBarcodeObservation], !observations.isEmpty else {
                    continuation.resume(returning: nil)
                    return
                }
                
                let barcodes = observations.compactMap { observation -> Barcode? in
                    guard let payload = observation.payloadStringValue else { return nil }
                    
                    return Barcode(
                        type: observation.symbology.rawValue,
                        payload: payload,
                        boundingBox: observation.boundingBox
                    )
                }
                
                continuation.resume(returning: barcodes.isEmpty ? nil : barcodes)
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                Self.logger.error("Vision barcode request failed: \((error as NSError).localizedDescription, privacy: .public)")
                continuation.resume(returning: nil)
            }
        }
    }
    
    nonisolated private static func detectHands(cgImage: CGImage) async -> [HandPose]? {
        await withCheckedContinuation { continuation in
            let request = VNDetectHumanHandPoseRequest { [continuation] request, error in
                if let error = error {
                    Self.logger.error("Vision hand-pose detection failed: \((error as NSError).localizedDescription, privacy: .public)")
                    continuation.resume(returning: nil)
                    return
                }

                guard let observations = request.results as? [VNHumanHandPoseObservation], !observations.isEmpty else {
                    continuation.resume(returning: nil)
                    return
                }

                let hands = observations.compactMap { observation -> HandPose? in
                    guard let recognizedPoints = try? observation.recognizedPoints(.all) else {
                        return nil
                    }

                    var fingerPoints: [String: CGPoint] = [:]
                    for (pointName, point) in recognizedPoints where point.confidence > 0.5 {
                        fingerPoints[pointName.rawValue.rawValue] = point.location
                    }

                    guard !fingerPoints.isEmpty else { return nil }

                    let chirality = observation.chirality == .left ? "left" : "right"

                    return HandPose(
                        confidence: observation.confidence,
                        chirality: chirality,
                        fingerPoints: fingerPoints
                    )
                }

                continuation.resume(returning: hands.isEmpty ? nil : hands)
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                Self.logger.error("Vision hand-pose request failed: \((error as NSError).localizedDescription, privacy: .public)")
                continuation.resume(returning: nil)
            }
        }
    }
    
    nonisolated private static func assessPhotoQuality(faces: [DetectedFace]?) -> PhotoQuality? {
        // Photo quality is based on face detection quality
        guard let faces, !faces.isEmpty else {
            return nil
        }
        
        let qualities = faces.compactMap { $0.quality }
        guard !qualities.isEmpty else { return nil }
        
        let avgQuality = qualities.reduce(0, +) / Float(qualities.count)
        
        var issues: [String] = []
        if avgQuality < 0.4 {
            issues.append("poor lighting or angle")
        } else if avgQuality < 0.6 {
            issues.append("slightly blurry")
        }
        
        return PhotoQuality(
            overallQuality: avgQuality,
            faceCount: faces.count,
            issues: issues
        )
    }
    
    // --- NEW: Scene Classification ("Vibe Check") ---
    nonisolated private static func classifyScene(cgImage: CGImage) async -> [String]? {
        await withCheckedContinuation { continuation in
            // This uses Apple's built-in taxonomy—super efficient
            let request = VNClassifyImageRequest { request, error in
                if let error = error {
                    Self.logger.error("Vision scene classification failed: \((error as NSError).localizedDescription, privacy: .public)")
                    continuation.resume(returning: nil)
                    return
                }
                
                guard let observations = request.results as? [VNClassificationObservation] else {
                    continuation.resume(returning: nil)
                    return
                }
                
                // Get top 3 classifications with decent confidence
                let labels = observations
                    .filter { $0.confidence > 0.6 } // 60% confidence
                    .prefix(3)
                    .map { $0.identifier.components(separatedBy: ", ").first ?? $0.identifier } // Clean up labels like "Office, Cubicle"
                
                continuation.resume(returning: labels.isEmpty ? nil : labels)
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                Self.logger.error("Vision scene-classification request failed: \((error as NSError).localizedDescription, privacy: .public)")
                continuation.resume(returning: nil)
            }
        }
    }
    
    // --- NEW: Saliency ("Main Character") ---
    nonisolated private static func getSaliencyRect(cgImage: CGImage) async -> CGRect? {
        // Returns the bounding box of the "most interesting" part of the image
        await withCheckedContinuation { continuation in
            let request = VNGenerateAttentionBasedSaliencyImageRequest { request, error in
                if let error = error {
                    Self.logger.error("Vision saliency detection failed: \((error as NSError).localizedDescription, privacy: .public)")
                    continuation.resume(returning: nil)
                    return
                }
                
                guard let observation = request.results?.first as? VNSaliencyImageObservation,
                      let salientObjects = observation.salientObjects,
                      let primary = salientObjects.first else {
                    continuation.resume(returning: nil)
                    return
                }
                
                // This is the CGRect (0-1) of the main subject
                continuation.resume(returning: primary.boundingBox)
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage)
            do {
                try handler.perform([request])
            } catch {
                Self.logger.error("Vision saliency request failed: \((error as NSError).localizedDescription, privacy: .public)")
                continuation.resume(returning: nil)
            }
        }
    }
    
    // MARK: - Convenience Methods
    
    /// Quick text recognition only
    func recognizeTextOnly(in image: UIImage) async throws -> String {
        let result = try await analyze(image: image, options: .textOnly)
        return result.fullText
    }
    
    /// Get user-friendly status message
    @MainActor
    var statusMessage: String {
        if let error = lastError {
            return error
        } else if isProcessing {
            return "Analyzing image..."
        } else {
            return "Ready to analyze"
        }
    }
    
    /// Check if Vision Framework is ready
    var isReady: Bool {
        objectDetector.isModelLoaded
    }
}

// MARK: - Error Types

nonisolated enum VisionError: LocalizedError {
    case invalidImage
    case analysisFailed
    case noResults
    
    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "The provided image is invalid or cannot be processed."
        case .analysisFailed:
            return "Vision analysis failed to complete."
        case .noResults:
            return "No analysis results were generated."
        }
    }
}
