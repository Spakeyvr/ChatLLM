//
//  DetectionResultsView.swift
//  ChatLLM
//
//  Created by Nevio on 10/27/25.
//

import SwiftUI

/// View for displaying combined vision analysis results (object detection + OCR)
struct VisionAnalysisResultsView: View {
    let result: VisionAnalysisResult
    let image: UIImage
    
    @State private var showDetails = false
    @State private var selectedTab: AnalysisTab = .objects
    
    enum AnalysisTab: String, CaseIterable {
        case objects = "Objects"
        case text = "Text"
        case all = "All"
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Summary header
            HStack(spacing: 8) {
                Image(systemName: "viewfinder.rectangular")
                    .font(.headline)
                    .foregroundStyle(.blue)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("Vision Analysis")
                        .font(.headline)
                    
                    Text("\(result.objects.count) objects · \(result.textBlocks.count) text blocks")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                Spacer()
                
                Button {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                        showDetails.toggle()
                    }
                } label: {
                    Image(systemName: showDetails ? "chevron.up.circle.fill" : "chevron.down.circle.fill")
                        .font(.title3)
                        .foregroundStyle(.blue)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.blue.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            
            // Quick preview
            if !showDetails {
                quickPreviewSection
            }
            
            // Detailed view
            if showDetails {
                detailedViewSection
            }
        }
        .padding(12)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
    
    private var quickPreviewSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Objects preview
            if !result.objects.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        Image(systemName: "cube.box")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        
                        ForEach(result.objects.prefix(5)) { detection in
                            DetectionChip(detection: detection)
                        }
                        
                        if result.objects.count > 5 {
                            Text("+\(result.objects.count - 5) more")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color.secondary.opacity(0.1))
                                .clipShape(Capsule())
                        }
                    }
                    .padding(.horizontal, 4)
                }
            }
            
            // Text preview
            if !result.textBlocks.isEmpty {
                HStack(spacing: 8) {
                    Image(systemName: "doc.text")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    Text(String(result.fullText.prefix(100)) + (result.fullText.count > 100 ? "..." : ""))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.orange.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
    }
    
    private var detailedViewSection: some View {
        VStack(spacing: 12) {
            // Tab selector
            if !result.objects.isEmpty && !result.textBlocks.isEmpty {
                Picker("View", selection: $selectedTab) {
                    ForEach(AnalysisTab.allCases, id: \.self) { tab in
                        Text(tab.rawValue).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
            }
            
            // Image with overlays
            if selectedTab == .all || selectedTab == .objects || selectedTab == .text {
                ImageWithAnalysisOverlay(
                    image: image,
                    detections: selectedTab == .text ? [] : result.objects,
                    textBlocks: selectedTab == .objects ? [] : result.textBlocks
                )
                .frame(maxHeight: 300)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            
            // Details list
            ScrollView {
                VStack(spacing: 12) {
                    // Objects list
                    if (selectedTab == .all || selectedTab == .objects) && !result.objects.isEmpty {
                        ObjectsListSection(objects: result.objects)
                    }
                    
                    // Text list
                    if (selectedTab == .all || selectedTab == .text) && !result.textBlocks.isEmpty {
                        TextBlocksListSection(textBlocks: result.textBlocks)
                    }
                }
            }
            .frame(maxHeight: 400)
        }
    }
}

// MARK: - Objects List Section

private struct ObjectsListSection: View {
    let objects: [DetectedObject]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "cube.box.fill")
                    .foregroundStyle(.blue)
                Text("Objects (\(objects.count))")
                    .font(.subheadline.weight(.semibold))
            }
            .padding(.horizontal, 8)
            
            ForEach(Array(objects.enumerated()), id: \.element.id) { index, detection in
                DetectionRow(index: index + 1, detection: detection)
            }
        }
    }
}

// MARK: - Text Blocks List Section

private struct TextBlocksListSection: View {
    let textBlocks: [RecognizedText]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "doc.text.fill")
                    .foregroundStyle(.orange)
                Text("Text Blocks (\(textBlocks.count))")
                    .font(.subheadline.weight(.semibold))
            }
            .padding(.horizontal, 8)
            
            ForEach(Array(textBlocks.enumerated()), id: \.element.id) { index, textBlock in
                TextBlockRow(index: index + 1, textBlock: textBlock)
            }
        }
    }
}

private struct TextBlockRow: View {
    let index: Int
    let textBlock: RecognizedText
    
    var body: some View {
        HStack(spacing: 12) {
            // Index badge
            Text("\(index)")
                .font(.caption.weight(.bold))
                .foregroundStyle(.white)
                .frame(width: 24, height: 24)
                .background(Circle().fill(Color.orange))
            
            // Text info
            VStack(alignment: .leading, spacing: 2) {
                Text(textBlock.text)
                    .font(.subheadline.weight(.medium))
                    .lineLimit(3)
                
                Text("Confidence: \(textBlock.confidencePercentage)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            Spacer()
            
            // Confidence indicator
            ConfidenceBar(confidence: textBlock.confidence)
        }
        .padding(10)
        .background(Color(uiColor: .tertiarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Legacy View (Object Detection Only)

/// View for displaying object detection results only (for backward compatibility)
struct DetectionResultsView: View {
    let detections: [DetectedObject]
    let image: UIImage
    
    @State private var showDetails = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Summary header
            HStack(spacing: 8) {
                Image(systemName: "viewfinder")
                    .font(.headline)
                    .foregroundStyle(.blue)
                
                Text("Detected \(detections.count) object\(detections.count == 1 ? "" : "s")")
                    .font(.headline)
                
                Spacer()
                
                Button {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                        showDetails.toggle()
                    }
                } label: {
                    Image(systemName: showDetails ? "chevron.up.circle.fill" : "chevron.down.circle.fill")
                        .font(.title3)
                        .foregroundStyle(.blue)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.blue.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            
            // Quick preview of top detections
            if !showDetails {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(detections.prefix(5)) { detection in
                            DetectionChip(detection: detection)
                        }
                        
                        if detections.count > 5 {
                            Text("+\(detections.count - 5) more")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color.secondary.opacity(0.1))
                                .clipShape(Capsule())
                        }
                    }
                    .padding(.horizontal, 4)
                }
            }
            
            // Detailed view
            if showDetails {
                VStack(spacing: 12) {
                    // Image with bounding boxes
                    ImageWithDetections(image: image, detections: detections)
                        .frame(maxHeight: 300)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    
                    // Detailed list
                    VStack(spacing: 8) {
                        ForEach(Array(detections.enumerated()), id: \.element.id) { index, detection in
                            DetectionRow(index: index + 1, detection: detection)
                        }
                    }
                    .padding(.vertical, 8)
                }
            }
        }
        .padding(12)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
}

// MARK: - Supporting Views

private struct DetectionChip: View {
    let detection: DetectedObject
    
    var body: some View {
        HStack(spacing: 4) {
            Text(detection.label)
                .font(.caption.weight(.medium))
            Text(detection.confidencePercentage)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color.blue.opacity(0.15))
        .clipShape(Capsule())
    }
}

private struct DetectionRow: View {
    let index: Int
    let detection: DetectedObject
    
    var body: some View {
        HStack(spacing: 12) {
            // Index badge
            Text("\(index)")
                .font(.caption.weight(.bold))
                .foregroundStyle(.white)
                .frame(width: 24, height: 24)
                .background(Circle().fill(Color.blue))
            
            // Detection info
            VStack(alignment: .leading, spacing: 2) {
                Text(detection.label)
                    .font(.subheadline.weight(.medium))
                
                Text("Confidence: \(detection.confidencePercentage)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            Spacer()
            
            // Confidence indicator
            ConfidenceBar(confidence: detection.confidence)
        }
        .padding(10)
        .background(Color(uiColor: .tertiarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

private struct ConfidenceBar: View {
    let confidence: Float
    
    var color: Color {
        if confidence >= 0.7 {
            return .green
        } else if confidence >= 0.5 {
            return .orange
        } else {
            return .red
        }
    }
    
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.secondary.opacity(0.2))
                
                RoundedRectangle(cornerRadius: 4)
                    .fill(color)
                    .frame(width: geometry.size.width * CGFloat(confidence))
            }
        }
        .frame(width: 60, height: 8)
    }
}

// Make this public so it can be used by other views
struct ImageWithDetections: View {
    let image: UIImage
    let detections: [DetectedObject]
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                // Overlay bounding boxes
                ForEach(detections) { detection in
                    BoundingBoxView(
                        detection: detection,
                        imageSize: image.size,
                        containerSize: geometry.size
                    )
                }
            }
        }
    }
}

// View for displaying combined vision analysis overlays (objects + text)
struct ImageWithAnalysisOverlay: View {
    let image: UIImage
    let detections: [DetectedObject]
    let textBlocks: [RecognizedText]
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                // Overlay bounding boxes for objects
                ForEach(detections) { detection in
                    BoundingBoxView(
                        detection: detection,
                        imageSize: image.size,
                        containerSize: geometry.size
                    )
                }
                
                // Overlay bounding boxes for text
                ForEach(textBlocks) { textBlock in
                    TextBoundingBoxView(
                        textBlock: textBlock,
                        imageSize: image.size,
                        containerSize: geometry.size
                    )
                }
            }
        }
    }
}

struct BoundingBoxView: View {
    let detection: DetectedObject
    let imageSize: CGSize
    let containerSize: CGSize
    
    var body: some View {
        let rect = convertBoundingBox(detection.boundingBox)
        
        ZStack(alignment: .topLeading) {
            // Bounding box
            Rectangle()
                .stroke(Color.blue, lineWidth: 2)
                .frame(width: rect.width, height: rect.height)
            
            // Label
            Text(detection.label)
                .font(.caption2.weight(.bold))
                .foregroundStyle(.white)
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(Color.blue)
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .offset(y: -20)
        }
        .position(x: rect.midX, y: rect.midY)
    }
    
    private func convertBoundingBox(_ box: CGRect) -> CGRect {
        // Vision uses normalized coordinates (0-1) with origin at bottom-left
        // Convert to SwiftUI coordinates with origin at top-left
        
        // Calculate actual image display size (maintaining aspect ratio)
        let imageAspect = imageSize.width / imageSize.height
        let containerAspect = containerSize.width / containerSize.height
        
        var displaySize: CGSize
        var offset: CGPoint = .zero
        
        if imageAspect > containerAspect {
            // Image is wider - fit to width
            displaySize = CGSize(
                width: containerSize.width,
                height: containerSize.width / imageAspect
            )
            offset.y = (containerSize.height - displaySize.height) / 2
        } else {
            // Image is taller - fit to height
            displaySize = CGSize(
                width: containerSize.height * imageAspect,
                height: containerSize.height
            )
            offset.x = (containerSize.width - displaySize.width) / 2
        }
        
        // Convert normalized box to pixel coordinates
        // Vision uses bottom-left origin, so we need to flip Y
        let x = box.origin.x * displaySize.width + offset.x
        let y = (1 - box.origin.y - box.height) * displaySize.height + offset.y
        let width = box.width * displaySize.width
        let height = box.height * displaySize.height
        
        return CGRect(x: x, y: y, width: width, height: height)
    }
}

// Bounding box view for text blocks
struct TextBoundingBoxView: View {
    let textBlock: RecognizedText
    let imageSize: CGSize
    let containerSize: CGSize
    
    var body: some View {
        let rect = convertBoundingBox(textBlock.boundingBox)
        
        ZStack(alignment: .topLeading) {
            // Bounding box
            Rectangle()
                .stroke(Color.orange, lineWidth: 2)
                .frame(width: rect.width, height: rect.height)
            
            // Label showing first few characters
            Text(String(textBlock.text.prefix(20)))
                .font(.caption2.weight(.bold))
                .foregroundStyle(.white)
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(Color.orange)
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .offset(y: -20)
        }
        .position(x: rect.midX, y: rect.midY)
    }
    
    private func convertBoundingBox(_ box: CGRect) -> CGRect {
        // Vision uses normalized coordinates (0-1) with origin at bottom-left
        // Convert to SwiftUI coordinates with origin at top-left
        
        // Calculate actual image display size (maintaining aspect ratio)
        let imageAspect = imageSize.width / imageSize.height
        let containerAspect = containerSize.width / containerSize.height
        
        var displaySize: CGSize
        var offset: CGPoint = .zero
        
        if imageAspect > containerAspect {
            // Image is wider - fit to width
            displaySize = CGSize(
                width: containerSize.width,
                height: containerSize.width / imageAspect
            )
            offset.y = (containerSize.height - displaySize.height) / 2
        } else {
            // Image is taller - fit to height
            displaySize = CGSize(
                width: containerSize.height * imageAspect,
                height: containerSize.height
            )
            offset.x = (containerSize.width - displaySize.width) / 2
        }
        
        // Convert normalized box to pixel coordinates
        // Vision uses bottom-left origin, so we need to flip Y
        let x = box.origin.x * displaySize.width + offset.x
        let y = (1 - box.origin.y - box.height) * displaySize.height + offset.y
        let width = box.width * displaySize.width
        let height = box.height * displaySize.height
        
        return CGRect(x: x, y: y, width: width, height: height)
    }
}

// MARK: - Compact Summary View

struct DetectionSummaryView: View {
    let summary: String
    
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "viewfinder.circle.fill")
                .font(.caption)
                .foregroundStyle(.blue)
            
            Text(summary)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color.blue.opacity(0.1))
        .clipShape(Capsule())
    }
}



// MARK: - Previews

#Preview("Detection Results") {
    let sampleDetections = [
        DetectedObject(label: "person", confidence: 0.95, boundingBox: CGRect(x: 0.2, y: 0.3, width: 0.3, height: 0.5)),
        DetectedObject(label: "dog", confidence: 0.87, boundingBox: CGRect(x: 0.5, y: 0.4, width: 0.25, height: 0.4)),
        DetectedObject(label: "chair", confidence: 0.72, boundingBox: CGRect(x: 0.1, y: 0.6, width: 0.2, height: 0.3))
    ]
    
    let sampleImage = UIImage(systemName: "photo")!
    
    return ScrollView {
        DetectionResultsView(detections: sampleDetections, image: sampleImage)
            .padding()
    }
}

#Preview("Detection Summary") {
    VStack(spacing: 16) {
        DetectionSummaryView(summary: "Detected: person, dog, chair")
        DetectionSummaryView(summary: "No objects detected")
        DetectionSummaryView(summary: "Detected: person (95.2%)")
    }
    .padding()
}
