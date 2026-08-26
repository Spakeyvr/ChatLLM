//
//  ImagePickerWithDetection.swift
//  ChatLLM
//
//  Created by Nevio on 10/27/25.
//

import SwiftUI
import PhotosUI

/// Image picker that automatically runs Vision Framework detection on selected images
struct ImagePickerWithDetection: View {
    @Binding var selectedImage: UIImage?
    @Binding var detectedObjects: [DetectedObject]?
    @Binding var isPresented: Bool
    
    @StateObject private var detector = VisionObjectDetector()
    @State private var selectedItem: PhotosPickerItem?
    @State private var isProcessing = false
    @State private var errorMessage: String?
    
    var onComplete: (UIImage, [DetectedObject]) -> Void
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                // Status indicator
                statusView
                
                // Image preview
                if let image = selectedImage {
                    imagePreview(image: image)
                }
                
                // Detection results
                if let detections = detectedObjects, !detections.isEmpty {
                    detectionsList(detections: detections)
                }
                
                Spacer()
                
                // Action buttons
                actionButtons
            }
            .padding()
            .navigationTitle("Add Image")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        isPresented = false
                    }
                }
                
                ToolbarItem(placement: .confirmationAction) {
                    Button("Add") {
                        if let image = selectedImage,
                           let detections = detectedObjects {
                            onComplete(image, detections)
                            isPresented = false
                        }
                    }
                    .disabled(selectedImage == nil || isProcessing)
                    .fontWeight(.semibold)
                }
            }
            .onChange(of: selectedItem) { _, newItem in
                Task {
                    await loadAndDetect(item: newItem)
                }
            }
        }
    }
    
    // MARK: - View Components
    
    @ViewBuilder
    private var statusView: some View {
        VStack(spacing: 8) {
            if isProcessing {
                HStack(spacing: 12) {
                    ProgressView()
                    Text("Detecting objects...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color.blue.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            } else if let error = errorMessage {
                HStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(error)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color.orange.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            } else if !detector.isModelLoaded {
                HStack(spacing: 12) {
                    ProgressView()
                    Text("Initializing Vision Framework...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color.secondary.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
        }
    }
    
    private func imagePreview(image: UIImage) -> some View {
        Image(uiImage: image)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(maxHeight: 300)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
            )
    }
    
    private func detectionsList(detections: [DetectedObject]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("Detected \(detections.count) object\(detections.count == 1 ? "" : "s")")
                    .font(.headline)
            }
            
            ScrollView {
                VStack(spacing: 8) {
                    ForEach(Array(detections.prefix(5).enumerated()), id: \.element.id) { index, detection in
                        HStack {
                            Text("\(index + 1).")
                                .font(.caption.weight(.bold))
                                .foregroundStyle(.secondary)
                                .frame(width: 20)
                            
                            Text(detection.label)
                                .font(.subheadline)
                            
                            Spacer()
                            
                            Text(detection.confidencePercentage)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color(uiColor: .secondarySystemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                    
                    if detections.count > 5 {
                        Text("+ \(detections.count - 5) more")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxHeight: 200)
        }
        .padding()
        .background(Color.green.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
    
    private var actionButtons: some View {
        VStack(spacing: 12) {
            PhotosPicker(
                selection: $selectedItem,
                matching: .images
            ) {
                Label(
                    selectedImage == nil ? "Select Image" : "Change Image",
                    systemImage: selectedImage == nil ? "photo" : "arrow.triangle.2.circlepath"
                )
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .disabled(isProcessing)
            
            if selectedImage != nil && !isProcessing {
                Button {
                    Task {
                        await redetect()
                    }
                } label: {
                    Label("Re-analyze Image", systemImage: "arrow.clockwise")
                        .font(.subheadline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.secondary.opacity(0.2))
                        .foregroundStyle(.primary)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
            }
        }
    }
    
    // MARK: - Image Loading and Detection
    
    private func loadAndDetect(item: PhotosPickerItem?) async {
        guard let item = item else { return }
        
        print("\nImagePicker: Starting to load and detect image...")
        
        isProcessing = true
        errorMessage = nil
        detectedObjects = nil
        
        do {
            guard let data = try await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: data) else {
                throw NSError(domain: "ImagePicker", code: -1,
                              userInfo: [NSLocalizedDescriptionKey: "Failed to load image"])
            }
            
            print("Image loaded: \(image.size.width)x\(image.size.height)")
            
            await MainActor.run {
                selectedImage = image
            }
            
            print("Calling detector.detectObjects()...")
            let detections = try await detector.detectObjects(in: image)
            print("Detection returned \(detections.count) objects")
            
            await MainActor.run {
                detectedObjects = detections
                isProcessing = false
                
                if detections.isEmpty {
                    errorMessage = "No objects detected in this image"
                }
            }
            
        } catch {
            await MainActor.run {
                errorMessage = error.localizedDescription
                isProcessing = false
            }
        }
    }
    
    private func redetect() async {
        guard let image = selectedImage else { return }
        
        isProcessing = true
        errorMessage = nil
        
        do {
            let detections = try await detector.detectObjects(in: image)
            
            await MainActor.run {
                detectedObjects = detections
                isProcessing = false
                
                if detections.isEmpty {
                    errorMessage = "No objects detected in this image"
                }
            }
        } catch {
            await MainActor.run {
                errorMessage = error.localizedDescription
                isProcessing = false
            }
        }
    }
}

// MARK: - Preview

#Preview {
    struct PreviewWrapper: View {
        @State private var image: UIImage?
        @State private var detections: [DetectedObject]?
        @State private var isPresented = true
        
        var body: some View {
            Text("Image Picker Preview")
                .sheet(isPresented: $isPresented) {
                    ImagePickerWithDetection(
                        selectedImage: $image,
                        detectedObjects: $detections,
                        isPresented: $isPresented
                    ) { image, detections in
                        print("Selected image with \(detections.count) detections")
                    }
                }
        }
    }
    
    return PreviewWrapper()
}
