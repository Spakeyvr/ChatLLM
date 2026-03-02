//
//  ChatView+ImageIntegration.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/27/25.
//
//  SAMPLE CODE: How to integrate image attachments into your existing ChatView
//
//  IMPORTANT - IMAGE PERSISTENCE:
//  =============================
//  Images must be stored persistently to survive app restarts and updates.
//  
//  The MessageAttachment model stores a RELATIVE URL (just the filename) instead
//  of absolute paths. This is critical because:
//  
//  1. iOS app containers can change location between launches
//  2. App updates may relocate the Documents directory
//  3. Absolute paths break when the container moves
//  
//  Solution: Store URL as `file:///filename.jpg` (relative), then use the
//  `actualFileURL` property which reconstructs the full path dynamically.
//  
//  This approach is backward-compatible with existing data!
//  
//  See MessageAttachment.swift for the implementation.
//

import SwiftUI
import SwiftData
import PhotosUI
import Foundation


// MARK: - Sample Integration

extension View {
    /// Sample: How to add image picker to your chat input area
    @ViewBuilder
    func sampleChatInputWithImagePicker(
        messageText: Binding<String>,
        showImagePicker: Binding<Bool>,
        onSendMessage: @escaping (String) -> Void,
        onSendWithImage: @escaping (String, UIImage, [DetectedObject]) -> Void
    ) -> some View {
        HStack(spacing: 12) {
            // Image attachment button
            Button {
                showImagePicker.wrappedValue = true
            } label: {
                Image(systemName: "photo.badge.plus")
                    .font(.title2)
                    .foregroundStyle(.blue)
            }
            .accessibilityLabel("Attach image")
            
            // Text input field
            TextField("Message", text: messageText, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...6)
            
            // Send button
            Button {
                if !messageText.wrappedValue.isEmpty {
                    onSendMessage(messageText.wrappedValue)
                    messageText.wrappedValue = ""
                }
            } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
                    .foregroundStyle(.blue)
            }
            .disabled(messageText.wrappedValue.isEmpty)
        }
        .padding()
    }
}

// MARK: - Complete Example View

struct ChatViewWithImageAttachments: View {
    @ObservedObject var viewModel: ChatViewModel
    
    // Image picker state
    @State private var showImagePicker = false
    @State private var selectedImage: UIImage?
    @State private var detectedObjects: [DetectedObject]?
    
    // Chat state
    @State private var messageText = ""
    
    var body: some View {
        VStack(spacing: 0) {
            // Message list
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(viewModel.conversation.messages.filter { $0.role != .system }) { message in
                        MessageBubbleWithAttachments(message: message)
                    }
                }
                .padding()
            }
            
            Divider()
            
            // Input area with image attachment support
            HStack(spacing: 12) {
                // Image attachment button
                Button {
                    showImagePicker = true
                } label: {
                    Image(systemName: "photo.badge.plus")
                        .font(.title2)
                        .foregroundStyle(.blue)
                }
                .accessibilityLabel("Attach image")
                
                // Text input
                TextField("Message", text: $messageText, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...6)
                
                // Send button
                Button {
                    sendMessage()
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.blue)
                }
                .disabled(messageText.isEmpty)
            }
            .padding()
        }
        .navigationTitle(viewModel.conversation.title)
        // Image picker sheet
        .sheet(isPresented: $showImagePicker) {
            ImagePickerWithDetection(
                selectedImage: $selectedImage,
                detectedObjects: $detectedObjects,
                isPresented: $showImagePicker
            ) { image, detections in
                // User confirmed image selection
                Task {
                    await viewModel.sendWithImage(
                        text: messageText.isEmpty ? "Analyze this image" : messageText,
                        image: image,
                        detections: detections
                    )
                    messageText = ""
                }
            }
        }
    }
    
    private func sendMessage() {
        guard !messageText.isEmpty else { return }
        
        Task {
            await viewModel.send(userText: messageText)
            messageText = ""
        }
    }
}

// MARK: - Message Bubble with Attachments

struct MessageBubbleWithAttachments: View {
    let message: Message
    
    var body: some View {
        HStack {
            if message.role == .user {
                Spacer()
            }
            
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 8) {
                // Show attachments first (for user messages)
                if message.role == .user {
                    ForEach(message.attachments, id: \.id) { attachment in
                        if attachment.type == .image {
                            // Note: In your actual app, use MessageImageAttachmentView from ChatView.swift
                            // Since this is a sample/integration file, we'll show a simple placeholder
                            SimpleImageAttachmentPreview(attachment: attachment)
                        }
                    }
                }
                
                // Then show text
                if !message.text.isEmpty {
                    Text(message.displayText)
                        .padding(12)
                        .background(
                            message.role == .user ? Color.blue.opacity(0.2) : Color.secondary.opacity(0.2)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                }
            }
            .frame(maxWidth: 300, alignment: message.role == .user ? .trailing : .leading)
            
            if message.role == .assistant {
                Spacer()
            }
        }
    }
}

// MARK: - Toolbar Integration Example

struct ChatViewToolbarExample: View {
    @ObservedObject var viewModel: ChatViewModel
    @State private var showImagePicker = false
    @State private var selectedImage: UIImage?
    @State private var detectedObjects: [DetectedObject]?
    @State private var messageText = ""
    
    var body: some View {
        Text("Chat View")
            .toolbar {
                // Bottom toolbar with image attachment
                ToolbarItemGroup(placement: .bottomBar) {
                    // Image button
                    Button {
                        showImagePicker = true
                    } label: {
                        Label("Attach Image", systemImage: "photo.badge.plus")
                    }
                    
                    Spacer()
                    
                    // Other actions...
                }
            }
            .sheet(isPresented: $showImagePicker) {
                ImagePickerWithDetection(
                    selectedImage: $selectedImage,
                    detectedObjects: $detectedObjects,
                    isPresented: $showImagePicker
                ) { image, detections in
                    Task {
                        await viewModel.sendWithImage(
                            text: messageText,
                            image: image,
                            detections: detections
                        )
                        messageText = ""
                    }
                }
            }
    }
}

// MARK: - Alternative: Floating Action Button Style

struct ChatViewWithFloatingImageButton: View {
    @ObservedObject var viewModel: ChatViewModel
    @State private var showImagePicker = false
    @State private var selectedImage: UIImage?
    @State private var detectedObjects: [DetectedObject]?
    
    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            // Main chat content
            ScrollView {
                // Your messages...
            }
            
            // Floating image attachment button
            Button {
                showImagePicker = true
            } label: {
                Image(systemName: "camera.fill")
                    .font(.title2)
                    .foregroundStyle(.white)
                    .frame(width: 56, height: 56)
                    .background(Circle().fill(Color.blue))
                    .shadow(radius: 4)
            }
            .padding(.trailing, 20)
            .padding(.bottom, 100) // Above keyboard area
            .sheet(isPresented: $showImagePicker) {
                ImagePickerWithDetection(
                    selectedImage: $selectedImage,
                    detectedObjects: $detectedObjects,
                    isPresented: $showImagePicker
                ) { image, detections in
                    Task {
                        await viewModel.sendWithImage(
                            text: "What's in this image?",
                            image: image,
                            detections: detections
                        )
                    }
                }
            }
        }
    }
}

// MARK: - Context Menu Integration

extension View {
    /// Add image attachment option to message context menu
    func messageContextMenu(
        message: Message,
        onAttachImage: @escaping () -> Void
    ) -> some View {
        contextMenu {
            if message.role == .user {
                Button {
                    onAttachImage()
                } label: {
                    Label("Add Image", systemImage: "photo.badge.plus")
                }
            }
            
            // Other context menu items...
        }
    }
}

// MARK: - Quick Actions

struct ImageAttachmentQuickActions: View {
    @Binding var showImagePicker: Bool
    @Binding var showCamera: Bool
    
    var body: some View {
        HStack(spacing: 20) {
            // From library
            Button {
                showImagePicker = true
            } label: {
                VStack(spacing: 8) {
                    Image(systemName: "photo.on.rectangle")
                        .font(.title)
                    Text("Photo Library")
                        .font(.caption)
                }
            }
            
            // From camera
            Button {
                showCamera = true
            } label: {
                VStack(spacing: 8) {
                    Image(systemName: "camera")
                        .font(.title)
                    Text("Take Photo")
                        .font(.caption)
                }
            }
        }
        .padding()
    }
}

// MARK: - Simple Image Preview (for samples)

/// A simple image attachment preview for the sample integration views
/// In your actual ChatView, use MessageImageAttachmentView from ChatView.swift
private struct SimpleImageAttachmentPreview: View {
    let attachment: MessageAttachment
    @State private var image: UIImage?
    @State private var isLoading = true
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let image = image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 200, maxHeight: 200)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                
                // Show detection info if available
                if let detectionInfo = attachment.detectionInfo {
                    Text(detectionInfo)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 4)
                }
            } else if isLoading {
                ProgressView()
                    .frame(width: 200, height: 150)
            } else {
                VStack(spacing: 4) {
                    Image(systemName: "photo")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("Image unavailable")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(width: 200, height: 150)
                .background(Color.secondary.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
        .task {
            await loadImage()
        }
    }
    
    private func loadImage() async {
        isLoading = true
        image = await attachment.loadImage()
        isLoading = false
    }
}

// MARK: - Previews

#Preview("Chat with Image Support") {
    // Keep this preview SwiftData-free to avoid macro expansion issues in some environments.
    // This shows a simple placeholder rather than constructing a full ChatViewModel/SwiftData stack.
    VStack(spacing: 12) {
        Image(systemName: "photo.badge.plus")
            .font(.largeTitle)
            .foregroundStyle(.blue)
        Text("Image attachment integration samples")
            .font(.headline)
        Text("Open this file to see how to integrate the picker and message attachments in your own ChatView.")
            .multilineTextAlignment(.center)
            .foregroundStyle(.secondary)
            .padding(.horizontal)
    }
    .padding()
}
