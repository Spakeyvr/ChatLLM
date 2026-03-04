//
//  ChatView.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI
import SwiftData
import UniformTypeIdentifiers
import UIKit
import PhotosUI
import Combine
import OSLog
import SafariServices

struct ChatView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(\.scenePhase) private var scenePhase

    @ObservedObject var viewModel: ChatViewModel

    @State private var inputText: String = ""
    @State private var editingMessage: Message?
    @State private var editedText: String = ""
    @State private var showEditSheet = false
    @State private var shareText: String = ""
    @State private var isSharePresented: Bool = false

    // Search toggle state
    @State private var forceSearch: Bool = false

    // Reasoning mode state
    @State private var showReasoningSettings: Bool = false

    // Backend bridge for checking reasoning availability
    @ObservedObject private var modelBackendBridge = ModelBackendBridge.shared

    // Camera state
    @State private var showCameraCapture = false

    // Pickers state
    @State private var selectedPickerItems: [PhotosPickerItem] = []

    // Network monitoring
    @ObservedObject private var networkMonitor = NetworkMonitor.shared
    @State private var isFileImporterPresented = false
    @State private var showPhotosPicker = false

    // Image attachment state
    @State private var showImagePicker = false
    @State private var selectedImage: UIImage?
    @State private var detectedObjects: [DetectedObject]?
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var fullImageAnalysis: VisionAnalysisResult?

    // Error handling states
    @State private var errorAlert: ErrorAlert?

    // Performance optimization states
    @State private var isViewAppearing = false
    @State private var scrollProxy: ScrollViewProxy?
    @State private var cachedSortedMessages: [Message] = []
    @State private var lastMessageCount: Int = 0

    // In-app browser state
    @State private var activeURL: URL?

    // Logging - lazy to avoid startup overhead
    private var logger: Logger {
        Logger(subsystem: Bundle.main.bundleIdentifier ?? "ChatView", category: "ChatView")
    }

    // Optimized message sorting with caching
    private var sortedMessages: [Message] {
        // Cache invalidation: only re-sort if message count changes
        let currentCount = viewModel.conversation.messages.count
        if currentCount == lastMessageCount && !cachedSortedMessages.isEmpty {
            return cachedSortedMessages
        }

        // Sort and cache synchronously for immediate use
        let sorted = Array(viewModel.conversation.messages).sorted { $0.order < $1.order }
        cachedSortedMessages = sorted
        lastMessageCount = currentCount

        return sorted
    }
    private var lastMessageID: UUID? { sortedMessages.last?.id }

    // Error handling helper
    private struct ErrorAlert: Identifiable {
        let id = UUID()
        let title: String
        let message: String
        let retry: (() -> Void)?
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        // Use ForEach with stable IDs for better performance
                        ForEach(sortedMessages, id: \.id) { msg in
                            MessageCellView(
                                message: msg,
                                viewModel: viewModel,
                                onEdit: { message in
                                    // Defer to allow context menu to dismiss cleanly
                                    menuSafe {
                                        editingMessage = message
                                        editedText = message.text
                                        showEditSheet = true
                                    }
                                },
                                onShare: { text in
                                    // Defer to allow context menu to dismiss cleanly
                                    menuSafe {
                                        shareText = text
                                        isSharePresented = true
                                    }
                                }
                            )
                        }

                        if viewModel.isGenerating,
                           let streamingID = viewModel.streamingMessageID,
                           !sortedMessages.contains(where: { $0.id == streamingID }) {
                            LoadingIndicatorView(isModelLoading: modelBackendBridge.modelManager?.isLoading ?? false)
                                .padding(.vertical, 8)
                        }
                    }
                    .padding()
                }
                .scrollDismissesKeyboard(.interactively)
                .onAppear {
                    Task { @MainActor in
                        scrollProxy = proxy
                        isViewAppearing = true
                        try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
                        if !Task.isCancelled {
                            scrollToBottomIfNeeded(proxy: proxy, animated: false)
                        }
                    }
                }
                // Scroll when streaming message changes
                .onChange(of: viewModel.streamingMessageID) { _, newID in
                    guard let id = newID,
                          sortedMessages.contains(where: { $0.id == id }) else { return }
                    scrollToMessage(id: id, proxy: proxy, animated: true)
                }
                // Scroll when the last message ID changes (more precise than count)
                .onChange(of: lastMessageID) { _, _ in
                    scrollToBottomIfNeeded(proxy: proxy, animated: true)
                }
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .principal) {
                NavigationTitleView(
                    title: viewModel.conversation.title,
                    isReasoningEnabled: viewModel.conversation.reasoningMode,
                    isSmartReasoningEnabled: viewModel.conversation.smartReasoningMode,
                    reasoningAvailable: modelBackendBridge.reasoningAvailable,
                    hasMessages: !viewModel.conversation.messages.isEmpty,
                    modelBackendBridge: modelBackendBridge
                )
            }
            ToolbarItemGroup(placement: .navigationBarTrailing) {
                ToolbarButtonsView(
                    isReasoningEnabled: viewModel.conversation.reasoningMode,
                    isSmartReasoningEnabled: viewModel.conversation.smartReasoningMode,
                    reasoningAvailable: modelBackendBridge.reasoningAvailable,
                    onReasoningTap: { showReasoningSettings = true }
                )
            }
        }
        .safeAreaInset(edge: .bottom) {
            VStack(spacing: 0) {
                // Image preview (if image is attached)
                if let image = selectedImage {
                    HStack(spacing: 12) {
                        // Thumbnail
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: 50, height: 50)
                            .clipShape(RoundedRectangle(cornerRadius: 8))

                        Text("Image attached")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)

                        Spacer()

                        // Remove button
                        Button {
                            withAnimation {
                                selectedImage = nil
                                detectedObjects = nil
                                fullImageAnalysis = nil
                            }
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.title3)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(10)
                    .background(Color(uiColor: .secondarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .padding(.horizontal, 16)
                    .padding(.bottom, 8)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }

                composer
                    .padding(.horizontal, 16)
                    .padding(.bottom, 12)
            }
            .background(Color.clear)
        }
        .sheet(isPresented: $showEditSheet) {
            EditMessageSheet(
                editedText: $editedText,
                isGenerating: viewModel.isGenerating,
                onSave: {
                    guard let message = editingMessage else { return }
                    // Defer to ensure sheet button tap completes
                    menuSafe {
                        Task {
                            await viewModel.editUserMessageAndRegenerate(from: message.id, newText: editedText)
                            showEditSheet = false
                        }
                    }
                },
                onCancel: { showEditSheet = false }
            )
        }
        .sheet(isPresented: $showReasoningSettings) {
            NavigationStack {
                ReasoningModeSettings(
                    isEnabled: Binding(
                        get: { viewModel.conversation.reasoningMode },
                        set: { newValue in
                            viewModel.setReasoningMode(newValue)
                        }
                    ),
                    isSmartEnabled: Binding(
                        get: { viewModel.conversation.smartReasoningMode },
                        set: { newValue in
                            viewModel.setSmartReasoningMode(newValue)
                        }
                    ),
                    reasoningAvailable: modelBackendBridge.reasoningAvailable,
                    thinkingModeInfo: modelBackendBridge.getThinkingModeInfo(),
                    onDismiss: { showReasoningSettings = false }
                )
                .navigationTitle(String(localized: "Reasoning Mode"))
                .navigationBarTitleDisplayMode(.inline)
            }
            .presentationDetents([.medium])
        }
        .sheet(isPresented: $isSharePresented, onDismiss: {
            shareText = ""
        }) {
            ActivityView(activityItems: [shareText as Any])
        }
        .photosPicker(
            isPresented: $showImagePicker,
            selection: $selectedPhotoItem,
            matching: .images,
            photoLibrary: .shared()
        )
        .onChange(of: selectedPhotoItem) { _, newItem in
            Task {
                guard let item = newItem else { return }
                do {
                    guard let data = try await item.loadTransferable(type: Data.self),
                          let image = UIImage(data: data) else {
                        return
                    }
                    await MainActor.run {
                        selectedImage = image
                        forceSearch = false
                    }
                    await runVisionAnalysis(on: image)
                } catch {
                    logger.error("Failed to load image: \(error.localizedDescription)")
                }
                selectedPhotoItem = nil
            }
        }
        .onChange(of: showCameraCapture) { _, isPresented in
            // When camera sheet is dismissed with a captured image, run Vision analysis
            guard !isPresented, let image = selectedImage else { return }
            detectedObjects = nil
            fullImageAnalysis = nil
            forceSearch = false
            Task { await runVisionAnalysis(on: image) }
        }
        .alert(item: $errorAlert) { alert in
            if let retry = alert.retry {
                return Alert(
                    title: Text(alert.title),
                    message: Text(alert.message),
                    primaryButton: .default(Text("Retry"), action: retry),
                    secondaryButton: .cancel()
                )
            } else {
                return Alert(
                    title: Text(alert.title),
                    message: Text(alert.message),
                    dismissButton: .default(Text("OK"))
                )
            }
        }
        .onChange(of: selectedPickerItems) { _, newItems in
            Task {
                do {
                    try await handlePickedPhotos(items: newItems)
                } catch {
                    await handleError(error, title: "Photo Import Failed", retry: nil)
                }
            }
        }
        .fileImporter(
            isPresented: $isFileImporterPresented,
            allowedContentTypes: [.image],
            allowsMultipleSelection: false
        ) { result in
            Task {
                do {
                    try await handleFileImporterResult(result)
                } catch {
                    await handleError(error, title: "File Import Failed", retry: nil)
                }
            }
        }
        .photosPicker(
            isPresented: $showPhotosPicker,
            selection: $selectedPickerItems,
            matching: .images,
            photoLibrary: .shared()
        )
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                break
            case .active:
                if isViewAppearing, let proxy = scrollProxy {
                    scrollToBottomIfNeeded(proxy: proxy, animated: false)
                }
            default:
                break
            }
        }
        // Reset force search when network becomes unavailable
        .onChange(of: networkMonitor.isConnected) { _, isConnected in
            if !isConnected && forceSearch {
                forceSearch = false
            }
        }
        // In-app browser: intercept link taps and present SafariView
        .environment(\.openURL, OpenURLAction { url in
            activeURL = url
            return .handled
        })
        .sheet(item: $activeURL) { url in
            SafariView(url: url)
                .ignoresSafeArea()
        }
        .sheet(isPresented: $showCameraCapture) {
            CameraCapturePicker(image: $selectedImage, isPresented: $showCameraCapture)
                .ignoresSafeArea()
        }
    }

    @ViewBuilder
    private var composer: some View {
        ComposerView(
            text: $inputText,
            placeholder: String(localized: "Message"),
            canSend: !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            isGenerating: viewModel.isGenerating,
            onSend: {
                Task { await sendIfPossible() }
            },
            onStop: {
                viewModel.cancelGeneration()
            },
            onClear: {
                inputText = ""
            },
            onCamera: {
                showCameraCapture = true
            },
            onPhotosPicker: {
                showImagePicker = true
            },
            onFileImporter: {
                isFileImporterPresented = true
            },
            forceSearch: $forceSearch,
            searchAvailable: networkMonitor.isConnected && selectedImage == nil
        )
    }

    // MARK: - Helper Methods

    private func sendIfPossible() async {
        let textToSend = inputText.trimmingCharacters(in: .whitespacesAndNewlines)

        // Check if we have an image to send
        if let image = selectedImage {
            let detections = detectedObjects ?? []
            let analysis = fullImageAnalysis
            guard !viewModel.isGenerating else { return }

            // Clear immediately for snappy feel, then send
            inputText = ""
            selectedImage = nil
            detectedObjects = nil
            fullImageAnalysis = nil

            Task {
                await viewModel.sendWithImage(
                    text: textToSend.isEmpty ? "Describe what you see" : textToSend,
                    image: image,
                    detections: detections,
                    analysisResult: analysis
                )
            }
            return
        }

        // Normal text-only message
        guard !viewModel.isGenerating, !textToSend.isEmpty else { return }

        let shouldSearch = forceSearch

        // Clear input immediately for snappy feel, then send
        inputText = ""
        forceSearch = false

        Task {
            await viewModel.send(userText: textToSend, forceSearch: shouldSearch)
        }
    }

    private func scrollToMessage(id: UUID, proxy: ScrollViewProxy, animated: Bool) {
        guard sortedMessages.contains(where: { $0.id == id }) else { return }
        if animated {
            withAnimation(.easeInOut(duration: 0.25)) {
                proxy.scrollTo(id, anchor: UnitPoint.bottom)
            }
        } else {
            proxy.scrollTo(id, anchor: UnitPoint.bottom)
        }
    }

    private func scrollToBottomIfNeeded(proxy: ScrollViewProxy, animated: Bool) {
        guard let lastMessage = sortedMessages.last else { return }
        scrollToMessage(id: lastMessage.id, proxy: proxy, animated: animated)
    }

    @MainActor
    private func handleError(_ error: Error, title: String, retry: (() -> Void)?) async {
        logger.error("Error in ChatView: \(error.localizedDescription)")

        let message: String
        if let localizedError = error as? LocalizedError {
            message = localizedError.errorDescription ?? error.localizedDescription
        } else {
            message = error.localizedDescription
        }

        errorAlert = ErrorAlert(
            title: title,
            message: message,
            retry: retry
        )
    }

    // MARK: - Image and File Handling

    private func handlePickedPhotos(items: [PhotosPickerItem]) async throws {
        guard !items.isEmpty else { return }
        for item in items {
            do {
                // Prefer Data to avoid large UIImage intermediates; ImageStore will downscale
                guard let data = try await item.loadTransferable(type: Data.self) else {
                    throw ImageError.noData
                }
                guard let image = UIImage(data: data) else {
                    throw ImageError.invalidFormat
                }
                await handleImageForOCR(image)
            } catch {
                logger.error("Failed to process photo item: \(error.localizedDescription)")
                throw error
            }
        }
        await MainActor.run { selectedPickerItems = [] }
    }

    private func handleFileImporterResult(_ result: Result<[URL], Error>) async throws {
        switch result {
        case .success(let urls):
            guard let url = urls.first else {
                throw ImageError.noData
            }
            guard url.startAccessingSecurityScopedResource() else {
                throw ImageError.accessDenied
            }
            defer { url.stopAccessingSecurityScopedResource() }

            let data = try Data(contentsOf: url)
            guard let image = UIImage(data: data) else {
                throw ImageError.invalidFormat
            }
            await handleImageForOCR(image)
        case .failure(let error):
            logger.error("File importer failed: \(error.localizedDescription)")
            throw error
        }
    }

    private func runVisionAnalysis(on image: UIImage) async {
        let analyzer = VisionAnalyzer()
        var options = AnalysisOptions.all
        options.minimumTextConfidence = 0.3
        options.useAccurateOCR = true
        if let result = try? await analyzer.analyze(image: image, options: options) {
            await MainActor.run {
                detectedObjects = result.objects
                fullImageAnalysis = result
            }
        } else {
            await MainActor.run {
                detectedObjects = []
                fullImageAnalysis = nil
            }
        }
    }

    private func handleImageForOCR(_ image: UIImage) async {
        do {
            _ = try await ImageStore.shared.save(image: image)
            let text = await viewModel.extractOCR(from: image)

            await MainActor.run {
                let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmedText.isEmpty {
                    errorAlert = ErrorAlert(
                        title: String(localized: "OCR Result"),
                        message: String(localized: "No text found in the selected image."),
                        retry: nil
                    )
                } else {
                    if inputText.isEmpty {
                        inputText = trimmedText
                    } else {
                        inputText += "\n" + trimmedText
                    }
                }
            }
        } catch {
            await handleError(error, title: "Image Processing Failed", retry: nil)
        }
    }

    // MARK: - Error Types

    private enum ImageError: LocalizedError {
        case noData
        case invalidFormat
        case accessDenied
        case ocrFailed

        var errorDescription: String? {
            switch self {
            case .noData:
                return String(localized: "Could not load the selected image.")
            case .invalidFormat:
                return String(localized: "The selected file is not a valid image.")
            case .accessDenied:
                return String(localized: "Access to the file was denied.")
            case .ocrFailed:
                return String(localized: "Failed to extract text from the image.")
            }
        }
    }

    // Small helper: run UI mutations after menus have time to dismiss
    private func menuSafe(_ action: @escaping () -> Void) {
        Task { @MainActor in
            // Yield once and add a tiny delay to ensure UIKit menu dismissal completes
            await Task.yield()
            try? await Task.sleep(nanoseconds: 150_000_000) // ~150 ms
            action()
        }
    }
}

// MARK: - Camera Capture Picker

private struct CameraCapturePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Binding var isPresented: Bool

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraCapturePicker
        init(_ parent: CameraCapturePicker) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let img = info[.originalImage] as? UIImage {
                parent.image = img
            }
            parent.isPresented = false
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.isPresented = false
        }
    }
}
