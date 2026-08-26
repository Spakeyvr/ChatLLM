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
import ImageIO

struct ChatView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(\.scenePhase) private var scenePhase

    @ObservedObject var viewModel: ChatViewModel
    var onNewChat: (() -> Void)? = nil

    @State private var inputText: String = ""
    @State private var editingMessage: Message?
    @State private var editedText: String = ""
    @State private var showEditSheet = false
    @State private var shareText: String = ""
    @State private var isSharePresented: Bool = false

    // Search toggle state
    @State private var forceSearch: Bool = false
    @AppStorage("disableToolCalls") private var disableToolCallsPreference: Bool = false

    // Backend bridge for checking reasoning availability
    @ObservedObject private var modelBackendBridge = ModelBackendBridge.shared

    // Camera state
    @State private var showCameraCapture = false

    // Network monitoring
    @ObservedObject private var networkMonitor = NetworkMonitor.shared
    @State private var isFileImporterPresented = false

    // Image attachment state
    @State private var showImagePicker = false
    @State private var selectedImage: UIImage?
    @State private var selectedImageToken = UUID()
    @State private var detectedObjects: [DetectedObject]?
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var fullImageAnalysis: VisionAnalysisResult?

    // Error handling states
    @State private var errorAlert: ErrorAlert?

    // Performance optimization states
    @State private var isViewAppearing = false
    @State private var scrollProxy: ScrollViewProxy?

    // In-app browser state
    @State private var activeURL: URL?

    private let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "ChatView", category: "ChatView")

    // Pure computed property — no side effects, no @State mutations during body evaluation.
    // Sorting a typical message list (~10–50 items) is negligible; SwiftUI diffs the result.
    private var sortedMessages: [Message] {
        viewModel.conversation.messages
            .filter { $0.role != .system }
            .sortedByOrder
    }
    private var lastMessageID: UUID? { sortedMessages.last?.id }

    private var shouldPrecomputeVisionAnalysis: Bool {
        guard modelBackendBridge.selectedBackend == .mlx else { return true }
        return !(modelBackendBridge.modelManager?.currentModel?.supportsNativeImages ?? false)
    }

    private var webSearchToggleAvailable: Bool {
        networkMonitor.isConnected &&
        selectedImage == nil &&
        modelBackendBridge.toolCallsAvailableForCurrentBackend &&
        viewModel.searchService != nil
    }

    private var toolCallsLockedDisabled: Bool {
        !modelBackendBridge.toolCallsAvailableForCurrentBackend
    }

    private var effectiveDisableToolCalls: Bool {
        toolCallsLockedDisabled || disableToolCallsPreference
    }

    private var disableToolCallsBinding: Binding<Bool> {
        Binding(
            get: { effectiveDisableToolCalls },
            set: { disableToolCallsPreference = $0 }
        )
    }

    // Error handling helper
    private struct ErrorAlert: Identifiable {
        let id = UUID()
        let title: String
        let message: String
        let retry: (() -> Void)?
    }

    static func shouldPresentInAppBrowser(for url: URL) -> Bool {
        guard let scheme = url.scheme?.lowercased() else {
            return false
        }

        return scheme == "http" || scheme == "https"
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
                    hasMessages: !sortedMessages.isEmpty,
                    modelBackendBridge: modelBackendBridge
                )
            }
            ToolbarItemGroup(placement: .navigationBarTrailing) {
                Button {
                    onNewChat?()
                } label: {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(canStartNewChat ? .primary : .secondary)
                }
                .disabled(!canStartNewChat)
                .accessibilityLabel(String(localized: "New Chat"))
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
                                clearSelectedImage()
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
                guard let image = await loadAndDownscaleImage(from: item) else {
                    logger.error("Failed to load image from picker item")
                    await MainActor.run {
                        selectedPhotoItem = nil
                    }
                    await handleError(ImageError.invalidFormat, title: "Photo Import Failed", retry: nil)
                    return
                }
                await prepareSelectedImage(image)
                await MainActor.run { selectedPhotoItem = nil }
            }
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
        .onChange(of: effectiveDisableToolCalls) { _, isDisabled in
            if isDisabled && forceSearch {
                forceSearch = false
            }
        }
        .onChange(of: webSearchToggleAvailable) { _, isAvailable in
            if !isAvailable && forceSearch {
                forceSearch = false
            }
        }
        .environment(\.openURL, OpenURLAction { url in
            if Self.shouldPresentInAppBrowser(for: url) {
                activeURL = url
                return .handled
            }

            return .systemAction(url)
        })
        .sheet(item: $activeURL) { url in
            SafariView(url: url)
                .ignoresSafeArea()
        }
        .sheet(isPresented: $showCameraCapture) {
            CameraCapturePicker(isPresented: $showCameraCapture) { image in
                Task {
                    await prepareSelectedImage(image)
                }
            }
                .ignoresSafeArea()
        }
    }

    @ViewBuilder
    private var composer: some View {
        ComposerView(
            text: $inputText,
            placeholder: String(localized: "Ask anything"),
            canSend: selectedImage != nil || !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
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
            searchAvailable: webSearchToggleAvailable,
            disableToolCalls: disableToolCallsBinding,
            toolCallsLockedDisabled: toolCallsLockedDisabled,
            isReasoningEnabled: Binding(
                get: { viewModel.conversation.reasoningMode },
                set: { viewModel.setReasoningMode($0) }
            ),
            isSmartReasoningEnabled: Binding(
                get: { viewModel.conversation.smartReasoningMode },
                set: { viewModel.setSmartReasoningMode($0) }
            ),
            reasoningAvailable: modelBackendBridge.reasoningAvailable
        )
    }

    // MARK: - Computed State

    /// True when the current conversation has at least one non-system message.
    /// Used to gray out the New Chat button when already on an empty chat.
    private var canStartNewChat: Bool {
        viewModel.conversation.messages.contains { $0.role != .system }
    }

    // MARK: - Helper Methods

    private func sendIfPossible() async {
        let textToSend = inputText.trimmingCharacters(in: .whitespacesAndNewlines)

        // Check if we have an image to send
        if let image = selectedImage {
            guard !viewModel.isGenerating else { return }

            var analysis = fullImageAnalysis
            var detections = detectedObjects ?? []
            if shouldPrecomputeVisionAnalysis && analysis == nil {
                analysis = await analyzeImage(image)
                detections = analysis?.objects ?? detections
            }

            // Clear immediately for snappy feel, then send
            inputText = ""
            clearSelectedImage()

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
        let shouldDisableToolCalls = effectiveDisableToolCalls

        // Clear input immediately for snappy feel, then send
        inputText = ""
        forceSearch = false

        Task {
            await viewModel.send(
                userText: textToSend,
                forceSearch: shouldSearch,
                disableToolCalls: shouldDisableToolCalls
            )
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

            guard let image = decodeThumbnailImage(
                from: url,
                maxPixelSize: 2_048
            ) else {
                throw ImageError.invalidFormat
            }
            await prepareSelectedImage(image)
        case .failure(let error):
            logger.error("File importer failed: \(error.localizedDescription)")
            throw error
        }
    }

    private func prepareSelectedImage(_ image: UIImage) async {
        let token = await MainActor.run {
            setSelectedImage(image)
        }
        let shouldAnalyze = await MainActor.run {
            shouldPrecomputeVisionAnalysis
        }
        if shouldAnalyze {
            await runVisionAnalysis(on: image, token: token)
        } else {
            await MainActor.run {
                guard selectedImageToken == token else { return }
                detectedObjects = nil
                fullImageAnalysis = nil
            }
        }
    }

    @MainActor
    @discardableResult
    private func setSelectedImage(_ image: UIImage) -> UUID {
        selectedImage = image
        detectedObjects = nil
        fullImageAnalysis = nil
        selectedImageToken = UUID()
        forceSearch = false
        return selectedImageToken
    }

    @MainActor
    private func clearSelectedImage() {
        selectedImage = nil
        detectedObjects = nil
        fullImageAnalysis = nil
        selectedImageToken = UUID()
    }

    private func analyzeImage(_ image: UIImage) async -> VisionAnalysisResult? {
        let analyzer = VisionAnalyzer()
        var options = AnalysisOptions.all
        options.minimumTextConfidence = 0.3
        options.useAccurateOCR = true
        return try? await analyzer.analyze(image: image, options: options)
    }

    private func runVisionAnalysis(on image: UIImage, token: UUID) async {
        if let result = await analyzeImage(image) {
            await MainActor.run {
                guard selectedImageToken == token else { return }
                detectedObjects = result.objects
                fullImageAnalysis = result
            }
        } else {
            await MainActor.run {
                guard selectedImageToken == token else { return }
                detectedObjects = []
                fullImageAnalysis = nil
            }
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

    // MARK: - Image Downscaling

    // Load a PhotosPickerItem directly through ImageIO at the pipeline budget.
    private func loadAndDownscaleImage(from item: PhotosPickerItem) async -> UIImage? {
        if let thumbnail = try? await loadThumbnailImage(from: item, maxPixelSize: 1_200) {
            return thumbnail
        }
        return nil
    }

    private func loadThumbnailImage(
        from item: PhotosPickerItem,
        maxPixelSize: Int
    ) async throws -> UIImage? {
        guard let data = try await item.loadTransferable(type: Data.self) else {
            return nil
        }
        return decodeThumbnailImage(from: data, maxPixelSize: maxPixelSize)
    }

    private func decodeThumbnailImage(from data: Data, maxPixelSize: Int) -> UIImage? {
        data.withUnsafeBytes { buffer in
            guard let baseAddress = buffer.baseAddress else { return nil }
            let cfData = CFDataCreate(kCFAllocatorDefault, baseAddress.assumingMemoryBound(to: UInt8.self), data.count)
            guard let cfData,
                  let source = CGImageSourceCreateWithData(cfData, nil) else {
                return nil
            }
            return decodeThumbnailImage(from: source, maxPixelSize: maxPixelSize)
        }
    }

    private func decodeThumbnailImage(from url: URL, maxPixelSize: Int) -> UIImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            return nil
        }
        return decodeThumbnailImage(from: source, maxPixelSize: maxPixelSize)
    }

    private func decodeThumbnailImage(
        from source: CGImageSource,
        maxPixelSize: Int
    ) -> UIImage? {
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: false,
            kCGImageSourceShouldCache: false,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize
        ]
        guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            return nil
        }
        return UIImage(cgImage: cgImage)
    }
}

// MARK: - Camera Capture Picker

private struct CameraCapturePicker: UIViewControllerRepresentable {
    @Binding var isPresented: Bool
    let onCapture: @MainActor (UIImage) -> Void

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
            guard let image = info[.originalImage] as? UIImage else {
                parent.isPresented = false
                return
            }
            parent.isPresented = false

            // UIImagePickerController only exposes a UIImage for camera stills.
            // Prepare a bounded thumbnail before handing it to chat state, so
            // the original capture is never retained by the SwiftUI view tree.
            Task { @MainActor in
                let pixelWidth = image.size.width * image.scale
                let pixelHeight = image.size.height * image.scale
                let maxSide = max(pixelWidth, pixelHeight)
                guard maxSide > 1_200 else {
                    parent.onCapture(image)
                    return
                }
                let ratio = 1_200 / maxSide
                let targetSize = CGSize(
                    width: (pixelWidth * ratio).rounded(),
                    height: (pixelHeight * ratio).rounded()
                )
                let thumbnail = await image.byPreparingThumbnail(ofSize: targetSize)
                parent.onCapture(thumbnail ?? image)
            }
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.isPresented = false
        }
    }
}
