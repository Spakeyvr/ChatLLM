//
//  ChatView.swift
//  ChatLLM
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
    @Environment(\.locale) private var locale
    @Environment(\.scenePhase) private var scenePhase

    @ObservedObject var viewModel: ChatViewModel
    var onNewChat: (() -> Void)? = nil

    @State private var inputText: String = ""
    @State private var composerFocusRequest = 0
    @State private var editingSession: MessageEditingSession?
    @State private var shareText: String = ""
    @State private var isSharePresented: Bool = false

    // Search toggle state
    @State private var forceSearch: Bool = false
    @AppStorage(AppSettingsKeys.disableToolCalls) private var disableToolCallsPreference: Bool = false

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
    @State private var selectedPhotoItem: PhotosPickerItem?

    // Error handling states
    @State private var errorAlert: ErrorAlert?

    // Performance optimization states
    @State private var isViewAppearing = false
    @State private var scrollPosition = ScrollPosition(edge: .bottom)

    // In-app browser state
    @State private var activeURL: URL?

    private let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "ChatView", category: "ChatView")

    // Pure computed property — no side effects, no @State mutations during body evaluation.
    // Sorting a typical message list (~10–50 items) is negligible; SwiftUI diffs the result.
    private var sortedMessages: [Message] {
        let messages = viewModel.conversation.messages
            .filter { $0.role != .system }
            .sortedByOrder

        guard let editingSession else {
            return messages
        }
        return messages.filter { $0.order < editingSession.messageOrder }
    }
    private var lastMessageID: UUID? { sortedMessages.last?.id }

    private var imageInputAvailable: Bool {
        modelBackendBridge.nativeImageInputAvailable
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

    private struct MessageEditingSession {
        let messageID: UUID
        let messageOrder: Int
        let previousInputText: String
    }

    struct ScrollFollowState: Equatable {
        static let nearBottomThreshold: CGFloat = 60

        let contentHeight: CGFloat
        let visibleHeight: CGFloat
        let distanceFromBottom: CGFloat

        var isNearBottom: Bool { distanceFromBottom <= Self.nearBottomThreshold }

        /// Re-pin to the bottom when a size change (not the reader's own scrolling) moved the
        /// bottom out of view while the reader was there. Any hidden overflow counts, not just
        /// overflow past the near-bottom threshold: a keyboard covering the last 40 points would
        /// otherwise leave the reply's action row hidden behind the composer.
        func shouldFollow(from old: ScrollFollowState) -> Bool {
            let sizeChanged = contentHeight != old.contentHeight || visibleHeight != old.visibleHeight
            return sizeChanged && old.isNearBottom && distanceFromBottom > 0.5
        }
    }

    private struct MessageEditingCommit {
        let messageID: UUID?
        let requiresTitleRegeneration: Bool
    }

    static func shouldPresentInAppBrowser(for url: URL) -> Bool {
        guard let scheme = url.scheme?.lowercased() else {
            return false
        }

        return scheme == "http" || scheme == "https"
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    // Use ForEach with stable IDs for better performance
                    ForEach(sortedMessages, id: \.id) { msg in
                        MessageCellView(
                            message: msg,
                            viewModel: viewModel,
                            onEdit: { message in
                                menuSafe {
                                    guard !viewModel.isGenerating else { return }

                                    let previousInputText = editingSession?.previousInputText ?? inputText
                                    editingSession = MessageEditingSession(
                                        messageID: message.id,
                                        messageOrder: message.order,
                                        previousInputText: previousInputText
                                    )
                                    inputText = message.text
                                    composerFocusRequest += 1
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
                        LoadingIndicatorView(activity: viewModel.streamingActivity)
                            .padding(.vertical, 8)
                    }
                }
                .padding()
            }
            .scrollDismissesKeyboard(.interactively)
            .scrollPosition($scrollPosition)
            // Short chats sit at the top edge; longer ones open at the latest message.
            // Edge-based scrolling (unlike scrolling a message to the bottom anchor) never
            // pushes short content past the top.
            .defaultScrollAnchor(.top, for: .alignment)
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            // Follow a growing reply (or a shrinking viewport) only while the reader is already
            // at the bottom; someone who scrolled up to read earlier messages stays put.
            .onScrollGeometryChange(for: ScrollFollowState.self) { geometry in
                // visibleRect spans the insets (navigation bar, composer, keyboard), so
                // measure against the inset-free container and the inset content bottom.
                ScrollFollowState(
                    contentHeight: geometry.contentSize.height,
                    visibleHeight: geometry.containerSize.height,
                    distanceFromBottom: geometry.contentSize.height + geometry.contentInsets.bottom
                        - geometry.visibleRect.maxY
                )
            } action: { old, new in
                if new.shouldFollow(from: old) {
                    scrollPosition.scrollTo(edge: .bottom)
                }
            }
            .onAppear {
                Task { @MainActor in
                    isViewAppearing = true
                    try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
                    if !Task.isCancelled {
                        scrollToBottom(animated: false)
                    }
                }
            }
            // Scroll when streaming message changes
            .onChange(of: viewModel.streamingMessageID) { _, newID in
                guard let id = newID,
                      sortedMessages.contains(where: { $0.id == id }) else { return }
                scrollToBottom(animated: true)
            }
            // Scroll when the last message ID changes (more precise than count)
            .onChange(of: lastMessageID) { _, _ in
                scrollToBottom(animated: true)
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .onAppear(perform: restoreComposerDraft)
        .onChange(of: inputText) { _, text in
            viewModel.composerDraft.recordText(text, preEditText: editingSession?.previousInputText)
        }
        .onChange(of: selectedImageToken) {
            viewModel.composerDraft.image = selectedImage
        }
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .principal) {
                NavigationTitleView(
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
                .accessibilityLabel("New Chat")
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
                    await handleError(ImageError.invalidFormat, title: String(localized: "Photo Import Failed", bundle: .appLocalized), retry: nil)
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
                    await handleError(error, title: String(localized: "File Import Failed", bundle: .appLocalized), retry: nil)
                }
            }
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                break
            case .active:
                if isViewAppearing {
                    scrollToBottom(animated: false)
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
            placeholder: String(localized: "Ask anything", bundle: .appLocalized, locale: locale),
            focusRequest: composerFocusRequest,
            isEditing: editingSession != nil,
            canSend: selectedImage != nil || !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            isGenerating: viewModel.isGenerating,
            onSend: {
                Task { await sendIfPossible() }
            },
            onStop: {
                viewModel.cancelGeneration()
            },
            onCancelEditing: cancelEditing,
            onCamera: {
                showCameraCapture = true
            },
            onPhotosPicker: {
                showImagePicker = true
            },
            onFileImporter: {
                isFileImporterPresented = true
            },
            imageInputAvailable: imageInputAvailable,
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
            reasoningAvailable: modelBackendBridge.reasoningAvailable,
            reasoningRequired: modelBackendBridge.reasoningRequired
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
            // An image picked before switching to a text-only model stays in
            // the composer; that model would answer without seeing it.
            guard imageInputAvailable else {
                await handleError(
                    ImageError.unsupportedByModel,
                    title: String(localized: "Can't Send Image", bundle: .appLocalized),
                    retry: nil
                )
                return
            }

            let replacementText = textToSend.isEmpty ? "Describe what you see" : textToSend
            guard let editingCommit = commitEditingIfNeeded(replacementText: replacementText) else { return }

            // Clear immediately for snappy feel, then send
            inputText = ""
            clearSelectedImage()

            Task {
                await viewModel.sendWithImage(
                    text: replacementText,
                    image: image,
                    regenerateTitle: editingCommit.requiresTitleRegeneration,
                    editedUserMessageID: editingCommit.messageID
                )
            }
            return
        }

        // Normal text-only message
        guard !viewModel.isGenerating, !textToSend.isEmpty else { return }
        guard let editingCommit = commitEditingIfNeeded(replacementText: textToSend) else { return }

        let shouldSearch = forceSearch
        let shouldDisableToolCalls = effectiveDisableToolCalls

        // Clear input immediately for snappy feel, then send
        inputText = ""
        forceSearch = false

        Task {
            await viewModel.send(
                userText: textToSend,
                forceSearch: shouldSearch,
                disableToolCalls: shouldDisableToolCalls,
                regenerateTitle: editingCommit.requiresTitleRegeneration,
                editedUserMessageID: editingCommit.messageID
            )
        }
    }

    private func cancelEditing() {
        guard let editingSession else { return }
        inputText = editingSession.previousInputText
        self.editingSession = nil
        composerFocusRequest += 1
    }

    private func commitEditingIfNeeded(replacementText: String) -> MessageEditingCommit? {
        guard let editingSession else {
            return MessageEditingCommit(messageID: nil, requiresTitleRegeneration: false)
        }
        guard let rewind = viewModel.rewindForEditingUserMessage(
            messageID: editingSession.messageID,
            replacementText: replacementText
        ) else {
            return nil
        }

        self.editingSession = nil
        return MessageEditingCommit(
            messageID: rewind.messageID,
            requiresTitleRegeneration: rewind.requiresTitleRegeneration
        )
    }

    private func scrollToBottom(animated: Bool) {
        guard !sortedMessages.isEmpty else { return }
        if animated {
            withAnimation(.easeInOut(duration: 0.25)) {
                scrollPosition.scrollTo(edge: .bottom)
            }
        } else {
            scrollPosition.scrollTo(edge: .bottom)
        }
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
        await MainActor.run {
            setSelectedImage(image)
        }
    }

    /// Brings back an unsent draft after this view was rebuilt for the same conversation.
    private func restoreComposerDraft() {
        let draft = viewModel.composerDraft
        if inputText.isEmpty, !draft.text.isEmpty {
            inputText = draft.text
        }
        if selectedImage == nil, let image = draft.image {
            Task { await prepareSelectedImage(image) }
        }
    }

    @MainActor
    private func setSelectedImage(_ image: UIImage) {
        selectedImage = image
        selectedImageToken = UUID()
        forceSearch = false
    }

    @MainActor
    private func clearSelectedImage() {
        selectedImage = nil
        selectedImageToken = UUID()
    }

    // MARK: - Error Types

    private enum ImageError: LocalizedError {
        case noData
        case invalidFormat
        case accessDenied
        case unsupportedByModel

        var errorDescription: String? {
            switch self {
            case .noData:
                return String(localized: "Could not load the selected image.", bundle: .appLocalized)
            case .invalidFormat:
                return String(localized: "The selected file is not a valid image.", bundle: .appLocalized)
            case .accessDenied:
                return String(localized: "Access to the file was denied.", bundle: .appLocalized)
            case .unsupportedByModel:
                return String(localized: "The selected model can't read images. Remove the image or switch to a model that supports images.", bundle: .appLocalized)
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
