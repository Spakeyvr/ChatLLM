import SwiftUI
import UIKit

// MARK: - Fullscreen Image View

struct FullscreenAttachmentImageView: View {
    let attachment: MessageAttachment
    @Binding var isPresented: Bool

    @State private var image: UIImage?
    @State private var finishedLoading = false

    var body: some View {
        Group {
            if let image {
                FullscreenImageView(
                    image: image,
                    isPresented: $isPresented
                )
            } else {
                // The cover has no backdrop of its own and can't be swiped away,
                // so placeholder states need the black backdrop and Done button too.
                ZStack(alignment: .topLeading) {
                    Color.black.ignoresSafeArea()
                    Group {
                        if finishedLoading {
                            ContentUnavailableView(
                                "Image unavailable",
                                systemImage: "photo.badge.exclamationmark"
                            )
                        } else {
                            ProgressView()
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    FullscreenToolbar(image: nil, isPresented: $isPresented)
                }
            }
        }
        .task {
            image = await attachment.loadImage()
            finishedLoading = true
        }
        // The viewer is always on black, so its controls use dark appearance.
        .environment(\.colorScheme, .dark)
    }
}

struct FullscreenImageView: View {
    let image: UIImage
    @Binding var isPresented: Bool
    
    @State private var dragOffset: CGSize = .zero
    
    var body: some View {
        // Read the safe area here, outside the sliding page: SwiftUI shrinks the safe
        // area of an offset view, so the page's black background stopped covering
        // the status bar as soon as a drag began.
        GeometryReader { proxy in
            let insets = proxy.safeAreaInsets
            ZStack(alignment: .top) {
                Color.black
                
                // The Zoomable View
                // The dismiss drag runs on the scroll view itself. A SwiftUI DragGesture
                // here had to wait for the scroll view's double-tap recognizer to fail,
                // so the image stuck for a moment and then jumped to the finger.
                ZoomableScrollView(
                    onDismissDragChanged: { translation in
                        // Only allow dragging down
                        if translation.height > 0 {
                            dragOffset = translation
                        }
                    },
                    onDismissDragEnded: { translation in
                        if translation.height > 150 {
                            slideAway(distance: proxy.size.height + insets.top + insets.bottom)
                        } else {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                                dragOffset = .zero
                            }
                        }
                    }
                ) {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
                .padding(insets)
                
                // Custom toolbar overlay
                FullscreenToolbar(image: image, isPresented: $isPresented)
                    .padding(.top, insets.top)
                    .padding(.leading, insets.leading)
                    .padding(.trailing, insets.trailing)
            }
            .ignoresSafeArea()
            // The background, image and buttons slide down together as one page.
            .offset(y: dragOffset.height)
        }
    }

    /// Finishes the swipe from where the finger let go, then removes the cover
    /// without its own slide, which would start the motion over from rest.
    private func slideAway(distance: CGFloat) {
        withAnimation(.easeOut(duration: 0.25)) {
            dragOffset.height = distance
        } completion: {
            var transaction = Transaction()
            transaction.disablesAnimations = true
            withTransaction(transaction) {
                isPresented = false
            }
        }
    }
}

private struct FullscreenToolbar: View {
    let image: UIImage?
    @Binding var isPresented: Bool

    var body: some View {
        HStack {
            Button {
                isPresented = false
            } label: {
                Text("Done")
                    .fontWeight(.semibold)
                    .frame(height: 30)
            }
            .buttonStyle(.glass)
            .buttonBorderShape(.capsule)
            .foregroundStyle(.white)

            Spacer()

            // A camera photo taken in the app exists only in the chat, so this is
            // the way to save or send it elsewhere.
            if let image {
                shareButton(for: image)
            }
        }
        .padding(.horizontal)
        .padding(.top, 8)
    }

    private func shareButton(for image: UIImage) -> some View {
        ShareLink(
            item: Image(uiImage: image),
            preview: SharePreview(String(localized: "Image", bundle: .appLocalized), image: Image(uiImage: image))
        ) {
            Image(systemName: "square.and.arrow.up")
                .fontWeight(.semibold)
                .frame(width: 30, height: 30)
        }
        .buttonStyle(.glass)
        .buttonBorderShape(.circle)
        .foregroundStyle(.white)
        .accessibilityLabel("Share")
        .accessibilityIdentifier("imageViewer.share")
    }
}

// MARK: - Zoomable ScrollView (UIKit Wrapper)

struct ZoomableScrollView<Content: View>: UIViewRepresentable {
    let content: Content
    var onDismissDragChanged: (CGSize) -> Void
    var onDismissDragEnded: (CGSize) -> Void
    
    init(
        onDismissDragChanged: @escaping (CGSize) -> Void,
        onDismissDragEnded: @escaping (CGSize) -> Void,
        @ViewBuilder content: () -> Content
    ) {
        self.onDismissDragChanged = onDismissDragChanged
        self.onDismissDragEnded = onDismissDragEnded
        self.content = content()
    }
    
    func makeUIView(context: Context) -> UIScrollView {
        let scrollView = UIScrollView()
        scrollView.delegate = context.coordinator
        scrollView.maximumZoomScale = 4.0
        scrollView.minimumZoomScale = 1.0
        scrollView.bouncesZoom = true
        scrollView.showsHorizontalScrollIndicator = false
        scrollView.showsVerticalScrollIndicator = false
        scrollView.backgroundColor = .clear
        
        let hostedView = context.coordinator.hostingController.view!
        hostedView.translatesAutoresizingMaskIntoConstraints = true
        hostedView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        hostedView.backgroundColor = .clear
        
        scrollView.addSubview(hostedView)

        let doubleTap = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleDoubleTap(_:))
        )
        doubleTap.numberOfTapsRequired = 2
        scrollView.addGestureRecognizer(doubleTap)

        let dismissPan = UIPanGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleDismissPan(_:))
        )
        dismissPan.delegate = context.coordinator
        scrollView.addGestureRecognizer(dismissPan)
        context.coordinator.updateAccessibilityValue(for: scrollView)
        return scrollView
    }
    
    func updateUIView(_ uiView: UIScrollView, context: Context) {
        context.coordinator.hostingController.rootView = content
        context.coordinator.onDismissDragChanged = onDismissDragChanged
        context.coordinator.onDismissDragEnded = onDismissDragEnded
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(hostingController: UIHostingController(rootView: content))
    }
    
    class Coordinator: NSObject, UIScrollViewDelegate, UIGestureRecognizerDelegate {
        var hostingController: UIHostingController<Content>
        var onDismissDragChanged: (CGSize) -> Void = { _ in }
        var onDismissDragEnded: (CGSize) -> Void = { _ in }
        
        init(hostingController: UIHostingController<Content>) {
            self.hostingController = hostingController
        }
        
        func viewForZooming(in scrollView: UIScrollView) -> UIView? {
            return scrollView.subviews.first
        }
        
        /// Like Photos: double-tap zooms in on the tapped spot, or back out when zoomed.
        @objc func handleDoubleTap(_ recognizer: UITapGestureRecognizer) {
            guard let scrollView = recognizer.view as? UIScrollView else { return }
            if scrollView.zoomScale > scrollView.minimumZoomScale {
                scrollView.setZoomScale(scrollView.minimumZoomScale, animated: true)
                return
            }
            guard let content = viewForZooming(in: scrollView) else { return }
            let scale = min(scrollView.maximumZoomScale, 2.5)
            let point = recognizer.location(in: content)
            let size = CGSize(
                width: scrollView.bounds.width / scale,
                height: scrollView.bounds.height / scale
            )
            let rect = CGRect(
                x: point.x - size.width / 2,
                y: point.y - size.height / 2,
                width: size.width,
                height: size.height
            )
            scrollView.zoom(to: rect, animated: true)
        }

        /// Drags the image down to dismiss it, starting with the finger's first movement.
        @objc func handleDismissPan(_ recognizer: UIPanGestureRecognizer) {
            // Measured in the window, because the scroll view moves with the drag.
            let translation = recognizer.translation(in: recognizer.view?.window)
            let size = CGSize(width: translation.x, height: translation.y)
            switch recognizer.state {
            case .changed:
                onDismissDragChanged(size)
            case .ended, .cancelled, .failed:
                onDismissDragEnded(size)
            default:
                break
            }
        }

        /// A zoomed image pans instead, and only a downward drag dismisses.
        func gestureRecognizerShouldBegin(_ gestureRecognizer: UIGestureRecognizer) -> Bool {
            guard let pan = gestureRecognizer as? UIPanGestureRecognizer,
                  let scrollView = pan.view as? UIScrollView else { return true }
            let velocity = pan.velocity(in: scrollView)
            return scrollView.zoomScale <= scrollView.minimumZoomScale && velocity.y > abs(velocity.x)
        }

        /// Tells VoiceOver (and UI tests) how far the image is zoomed.
        func updateAccessibilityValue(for scrollView: UIScrollView) {
            scrollView.accessibilityValue = Double(scrollView.zoomScale)
                .formatted(.percent.precision(.fractionLength(0)).locale(AppLanguage.current.locale))
        }

        func scrollViewDidZoom(_ scrollView: UIScrollView) {
            updateAccessibilityValue(for: scrollView)
            guard let view = scrollView.subviews.first else { return }
            
            let offsetX = max((scrollView.bounds.width - view.frame.width) * 0.5, 0)
            let offsetY = max((scrollView.bounds.height - view.frame.height) * 0.5, 0)
            
            view.center = CGPoint(
                x: scrollView.contentSize.width * 0.5 + offsetX,
                y: scrollView.contentSize.height * 0.5 + offsetY
            )
        }
    }
}
