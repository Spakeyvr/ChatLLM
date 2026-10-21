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
    
    @State private var isDraggingToDismiss = false
    @State private var dragOffset: CGSize = .zero
    
    var body: some View {
        ZStack {
            // Fixed: Explicit type casting for the opacity calculation
            Color.black
                .opacity(calculateBackdropOpacity())
                .ignoresSafeArea()
            
            // The Zoomable View
            ZoomableScrollView {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }
            .offset(y: dragOffset.height)
            .gesture(
                DragGesture()
                    .onChanged { value in
                        // Only allow dragging down
                        if value.translation.height > 0 {
                            isDraggingToDismiss = true
                            dragOffset = value.translation
                        }
                    }
                    .onEnded { value in
                        if value.translation.height > 150 {
                            isPresented = false
                        } else {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                                isDraggingToDismiss = false
                                dragOffset = .zero
                            }
                        }
                    }
            )
            
            // Custom toolbar overlay
            VStack {
                FullscreenToolbar(image: image, isPresented: $isPresented)
                    .opacity(isDraggingToDismiss ? 0 : 1)
                
                Spacer()
            }
        }
    }
    
    // Helper to keep the view body clean and typesafe
    private func calculateBackdropOpacity() -> Double {
        guard isDraggingToDismiss else { return 1.0 }
        // Cast to Double to avoid "Ambiguous use of operator '/'"
        let dragHeight = Double(abs(dragOffset.height))
        let threshold = 300.0
        return max(0.4, 1.0 - (dragHeight / threshold))
    }
}

private struct FullscreenToolbar: View {
    let image: UIImage?
    @Binding var isPresented: Bool

    var body: some View {
        HStack {
            Button("Done") { isPresented = false }
                .buttonStyle(.glass)
                .fontWeight(.semibold)
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
        }
        .buttonStyle(.glass)
        .foregroundStyle(.white)
        .accessibilityLabel("Share")
        .accessibilityIdentifier("imageViewer.share")
    }
}

// MARK: - Zoomable ScrollView (UIKit Wrapper)

struct ZoomableScrollView<Content: View>: UIViewRepresentable {
    let content: Content
    
    init(@ViewBuilder content: () -> Content) {
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
        context.coordinator.updateAccessibilityValue(for: scrollView)
        return scrollView
    }
    
    func updateUIView(_ uiView: UIScrollView, context: Context) {
        context.coordinator.hostingController.rootView = content
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(hostingController: UIHostingController(rootView: content))
    }
    
    class Coordinator: NSObject, UIScrollViewDelegate {
        var hostingController: UIHostingController<Content>
        
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
