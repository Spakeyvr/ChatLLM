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
                    FullscreenDoneButton(isPresented: $isPresented)
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
                FullscreenDoneButton(isPresented: $isPresented)
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

private struct FullscreenDoneButton: View {
    @Binding var isPresented: Bool

    var body: some View {
        HStack {
            Button("Done") { isPresented = false }
                .buttonStyle(.glass)
                .fontWeight(.semibold)
                .foregroundStyle(.white)

            Spacer()
        }
        .padding(.horizontal)
        .padding(.top, 8)
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
        return scrollView
    }
    
    func updateUIView(_ uiView: UIScrollView, context: Context) {
        context.coordinator.hostingController.rootView = content
        
        if uiView.subviews.first != nil {
             // Ensure the content size fits the scroll view initially
             // This is sometimes needed if the content changes dynamically
        }
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
        
        func scrollViewDidZoom(_ scrollView: UIScrollView) {
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
