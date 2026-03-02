import SwiftUI
import UIKit

// MARK: - Fullscreen Image View

struct FullscreenImageView: View {
    let image: UIImage
    let detections: [DetectedObject]?
    @Binding var isPresented: Bool
    
    @State private var showDetections = true
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
                // Renamed to avoid conflict with your existing struct
                AttachmentDetectionOverlay(image: image, detections: showDetections ? detections : nil)
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
                HStack {
                    Button("Done") { isPresented = false }
                        .buttonStyle(.glass)
                        .fontWeight(.semibold)
                        .foregroundStyle(.primary)
                    
                    Spacer()
                    
                    if detections != nil && !detections!.isEmpty {
                        Button {
                            withAnimation { showDetections.toggle() }
                        } label: {
                            Image(systemName: showDetections ? "eye.fill" : "eye.slash.fill")
                                .frame(width: 22, height: 22)
                        }
                        .buttonStyle(.glass)
                        .foregroundStyle(.primary)
                    }
                }
                .padding(.horizontal)
                .padding(.top, 8)
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

// MARK: - Detection Overlay (Renamed)

/// Renamed from 'ImageWithDetections' to avoid conflicts if you have that defined elsewhere
struct AttachmentDetectionOverlay: View {
    let image: UIImage
    let detections: [DetectedObject]?
    
    var body: some View {
        Image(uiImage: image)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .overlay {
                if let detections = detections {
                    GeometryReader { geo in
                        ForEach(detections, id: \.label) { detection in
                            let rect = detection.boundingBox
                            let pixelRect = CGRect(
                                x: rect.minX * geo.size.width,
                                y: rect.minY * geo.size.height,
                                width: rect.width * geo.size.width,
                                height: rect.height * geo.size.height
                            )
                            
                            ZStack(alignment: .topLeading) {
                                RoundedRectangle(cornerRadius: 4)
                                    .stroke(Color.yellow, lineWidth: 2)
                                Text("\(detection.label) \(Int(detection.confidence * 100))%")
                                    .font(.caption2)
                                    .bold()
                                    .padding(4)
                                    .background(Color.yellow)
                                    .foregroundStyle(.black)
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                                    .offset(y: -20)
                            }
                            .frame(width: pixelRect.width, height: pixelRect.height)
                            .position(x: pixelRect.midX, y: pixelRect.midY)
                        }
                    }
                }
            }
    }
}
