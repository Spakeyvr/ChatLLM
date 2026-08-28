// Deterministic UI fixture that exercises the actual message bubbles without a model download.
#if DEBUG
import SwiftUI
import SwiftData

struct MarkdownRenderingPreview: View {
    @Environment(\.modelContext) private var modelContext
    @State private var viewModel: ChatViewModel?
    @State private var stage = 0

    private static let paragraphs = "First **bold** word.\nNext line.\n\nSecond paragraph with _italic_ text and `first_name`."
    private static let blocks = "## Live heading\n\nFirst **bold** word.\n\n- First item\n- Second item\n\n```swift\nlet first_name = 1\n```"

    var body: some View {
        VStack {
            HStack {
                Text(stage < 2 ? "Streaming" : "Finished")
                    .accessibilityIdentifier("markdown.status")
                Spacer()
                Button(stage == 0 ? "Add blocks" : "Finish") {
                    guard let viewModel, let message = viewModel.conversation.messages.first else { return }
                    stage += 1
                    message.text = Self.blocks + (stage >= 2 ? "\n\nLast **paragraph**." : "")
                    if message.isReasoningMode { message.finalAnswer = message.text }
                    message.isFinal = stage >= 2
                    viewModel.isGenerating = stage < 2
                    if stage >= 2 { viewModel.streamingMessageID = nil }
                }
                .accessibilityIdentifier("markdown.advance")
                .disabled(stage >= 2)
            }
            .padding()
            ScrollView {
                if let viewModel, let message = viewModel.conversation.messages.first {
                    MessageBubble(message: message, viewModel: viewModel)
                        .padding(.horizontal, 12)
                }
            }
        }
        .task {
            guard viewModel == nil else { return }
            let reasoning = ProcessInfo.processInfo.arguments.contains("-ui-test-markdown-reasoning")
            let conversation = Conversation(title: "Markdown Preview", reasoningMode: reasoning)
            let message = Message(role: .assistant, text: Self.paragraphs, order: 0,
                                  conversation: conversation, isReasoningMode: reasoning)
            if reasoning {
                message.reasoning = "Checking the formatting."
                message.finalAnswer = message.text
            }
            conversation.messages = [message]
            let model = ChatViewModel(generator: OnDeviceLLMGenerator(), context: modelContext, conversation: conversation)
            model.isGenerating = true
            model.streamingMessageID = message.id
            viewModel = model
        }
    }
}
#endif
