# ChatLLM

ChatLLM is a privacy-focused, on-device AI chat application for Apple platforms. It leverages Apple's Foundation Models framework to power conversational AI directly on your device, ensuring your data stays local without cloud dependencies. The app supports persistent chat history, advanced reasoning capabilities, web search integration for real-time information, and image analysis (including OCR and object detection). Built with SwiftUI and SwiftData, it's designed for seamless performance on iOS, iPadOS, and macOS.

This app is ideal for developers, students, and users who want a fast, offline-capable chatbot for tasks like code assistance, explanations, debugging, and casual queries—all while maintaining full control over your data.

## Key Features

- **ChatLLM**: Uses Apple's SystemLanguageModel for instant, private responses. Supports streaming token-by-token generation for a natural chat experience.
  
- **Reasoning Modes**:
  - **Manual Reasoning**: Always shows the AI's step-by-step thinking process before the final answer—great for complex problem-solving, code debugging, or logical analysis.
  - **Smart Reasoning**: Automatically detects when reasoning is needed (e.g., for multi-step tasks like "Optimize this Swift code" or "Debug my algorithm") using a lightweight on-device evaluation. Falls back to direct answers for simple queries.

- **Web Search Integration**: When the AI identifies a need for up-to-date info (e.g., "What's the latest iOS version?" or "Recent news on WWDC"), it triggers a search via the Tavily API. Results are injected transparently as hidden sources, with a UI button to view citations. No API key is required for core chat, but it enhances accuracy for factual queries.

- **Image Analysis**:
  - **OCR (Optical Character Recognition)**: Extract text from images using Apple's Vision framework and include it in your prompts.
  - **Object Detection**: Powered by YOLOv3 integration, detects and describes objects in images (e.g., "What do you see in this photo?"). Supports attaching images directly to messages.

- **Persistent Chat History**: Conversations are saved locally with SwiftData. Features include:
  - Auto-generated titles based on content (using on-device AI).
  - Edit user messages and regenerate AI responses.
  - Delete or trim messages.
  - Share individual messages or entire chats.

- **Customization and Controls**:
  - Custom system prompts per conversation to guide AI behavior.
  - Text-to-speech for reading responses aloud.
  - Regenerate options: "Try again," "More concise," or "More formal."
  - Secure storage of Tavily API key in Keychain, synced with app settings.

- **Performance and Reliability**:
  - Handles generation timeouts, cancellations, and errors gracefully.
  - Cleans up glitched or repetitive AI outputs (common in streaming).
  - Optimized for concurrency with async/await; prevents issues like concurrent model sessions.

- **UI/UX Highlights**:
  - Modern SwiftUI interface with message bubbles, swipe actions, and context menus.
  - Visual indicators for reasoning (e.g., "Thinking..." animations) and sources.
  - Accessibility support, including dynamic type and VoiceOver.

## Requirements

- **Platforms**: iOS 26.0+ (requires Apple Intelligence-compatible devices like iPhone 15 Pro+).
- **Optional**: Tavily API key (free tier at [tavily.com](https://tavily.com)) for web search. Without it, the app works fully on-device.

No third-party dependencies beyond Apple's frameworks—keeps it lightweight and secure.

## Usage

1. **Start Chatting**:
   - Type a message in the composer at the bottom and send.
   - The AI responds in real-time. For images, tap the camera icon to attach and detect objects/OCR.

2. **Enable Reasoning**:
   - Tap the brain icon in the toolbar to open settings.
   - Toggle "Always Use Reasoning" for step-by-step breakdowns or "Smart Reasoning Mode" for automatic detection.

3. **Web Search**:
   - Ask about current events (e.g., "Who won the latest election?").
   - If configured with a Tavily key (via Settings > API Keys), the AI searches and cites sources (tap "Sources" button to view).

4. **Image Prompts**:
   - Attach an image via Photos or Files.
   - The app detects objects (e.g., "person, car") and extracts text, then prompts the AI (e.g., "Describe this image").

5. **Manage Conversations**:
   - Edit: Swipe left on user messages > "Edit" > Modify and regenerate.
   - Regenerate: For AI responses, tap the refresh icon > Choose style.
   - Delete: Swipe > "Delete" or use context menu.
   - Customize: Tap gear icon for system prompt.

6. **Settings**:
   - Access via app menu or conversation toolbar.
   - Enter Tavily API key for search (stored securely).

Example Queries:
- Simple: "What is SwiftUI?"
- Reasoning: "How do I fix this layout bug in my app?"
- Search: "Latest Apple Intelligence features."
- Image: Attach photo > "What's in this picture?"

## Architecture Overview

- **Data Layer**: SwiftData models (`Conversation`, `Message`) for local persistence. Messages track role (user/assistant/system), order, reasoning flags, and attachments.
- **Business Logic**: `ChatViewModel` (ObservableObject) handles streaming, prompt building, reasoning evaluation, search injection, and saving. Uses async/await for concurrency.
- **AI Generation**: `OnDeviceLLMGenerator` wraps Foundation Models for responses and streaming. Includes glitch detection and cumulative stream handling.
- **Services**: `TavilySearchService` for API calls; Vision for OCR; YOLOv3 for object detection (integrated in image picker).
- **UI**: `ChatView` with optimized SwiftUI components (e.g., cached markdown rendering, animations for reasoning bubbles).

The app emphasizes on-device processing for speed and privacy, with fallbacks (e.g., heuristics for unavailable LLM).

## Contributing

Contributions are welcome! Focus on:
- Enhancing image detection (e.g., more models).
- Additional tools (e.g., math solver integration).
- macOS/iPadOS optimizations.
- Bug fixes for edge cases like long streams.

Fork the repo, create a branch, and submit a pull request. Follow Swift best practices and test on multiple devices.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Apple's SwiftUI, SwiftData, Foundation Models, and Vision frameworks.
- Tavily for AI-optimized search API.
- YOLOv3 for object detection (open-source model).

For questions or issues, open a GitHub discussion or issue. Enjoy chatting on-device!
