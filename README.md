# ChatLLM

ChatLLM is a SwiftUI iOS app for chatting with on-device language models. It combines Apple's Foundation Models APIs with optional downloadable MLX models, image attachments, reasoning controls, and Tavily-powered web search.

## Highlights

- On-device chat UI built with SwiftUI and SwiftData
- Two model backends:
  - Apple Foundation Models / Apple Intelligence
  - Local MLX models managed inside the app
- Downloadable multimodal Qwen 3.5 MLX models
- Image attachments with native multimodal support when available
- Vision-based image analysis fallback when native image support is unavailable
- Manual reasoning mode and smart reasoning mode
- Tavily web search integration for time-sensitive questions
- Markdown and LaTeX rendering in chat responses
- Conversation history, export, and basic data-management settings

## Current model support

The repository currently exposes these downloadable MLX models:

- `Qwen 3.5 2B (4-bit)`
- `Qwen 3.5 4B (4-bit)`

Both are configured as multimodal models with reasoning support.

## Requirements

- macOS with a recent Xcode version that supports:
  - Swift 6
  - iOS 26.0 SDK/runtime
  - Apple's Foundation Models APIs
- An iPhone or iPad for realistic testing

Notes:

- The project target is currently set to `iOS 26.0`.
- Some MLX multimodal paths are explicitly marked as unsupported on the iOS Simulator.
- If the Foundation Models backend is unavailable on the current device/configuration, the UI still runs and the app shows mocked responses instead of real model output.

## Dependencies

Swift Package Manager dependencies are resolved through the Xcode workspace. The project currently pulls in:

- `mlx-swift`
- `mlx-swift-lm`
- `swift-transformers`
- `swift-jinja`
- Apple async/collections/crypto support packages

## Getting started

1. Open [ChatLLM.xcodeproj](/Users/nevioknogler/Desktop/ChatLLM/ChatLLM.xcodeproj) in Xcode.
2. Let Xcode resolve Swift Package Manager dependencies.
3. Select an iPhone or iPad target. Prefer a physical device if you want to test MLX multimodal behavior.
4. Build and run the `On-Device_LLM_Chat` app target.

## Using the app

### Apple backend

The default backend uses Apple's on-device language model when the platform makes it available.

### MLX backend

The app can also run local MLX models. From the app UI, switch to the MLX backend and download one of the supported Qwen packages. Model files are stored under the app's Documents directory.

### Web search

Web search is optional. To enable it:

1. Open Settings.
2. Add a Tavily API key.
3. Ask a time-sensitive question or explicitly ask the assistant to search.

### Images

You can attach images to prompts. The app will either:

- pass the image directly to a multimodal MLX model, or
- fall back to Apple Vision analysis and include the extracted context in the prompt

## Data and storage

- Conversations are stored locally with SwiftData.
- Image attachments are stored on-device.
- Tavily API keys are handled in app settings and keychain-related code exists in the chat view model.
- Chats can be exported from Settings as a plain-text conversation dump.

## GitHub commit relay

If you want Discord notifications whenever a GitHub push happens, there is a small webhook relay script in [Scripts/github_discord_relay.py](/Users/nevioknogler/Desktop/ChatLLM/Scripts/github_discord_relay.py).

Setup notes:

1. Start the relay with `DISCORD_WEBHOOK_URL` and `GITHUB_WEBHOOK_SECRET` set.
2. Expose the relay over HTTPS.
3. In GitHub, add a webhook that points to `/github`, uses `application/json`, and subscribes to the `Pushes` event.

More detailed instructions are in [Scripts/README.md](/Users/nevioknogler/Desktop/ChatLLM/Scripts/README.md).

## Project structure

- [On-Device_LLM_Chat](/Users/nevioknogler/Desktop/ChatLLM/On-Device_LLM_Chat): main app source
- [On-Device_LLM_ChatTests](/Users/nevioknogler/Desktop/ChatLLM/On-Device_LLM_ChatTests): unit tests
- [On-Device_LLM_ChatUITests](/Users/nevioknogler/Desktop/ChatLLM/On-Device_LLM_ChatUITests): UI tests
- [ChatLLM.xcodeproj](/Users/nevioknogler/Desktop/ChatLLM/ChatLLM.xcodeproj): Xcode project

## Known limitations

- The app currently exposes only English in the language picker, even though localized string files for other languages exist in the repository.
- MLX multimodal loading has simulator limitations.
- MLX model downloads are large, so first-run setup for local models is bandwidth- and storage-heavy.
- Web search depends on a valid Tavily API key and network access.

## License

This project includes a license file at [LICENSE](/Users/nevioknogler/Desktop/ChatLLM/On-Device_LLM_Chat/LICENSE).
