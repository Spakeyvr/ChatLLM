# ChatLLM

ChatLLM is a SwiftUI iPhone and iPad app for fully local chat with on-device language models.

It supports two backends:

- `Apple Intelligence` via `FoundationModels`
- `MLX` via locally downloaded Qwen 3.5 multimodal models and SmolLM3 text models

The app also includes optional web search with Tavily, image attachments, Vision-based image analysis, SwiftData-backed conversation history, onboarding, model management, and UI/UI automation tests.

## Features

- On-device chat with Apple Foundation Models when supported by the device
- Local MLX model execution with downloadable Qwen 3.5 and SmolLM3 models
- Per-chat backend/model selection before a conversation starts
- Reasoning mode and smart reasoning support
- Optional Tavily-powered web search for time-sensitive answers
- Image attachments with Vision analysis fallback for OCR, objects, faces, barcodes, scene labels, and saliency
- Markdown and LaTeX rendering in assistant responses
- SwiftData conversation persistence
- Prompt presets, appearance settings, MLX tuning controls, and privacy controls
- Chat export and bulk deletion tools

## Requirements

- Xcode 26 or newer
- iOS 26.0 SDK
- iPhone or iPad target
- A physical device is recommended for real model testing

Notes:

- The project deployment target is `iOS 26.0`.
- `FoundationModels` requires device support for Apple Intelligence.
- MLX model downloads and execution are supported on Apple-silicon iPhone and iPad simulators as well as physical devices. A physical device is still recommended for representative memory and performance testing.

## Included MLX Models

The app currently exposes these downloadable local models:

- `Qwen 3.5 4B (4-bit hybrid)` about `2.66 GB`
- `Qwen 3.5 2B (4-bit)` about `1.75 GB`
- `Qwen 3.5 0.8B (4-bit)` about `625 MB`
- `SmolLM3 3B (4-bit)` about `1.75 GB`

The Qwen models are configured as multimodal MLX models with reasoning support and native image support. SmolLM3 is configured as a text-only MLX model with reasoning support.

## Optional Setup

### Tavily Web Search

Web search is optional. To enable it:

1. Get an API key from `https://tavily.com`
2. Open the app
3. Enter the key during onboarding or later in `Settings > Tavily Search`

The key is stored locally in Keychain.

### MLX Models

MLX is optional. You can use the app immediately with Apple Foundation Models on supported devices.

To use MLX:

1. Launch the app
2. Open onboarding or `Manage Models`
3. Download one of the available MLX models
4. Start a new chat and select the MLX backend/model before sending the first message

## Running the Project

1. Open [ChatLLM.xcodeproj](/Users/nevio/Desktop/Projects/ChatLLM/ChatLLM.xcodeproj)
2. Select the `On-Device_LLM_Chat` scheme
3. Choose an iPhone or iPad simulator/device
4. Build and run

Swift Package dependencies are resolved through Xcode. The project includes MLX-related packages plus supporting Apple and Hugging Face packages through SwiftPM.

## Project Structure

- [On-Device_LLM_Chat](/Users/nevio/Desktop/Projects/ChatLLM/On-Device_LLM_Chat): main app target
- [On-Device_LLM_ChatTests](/Users/nevio/Desktop/Projects/ChatLLM/On-Device_LLM_ChatTests): unit tests
- [On-Device_LLM_ChatUITests](/Users/nevio/Desktop/Projects/ChatLLM/On-Device_LLM_ChatUITests): UI tests
- [Vendor/mlx-swift-lm](/Users/nevio/Desktop/Projects/ChatLLM/Vendor/mlx-swift-lm): local MLX package source

Key app files:

- [ContentView.swift](/Users/nevio/Desktop/Projects/ChatLLM/On-Device_LLM_Chat/ContentView.swift): app shell, sidebar, chat selection, settings/export flow
- [ChatView.swift](/Users/nevio/Desktop/Projects/ChatLLM/On-Device_LLM_Chat/ChatView.swift): conversation screen and composer integration
- [ChatViewModel.swift](/Users/nevio/Desktop/Projects/ChatLLM/On-Device_LLM_Chat/ChatViewModel.swift): message flow, streaming, persistence, OCR helpers
- [ModelBackendBridge.swift](/Users/nevio/Desktop/Projects/ChatLLM/On-Device_LLM_Chat/ModelBackendBridge.swift): backend selection and capability gating
- [MLXModelManager.swift](/Users/nevio/Desktop/Projects/ChatLLM/On-Device_LLM_Chat/MLXModelManager.swift): MLX model download, loading, memory handling, and inference sessions
- [VisionAnalyzer.swift](/Users/nevio/Desktop/Projects/ChatLLM/On-Device_LLM_Chat/VisionAnalyzer.swift): Vision-based image analysis pipeline
- [TavilySearchService.swift](/Users/nevio/Desktop/Projects/ChatLLM/On-Device_LLM_Chat/TavilySearchService.swift): optional web search integration

## Architecture Overview

- `SwiftUI` drives the entire interface
- `SwiftData` stores conversations, messages, and attachments
- `FoundationModels` powers Apple Intelligence chats when available
- `MLX` powers downloadable local Qwen models
- `Vision` handles OCR and image analysis fallback
- `UserDefaults` and Keychain store user settings and the Tavily API key

## Testing

The repository includes:

- unit tests in `On-Device_LLM_ChatTests`
- UI tests in `On-Device_LLM_ChatUITests`

Run them from Xcode with `Product > Test`.

## Notes for Contributors

- Backend selection is effectively locked once a conversation has messages
- Web search depends on both network connectivity and a configured Tavily key
- MLX behavior is memory-sensitive and includes device-specific tuning
- Image analysis may be precomputed when the selected model does not support native images

## License

See [LICENSE](LICENSE).
