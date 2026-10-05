# ChatLLM

ChatLLM is a SwiftUI iPhone and iPad app for fully local chat with on-device language models.

It supports two backends:

- `Apple Intelligence` via `FoundationModels`
- `MLX` via locally downloaded Qwen 3.5 multimodal models and LFM2.5 text models

The app also includes optional web search with Tavily, image attachments, Vision-based image analysis, SwiftData-backed conversation history, onboarding, model management, and UI/UI automation tests.

## Features

- On-device chat with Apple Foundation Models when supported by the device
- Local MLX model execution with downloadable Qwen 3.5 and LFM2.5 models
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
- `LFM2.5 2.6B (4-bit)` about `1.60 GB`

The Qwen models are configured as multimodal MLX models with reasoning support and native image support. LFM2.5 is text-only with always-on reasoning and native tool calling.

LFM2.5 uses [LiquidAI/LFM2.5-2.6B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-2.6B-MLX-4bit).
This official 4-bit package retains LiquidAI's 6-bit embedding exception. Its
131,072-token architecture limit remains subject to the app's device memory and
context limits. Existing SmolLM selections migrate to LFM; users download the new
weights through Manage Models. Conversation history is preserved.

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

1. Open [ChatLLM.xcodeproj](/Users/nevio/Developer/ChatLLM/ChatLLM.xcodeproj)
2. Select the `ChatLLM` scheme
3. Choose an iPhone or iPad simulator/device
4. Build and run

Swift Package dependencies are resolved through Xcode. The project includes MLX-related packages plus supporting Apple and Hugging Face packages through SwiftPM.

## Project Structure

- [ChatLLM](/Users/nevio/Developer/ChatLLM/ChatLLM): main app target
- [ChatLLMTests](/Users/nevio/Developer/ChatLLM/ChatLLMTests): unit tests
- [ChatLLMUITests](/Users/nevio/Developer/ChatLLM/ChatLLMUITests): UI tests
- [Vendor/mlx-swift-lm](/Users/nevio/Developer/ChatLLM/Vendor/mlx-swift-lm): local MLX package source

Key app files:

- [ContentView.swift](/Users/nevio/Developer/ChatLLM/ChatLLM/ContentView.swift): app shell, sidebar, chat selection, settings/export flow
- [ChatView.swift](/Users/nevio/Developer/ChatLLM/ChatLLM/ChatView.swift): conversation screen and composer integration
- [ChatViewModel.swift](/Users/nevio/Developer/ChatLLM/ChatLLM/ChatViewModel.swift): message flow, streaming, persistence, OCR helpers
- [ModelBackendBridge.swift](/Users/nevio/Developer/ChatLLM/ChatLLM/ModelBackendBridge.swift): backend selection and capability gating
- [MLXModelManager.swift](/Users/nevio/Developer/ChatLLM/ChatLLM/MLXModelManager.swift): MLX model download, loading, memory handling, and inference sessions
- [VisionAnalyzer.swift](/Users/nevio/Developer/ChatLLM/ChatLLM/VisionAnalyzer.swift): Vision-based image analysis pipeline
- [TavilySearchService.swift](/Users/nevio/Developer/ChatLLM/ChatLLM/TavilySearchService.swift): optional web search integration

## Architecture Overview

- `SwiftUI` drives the entire interface
- `SwiftData` stores conversations, messages, and attachments
- `FoundationModels` powers Apple Intelligence chats when available
- `MLX` powers downloadable local Qwen and LFM models
- `Vision` handles OCR and image analysis fallback
- `UserDefaults` and Keychain store user settings and the Tavily API key

## Testing

The repository includes:

- unit tests in `ChatLLMTests`
- UI tests in `ChatLLMUITests`

Run them from Xcode with `Product > Test`. For a faster unit-only check, add
`-only-testing:ChatLLMTests` to the simulator `xcodebuild test` command.

Unit tests construct chat view models through the fixture factories in
`ChatLLMTests+Factories.swift`. Each test owns a backend bridge, a disposable
preferences domain, and temporary model storage; fixture model managers do not
restore background downloads, and fixture view models do not read the Keychain.
Use these factories instead of production singletons when adding tests.

Cancellation tests cover preparation, streaming, reasoning, pending web searches,
regeneration, and switching chats. Math tests load the actual bundled WebView and
verify environment rendering and local font loading.

### Real LFM simulator checks

Download LFM2.5 through Manage Models in the chosen simulator first. These opt-in
tests use actual model weights and GPU generation; the search integration mocks
only Tavily's HTTP response so no API key is needed. Run with parallel testing
disabled to use that simulator's installed model directory:

```sh
TEST_RUNNER_CHATLLM_RUN_LFM_INTEGRATION=1 xcodebuild \
  -project ChatLLM.xcodeproj -scheme ChatLLM \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -parallel-testing-enabled NO \
  '-only-testing:ChatLLMTests/ChatLLMTests/lfmRealModelSearchRoundTrip()' \
  -only-testing:ChatLLMUITests/ChatLLMLFMIntegrationTests test
```

The UI check verifies a real answer, the reasoning sheet, and a contextual
follow-up. Physical-device memory/performance and authenticated Tavily access
require separate device/network checks.

App and test targets treat compiler warnings as errors. The vendored Cmlx target
has one explicit exception for upstream 64-to-32-bit conversion diagnostics;
this is a scoped diagnostic policy, not a claim that upstream narrowing was
eliminated. Deprecation and Swift concurrency diagnostics remain enabled.
See `Vendor/mlx-swift/README-ChatLLM.md` for the local compatibility patches.

App and test targets skip App Intents metadata extraction because they declare
no App Intents. Remove `LM_SKIP_METADATA_EXTRACTION` from a target if adding
App Intents or App Shortcuts to it.

## Notes for Contributors

- Backend selection is effectively locked once a conversation has messages
- Web search depends on both network connectivity and a configured Tavily key
- MLX behavior is memory-sensitive and includes device-specific tuning
- Image analysis may be precomputed when the selected model does not support native images

## License

See [LICENSE](LICENSE).
