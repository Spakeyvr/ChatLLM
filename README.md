# ChatLLM

ChatLLM is an iPhone and iPad app for chatting with language models that run entirely on your device. It works with Apple Intelligence on supported hardware, and it can also download open models and run them locally with MLX. Your conversations stay on the device.

The app is in beta, so expect some rough edges.

## Features

- **Two backends.** Use Apple Intelligence through Apple's Foundation Models framework, or download an open model and run it with MLX. You choose the backend and model for each chat.
- **Image input.** Attach photos from the camera, your library, or Files when the selected model supports images.
- **Reasoning.** Models that can reason show their thinking separately from the answer. You can turn reasoning on for a chat, or let the app decide when it is useful.
- **Optional web search.** Add your own Tavily API key and the model can search the web when a question needs current information. Answers that used search list their sources.
- **Rich formatting.** Replies support Markdown and LaTeX math. Everything needed to render them ships with the app, so it works offline.
- **Chat history.** Chats are saved on the device and titled automatically. You can export them, delete them in bulk, or have old chats removed after a set time.
- **Customization.** Settings cover response style, appearance, text size, haptics, interface language (English, German, or Spanish), and generation options for MLX models.

## Models

Apple Intelligence needs no download. It is available when the device supports it and Apple Intelligence is turned on in system settings.

MLX models are downloaded from Hugging Face inside the app, under Manage Models. The catalog includes Qwen 3.5 models in several sizes, which can read images, and LFM2.5, which is text only. Downloads continue in the background if you leave the app.

Larger models need more memory. On iPhone, the app checks the device's RAM and only offers models and features (such as web search) that the device can handle. You can turn these checks off under Settings > Advanced, but the app may become slow or crash.

## Requirements

- iOS or iPadOS 27 or later
- Xcode 27 or newer to build the app
- A device with Apple Intelligence support, if you want to use that backend

MLX models also run in the iOS Simulator on Apple silicon Macs. Memory use and speed in the simulator are not representative of a real device, so test those on hardware.

## Building

Open the project in Xcode:

```bash
open ChatLLM.xcodeproj
```

Select the `ChatLLM` scheme, pick a simulator or connected device, and run. Xcode resolves the Swift packages automatically.

To check that the app compiles from the command line:

```bash
xcodebuild -project ChatLLM.xcodeproj -scheme ChatLLM -destination 'generic/platform=iOS Simulator' build
```

Warnings are treated as errors, so any new warning will fail the build.

## Testing

Run the tests with Product > Test in Xcode, or from the terminal:

```bash
xcodebuild -project ChatLLM.xcodeproj -scheme ChatLLM -destination 'platform=iOS Simulator,name=iPhone 17 Pro' test
```

Replace the simulator name with one installed on your Mac. Unit tests use Swift Testing and live in `ChatLLMTests/`. UI tests use XCTest and live in `ChatLLMUITests/`. The tests use mocks and launch flags, so they do not need downloaded models, network access, or an API key.

## Web search setup

1. Create an API key at [tavily.com](https://tavily.com).
2. Enter it during onboarding or later under Settings > Web Search.

The key is stored in the Keychain. It is only sent to Tavily, together with your search queries.

## Project structure

```
ChatLLM/            App source code
  RenderingAssets/  Bundled Markdown and math rendering files
ChatLLMTests/       Unit tests
ChatLLMUITests/     UI tests
Vendor/             Local copies of the MLX Swift packages
Scripts/            Helper scripts
```

The packages in `Vendor/` include a few local patches. `Vendor/mlx-swift/README-ChatLLM.md` describes each change.

## Translations

The interface is available in English, German, and Spanish. See [LOCALIZATION.md](LOCALIZATION.md) for how to add or update strings.

## Support

Email [chatllm@icloud.com](mailto:chatllm@icloud.com), or join the Discord server linked under Settings > About & Support in the app.

## License

ChatLLM is released under the Apache 2.0 license. See [LICENSE](LICENSE).
