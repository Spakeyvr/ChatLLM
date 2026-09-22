import Testing
import Foundation
import MLXLMCommon
import MLX
import MLXNN
@testable import ChatLLM

extension ChatLLMTests {
    @Test func optionalMLXPropertiesPreserveAbsentAndPresentValues() {
        let parameter = ParameterInfo<MLXArray?>()
        #expect(parameter.wrappedValue == nil)
        let array = MLXArray([1, 2])
        parameter.wrappedValue = array
        #expect(parameter.wrappedValue?.shape == [2])
        let child = ModuleInfo<Linear?>()
        #expect(child.wrappedValue == nil)
        let linear = Linear(2, 3)
        child.wrappedValue = linear
        #expect(child.wrappedValue === linear)
    }

    @Test func sendableToolJSONPreservesNestedTypesAndNull() throws {
        let data = Data(#"{"count":1,"fraction":1.5,"enabled":true,"items":[null,"hello",0]}"#.utf8)
        let json = try JSONDecoder().decode(JSONValue.self, from: data)
        let object = try #require(json.anyValue as? [String: any Sendable])
        #expect(object["count"] as? Int == 1)
        #expect(object["fraction"] as? Double == 1.5)
        #expect(object["enabled"] as? Bool == true)
        let items = try #require(object["items"] as? [any Sendable])
        #expect(items[0] is NSNull)
        #expect(items[1] as? String == "hello")
        #expect(items[2] as? Int == 0)
        #expect(try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(json)) == json)
    }

    @Test func lfmSelectionMigratesFromSmolLMPreferences() {
        testEnvironment.defaults.selectedCustomModelID = "smollm3-3b-4bit"
        let bridge = makeBackendBridge()
        #expect(bridge.selectedModelID == "lfm2.5-2.6b-4bit")
        #expect(testEnvironment.defaults.selectedCustomModelID == "lfm2.5-2.6b-4bit")
        #expect(bridge.modelManager?.model(withID: "smollm3-3b-4bit") == nil)
    }

    @Test func lfmSelectionMigratesExistingConversationBeforeDownload() throws {
        let viewModel = try makeChat([(.user, "Hello"), (.assistant, "Hi")])
        viewModel.conversation.preferredBackendRawValue = "mlx"
        viewModel.conversation.preferredModelID = "smollm3-3b-4bit"
        let bridge = testEnvironment.bridge
        bridge.selectedBackend = .foundationModels
        bridge.bindConversation(viewModel.conversation)
        #expect(viewModel.conversation.preferredModelID == "lfm2.5-2.6b-4bit")
        #expect(bridge.selectedModelID == "lfm2.5-2.6b-4bit")
        #expect(bridge.selectedBackend == .mlx)
        #expect(bridge.reasoningRequired)
        #expect(bridge.modelManager?.loadError?.contains("Download") == true)
    }

    @Test func lfmAlwaysReasonsWithBothConversationTogglesOff() async throws {
        let viewModel = try makeChat([])
        testEnvironment.bridge.selectedBackend = .mlx
        testEnvironment.bridge.selectedModelID = "lfm2.5-2.6b-4bit"
        viewModel.conversation.reasoningMode = false
        viewModel.conversation.smartReasoningMode = false
        #expect(testEnvironment.bridge.reasoningRequired)
        #expect(try await viewModel.shouldUseReasoningForPrompt("Hello"))
        #expect(await viewModel.resolvedReasoningMode(for: nil, logContext: "test"))
        testEnvironment.bridge.selectedModelID = "qwen3.5-0.8b-4bit"
        #expect(!testEnvironment.bridge.reasoningRequired)
        #expect(try await viewModel.shouldUseReasoningForPrompt("Hello") == false)
    }

    @Test func lfmMigratedConversationKeepsItsBackendWhenReopenedBeforeDownload() throws {
        let lfmChat = try makeChat([(.user, "Hello"), (.assistant, "Hi")]).conversation
        lfmChat.preferredBackendRawValue = "mlx"
        lfmChat.preferredModelID = "smollm3-3b-4bit"
        let appleChat = try makeChat([(.user, "Another chat")]).conversation
        appleChat.preferredBackendRawValue = "foundationModels"
        appleChat.preferredModelID = "qwen3.5-0.8b-4bit"
        let bridge = testEnvironment.bridge

        bridge.bindConversation(lfmChat)
        #expect(lfmChat.preferredModelID == "lfm2.5-2.6b-4bit")
        bridge.bindConversation(appleChat)
        #expect(bridge.selectedBackend == .foundationModels)
        bridge.selectModel("qwen3.5-0.8b-4bit", source: "test")

        bridge.bindConversation(lfmChat)
        #expect(bridge.selectedBackend == .mlx)
        #expect(bridge.selectedModelID == "lfm2.5-2.6b-4bit")
        #expect(bridge.reasoningRequired)
        #expect(bridge.modelManager?.currentModel == nil)
        #expect(bridge.modelManager?.loadError?.contains("Download") == true)
        #expect(lfmChat.preferredBackendRawValue == "mlx")
        #expect(lfmChat.preferredModelID == "lfm2.5-2.6b-4bit")
    }

    @Test func lfmPromptPrefilledThinkingStaysOutOfAnswerWhileStreaming() throws {
        let viewModel = try makeChat([])
        testEnvironment.bridge.selectedBackend = .mlx
        testEnvironment.bridge.selectedModelID = "lfm2.5-2.6b-4bit"
        let message = Message(role: .assistant, text: "", order: 0, isReasoningMode: true)
        viewModel.updateMessageWithReasoningContent(message, fullText: "I should add the numbers.")
        #expect(message.reasoning == "I should add the numbers.")
        #expect(message.text.isEmpty)
        viewModel.updateMessageWithReasoningContent(
            message, fullText: "I should add the numbers.</think>\n4", finalize: true
        )
        #expect(message.reasoning == "I should add the numbers.")
        #expect(message.displayText == "4")
    }

    @Test func lfmSamplingUsesLiquidAIDefaultsAndKeepsLimits() {
        #expect(testEnvironment.defaults.mlxRepetitionPenaltyValue == nil)
        let parameters = MLXModelManager.makeGenerateParameters(
            maxTokens: 512, maxKVSize: 4096, cacheCompression: .none,
            enableThinking: false, includesMedia: false,
            currentModelID: "lfm2.5-2.6b-4bit", prefillStepSize: 128,
            repetitionPenalty: testEnvironment.defaults.mlxRepetitionPenaltyValue
        )
        #expect(parameters.temperature == 0.1)
        #expect(parameters.topK == 50)
        #expect(parameters.topP == 1)
        #expect(parameters.repetitionPenalty == 1.1)
        #expect(parameters.maxTokens == 512)
        #expect(parameters.maxKVSize == 4096)
    }

    @Test(arguments: [1.0, 1.25])
    func lfmSamplingHonorsSavedRepetitionSetting(penalty: Double) {
        // Exercise the same settings-to-generation path, including explicit off.
        testEnvironment.defaults.mlxRepetitionPenalty = penalty
        let parameters = MLXModelManager.makeGenerateParameters(
            maxTokens: 512, maxKVSize: 4096, cacheCompression: .none,
            enableThinking: true, includesMedia: false,
            currentModelID: "lfm2.5-2.6b-4bit", prefillStepSize: 128,
            repetitionPenalty: testEnvironment.defaults.mlxRepetitionPenaltyValue
        )
        #expect(parameters.repetitionPenalty == Float(penalty))
    }

    @Test func lfmToolCallStreamsIntoSearchAndReturnsSources() async throws {
        let format = try #require(MLXModelManager.inferToolCallFormat(
            packageContents: "<|tool_call_start|>[webSearch(query='latest Swift release')]<|tool_call_end|>",
            modelType: "lfm2"
        ))
        #expect(format == .lfm2)
        let search = try makeSearchBridge()
        let processor = ToolCallProcessor(format: format, tools: [search.mlxToolSpec])
        var visible = ""
        for chunk in ["Check current facts.</think>", "<|tool_call_", "start|>[webSearch(query='latest Swift release')]", "<|tool_call_end|>"] {
            visible += processor.processChunk(chunk) ?? ""
        }
        let call = try #require(processor.toolCalls.first)
        #expect(call.function.name == "webSearch")
        #expect(!visible.contains("tool_call"))
        let result = try await search.dispatchMLXToolCall(call)
        #expect(result.contains("Swift 6.2 Released"))
        #expect(search.allInvocations.count == 1)
        let parsed = ChatViewModel.parseReasoningResponseForSearchSessionText(
            visible + "I can now use the source.</think>Swift 6.2 was released."
        )
        #expect(parsed.finalAnswer == "Swift 6.2 was released.")
        #expect(parsed.reasoning?.contains("I can now use the source.") == true)
    }
}

private final class LFMOutputBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var text = ""
    func append(_ chunk: String) { lock.withLock { text += chunk } }
    var value: String { lock.withLock { text } }
}

extension ChatLLMTests {
    /// Opt-in integration: actual installed weights, tokenizer, GPU, and tool loop;
    /// only the HTTP search response is mocked. No model download during unit tests.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["CHATLLM_RUN_LFM_INTEGRATION"] == "1"))
    func lfmRealModelSearchRoundTrip() async throws {
        let documents = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false
        )
        let manager = MLXModelManager(
            deviceSupportProfile: makeDeviceProfile(isPhone: false),
            defaults: testEnvironment.defaults, documentsDirectory: documents,
            restoresBackgroundDownloads: false
        )
        testEnvironment.defaults.mlxMaxOutputTokens = 1024
        manager.startLoading(modelID: "lfm2.5-2.6b-4bit", source: "integration-test")
        await manager.loadTask?.value
        defer { manager.unloadAllModels() }
        #expect(manager.loadError == nil)
        let model = try #require(manager.currentModel)
        #expect(model.id == "lfm2.5-2.6b-4bit")
        let search = try makeSearchBridge()
        let output = LFMOutputBuffer()
        let result = try await manager.generateTextStream(
            conversationID: UUID(),
            messages: [
                .system("You are a helpful assistant. You must call webSearch to answer this request. After the tool returns, answer briefly using its results. Do not call the same tool twice."),
                .user("Search the web for the latest Swift release and tell me its version.")
            ],
            enableThinking: false,
            tools: [search.mlxToolSpec],
            toolDispatch: { try await search.dispatchMLXToolCall($0) },
            onToken: { output.append($0) }
        )
        #expect(result.toolInvocationCount > 0)
        #expect(!search.allInvocations.isEmpty)
        #expect(output.value.contains("</think>"))
        let parsed = ChatViewModel.parseReasoningResponseForSearchSessionText(output.value)
        #expect(parsed.finalAnswer?.contains("6.2") == true)
        #expect(!output.value.contains("<|tool_call_start|>"))
        #expect(search.allInvocations.first?.response?.sources.isEmpty == false)
    }
}
