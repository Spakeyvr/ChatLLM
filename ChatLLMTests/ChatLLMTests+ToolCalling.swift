//
//  ChatLLMTests+ToolCalling
//  ChatLLMTests
//
//  Split out of ChatLLMTests.swift; part of the single @Suite(.serialized) ChatLLMTests suite.
//

import Testing
import Foundation
import MLXLMCommon
import SwiftUI
import SwiftData
import WebKit
import UIKit
import FoundationModels
@testable import ChatLLM

extension ChatLLMTests {
    @Test func qwenToolTemplateInspectionPrefersXMLFunctionFormat() {
        let packageContents = """
        <tool_call>
        <function=webSearch>
        <parameter=query>
        latest OpenAI model
        </parameter>
        </function>
        </tool_call>
        """

        let inferredFormat = MLXModelManager.inferToolCallFormat(
            packageContents: packageContents,
            modelType: "qwen3_5"
        )

        #expect(inferredFormat == .xmlFunction)
        #expect(MLXModelManager.usesWrappedXMLToolCallTemplate(packageContents: packageContents))
    }

    @Test func smolLM3ToolTemplateInspectionPrefersJSONToolCalls() {
        let packageContents = """
        <tool_call>
        {"name": "webSearch", "arguments": {"query": "latest Swift release"}}
        </tool_call>
        """

        let inferredFormat = MLXModelManager.inferToolCallFormat(
            packageContents: packageContents,
            modelType: "smollm3"
        )

        #expect(inferredFormat == .json)
        #expect(!MLXModelManager.usesWrappedXMLToolCallTemplate(packageContents: packageContents))
    }

    @Test func wrappedXMLToolCallStreamFilterSuppressesToolMarkup() async {
        let filter = WrappedXMLToolCallStreamFilter()

        let firstChunk = await filter.consume("I will check that.\n<tool")
        let secondChunk = await filter.consume("_call>\n<function=webSearch>")
        let thirdChunk = await filter.consume("\n<parameter=query>\nlatest OpenAI model\n</parameter>")
        await filter.didDispatchToolCall()
        let trailingChunk = await filter.finish()

        #expect(firstChunk == "I will check that.\n")
        #expect(secondChunk == nil)
        #expect(thirdChunk == nil)
        #expect(trailingChunk == nil)
    }

    @Test func wrappedXMLToolCallStreamFilterResumesAfterClosedWrapperWithoutToolEvent() async {
        let filter = WrappedXMLToolCallStreamFilter()

        let visible = await filter.consume(
            "Before <tool_call><function=webSearch></function></tool_call>After"
        )
        let trailing = await filter.finish()

        #expect(visible == "Before After")
        #expect(trailing == nil)
    }

    @Test func wrappedXMLToolCallStreamFilterRecoversTrailingTextAfterMalformedOuterWrapper() async {
        let filter = WrappedXMLToolCallStreamFilter()

        let visible = await filter.consume(
            "Before <tool_call><function=webSearch></function>After"
        )
        let trailing = await filter.finish()

        #expect(visible == "Before ")
        #expect(trailing == "After")
    }

    @Test func qwenToolArgumentNormalizationUnwrapsJSONValueStrings() {
        let normalized = MLXModelManager.normalizedToolArguments([
            "query": JSONValue.string("OpenAI latest model")
        ])

        #expect(normalized["query"] as? String == "OpenAI latest model")
    }

    @Test func mlxToolLoopDisablesToolsOnlyAfterSearchLimitResponse() {
        #expect(
            MLXModelManager.shouldDisableTools(
                after: AppWebSearchToolBridge.searchLimitToolResponse(limit: AppWebSearchToolBridge.maxInvocations)
            )
        )
        #expect(
            !MLXModelManager.shouldDisableTools(
                after: AppWebSearchToolBridge.invalidArgumentsToolResponse()
            )
        )
    }

    @Test func qwenWrappedToolResponseContentUsesExpectedEnvelope() {
        let content = MLXModelManager.wrappedToolResponseContent("Search result body")

        #expect(content == "<tool_response>\nSearch result body\n</tool_response>")
        #expect(content.starts(with: "<tool_response>"))
        #expect(content.hasSuffix("</tool_response>"))
    }

    @Test func qwenToolResponsesReuseWrappedUserPromptCompatibilityPath() {
        let manager = MLXModelManager()
        let model = try! #require(manager.model(withID: "qwen3.5-2b-4bit"))

        let content = MLXModelManager.toolResponsePromptContent(
            for: model,
            toolResult: "Search result body"
        )
        let role = MLXModelManager.toolResponsePromptRole(for: model)

        #expect(content == "<tool_response>\nSearch result body\n</tool_response>")
        #expect(role == .user)
    }

    @Test func qwenToolResponsesRequireExplicitMessageHistoryAcrossToolLoop() {
        let manager = MLXModelManager()
        let model = try! #require(manager.model(withID: "qwen3.5-2b-4bit"))

        #expect(MLXModelManager.requiresExplicitMessageHistoryForToolLoop(for: model))
    }

    @Test func boundedExplicitToolLoopReleasesAllPersistentSessionsBeforePrefill() {
        let scope = MLXInferenceWorker.persistentSessionReleaseScope(
            usesExplicitMessageHistory: true,
            maxKVSize: 4096
        )

        #expect(scope == .all)
    }

    @Test func unboundedExplicitToolLoopOnlyReleasesCurrentConversationSession() {
        let scope = MLXInferenceWorker.persistentSessionReleaseScope(
            usesExplicitMessageHistory: true,
            maxKVSize: nil
        )

        #expect(scope == .conversation)
    }

    @Test func nativeToolLoopPreservesPersistentSessions() {
        let scope = MLXInferenceWorker.persistentSessionReleaseScope(
            usesExplicitMessageHistory: false,
            maxKVSize: 4096
        )

        #expect(scope == .none)
    }

    @Test func mlxGenerationCommitRequiresMatchingLoadAndRevision() {
        let loadID = UUID()

        #expect(MLXInferenceWorker.generationStateIsCurrent(
            expectedLoadID: loadID,
            currentLoadID: loadID,
            expectedStateRevision: 4,
            currentStateRevision: 4
        ))
        #expect(!MLXInferenceWorker.generationStateIsCurrent(
            expectedLoadID: loadID,
            currentLoadID: UUID(),
            expectedStateRevision: 4,
            currentStateRevision: 4
        ))
        #expect(!MLXInferenceWorker.generationStateIsCurrent(
            expectedLoadID: loadID,
            currentLoadID: loadID,
            expectedStateRevision: 4,
            currentStateRevision: 5
        ))
    }

    @Test func mlxVisibleMessageSignatureStoresOnlyFixedSizeContentIdentity() {
        let first = MLXVisibleMessageSignature(
            message: Chat.Message(role: .user, content: "A long conversation message")
        )
        let same = MLXVisibleMessageSignature(
            message: Chat.Message(role: .user, content: "A long conversation message")
        )
        let changed = MLXVisibleMessageSignature(
            message: Chat.Message(role: .user, content: "A different conversation message")
        )

        #expect(first == same)
        #expect(first != changed)
        #expect(first.contentFingerprint.count == 32)
    }

    @Test func smolLM3ToolResponsesUseNativeToolRole() {
        let manager = MLXModelManager()
        let model = try! #require(manager.model(withID: "smollm3-3b-4bit"))

        let content = MLXModelManager.toolResponsePromptContent(
            for: model,
            toolResult: "Search result body"
        )
        let role = MLXModelManager.toolResponsePromptRole(for: model)

        #expect(content == "Search result body")
        #expect(role == .tool)
        #expect(!MLXModelManager.requiresExplicitMessageHistoryForToolLoop(for: model))
    }

    @Test func nonQwenModelsKeepNativeToolRoleForToolResponses() {
        let nonQwenModel = MLXModelManager.MLXModelInfo(
            id: "test-model",
            name: "Test",
            localDirName: "TestModel",
            hfRepoId: "example/test-model",
            parameters: "1B",
            downloadSizeLabel: "1 GB",
            loadPolicy: .qwenMultimodal,
            description: "Test model",
            contextLength: 4096,
            isAvailable: true,
            supportsReasoning: false,
            supportsNativeImages: false,
            requiredProcessorClass: nil,
            minimumPhoneMemoryBytes: nil,
            minimumPhoneMemoryForToolCallsBytes: nil
        )

        let content = MLXModelManager.toolResponsePromptContent(
            for: nonQwenModel,
            toolResult: "Search result body"
        )
        let role = MLXModelManager.toolResponsePromptRole(for: nonQwenModel)

        #expect(content == "Search result body")
        #expect(role == .tool)
        #expect(!MLXModelManager.requiresExplicitMessageHistoryForToolLoop(for: nonQwenModel))
    }

    @Test func mlxToolLoopHasHardStopResponse() {
        let response = MLXModelManager.excessiveToolCallToolResponse(
            maximum: MLXModelManager.maxToolInvocationsPerResponse
        )

        #expect(response.contains("limit reached"))
        #expect(response.contains("\(MLXModelManager.maxToolInvocationsPerResponse)"))
    }
}
