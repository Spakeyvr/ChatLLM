import Foundation
import Testing
import SwiftData
@testable import ChatLLM

extension ChatLLMTests {
    private func waitUntil(_ condition: @MainActor () -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !condition(), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(condition())
    }

    @Test func independentFixturesDoNotShareBackendPreferencesOrModelStorage() throws {
        let other = ChatTestEnvironment()
        testEnvironment.defaults.set("mlx", forKey: "selectedLLMBackend")
        testEnvironment.defaults.set(true, forKey: "disableToolCalls")
        testEnvironment.bridge.selectedBackend = .mlx
        let models = testEnvironment.directory.appendingPathComponent("Models")
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        try Data("fixture".utf8).write(to: models.appendingPathComponent("marker"))

        #expect(other.bridge.selectedBackend == .foundationModels)
        #expect(!other.defaults.disableToolCalls)
        #expect(other.bridge.modelManager?.availableModels.allSatisfy { !$0.isAvailable } == true)
        #expect(other.directory != testEnvironment.directory)
        #expect(!FileManager.default.fileExists(atPath: other.directory.appendingPathComponent("Models/marker").path))
    }

    @Test func cancellingForcedSearchBeforeOutputRemovesPlaceholderWithoutSearchError() async throws {
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        let viewModel = try makeViewModel(generator: ControlledMarkdownGenerator(stream: stream))
        viewModel.searchService = try makeSearchService()
        let task = Task { await viewModel.send(userText: "Find something", forceSearch: true) }
        defer { continuation.finish(); task.cancel() }
        try await waitUntil { viewModel.streamingMessageID != nil }
        #expect(await viewModel.cancelGenerationAndWait())
        await task.value
        #expect(viewModel.conversation.messages.map(\.role) == [.user])
        #expect(viewModel.currentStreamTask == nil)
        #expect(!viewModel.isGenerating)
    }

    @Test func cancellingDuringReasoningPreservesPartialAnswerAndFinishesCapture() async throws {
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        let viewModel = try makeViewModel(generator: ControlledMarkdownGenerator(stream: stream))
        let message = Message(role: .assistant, text: "", order: 0,
                              conversation: viewModel.conversation, isReasoningMode: true)
        viewModel.conversation.messages = [message]
        let task = Task { await viewModel.streamAssistant(into: message, basedOnHistoryUpTo: 0) }
        defer { continuation.finish(); task.cancel() }
        continuation.yield("<thinking>Working through the problem")
        try await waitUntil { !(message.reasoning?.isEmpty ?? true) }
        #expect(await viewModel.cancelGenerationAndWait())
        #expect(await task.value == .cancelled)
        #expect(message.reasoning?.contains("Working through the problem") == true)
        #expect(message.reasoningCompletedAt != nil)
        #expect(message.isFinal)
        #expect(message.generationError == nil)
    }

    @Test func parentCancellationCleansUpTheTrackedStream() async throws {
        let viewModel = try makeViewModel(generator: BlockingLLMGenerator())
        let task = Task { await viewModel.send(userText: "Hello") }
        try await waitUntil { viewModel.streamingMessageID != nil }
        task.cancel()
        await task.value
        #expect(viewModel.currentStreamTask == nil)
        #expect(!viewModel.isGenerating)
        #expect(viewModel.conversation.messages.map(\.role) == [.user])
    }

    @Test func immediateRegenerationAfterStoppingCannotReceiveOldTokens() async throws {
        let (oldStream, oldContinuation) = AsyncThrowingStream<String, Error>.makeStream()
        let (newStream, newContinuation) = AsyncThrowingStream<String, Error>.makeStream()
        let generator = SequencedTestGenerator(streams: [oldStream, newStream])
        let viewModel = try makeViewModel(generator: generator)
        let task = Task { await viewModel.send(userText: "Hello") }
        defer { oldContinuation.finish(); newContinuation.finish(); task.cancel() }
        oldContinuation.yield("Original answer")
        try await waitUntil { viewModel.conversation.messages.contains { $0.text == "Original answer" } }
        let message = try #require(viewModel.conversation.messages.first { $0.role == .assistant })
        viewModel.cancelGeneration()
        let regenerate = Task { await viewModel.regenerateAfterAssistant(messageID: message.id) }
        defer { regenerate.cancel() }
        try await waitUntil { generator.startedCount == 2 }
        oldContinuation.yield(" STALE")
        newContinuation.yield("Replacement answer")
        newContinuation.finish()
        await regenerate.value
        await task.value
        #expect(message.text == "Replacement answer")
        #expect(message.isFinal)
        #expect(!viewModel.isGenerating)
        #expect(viewModel.currentStreamTask == nil)
    }

    @Test func leavingChatCancelsDelayedRegenerationAndRejectsFutureSends() async throws {
        let generator = CapturingFoundationGenerator()
        let viewModel = try makeViewModel(generator: generator)
        let message = Message(role: .assistant, text: "Existing answer", order: 0,
                              conversation: viewModel.conversation, isFinal: true)
        viewModel.conversation.messages = [message]
        viewModel.scheduleRegeneration(messageID: message.id, instruction: nil)
        viewModel.deactivate()
        try await Task.sleep(for: .milliseconds(350))
        await viewModel.send(userText: "Late action")
        await viewModel.regenerateAfterAssistant(messageID: message.id)
        #expect(generator.request == nil)
        #expect(message.text == "Existing answer")
        #expect(viewModel.conversation.messages.count == 1)
    }

    @Test func switchingConversationsRejectsLateOutputFromTheOldChat() async throws {
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        let oldChat = try makeViewModel(generator: ControlledMarkdownGenerator(stream: stream))
        let task = Task { await oldChat.send(userText: "First chat") }
        defer { continuation.finish(); task.cancel() }
        continuation.yield("Partial answer")
        try await waitUntil { oldChat.conversation.messages.contains { $0.text == "Partial answer" } }
        oldChat.deactivate()
        let newChat = try makeViewModel(generator: CapturingFoundationGenerator())
        testEnvironment.bridge.bindConversation(newChat.conversation)
        continuation.yield(" should never appear")
        continuation.finish()
        await task.value
        await newChat.send(userText: "Second chat")
        #expect(oldChat.conversation.messages.first { $0.role == .assistant }?.text == "Partial answer")
        #expect(newChat.conversation.messages.first { $0.role == .assistant }?.text == "A plain answer.")
        #expect(!newChat.isGenerating)
    }

    @Test func cancellingQueuedFoundationSessionDoesNotWaitForTheCurrentSession() async throws {
        let gate = FoundationModelsGate()
        try await gate.acquire()
        let waiter = Task { try await gate.acquire() }
        await Task.yield()
        waiter.cancel()
        let viewModel = try makeViewModel()
        let cancelled = try await viewModel.withTimeout(.seconds(1)) {
            do { try await waiter.value; return false }
            catch is CancellationError { return true }
        }
        #expect(cancelled)
        await gate.release()
        try await gate.acquire()
        await gate.release()
    }
}

extension ChatLLMTests {
    @Test func stoppingDuringSearchCancelsTheNetworkRequestWithoutShowingFailure() async throws {
        BlockingSearchURLProtocol.state.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [BlockingSearchURLProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let viewModel = try makeViewModel(generator: SearchCallingTestGenerator())
        viewModel.searchService = try TavilySearchService(apiKey: "test-key", session: session)
        let task = Task { await viewModel.send(userText: "Search now", forceSearch: true) }
        defer { task.cancel() }
        try await waitUntil { BlockingSearchURLProtocol.state.didStart }
        #expect(await viewModel.cancelGenerationAndWait())
        await task.value
        try await waitUntil { BlockingSearchURLProtocol.state.didStop }
        #expect(!viewModel.isGenerating)
        #expect(viewModel.conversation.messages.allSatisfy { $0.generationError == nil })
        #expect(viewModel.conversation.messages.map(\.role) == [.user])
    }

    @Test func stoppingBeforeStreamingWaitsForPreparationToUnwind() async throws {
        let viewModel = try makeViewModel()
        var started = false
        var unwound = false
        let task = Task {
            await viewModel.runGeneration { _ in
                started = true
                defer { unwound = true }
                try? await Task.sleep(for: .seconds(30))
            }
        }
        defer { task.cancel() }
        try await waitUntil { started }
        #expect(viewModel.currentStreamTask == nil)
        #expect(await viewModel.cancelGenerationAndWait())
        #expect(unwound)
        await task.value
        #expect(!viewModel.isGenerating)
    }
}

extension ChatLLMTests {
    @Test func backendResetOnlyCancelsItsOwnChatsAndRunsBeforeReturning() throws {
        let viewModel = try makeViewModel()
        let id = try #require(viewModel.beginGenerationLifecycle())
        let other = ChatTestEnvironment()
        NotificationCenter.default.post(name: .modelPipelineWillReset, object: other.bridge)
        #expect(viewModel.isGenerationActive(id))
        NotificationCenter.default.post(name: .modelPipelineWillReset, object: testEnvironment.bridge)
        #expect(!viewModel.isGenerating)
        #expect(!viewModel.isGenerationActive(id))
    }
}
