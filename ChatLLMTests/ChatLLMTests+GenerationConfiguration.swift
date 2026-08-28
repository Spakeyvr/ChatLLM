//
//  ChatLLMTests+GenerationConfiguration
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
    @Test func cachePolicyKeepsNormalTurnsOnPersistentSimpleCacheWhenRotorQuantIsOffOnHighMemoryDevices() {
        let cachePolicy = MLXModelManager.cachePolicy(
            isEnabled: false,
            hasTools: true,
            prefersBoundedCache: false,
            memoryConstrained: false
        )

        #expect(cachePolicy == .persistentSimple)
    }

    @Test func cachePolicyUsesBoundedRotatingFallbackWhenMemoryConstrained() {
        let cachePolicy = MLXModelManager.cachePolicy(
            isEnabled: true,
            hasTools: true,
            prefersBoundedCache: false,
            memoryConstrained: true
        )

        #expect(cachePolicy == .boundedRotating(maxKVSize: 4096))
    }

    @Test func cachePolicyUsesBoundedRotatingCacheOnLowMemoryDevices() {
        let cachePolicy = MLXModelManager.cachePolicy(
            isEnabled: true,
            hasTools: false,
            prefersBoundedCache: true,
            memoryConstrained: false
        )

        #expect(cachePolicy == .boundedRotating(maxKVSize: 4096))
    }

    @Test func generationConfigurationUsesDensePersistentCacheWhenRotorQuantIsNotPreferred() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: true,
            preferRotorQuant: false,
            hasTools: true,
            hasMedia: false,
            memoryConstrained: false,
            prefersBoundedCache: false,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 32768
        )

        #expect(configuration.maxTokens == 2048)
        #expect(configuration.maxKVSize == nil)
        #expect(configuration.cachePolicy == .persistentSimple)
        #expect(configuration.cacheCompression == .none)
    }

    @Test func generationConfigurationRespectsConfiguredToolOutputLimitWhenRotorQuantIsOff() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: false,
            preferRotorQuant: false,
            hasTools: true,
            hasMedia: false,
            memoryConstrained: false,
            prefersBoundedCache: false,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 32768
        )

        #expect(configuration.maxTokens == 2048)
        #expect(configuration.maxKVSize == nil)
        #expect(configuration.cachePolicy == .persistentSimple)
        #expect(configuration.cacheCompression == .none)
    }

    @Test func generationConfigurationKeepsToolAndMediaTurnsOnSimpleCacheWhenRotorQuantIsOn() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: true,
            preferRotorQuant: true,
            hasTools: true,
            hasMedia: true,
            memoryConstrained: false,
            prefersBoundedCache: false,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 32768
        )

        #expect(configuration.maxTokens == 2048)
        #expect(configuration.maxKVSize == nil)
        #expect(configuration.cachePolicy == .persistentSimple)
        #expect(configuration.cacheCompression == .none)
    }

    @Test func generationConfigurationUsesBoundedRotatingCacheOnLowMemoryDevicesWithTools() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: true,
            preferRotorQuant: false,
            hasTools: true,
            hasMedia: false,
            memoryConstrained: false,
            prefersBoundedCache: true,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 8192
        )

        #expect(configuration.maxTokens == 2048)
        #expect(configuration.maxKVSize == 4096)
        #expect(configuration.cachePolicy == .boundedRotating(maxKVSize: 4096))
        #expect(configuration.cacheCompression == .none)
    }

    @Test func generationConfigurationUsesBoundedRotatingCacheOnLowMemoryDevicesWithMedia() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: true,
            preferRotorQuant: false,
            hasTools: false,
            hasMedia: true,
            memoryConstrained: false,
            prefersBoundedCache: true,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 8192
        )

        #expect(configuration.maxTokens == 2048)
        #expect(configuration.maxKVSize == 4096)
        #expect(configuration.cachePolicy == .boundedRotating(maxKVSize: 4096))
        #expect(configuration.cacheCompression == .none)
    }

    @Test func generationConfigurationClampsLowMemoryMaxKVSizeToConfiguredContextWindow() {
        let configuration = MLXModelManager.generationConfiguration(
            isEnabled: true,
            preferRotorQuant: false,
            hasTools: false,
            hasMedia: false,
            memoryConstrained: false,
            prefersBoundedCache: true,
            configuredMaxOutputTokens: 2048,
            configuredContextWindow: 2048
        )

        #expect(configuration.maxKVSize == 2048)
        #expect(configuration.cachePolicy == .boundedRotating(maxKVSize: 2048))
        #expect(configuration.cacheCompression == .none)
    }

    @Test func promptTokenCostFallsBackToHeuristicWhenTokenizerCountUnavailable() {
        let cost = ChatViewModel.promptTokenCost(
            tokenizedContentTokenCount: nil,
            fallbackContent: "1234567890"
        )

        #expect(cost == 24)
    }

    @Test func promptTokenCostUsesTokenizerCountWhenAvailable() {
        let cost = ChatViewModel.promptTokenCost(
            tokenizedContentTokenCount: 40,
            fallbackContent: "short"
        )

        #expect(cost == 52)
    }

    @Test func mlxSessionReusePolicyAllowsKeyedSessionReuseAcrossModes() {
        #expect(MLXModelManager.shouldPersistSessionAcrossTurns(enableThinking: false, hasTools: false, hasMedia: false))
        #expect(MLXModelManager.shouldPersistSessionAcrossTurns(enableThinking: true, hasTools: false, hasMedia: false))
        #expect(MLXModelManager.shouldPersistSessionAcrossTurns(enableThinking: false, hasTools: true, hasMedia: false))
        #expect(MLXModelManager.shouldPersistSessionAcrossTurns(enableThinking: false, hasTools: false, hasMedia: true))
    }
}
