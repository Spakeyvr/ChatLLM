//
//  ChatLLMTests+TextCleaning
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
    @Test func finalTextCleaningPreservesCamelCaseAndCodeIdentifiers() {
        let input = "Use toFixed(2), autoFocus, isEnabled, and beginIndex exactly as written in this code example."

        let cleaned = ChatViewModel.cleanGlitchedText(input)

        #expect(cleaned.contains("toFixed(2)"))
        #expect(cleaned.contains("autoFocus"))
        #expect(cleaned.contains("isEnabled"))
        #expect(cleaned.contains("beginIndex"))
    }

    @Test func finalTextCleaningPreservesMarkdownParagraphBoundaries() {
        let input = """
        Here's a surprising fact:

        **Some wasps turn caterpillars into zombies!**

        These wasps attach their eggs to a caterpillar and inject venom.

        It's basically nature's ultimate horror movie.

        Did I surprise you? 😁
        """

        #expect(ChatViewModel.cleanGlitchedText(input) == input)
    }

    @Test func finalTextCleaningPreservesRepeatedNumbersAndCodeSeparators() {
        let input = "The total is 10000000, the ratio is 0.33333333, and the code separator is print(\"----------\")."

        #expect(ChatViewModel.cleanGlitchedText(input) == input)
    }

}
