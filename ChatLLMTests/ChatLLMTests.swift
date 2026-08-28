//
//  ChatLLMTests.swift
//  ChatLLMTests
//
//  Created by Nevio on 10/24/25.
//
//  Root of the ChatLLMTests suite. The individual @Test methods live in the
//  ChatLLMTests+Topic.swift extension files, and shared mocks live in
//  ChatLLMTestHelpers.swift.
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

@MainActor
@Suite(.serialized)
struct ChatLLMTests {
}
