import XCTest

/// Opt in with TEST_RUNNER_CHATLLM_RUN_LFM_INTEGRATION=1 when the official
/// LiquidAI model is installed in the simulator app's Documents/Models folder.
final class ChatLLMLFMIntegrationTests: XCTestCase {
    @MainActor
    func testRealLFMReasoningAndFollowUp() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["CHATLLM_RUN_LFM_INTEGRATION"] == "1")
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launchArguments = [
            "-hasCompletedOnboarding", "YES", "-selectedLLMBackend", "mlx",
            "-selectedCustomModelID", "lfm2.5-2.6b-4bit",
            "-reasoningModeDefault", "NO", "-mlxMaxOutputTokens", "512"
        ]
        app.launch()
        let newChat = app.buttons["New Chat"]
        XCTAssertTrue(newChat.waitForExistence(timeout: 10))
        newChat.tap()
        let editor = app.textViews.firstMatch
        XCTAssertTrue(editor.waitForExistence(timeout: 60), app.debugDescription)
        editor.tap()
        editor.typeText("What is 17 + 25? Answer with just the number.")
        let send = app.buttons["Send"]
        XCTAssertTrue(send.waitForExistence(timeout: 60))
        let ready = NSPredicate(format: "enabled == true")
        expectation(for: ready, evaluatedWith: send)
        waitForExpectations(timeout: 60)
        send.tap()
        let thought = app.buttons["message.thought"].firstMatch
        XCTAssertTrue(thought.waitForExistence(timeout: 120), app.debugDescription)
        XCTAssertTrue(app.staticTexts["42"].waitForExistence(timeout: 120), app.debugDescription)
        thought.tap()
        XCTAssertTrue(app.navigationBars["Reasoning"].waitForExistence(timeout: 5))
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "lfm-real-reasoning"
        screenshot.lifetime = .keepAlways
        add(screenshot)
        XCTAssertFalse(app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "</think>")).firstMatch.exists)
        let reasoningBar = app.navigationBars["Reasoning"]
        reasoningBar.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.1))
            .press(forDuration: 0.1, thenDragTo: app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.95)))
        XCTAssertTrue(reasoningBar.waitForNonExistence(timeout: 5))
        editor.tap()
        editor.typeText("Now subtract 2 from that result. Answer with just the number.")
        send.tap()
        XCTAssertTrue(app.staticTexts["40"].waitForExistence(timeout: 120), app.debugDescription)
        let answer = XCTAttachment(screenshot: app.screenshot())
        answer.name = "lfm-real-follow-up"
        answer.lifetime = .keepAlways
        add(answer)
    }
}
