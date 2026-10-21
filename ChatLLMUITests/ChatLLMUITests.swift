//
//  ChatLLMUITests.swift
//  ChatLLMUITests
//
//  Created by Nevio on 10/24/25.
//

import XCTest
import CoreGraphics

final class ChatLLMUITests: XCTestCase {
    private let firstMLXModelID = "qwen3.5-4b-4bit-hybrid"
    private let secondMLXModelID = "qwen3.5-2b-4bit"

    @MainActor
    func testMarkdownRendersDuringAndAfterStreamingInBothBubbleTypes() throws {
        for reasoning in [false, true] {
            let app = XCUIApplication()
            app.launchArguments = ["-ui-test-reset-app-state", "-ui-test-markdown"]
            if reasoning { app.launchArguments.append("-ui-test-markdown-reasoning") }
            app.launch()

            // Each paragraph is its own element now, and a soft break stays inside one.
            let firstParagraph = app.staticTexts
                .matching(NSPredicate(format: "label CONTAINS %@", "First bold word"))
                .firstMatch
            XCTAssertTrue(firstParagraph.waitForExistence(timeout: 5))
            XCTAssertTrue(firstParagraph.label.contains("Next line."))
            XCTAssertFalse(firstParagraph.label.contains("Second paragraph"))

            let secondParagraph = app.staticTexts
                .matching(NSPredicate(format: "label CONTAINS %@", "Second paragraph with italic text"))
                .firstMatch
            XCTAssertTrue(secondParagraph.exists)
            XCTAssertTrue(secondParagraph.label.contains("first_name"))
            XCTAssertEqual(app.staticTexts["markdown.status"].label, "Streaming")
            captureSettingsScreenshot("markdown-\(reasoning ? "reasoning" : "standard")-paragraphs", in: app)

            app.buttons["markdown.advance"].tap()
            // Headings, lists and code are laid out natively — no WebView involved.
            XCTAssertTrue(app.staticTexts["Live heading"].waitForExistence(timeout: 5))
            XCTAssertTrue(app.staticTexts["First item"].exists)
            XCTAssertTrue(app.staticTexts["Second item"].exists)
            XCTAssertTrue(app.staticTexts["let first_name = 1"].exists)
            XCTAssertFalse(app.webViews.firstMatch.exists)
            XCTAssertEqual(app.staticTexts["markdown.status"].label, "Streaming")
            captureSettingsScreenshot("markdown-\(reasoning ? "reasoning" : "standard")-streaming", in: app)

            app.buttons["markdown.advance"].tap()
            XCTAssertTrue(app.staticTexts["Last paragraph."].waitForExistence(timeout: 5))
            XCTAssertEqual(app.staticTexts["markdown.status"].label, "Finished")
            XCTAssertTrue(app.staticTexts["First item"].exists)
            XCTAssertFalse(app.webViews.firstMatch.exists)
            captureSettingsScreenshot("markdown-\(reasoning ? "reasoning" : "standard")-finished", in: app)
            app.terminate()
        }
    }

    @MainActor
    func testMathStillRendersThroughTheBundledDocument() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-ui-test-reset-app-state", "-ui-test-markdown", "-ui-test-markdown-math"]
        app.launch()

        let webView = app.webViews.firstMatch
        XCTAssertTrue(webView.waitForExistence(timeout: 10))
        captureSettingsScreenshot("markdown-math", in: app)
        app.terminate()
    }

    private func launchApp(
        resetAppState: Bool = true,
        fakeMLXDownloads: Bool = false
    ) -> XCUIApplication {
        let app = XCUIApplication()
        if resetAppState {
            app.launchArguments.append("-ui-test-reset-app-state")
        }
        if fakeMLXDownloads {
            app.launchArguments.append("-ui-test-fake-mlx-downloads")
        }
        app.launch()
        return app
    }

    private func element(in app: XCUIApplication, identifier: String) -> XCUIElement {
        app.descendants(matching: .any)[identifier]
    }

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    func testFreshLaunchShowsOnboarding() throws {
        let app = launchApp()

        XCTAssertTrue(app.otherElements["onboarding.root"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["onboarding.intro.title"].exists)
        XCTAssertFalse(app.otherElements["content.root"].exists)
    }

    @MainActor
    func testSkippingOnboardingPersistsCompletionAcrossLaunches() throws {
        let app = launchApp()

        app.buttons["onboarding.skip"].tap()
        XCTAssertTrue(app.otherElements["content.root"].waitForExistence(timeout: 5))

        app.terminate()

        let relaunchedApp = launchApp(resetAppState: false)
        XCTAssertFalse(relaunchedApp.otherElements["onboarding.root"].waitForExistence(timeout: 2))
        XCTAssertTrue(relaunchedApp.otherElements["content.root"].waitForExistence(timeout: 5))
    }

    @MainActor
    func testTavilyKeyEntryDoesNotShowOnboardingAgain() throws {
        let app = launchApp()

        app.buttons["onboarding.intro.continue"].tap()

        let keyField = app.secureTextFields["onboarding.tavily.key"]
        XCTAssertTrue(keyField.waitForExistence(timeout: 5))
        keyField.tap()
        keyField.typeText("tvly-test-key")

        app.buttons["onboarding.tavily.continue"].tap()
        app.buttons["onboarding.finish"].tap()
        XCTAssertTrue(app.otherElements["content.root"].waitForExistence(timeout: 5))

        app.terminate()

        let relaunchedApp = launchApp(resetAppState: false)
        XCTAssertFalse(relaunchedApp.otherElements["onboarding.root"].waitForExistence(timeout: 2))
        XCTAssertTrue(relaunchedApp.otherElements["content.root"].waitForExistence(timeout: 5))
    }

    @MainActor
    func testStartingMLXDownloadAllowsFinishingOnboarding() throws {
        let app = launchApp(fakeMLXDownloads: true)

        app.buttons["onboarding.intro.continue"].tap()
        app.buttons["onboarding.tavily.continue"].tap()

        let downloadButton = app.buttons["onboarding.mlx.download"]
        XCTAssertTrue(downloadButton.waitForExistence(timeout: 5))
        downloadButton.tap()

        XCTAssertTrue(element(in: app, identifier: "onboarding.mlx.progress.\(firstMLXModelID)").waitForExistence(timeout: 5))

        app.buttons["onboarding.finish"].tap()
        XCTAssertTrue(app.otherElements["content.root"].waitForExistence(timeout: 5))
    }

    @MainActor
    func testChangingSelectedModelDoesNotMoveActiveDownloadProgress() throws {
        let app = launchApp(fakeMLXDownloads: true)

        app.buttons["onboarding.intro.continue"].tap()
        app.buttons["onboarding.tavily.continue"].tap()

        let downloadButton = app.buttons["onboarding.mlx.download"]
        XCTAssertTrue(downloadButton.waitForExistence(timeout: 5))
        downloadButton.tap()

        let firstProgress = element(in: app, identifier: "onboarding.mlx.progress.\(firstMLXModelID)")
        XCTAssertTrue(firstProgress.waitForExistence(timeout: 5))

        let secondModelButton = app.buttons["onboarding.mlx.model.\(secondMLXModelID)"]
        XCTAssertTrue(secondModelButton.waitForExistence(timeout: 5))
        secondModelButton.tap()

        XCTAssertTrue(firstProgress.waitForExistence(timeout: 2))
        XCTAssertFalse(element(in: app, identifier: "onboarding.mlx.progress.\(secondMLXModelID)").exists)
    }

    @MainActor
    func testDownloadedModelIsMarkedAndNamedPlainlyInTheMenu() throws {
        let app = launchApp(fakeMLXDownloads: true)
        XCTAssertTrue(app.buttons["onboarding.skip"].waitForExistence(timeout: 5))
        app.buttons["onboarding.skip"].tap()
        XCTAssertTrue(app.textViews.firstMatch.waitForExistence(timeout: 5))

        app.navigationBars.staticTexts["Apple Intelligence"].firstMatch.tap()
        app.buttons["Manage Models"].tap()
        let download = app.buttons["modelSelection.download.qwen3.5-0.8b-4bit"]
        XCTAssertTrue(download.waitForExistence(timeout: 5))
        download.tap()
        XCTAssertTrue(element(in: app, identifier: "modelSelection.downloaded.qwen3.5-0.8b-4bit").waitForExistence(timeout: 20))
        // The icon-only row buttons are named for VoiceOver.
        XCTAssertTrue(app.buttons["Delete"].exists)
        XCTAssertTrue(app.buttons["Model Information"].firstMatch.exists)
        app.buttons["Done"].tap()

        app.navigationBars.staticTexts["Apple Intelligence"].firstMatch.tap()
        XCTAssertTrue(app.buttons["Qwen 3.5 0.8B (4-bit)"].waitForExistence(timeout: 3))
    }

    @MainActor
    func testSearchOnlyResponseShowsSourcesWithoutThoughtSummary() throws {
        let app = XCUIApplication()
        app.launchArguments += [
            "-ui-test-reset-app-state",
            "-ui-test-web-search-demo",
            "-ui-test-web-search-without-reasoning"
        ]
        app.launch()

        XCTAssertTrue(app.buttons["message.sources"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["message.thought"].exists)
    }

    @MainActor
    func testReasonedResponseShowsTappableThoughtDuration() throws {
        let app = XCUIApplication()
        app.launchArguments += [
            "-ui-test-reset-app-state",
            "-ui-test-web-search-demo"
        ]
        app.launch()

        let thoughtSummary = app.buttons["message.thought"]
        XCTAssertTrue(thoughtSummary.waitForExistence(timeout: 5))
        XCTAssertEqual(thoughtSummary.label, "Thought for 1 minute and 8 seconds")
        XCTAssertTrue(app.buttons["message.sources"].exists)

        thoughtSummary.tap()
        XCTAssertTrue(app.navigationBars["Reasoning"].waitForExistence(timeout: 5))
        XCTAssertEqual(
            app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "</think>")).count,
            0
        )
    }

    @MainActor
    func testShortChatStaysBelowTheTopEdgeAfterPresentingASheet() throws {
        let app = XCUIApplication()
        app.launchArguments += [
            "-ui-test-reset-app-state",
            "-ui-test-web-search-demo"
        ]
        app.launch()

        let question = app.staticTexts
            .matching(NSPredicate(format: "label CONTAINS %@", "What changed in SwiftUI web views"))
            .firstMatch
        XCTAssertTrue(question.waitForExistence(timeout: 5))
        XCTAssertTrue(question.isHittable)

        // Presenting and dismissing a sheet must not re-apply a scroll that hides the first message.
        let thoughtSummary = app.buttons["message.thought"]
        XCTAssertTrue(thoughtSummary.waitForExistence(timeout: 5))
        thoughtSummary.tap()
        let reasoningNavigationBar = app.navigationBars["Reasoning"]
        XCTAssertTrue(reasoningNavigationBar.waitForExistence(timeout: 5))
        let start = reasoningNavigationBar.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        let end = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.95))
        start.press(forDuration: 0.1, thenDragTo: end)
        XCTAssertTrue(reasoningNavigationBar.waitForNonExistence(timeout: 5))

        XCTAssertTrue(question.isHittable)
    }

    @MainActor
    func testLongChatOpensAtTheLatestMessage() throws {
        let app = XCUIApplication()
        app.launchArguments += [
            "-ui-test-reset-app-state",
            "-ui-test-web-search-demo",
            "-ui-test-long-chat"
        ]
        app.launch()

        XCTAssertTrue(app.buttons["message.sources"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["message.sources"].isHittable)
        XCTAssertFalse(app.staticTexts["Earlier message 1"].isHittable)
    }

    @MainActor
    func testReturningToTheAppKeepsTheReadersScrollPosition() throws {
        let app = XCUIApplication()
        app.launchArguments += [
            "-ui-test-reset-app-state",
            "-ui-test-web-search-demo",
            "-ui-test-long-chat"
        ]
        app.launch()

        XCTAssertTrue(app.buttons["message.sources"].waitForExistence(timeout: 5))
        let firstMessage = app.staticTexts["Earlier message 1"]
        for _ in 0..<6 where !firstMessage.isHittable {
            app.scrollViews.firstMatch.swipeDown(velocity: .fast)
        }
        XCTAssertTrue(firstMessage.isHittable)

        XCUIDevice.shared.press(.home)
        app.activate()

        XCTAssertTrue(app.staticTexts["Earlier message 1"].waitForExistence(timeout: 5))
        XCTAssertTrue(firstMessage.isHittable)
        XCTAssertFalse(app.buttons["message.sources"].isHittable)
    }

    @MainActor
    func testReasoningSheetRendersContentExposedByExpansion() throws {
        let app = XCUIApplication()
        app.launchArguments += [
            "-ui-test-reset-app-state",
            "-ui-test-web-search-demo",
            "-ui-test-long-reasoning-demo"
        ]
        app.launch()

        let thoughtSummary = app.buttons["message.thought"]
        XCTAssertTrue(thoughtSummary.waitForExistence(timeout: 5))
        thoughtSummary.tap()

        let reasoningNavigationBar = app.navigationBars["Reasoning"]
        XCTAssertTrue(reasoningNavigationBar.waitForExistence(timeout: 5))
        let start = reasoningNavigationBar.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.15))
        let end = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.08))
        start.press(forDuration: 0.1, thenDragTo: end)

        let newlyExposedText = app.staticTexts.matching(
            NSPredicate(
                format: "label CONTAINS %@",
                "This newly exposed reasoning must appear immediately when the sheet expands."
            )
        ).firstMatch
        XCTAssertTrue(newlyExposedText.waitForExistence(timeout: 5))
        XCTAssertTrue(newlyExposedText.isHittable)
    }

    @MainActor
    func testOpeningAnotherChatAfterReturningToListShowsIt() throws {
        let fixture = XCUIApplication()
        fixture.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo", "-ui-test-second-chat"]
        fixture.launch()
        XCTAssertTrue(fixture.buttons["message.sources"].waitForExistence(timeout: 5))
        fixture.terminate()

        // Relaunch without the demo selection so the app opens into a fresh chat.
        let app = launchApp(resetAppState: false)
        openSidebar(in: app)
        let secondChatRow = app.staticTexts["Second Chat"]
        XCTAssertTrue(secondChatRow.waitForExistence(timeout: 5))
        secondChatRow.tap()
        XCTAssertTrue(app.staticTexts["Second chat reply."].waitForExistence(timeout: 5))

        openSidebar(in: app)
        let demoChatRow = app.staticTexts["Web Search Preview"]
        XCTAssertTrue(demoChatRow.waitForExistence(timeout: 5))
        demoChatRow.tap()

        XCTAssertTrue(app.buttons["message.sources"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.staticTexts["Preparing chat…"].exists)
    }

    @MainActor
    func testSelectingSearchResultClosesDrawerAndKeyboard() throws {
        let fixture = XCUIApplication()
        fixture.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo", "-ui-test-second-chat"]
        fixture.launch()
        XCTAssertTrue(fixture.buttons["message.sources"].waitForExistence(timeout: 5))
        fixture.terminate()

        let app = launchApp(resetAppState: false)
        openSidebar(in: app)
        app.buttons["Search"].firstMatch.tap()
        let searchField = app.searchFields.firstMatch
        XCTAssertTrue(searchField.waitForExistence(timeout: 5))
        searchField.typeText("Second")

        let result = app.staticTexts["Second Chat"]
        XCTAssertTrue(result.waitForExistence(timeout: 5))
        result.tap()

        XCTAssertTrue(app.staticTexts["Second chat reply."].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["sidebar.dismiss"].exists)
        let keyboardGone = NSPredicate(format: "exists == false")
        expectation(for: keyboardGone, evaluatedWith: app.keyboards.firstMatch)
        waitForExpectations(timeout: 3)
    }

    @MainActor
    func testRenamingAChatKeepsItsPlaceInTheSidebar() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo", "-ui-test-second-chat"]
        app.launch()
        XCTAssertTrue(app.textViews.firstMatch.waitForExistence(timeout: 5))

        openSidebar(in: app)
        let olderRow = app.staticTexts["Web Search Preview"]
        XCTAssertTrue(olderRow.waitForExistence(timeout: 5))
        olderRow.press(forDuration: 1.0)
        let rename = app.buttons["Rename"]
        XCTAssertTrue(rename.waitForExistence(timeout: 3))
        rename.tap()

        let field = app.alerts.textFields.firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: 3))
        field.tap()
        let existing = field.value as? String ?? ""
        field.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: existing.count) + "Swift Notes")
        app.alerts.buttons["Save"].tap()

        let renamedRow = app.staticTexts["Swift Notes"]
        XCTAssertTrue(renamedRow.waitForExistence(timeout: 3))
        let newerRow = app.staticTexts["Second Chat"]
        XCTAssertLessThan(newerRow.frame.minY, renamedRow.frame.minY)
    }

    @MainActor
    func testRenameFieldCanBeClearedInOneTap() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo"]
        app.launch()
        XCTAssertTrue(app.textViews.firstMatch.waitForExistence(timeout: 5))

        openSidebar(in: app)
        let row = app.staticTexts["Web Search Preview"]
        XCTAssertTrue(row.waitForExistence(timeout: 5))
        row.press(forDuration: 1.0)
        app.buttons["Rename"].tap()

        let field = app.alerts.textFields.firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: 3))
        field.tap()
        app.alerts.buttons["Clear text"].tap()
        field.typeText("Web Notes\n")

        XCTAssertTrue(app.staticTexts["Web Notes"].waitForExistence(timeout: 3))
        XCTAssertFalse(app.alerts.firstMatch.exists)
    }

    @MainActor
    func testDeletingTheOpenChatOpensABlankChat() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo", "-ui-test-second-chat"]
        app.launch()
        let demoMessage = app.staticTexts["What changed in SwiftUI web views this year?"]
        XCTAssertTrue(demoMessage.waitForExistence(timeout: 5))

        openSidebar(in: app)
        let demoChatRow = app.staticTexts["Web Search Preview"]
        XCTAssertTrue(demoChatRow.waitForExistence(timeout: 5))
        demoChatRow.swipeLeft()
        let delete = app.buttons["Delete"]
        XCTAssertTrue(delete.waitForExistence(timeout: 3))
        delete.tap()

        // The sidebar stays open for further edits while a blank chat takes the open chat's place.
        XCTAssertTrue(app.buttons["sidebar.dismiss"].exists)
        XCTAssertTrue(app.staticTexts["Second Chat"].exists)
        app.buttons["sidebar.dismiss"].tap()

        XCTAssertTrue(app.textViews.firstMatch.waitForExistence(timeout: 5))
        XCTAssertFalse(demoMessage.exists)
        XCTAssertFalse(app.staticTexts["Select a Conversation"].exists)
    }

    @MainActor
    func testDeletingAllChatsFromADraftKeepsTheComposer() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo"]
        app.launch()
        XCTAssertTrue(app.staticTexts["What changed in SwiftUI web views this year?"].waitForExistence(timeout: 5))

        // Open a blank draft, then delete every saved chat from Settings.
        openSidebar(in: app)
        app.buttons["New Chat"].firstMatch.tap()
        openSettings(in: app)
        app.buttons["settings.privacy"].tap()
        app.buttons["settings.deleteAllChats"].tap()
        let confirm = app.alerts.buttons["Delete"]
        XCTAssertTrue(confirm.waitForExistence(timeout: 3))
        confirm.tap()

        // The draft stays usable instead of being replaced by a dead placeholder.
        if app.buttons["sidebar.dismiss"].waitForExistence(timeout: 3) {
            app.buttons["sidebar.dismiss"].tap()
        }
        XCTAssertTrue(app.textViews.firstMatch.waitForExistence(timeout: 5))
        XCTAssertFalse(app.staticTexts["Start a New Chat"].exists)
    }

    @MainActor
    func testUnavailableWebSearchExplainsWhy() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo"]
        app.launch()
        XCTAssertTrue(app.textViews.firstMatch.waitForExistence(timeout: 5))

        // No Tavily key is configured after a reset.
        app.buttons["Add attachment or web search"].tap()
        let reason = app.staticTexts["composer.webSearchUnavailableReason"]
        XCTAssertTrue(reason.waitForExistence(timeout: 3))
        XCTAssertTrue(reason.label.contains("Tavily"))
    }

    @MainActor
    func testEditingTheFirstMessageKeepsTheModelLocked() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo"]
        app.launch()
        let message = app.staticTexts["What changed in SwiftUI web views this year?"]
        XCTAssertTrue(message.waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["Manage Models"].exists)

        message.press(forDuration: 1.0)
        app.buttons["Edit"].tap()
        XCTAssertTrue(app.buttons["chat.edit.cancel"].waitForExistence(timeout: 3))

        // The title stays a plain label rather than the model menu.
        app.navigationBars.staticTexts["Apple Intelligence"].firstMatch.tap()
        XCTAssertFalse(app.buttons["Manage Models"].waitForExistence(timeout: 2))
    }

    @MainActor
    func testImageAttachmentOpensFullScreenAndCloses() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo", "-ui-test-image-message"]
        app.launch()
        let thumbnail = app.images.firstMatch
        XCTAssertTrue(thumbnail.waitForExistence(timeout: 5))

        thumbnail.tap()
        let done = app.buttons["Done"]
        XCTAssertTrue(done.waitForExistence(timeout: 3))
        done.tap()
        XCTAssertFalse(done.waitForExistence(timeout: 2))
        XCTAssertTrue(app.textViews.firstMatch.exists)
    }

    @MainActor
    func testDoubleTappingAFullScreenImageZoomsInAndOut() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo", "-ui-test-image-message"]
        app.launch()
        let thumbnail = app.images.firstMatch
        XCTAssertTrue(thumbnail.waitForExistence(timeout: 5))
        thumbnail.tap()
        XCTAssertTrue(app.buttons["Done"].waitForExistence(timeout: 3))

        let viewer = app.scrollViews.firstMatch
        XCTAssertTrue(viewer.waitForExistence(timeout: 3))
        XCTAssertEqual(viewer.value as? String, "100%")
        viewer.doubleTap()
        expectation(for: NSPredicate(format: "value == %@", "250%"), evaluatedWith: viewer)
        waitForExpectations(timeout: 3)
        viewer.doubleTap()
        expectation(for: NSPredicate(format: "value == %@", "100%"), evaluatedWith: viewer)
        waitForExpectations(timeout: 3)

        // At normal size a downward swipe still dismisses the viewer.
        let start = viewer.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.4))
        start.press(forDuration: 0.05, thenDragTo: start.withOffset(CGVector(dx: 0, dy: 300)))
        XCTAssertTrue(app.buttons["Done"].waitForNonExistence(timeout: 3))
    }

    @MainActor
    func testSwipingRightFromAMessageOpensTheSidebar() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo"]
        app.launch()
        let userMessage = app.staticTexts["What changed in SwiftUI web views this year?"]
        XCTAssertTrue(userMessage.waitForExistence(timeout: 5))

        // Messages have no swipe actions, so a rightward swipe that starts on
        // one belongs to the sidebar drawer.
        let start = userMessage.coordinate(withNormalizedOffset: CGVector(dx: 0.1, dy: 0.5))
        let end = start.withOffset(CGVector(dx: app.frame.width * 0.7, dy: 0))
        start.press(forDuration: 0.05, thenDragTo: end)

        XCTAssertTrue(app.buttons["sidebar.dismiss"].waitForExistence(timeout: 3))
    }

    @MainActor
    func testLongPressingAMessageOffersCopyAndShare() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo"]
        app.launch()
        let userMessage = app.staticTexts["What changed in SwiftUI web views this year?"]
        XCTAssertTrue(userMessage.waitForExistence(timeout: 5))

        userMessage.press(forDuration: 1.0)

        XCTAssertTrue(app.buttons["Copy"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.buttons["Share"].exists)
        XCTAssertTrue(app.buttons["Edit"].exists)
    }

    @MainActor
    func testLaunchPerformance() throws {
        measure(metrics: [XCTApplicationLaunchMetric()]) {
            let app = XCUIApplication()
            app.launchArguments.append("-ui-test-reset-app-state")
            app.launch()
        }
    }

    /// The drawer's sidebar and chat navigation bars stay in the hierarchy under the
    /// settings sheet, so target the system back button explicitly.
    @MainActor
    private func settingsBackButton(in app: XCUIApplication) -> XCUIElement {
        app.navigationBars.buttons.matching(identifier: "BackButton").firstMatch
    }

    /// On iPhone the chat list lives in a drawer behind the chat; slide it out.
    @MainActor
    private func openSidebar(in app: XCUIApplication) {
        let toggle = app.buttons["sidebar.toggle"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 5))
        toggle.tap()
        XCTAssertTrue(app.buttons["sidebar.dismiss"].waitForExistence(timeout: 5))
    }

    @MainActor
    private func openSettings(in app: XCUIApplication) {
        let skip = app.buttons["onboarding.skip"]
        if skip.waitForExistence(timeout: 2) { skip.tap() }
        if !app.buttons["sidebar.dismiss"].exists { openSidebar(in: app) }
        let settings = app.buttons["Settings"]
        XCTAssertTrue(settings.waitForExistence(timeout: 5))
        settings.tap()
        XCTAssertTrue(app.buttons["settings.appearance"].waitForExistence(timeout: 5))
    }

    @MainActor
    private func toggle(_ element: XCUIElement) {
        // iOS 26 exposes the labeled SwiftUI row and the UISwitch separately.
        // Tapping the center of the outer row hits its label, not the switch.
        let control = element.switches.firstMatch
        if control.exists { control.tap() } else { element.tap() }
    }

    @MainActor
    private func captureSettingsScreenshot(_ name: String, in app: XCUIApplication) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    @MainActor
    private func settingsBackgroundBrightness(in app: XCUIApplication) throws -> Double {
        let image = try XCTUnwrap(app.screenshot().image.cgImage)
        // Sample the list background at its left edge, away from text and cards.
        let sample = try XCTUnwrap(image.cropping(to: CGRect(x: 0, y: image.height / 2, width: 1, height: 1)))
        var bytes = [UInt8](repeating: 0, count: 4)
        try bytes.withUnsafeMutableBytes { buffer in
            let context = try XCTUnwrap(CGContext(
                data: buffer.baseAddress, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ))
            context.draw(sample, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        }
        return Double(Int(bytes[0]) + Int(bytes[1]) + Int(bytes[2])) / 3
    }

    @MainActor
    func testSettingsSwipeDismissKeepsChanges() throws {
        let app = launchApp()
        openSettings(in: app)
        app.buttons["settings.chat"].tap()
        let sendOnReturn = app.switches["settings.sendOnReturn"]
        let original = sendOnReturn.value as? String
        toggle(sendOnReturn)
        settingsBackButton(in: app).tap()
        let navigationBar = app.navigationBars["Settings"]
        navigationBar.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.1))
            .press(forDuration: 0.1, thenDragTo: app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.95)))
        XCTAssertTrue(app.buttons["Settings"].waitForExistence(timeout: 5))
        openSettings(in: app)
        app.buttons["settings.chat"].tap()
        XCTAssertNotEqual(app.switches["settings.sendOnReturn"].value as? String, original)
    }

    @MainActor
    func testSendOnReturnLabelsTheKeyboardKeySend() throws {
        let app = launchApp()
        openSettings(in: app)
        app.buttons["settings.chat"].tap()
        toggle(app.switches["settings.sendOnReturn"])
        settingsBackButton(in: app).tap()
        app.buttons["settings.done"].tap()
        let dismissSidebar = app.buttons["sidebar.dismiss"]
        if dismissSidebar.waitForExistence(timeout: 2) { dismissSidebar.tap() }

        let composer = app.textViews.firstMatch
        XCTAssertTrue(composer.waitForExistence(timeout: 5))
        composer.tap()
        let sendKey = app.keyboards.buttons.matching(NSPredicate(format: "label ==[c] %@", "send")).firstMatch
        XCTAssertTrue(sendKey.waitForExistence(timeout: 5))
    }

    @MainActor
    func testInterfaceLanguageSwitchesImmediatelyAndPersists() throws {
        let app = launchApp()
        openSettings(in: app)
        app.buttons["settings.appearance"].tap()

        for (name, title, appearance, done) in [
            ("Deutsch", "Einstellungen", "Darstellung", "Fertig"),
            ("Español", "Ajustes", "Apariencia", "Listo"),
            ("English", "Settings", "Appearance", "Done")
        ] {
            app.buttons["settings.language"].tap()
            app.buttons[name].tap()
            XCTAssertTrue(app.navigationBars[appearance].waitForExistence(timeout: 5))
            captureSettingsScreenshot("appearance-\(name)", in: app)
            settingsBackButton(in: app).tap()
            XCTAssertTrue(app.navigationBars[title].waitForExistence(timeout: 5))
            XCTAssertTrue(app.buttons["settings.appearance"].label.contains(appearance))
            XCTAssertEqual(app.buttons["settings.done"].label, done)
            captureSettingsScreenshot("language-\(name)", in: app)
            app.buttons["settings.appearance"].tap()
        }

        app.buttons["settings.language"].tap()
        app.buttons["Español"].tap()
        settingsBackButton(in: app).tap()
        app.buttons["settings.done"].tap()
        app.terminate()
        let relaunched = launchApp(resetAppState: false)
        relaunched.buttons["sidebar.toggle"].tap()
        let settings = relaunched.buttons["Ajustes"]
        XCTAssertTrue(settings.waitForExistence(timeout: 5))
        settings.tap()
        XCTAssertTrue(relaunched.navigationBars["Ajustes"].waitForExistence(timeout: 5))
        XCTAssertEqual(relaunched.buttons["settings.done"].label, "Listo")
        relaunched.buttons["settings.appearance"].tap()
        XCTAssertTrue(relaunched.buttons["settings.language"].label.contains("Español"))
    }

    @MainActor
    func testLocalizedOnboardingAndModelSetup() throws {
        for (language, welcome, localModels) in [
            ("de", "Willkommen bei ChatLLM", "Lokale Modelle"),
            ("es", "Te damos la bienvenida a ChatLLM", "Modelos locales")
        ] {
            let app = XCUIApplication()
            app.launchArguments = [
                "-ui-test-reset-app-state", "-ui-test-fake-mlx-downloads",
                "-appLanguage", language
            ]
            app.launch()
            XCTAssertTrue(app.staticTexts[welcome].waitForExistence(timeout: 5))
            XCTAssertTrue(app.buttons["onboarding.intro.continue"].isHittable)
            captureSettingsScreenshot("onboarding-\(language)", in: app)
            app.scrollViews.firstMatch.swipeUp()
            let finalCard = language == "de"
                ? "Füge einen Tavily-API-Schlüssel hinzu, damit der Assistent im Web nach aktuellen Informationen suchen kann."
                : "Añade una clave de API de Tavily para que el asistente busque información actual en la web."
            XCTAssertTrue(app.staticTexts[finalCard].isHittable)
            XCTAssertTrue(app.buttons["onboarding.intro.continue"].isHittable)
            captureSettingsScreenshot("onboarding-scrolled-\(language)", in: app)
            app.buttons["onboarding.intro.continue"].tap()
            XCTAssertTrue(app.buttons["onboarding.tavily.continue"].waitForExistence(timeout: 5))
            app.buttons["onboarding.tavily.continue"].tap()
            XCTAssertTrue(app.staticTexts[localModels].waitForExistence(timeout: 5))
            XCTAssertTrue(app.buttons["onboarding.finish"].isHittable)
            captureSettingsScreenshot("models-\(language)", in: app)
            app.terminate()
        }
    }

    @MainActor
    func testSettingsAppearanceAndDetailNavigation() throws {
        let app = launchApp()
        openSettings(in: app)
        let systemBrightness = try settingsBackgroundBrightness(in: app)
        let oppositeAppearance = systemBrightness > 128 ? "Dark" : "Light"
        captureSettingsScreenshot("settings-light", in: app)
        app.buttons["settings.appearance"].tap()
        app.segmentedControls["settings.colorScheme"].buttons[oppositeAppearance].tap()
        captureSettingsScreenshot("appearance-dark", in: app)
        app.sliders["settings.messageTextSize"].adjust(toNormalizedSliderPosition: 1)
        XCTAssertEqual(app.sliders["settings.messageTextSize"].value as? String, "22 points")
        settingsBackButton(in: app).tap()
        XCTAssertTrue(app.buttons["settings.appearance"].label.contains(oppositeAppearance))
        XCTAssertGreaterThan(abs(try settingsBackgroundBrightness(in: app) - systemBrightness), 100)
        captureSettingsScreenshot("settings-dark", in: app)

        app.buttons["settings.appearance"].tap()
        app.segmentedControls["settings.colorScheme"].buttons["System"].tap()
        app.buttons["Reset Text Size"].tap()
        XCTAssertEqual(app.sliders["settings.messageTextSize"].value as? String, "16 points")
        captureSettingsScreenshot("appearance-light", in: app)
        settingsBackButton(in: app).tap()
        XCTAssertEqual(try settingsBackgroundBrightness(in: app), systemBrightness, accuracy: 5)

        for (id, title) in [("chat", "Chat"), ("webSearch", "Web Search"), ("privacy", "Privacy & Data"),
                            ("about", "About & Support"), ("advanced", "Advanced")] {
            app.buttons["settings.\(id)"].tap()
            XCTAssertTrue(app.navigationBars[title].waitForExistence(timeout: 3))
            captureSettingsScreenshot("settings-\(id)", in: app)
            settingsBackButton(in: app).tap()
        }
        app.buttons["settings.advanced"].tap()
        for (id, title) in [("mlx", "On-Device Models"), ("developer", "Developer")] {
            app.buttons["settings.\(id)"].tap()
            XCTAssertTrue(app.navigationBars[title].waitForExistence(timeout: 3))
            captureSettingsScreenshot("settings-\(id)", in: app)
            if id == "mlx" {
                app.buttons["About RotorQuant"].tap()
                XCTAssertTrue(app.navigationBars["RotorQuant"].waitForExistence(timeout: 3))
                captureSettingsScreenshot("settings-rotorQuant", in: app)
                settingsBackButton(in: app).tap()
            }
            settingsBackButton(in: app).tap()
        }
    }

    @MainActor
    func testSettingsPagesRemainScrollable() throws {
        let app = launchApp()
        openSettings(in: app)
        // Also run this flow with the simulator at its largest accessibility
        // text size; keep screenshots for checking wraps and control spacing.
        captureSettingsScreenshot("settings-scroll-root", in: app)
        for id in ["appearance", "chat"] {
            app.buttons["settings.\(id)"].tap()
            captureSettingsScreenshot("settings-scroll-\(id)", in: app)
            app.swipeUp()
            captureSettingsScreenshot("settings-scroll-\(id)-bottom", in: app)
            settingsBackButton(in: app).tap()
        }
        let about = app.buttons["settings.about"]
        for _ in 0..<4 where !about.isHittable { app.swipeUp() }
        XCTAssertTrue(about.isHittable)
        captureSettingsScreenshot("settings-scroll-root-bottom", in: app)
        about.tap()
        XCTAssertTrue(app.navigationBars["About & Support"].waitForExistence(timeout: 3))
    }

    @MainActor
    func testSettingsCategoriesAndImmediatePersistence() throws {
        let app = launchApp()
        openSettings(in: app)
        for id in ["appearance", "chat", "webSearch", "privacy", "advanced", "about"] {
            XCTAssertTrue(app.buttons["settings.\(id)"].isHittable)
        }
        XCTAssertFalse(app.buttons["Save"].exists)
        XCTAssertFalse(app.buttons["Cancel"].exists)

        app.buttons["settings.chat"].tap()
        let sendOnReturn = app.switches["settings.sendOnReturn"]
        XCTAssertTrue(sendOnReturn.waitForExistence(timeout: 3))
        XCTAssertEqual(sendOnReturn.value as? String, "0")
        toggle(sendOnReturn)
        XCTAssertEqual(sendOnReturn.value as? String, "1")
        // Terminate without Done: the toggle must already be saved.
        app.terminate()
        let relaunched = launchApp(resetAppState: false)
        openSettings(in: relaunched)
        relaunched.buttons["settings.chat"].tap()
        XCTAssertEqual(relaunched.switches["settings.sendOnReturn"].value as? String, "1")
    }

    @MainActor
    func testSettingsResponsePreferencesSaveAndCancel() throws {
        let app = launchApp()
        openSettings(in: app)
        app.buttons["settings.chat"].tap()
        app.buttons["settings.responsePreferences"].tap()
        let editor = app.textViews["settings.preferencesEditor"]
        XCTAssertTrue(editor.waitForExistence(timeout: 3))
        editor.tap()
        editor.typeText("Discard this draft")
        app.buttons["Cancel"].tap()
        XCTAssertTrue(app.buttons["settings.responsePreferences"].label.contains("Not Set"))

        app.buttons["settings.responsePreferences"].tap()
        editor.tap()
        editor.typeText("Keep answers concise.")
        app.buttons["settings.savePreferences"].tap()
        XCTAssertTrue(app.buttons["settings.responsePreferences"].label.contains("Custom"))
        app.terminate()
        let relaunched = launchApp(resetAppState: false)
        openSettings(in: relaunched)
        relaunched.buttons["settings.chat"].tap()
        relaunched.buttons["settings.responsePreferences"].tap()
        XCTAssertEqual(relaunched.textViews["settings.preferencesEditor"].value as? String, "Keep answers concise.")
    }

    @MainActor
    func testSettingsAPIKeySaveCancelAndRemove() throws {
        let app = launchApp()
        openSettings(in: app)
        app.buttons["settings.webSearch"].tap()
        app.buttons["settings.editAPIKey"].tap()
        let key = app.secureTextFields["settings.apiKeyEditor"]
        XCTAssertTrue(key.waitForExistence(timeout: 3))
        key.tap()
        key.typeText("tvly-settings-test-key")
        app.buttons["Cancel"].tap()
        XCTAssertFalse(app.buttons["settings.testAPIKey"].exists)

        app.buttons["settings.editAPIKey"].tap()
        key.tap()
        key.typeText("tvly-settings-test-key")
        app.buttons["settings.saveAPIKey"].tap()
        XCTAssertTrue(app.buttons["settings.testAPIKey"].waitForExistence(timeout: 3))
        app.terminate()
        let relaunched = launchApp(resetAppState: false)
        openSettings(in: relaunched)
        // A saved key hides the row's "Set Up" prompt.
        XCTAssertFalse(relaunched.buttons["settings.webSearch"].label.contains("Set Up"))
        relaunched.buttons["settings.webSearch"].tap()
        relaunched.buttons["settings.removeAPIKey"].tap()
        relaunched.alerts.buttons["Cancel"].tap()
        XCTAssertTrue(relaunched.buttons["settings.testAPIKey"].exists)
        relaunched.buttons["settings.removeAPIKey"].tap()
        relaunched.alerts.buttons["Remove"].tap()
        XCTAssertFalse(relaunched.buttons["settings.testAPIKey"].exists)
    }

    @MainActor
    func testSettingsMemoryWarningAndResetRequireConfirmation() throws {
        let app = launchApp()
        openSettings(in: app)
        app.buttons["settings.advanced"].tap()
        app.buttons["settings.developer"].tap()
        let memory = app.switches["settings.disableRAMPrecautions"]
        toggle(memory)
        XCTAssertTrue(app.alerts["Disable RAM Precautions?"].waitForExistence(timeout: 3))
        app.alerts.buttons["Cancel"].tap()
        XCTAssertEqual(memory.value as? String, "0")
        toggle(memory)
        app.alerts.buttons["Disable"].tap()
        XCTAssertEqual(memory.value as? String, "1")
        settingsBackButton(in: app).tap()
        app.buttons["settings.reset"].tap()
        app.alerts.buttons["Cancel"].tap()
        app.buttons["settings.developer"].tap()
        XCTAssertEqual(memory.value as? String, "1")
        settingsBackButton(in: app).tap()
        app.buttons["settings.reset"].tap()
        app.alerts.buttons["Reset"].tap()
        app.buttons["settings.developer"].tap()
        XCTAssertEqual(memory.value as? String, "0")
    }

    @MainActor
    func testSettingsPrivacyEmptyStateAndRetentionPicker() throws {
        let app = launchApp()
        openSettings(in: app)
        app.buttons["settings.privacy"].tap()
        XCTAssertFalse(app.buttons["settings.exportChats"].isEnabled)
        XCTAssertFalse(app.buttons["settings.deleteAllChats"].isEnabled)
        XCTAssertFalse(app.buttons["settings.deleteOtherChats"].isEnabled)
        XCTAssertFalse(app.buttons["settings.retention"].exists)
        toggle(app.switches["settings.autoDelete"])
        XCTAssertTrue(app.buttons["settings.retention"].waitForExistence(timeout: 3))
        toggle(app.switches["settings.autoDelete"])
        XCTAssertFalse(app.buttons["settings.retention"].exists)
    }

    @MainActor
    func testAutoDeleteRemovesOldChatsWhenTheAppReturns() throws {
        let fixture = XCUIApplication()
        fixture.launchArguments = [
            "-ui-test-reset-app-state", "-ui-test-web-search-demo",
            "-ui-test-second-chat", "-ui-test-stale-second-chat"
        ]
        fixture.launch()
        XCTAssertTrue(fixture.buttons["message.sources"].waitForExistence(timeout: 5))
        fixture.terminate()

        let app = launchApp(resetAppState: false)
        openSidebar(in: app)
        XCTAssertTrue(app.staticTexts["Second Chat"].waitForExistence(timeout: 5))
        openSettings(in: app)
        app.buttons["settings.privacy"].tap()
        toggle(app.switches["settings.autoDelete"])
        XCTAssertTrue(app.buttons["settings.retention"].waitForExistence(timeout: 3))
        settingsBackButton(in: app).tap()
        app.buttons["settings.done"].tap()

        XCUIDevice.shared.press(.home)
        app.activate()

        if !app.buttons["sidebar.dismiss"].exists { openSidebar(in: app) }
        XCTAssertTrue(app.staticTexts["Web Search Preview"].waitForExistence(timeout: 5))
        let staleRow = app.staticTexts["Second Chat"]
        XCTAssertTrue(staleRow.waitForNonExistence(timeout: 5))
    }

    @MainActor
    func testSettingsDeletionCancelsThenDeletesWithoutASecondPrompt() throws {
        let fixture = XCUIApplication()
        fixture.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo"]
        fixture.launch()
        XCTAssertTrue(fixture.buttons["message.sources"].waitForExistence(timeout: 5))
        fixture.terminate()

        let app = launchApp(resetAppState: false)
        openSettings(in: app)
        app.buttons["settings.privacy"].tap()
        let delete = app.buttons["settings.deleteAllChats"]
        XCTAssertTrue(delete.isEnabled)
        delete.tap()
        XCTAssertTrue(app.alerts["Delete All Chats?"].waitForExistence(timeout: 3))
        app.alerts.buttons["Cancel"].tap()
        XCTAssertTrue(delete.isEnabled)
        delete.tap()
        app.alerts.buttons["Delete"].tap()

        XCTAssertTrue(app.buttons["Settings"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.alerts.firstMatch.exists)
        openSettings(in: app)
        app.buttons["settings.privacy"].tap()
        XCTAssertFalse(app.buttons["settings.deleteAllChats"].isEnabled)
        XCTAssertFalse(app.buttons["settings.exportChats"].isEnabled)
    }

    @MainActor
    func testSettingsExportPresentsShareSheetAfterClosingSettings() throws {
        let fixture = XCUIApplication()
        fixture.launchArguments = ["-ui-test-reset-app-state", "-ui-test-web-search-demo"]
        fixture.launch()
        XCTAssertTrue(fixture.buttons["message.sources"].waitForExistence(timeout: 5))
        fixture.terminate()

        let app = launchApp(resetAppState: false)
        openSettings(in: app)
        app.buttons["settings.privacy"].tap()
        app.buttons["settings.exportChats"].tap()
        XCTAssertTrue(app.otherElements["ActivityListView"].waitForExistence(timeout: 5))
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "settings-export-share-sheet"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
