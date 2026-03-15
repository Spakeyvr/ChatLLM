//
//  On_Device_LLM_ChatUITests.swift
//  On-Device_LLM_ChatUITests
//
//  Created by Nevio on 10/24/25.
//

import XCTest

final class On_Device_LLM_ChatUITests: XCTestCase {
    private let firstMLXModelID = "qwen3.5-4b-4bit"
    private let secondMLXModelID = "qwen3.5-2b-4bit"

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

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.

        // In UI tests it is usually best to stop immediately when a failure occurs.
        continueAfterFailure = false

        // In UI tests it’s important to set the initial state - such as interface orientation - required for your tests before they run. The setUp method is a good place to do this.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
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

        XCTAssertTrue(app.otherElements["onboarding.mlx.progress.\(firstMLXModelID)"].waitForExistence(timeout: 5))

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

        let firstProgress = app.otherElements["onboarding.mlx.progress.\(firstMLXModelID)"]
        XCTAssertTrue(firstProgress.waitForExistence(timeout: 5))

        let secondModelButton = app.buttons["onboarding.mlx.model.\(secondMLXModelID)"]
        XCTAssertTrue(secondModelButton.waitForExistence(timeout: 5))
        secondModelButton.tap()

        XCTAssertTrue(firstProgress.waitForExistence(timeout: 2))
        XCTAssertFalse(app.otherElements["onboarding.mlx.progress.\(secondMLXModelID)"].exists)
    }

    @MainActor
    func testLaunchPerformance() throws {
        // This measures how long it takes to launch your application.
        measure(metrics: [XCTApplicationLaunchMetric()]) {
            XCUIApplication().launch()
        }
    }
}
