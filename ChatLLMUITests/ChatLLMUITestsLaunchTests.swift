//
//  ChatLLMUITestsLaunchTests.swift
//  ChatLLMUITests
//
//  Created by Nevio on 10/24/25.
//

import XCTest

final class ChatLLMUITestsLaunchTests: XCTestCase {

    override class var runsForEachTargetApplicationUIConfiguration: Bool {
        true
    }

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    func testLaunch() throws {
        let app = XCUIApplication()
        app.launchArguments.append("-ui-test-reset-app-state")
        app.launch()

        XCTAssertTrue(app.otherElements["onboarding.root"].waitForExistence(timeout: 5))

        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "Launch Screen"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
