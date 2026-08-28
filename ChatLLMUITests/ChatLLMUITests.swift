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
    func testLaunchPerformance() throws {
        measure(metrics: [XCTApplicationLaunchMetric()]) {
            let app = XCUIApplication()
            app.launchArguments.append("-ui-test-reset-app-state")
            app.launch()
        }
    }

    @MainActor
    private func openSettings(in app: XCUIApplication) {
        let skip = app.buttons["onboarding.skip"]
        if skip.waitForExistence(timeout: 2) { skip.tap() }
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
    func testSettingsAppearanceAndDetailNavigation() throws {
        let app = launchApp()
        openSettings(in: app)
        let systemBrightness = try settingsBackgroundBrightness(in: app)
        let oppositeAppearance = systemBrightness > 128 ? "Dark" : "Light"
        captureSettingsScreenshot("settings-light", in: app)
        app.buttons["settings.appearance"].tap()
        app.buttons["settings.colorScheme"].tap()
        app.buttons[oppositeAppearance].tap()
        captureSettingsScreenshot("appearance-dark", in: app)
        app.sliders["settings.messageTextSize"].adjust(toNormalizedSliderPosition: 1)
        XCTAssertEqual(app.sliders["settings.messageTextSize"].value as? String, "22 points")
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertTrue(app.buttons["settings.appearance"].label.contains(oppositeAppearance))
        XCTAssertGreaterThan(abs(try settingsBackgroundBrightness(in: app) - systemBrightness), 100)
        captureSettingsScreenshot("settings-dark", in: app)

        app.buttons["settings.appearance"].tap()
        app.buttons["settings.colorScheme"].tap()
        app.buttons["System"].tap()
        app.buttons["Reset Text Size"].tap()
        XCTAssertEqual(app.sliders["settings.messageTextSize"].value as? String, "16 points")
        captureSettingsScreenshot("appearance-light", in: app)
        app.navigationBars.buttons.element(boundBy: 0).tap()
        XCTAssertEqual(try settingsBackgroundBrightness(in: app), systemBrightness, accuracy: 5)

        for (id, title) in [("chat", "Chat"), ("webSearch", "Web Search"), ("privacy", "Privacy & Data"),
                            ("about", "About & Support"), ("advanced", "Advanced")] {
            app.buttons["settings.\(id)"].tap()
            XCTAssertTrue(app.navigationBars[title].waitForExistence(timeout: 3))
            captureSettingsScreenshot("settings-\(id)", in: app)
            app.navigationBars.buttons.element(boundBy: 0).tap()
        }
        app.buttons["settings.advanced"].tap()
        for (id, title) in [("mlx", "On-Device Models"), ("imageAnalysis", "Image Analysis"), ("developer", "Developer")] {
            app.buttons["settings.\(id)"].tap()
            XCTAssertTrue(app.navigationBars[title].waitForExistence(timeout: 3))
            captureSettingsScreenshot("settings-\(id)", in: app)
            if id == "mlx" {
                app.buttons["About RotorQuant"].tap()
                XCTAssertTrue(app.navigationBars["RotorQuant"].waitForExistence(timeout: 3))
                captureSettingsScreenshot("settings-rotorQuant", in: app)
                app.navigationBars.buttons.element(boundBy: 0).tap()
            }
            app.navigationBars.buttons.element(boundBy: 0).tap()
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
            app.navigationBars.buttons.element(boundBy: 0).tap()
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
        XCTAssertTrue(relaunched.buttons["settings.webSearch"].label.contains("Configured"))
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
        app.navigationBars.buttons.element(boundBy: 0).tap()
        app.buttons["settings.reset"].tap()
        app.alerts.buttons["Cancel"].tap()
        app.buttons["settings.developer"].tap()
        XCTAssertEqual(memory.value as? String, "1")
        app.navigationBars.buttons.element(boundBy: 0).tap()
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
