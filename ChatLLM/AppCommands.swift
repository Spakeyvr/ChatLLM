//
//  AppCommands.swift
//  ChatLLM
//
//  Menu bar items and keyboard shortcuts for a hardware keyboard.
//

import SwiftUI

/// What the menu bar can do in the window the user is working in.
struct WindowCommandActions {
    let newChat: () -> Void
    let showSettings: () -> Void
}

extension FocusedValues {
    @Entry var windowCommandActions: WindowCommandActions?
}

/// Replaces the system's New Window and Settings items. Without this, Command-comma
/// opens the app's page in the Settings app whenever no text field has focus.
struct ChatLLMCommands: Commands {
    @FocusedValue(\.windowCommandActions) private var actions

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Chat") { actions?.newChat() }
                .keyboardShortcut("n")
                .disabled(actions == nil)
        }
        CommandGroup(replacing: .appSettings) {
            Button("Settings…") { actions?.showSettings() }
                .keyboardShortcut(",")
                .disabled(actions == nil)
        }
    }
}
