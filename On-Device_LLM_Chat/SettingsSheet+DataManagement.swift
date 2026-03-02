//
//  SettingsSheet+DataManagement.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI
import UIKit

extension SettingsSheet {

    // MARK: - Data Management Section

    @ViewBuilder
    var dataManagementSection: some View {
        Section(header: Text(String(localized: "Data Management"))) {
            // Export conversations
            Button {
                exportConversations()
            } label: {
                Label {
                    Text(String(localized: "Export All Chats"))
                        .foregroundStyle(.primary)
                } icon: {
                    Image(systemName: "square.and.arrow.up")
                        .symbolRenderingMode(.monochrome)
                        .foregroundStyle(.blue)
                }
            }
            .buttonStyle(.plain)
            .disabled(!hasChats)

            // Make text white (primary) and keep trash icon red by avoiding destructive role.
            Button {
                if confirmBeforeDeletingChats {
                    pendingAction = .deleteAll
                } else {
                    onDeleteAll()
                }
            } label: {
                Label {
                    Text(String(localized: "Delete All Chats"))
                        .foregroundStyle(.primary)
                } icon: {
                    Image(systemName: "trash")
                        .symbolRenderingMode(.monochrome)
                        .foregroundStyle(.red)
                }
            }
            .buttonStyle(.plain)
            .disabled(!hasChats)

            // Respect confirmation preference and gray out when only one or no chats
            Button {
                if confirmBeforeDeletingChats {
                    pendingAction = .deleteAllExceptCurrent
                } else {
                    onDeleteAllExceptCurrent()
                }
            } label: {
                Label {
                    Text(String(localized: "Delete All Except Current"))
                        .foregroundStyle(.primary)
                } icon: {
                    Image(systemName: "trash")
                        .symbolRenderingMode(.monochrome)
                        .foregroundStyle(.red)
                }
            }
            .buttonStyle(.plain)
            .disabled(!canDeleteAllExceptCurrent)

            Button {
                pendingAction = .resetSettings
            } label: {
                Label {
                    Text(String(localized: "Reset Settings"))
                        .foregroundStyle(.primary)
                } icon: {
                    Image(systemName: "arrow.counterclockwise")
                        .symbolRenderingMode(.monochrome)
                        .foregroundStyle(.red)
                }
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Actions

    func deletePreset(_ preset: SystemPromptPreset) {
        var items = customPresets
        items.removeAll { $0.id == preset.id }
        updateCustomPresets(items)
    }

    func saveCurrentAsPreset() {
        let trimmed = defaultSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        editingPreset = SystemPromptPreset(name: "", text: trimmed)
        isEditingSheetPresented = true
    }

    func exportConversations() {
        onExportChats()
    }

    func resetSettings() {
        // Restore defaults
        appAppearance = "system"
        appLanguage = "en"
        defaultSystemPrompt = ""
        customPresetsData = Data()
        tavilyApiKey = ""

        // QoL defaults
        sendOnReturn = false
        enableHaptics = true
        reasoningModeDefault = false
        messageFontSize = 16.0
        autoDeleteOldChats = false
        autoDeleteDays = 30

        // Vision Framework defaults
        visionConfidenceThreshold = 0.5
    }

    func openDiscordInvite() {
        let discordInviteURL = "https://discord.gg/PGNCC4Vy7T"

        guard let url = URL(string: discordInviteURL) else {
            print("Invalid Discord invite URL")
            return
        }

        UIApplication.shared.open(url, options: [:]) { success in
            if !success {
                print("Failed to open Discord invite")
            }
        }
    }

    func openTavilySignUp() {
        let tavilyURL = "https://tavily.com/"

        guard let url = URL(string: tavilyURL) else {
            print("Invalid Tavily URL")
            return
        }

        UIApplication.shared.open(url, options: [:]) { success in
            if !success {
                print("Failed to open Tavily")
            }
        }
    }
}
