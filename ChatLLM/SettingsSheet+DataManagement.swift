//
//  SettingsSheet+DataManagement.swift
//  ChatLLM
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
            Button {
                exportConversations()
            } label: {
                Label {
                    Text(String(localized: "Export All Chats"))
                        .foregroundStyle(.primary)
                } icon: {
                    Image(systemName: "square.and.arrow.up")
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(.blue)
                }
            }
            .buttonStyle(.plain)
            .disabled(!hasChats)

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
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(.red)
                }
            }
            .buttonStyle(.plain)
            .disabled(!hasChats)

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
                        .symbolRenderingMode(.hierarchical)
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
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(.red)
                }
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Actions

    func exportConversations() {
        onExportChats()
    }

    func resetSettings() {
        settings.resetToDefaults()
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
