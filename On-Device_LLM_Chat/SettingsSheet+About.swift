//
//  SettingsSheet+About.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI
import UIKit

// MARK: - Feature Row Helper

struct FeatureRow: View {
    let icon: String
    let title: String
    let description: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(.blue)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                Text(description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}

// MARK: - SettingsSheet About Extension

extension SettingsSheet {
    @ViewBuilder
    var aboutSection: some View {
        // MARK: Support & Feedback
        Section(
            header: Text(String(localized: "Support & Feedback"))
                .textCase(nil)
                .font(.headline)
                .padding(.top, 4)
                .padding(.bottom, -4),
        ) {
            Button {
                openDiscordInvite()
            } label: {
                Label {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(String(localized: "Join Discord Community"))
                            .foregroundStyle(.primary)
                        Text(String(localized: "Get help and share feedback"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                            .padding(1)
                    }
                } icon: {
                    Image(systemName: "bubble.left.and.bubble.right.fill")
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(.blue)
                }
            }
            .buttonStyle(.plain)
            .listRowInsets(EdgeInsets(top: 4, leading: 16, bottom: 4, trailing: 16))
        }

        // MARK: About
        Section(
            header: Text(String(localized: "About"))
                .textCase(nil)
                .font(.headline)
                .padding(.top, 4)
                .padding(.bottom, -1)
        ) {
            VStack(alignment: .leading, spacing: 10) {
                // App version info
                VStack(alignment: .leading, spacing: 5) {
                    Text(String(localized: "Beta Build"))
                        .font(.headline)
                    Text(String(localized: "This is a beta version. Please expect bugs and report any issues on Discord or through the Support Email."))
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Divider()

                // Email contact
                VStack(alignment: .leading, spacing: 5) {
                    Text(String(localized: "Email Support"))
                        .font(.headline)
                    Text("Have any issues and can't use Discord? Contact me at chatllm@icloud.com")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .tint(.blue)
                        .onTapGesture {
                            if let url = URL(string: "mailto:chatllm@icloud.com") {
                                UIApplication.shared.open(url)
                            }
                        }
                }
                Divider()

                // Features list
                Text(String(localized: "Key Features"))
                    .font(.headline)
                    .padding(.bottom, 4)

                FeatureRow(
                    icon: "cpu",
                    title: String(localized: "Apple Intelligence"),
                    description: String(localized: "Powered by Apple's on-device large language model")
                )

                FeatureRow(
                    icon: "brain.head.profile",
                    title: String(localized: "Reasoning Mode"),
                    description: String(localized: "Advanced step-by-step problem solving")
                )

                FeatureRow(
                    icon: "sparkles",
                    title: String(localized: "Smart Reasoning"),
                    description: String(localized: "Automatically enables reasoning for complex queries")
                )

                FeatureRow(
                    icon: "text.alignleft",
                    title: String(localized: "Custom System Prompts"),
                    description: String(localized: "Personalize the assistant's behavior and tone")
                )

                FeatureRow(
                    icon: "globe",
                    title: String(localized: "Tavily Search"),
                    description: String(localized: "Web search integration for real-time information")
                )

                FeatureRow(
                    icon: "viewfinder.circle",
                    title: String(localized: "Object Detection"),
                    description: String(localized: "Apple Vision Framework object recognition")
                )
            }
            .padding(.top, 0)
            .padding(.bottom, 4)
        }
    }
}
