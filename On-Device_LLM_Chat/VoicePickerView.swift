//
//  VoicePickerView.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI

// MARK: - Voice Picker View

struct VoicePickerView: View {
    @ObservedObject private var speaker = Speaker.shared
    @State private var testingVoiceID: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Group voices by quality
            let premiumVoices = speaker.availableVoices.filter { $0.quality == .premium }
            let enhancedVoices = speaker.availableVoices.filter { $0.quality == .enhanced }
            let hasQualityVoices = !premiumVoices.isEmpty || !enhancedVoices.isEmpty

            if hasQualityVoices {
                VStack(alignment: .leading, spacing: 16) {
                    // Premium voices section
                    if !premiumVoices.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Premium Voices (Recommended)")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundStyle(.purple)

                            ForEach(premiumVoices) { voice in
                                voiceRow(for: voice)
                            }
                        }
                    }

                    // Enhanced voices section
                    if !enhancedVoices.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Enhanced Voices")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundStyle(.blue)

                            ForEach(enhancedVoices) { voice in
                                voiceRow(for: voice)
                            }
                        }
                    }

                    // View all voices navigation link
                    NavigationLink {
                        AllVoicesView(testingVoiceID: $testingVoiceID)
                    } label: {
                        HStack {
                            Label(String(localized: "View All Voices"), systemImage: "list.bullet")
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 7)
                        .padding(.horizontal, 12)
                        .overlay(alignment: .top) {
                            Rectangle()
                                .fill(Color.secondary.opacity(0.2))
                                .frame(height: 0.5)
                                .offset(y: -6) // sits just above the row's content
                        }
                    }
                    .buttonStyle(.plain)
                    .padding(.top, 4)
                }
            }
        }
    }

    private func voiceRow(for voice: VoiceOption) -> some View {
        HStack(spacing: 12) {
            Button(action: {
                Speaker.shared.setVoice(identifier: voice.identifier)
            }) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(voice.name)
                            .font(.body)
                            .foregroundStyle(.primary)

                        HStack(spacing: 6) {
                            Text(voice.language)
                                .font(.caption)
                                .foregroundStyle(.secondary)

                        }
                    }

                    Spacer()

                    if voice.identifier == speaker.currentVoiceID {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    } else {
                        Image(systemName: "circle")
                            .foregroundStyle(.secondary.opacity(0.3))
                    }
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 12)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(voice.identifier == speaker.currentVoiceID ? Color.accentColor.opacity(0.1) : Color.clear)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(voice.identifier == speaker.currentVoiceID ? Color.accentColor : Color.secondary.opacity(0.2), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)

            // Test voice button
            Button(action: {
                if testingVoiceID == voice.identifier {
                    stopVoice()
                } else {
                    testVoice(voice)
                }
            }) {
                Image(systemName: testingVoiceID == voice.identifier ? "pause.circle" : "play.circle")
                    .foregroundStyle(testingVoiceID == voice.identifier ? .secondary : .secondary)
                    .imageScale(.large)
            }
            .buttonStyle(.plain)
            .disabled(testingVoiceID != nil && testingVoiceID != voice.identifier)
        }
    }

    private func testVoice(_ voice: VoiceOption) {
        guard testingVoiceID == nil else { return }
        testingVoiceID = voice.identifier

        // Temporarily set this voice
        Speaker.shared.setVoice(identifier: voice.identifier)

        // Test speech
        Speaker.shared.toggleSpeak(text: "Hello! This is how I sound.")

        // Reset testing state after a delay
        Task {
            try? await Task.sleep(nanoseconds: 3_000_000_000) // 3 seconds
            if testingVoiceID == voice.identifier {
                testingVoiceID = nil
            }
        }
    }

    private func stopVoice() {
        Speaker.shared.stop()
        testingVoiceID = nil
    }
}

// MARK: - All Voices View

struct AllVoicesView: View {
    @ObservedObject private var speaker = Speaker.shared
    @Binding var testingVoiceID: String?

    var body: some View {
        List {
            // Group voices by quality
            let premiumVoices = speaker.availableVoices.filter { $0.quality == .premium }
            let enhancedVoices = speaker.availableVoices.filter { $0.quality == .enhanced }
            let defaultVoices = speaker.availableVoices.filter { $0.quality == .default }

            // Premium voices section
            if !premiumVoices.isEmpty {
                Section {
                    ForEach(premiumVoices) { voice in
                        voiceRow(for: voice)
                    }
                } header: {
                    Text("Premium Voices (Recommended)")
                }
            }

            // Enhanced voices section
            if !enhancedVoices.isEmpty {
                Section {
                    ForEach(enhancedVoices) { voice in
                        voiceRow(for: voice)
                    }
                } header: {
                    Text("Enhanced Voices")
                }
            }

            // Default voices section
            if !defaultVoices.isEmpty {
                Section {
                    ForEach(defaultVoices) { voice in
                        voiceRow(for: voice)
                    }
                } header: {
                    Text("Standard Voices (Not Recommended)")
                }
            }
        }
        .navigationTitle(String(localized: "All Voices"))
        .navigationBarTitleDisplayMode(.inline)
    }

    private func voiceRow(for voice: VoiceOption) -> some View {
        HStack(spacing: 12) {
            Button(action: {
                Speaker.shared.setVoice(identifier: voice.identifier)
            }) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            Text(voice.name)
                                .font(.body)
                                .foregroundStyle(.primary)

                        }

                        Text(voice.language)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    if voice.identifier == speaker.currentVoiceID {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    }
                }
            }
            .buttonStyle(.plain)

            // Test voice button
            Button(action: {
                if testingVoiceID == voice.identifier {
                    stopVoice()
                } else {
                    testVoice(voice)
                }
            }) {
                Image(systemName: testingVoiceID == voice.identifier ? "pause.circle" : "play.circle")
                    .foregroundStyle(testingVoiceID == voice.identifier ? .red : .secondary)
                    .imageScale(.large)
            }
            .buttonStyle(.plain)
            .disabled(testingVoiceID != nil && testingVoiceID != voice.identifier)
        }
    }

    private func testVoice(_ voice: VoiceOption) {
        guard testingVoiceID == nil else { return }
        testingVoiceID = voice.identifier

        // Temporarily set this voice
        Speaker.shared.setVoice(identifier: voice.identifier)

        // Test speech
        Speaker.shared.toggleSpeak(text: "Hello! This is how I sound.")

        // Reset testing state after a delay
        Task {
            try? await Task.sleep(nanoseconds: 3_000_000_000) // 3 seconds
            if testingVoiceID == voice.identifier {
                testingVoiceID = nil
            }
        }
    }

    private func stopVoice() {
        Speaker.shared.stop()
        testingVoiceID = nil
    }
}
