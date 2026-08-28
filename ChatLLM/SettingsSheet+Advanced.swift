import SwiftUI

struct AdvancedSettingsView: View {
    @Binding var settings: AppSettingsDraft
    @State private var showingResetConfirmation = false

    var body: some View {
        Form {
            Section {
                NavigationLink {
                    MLXSettingsView(settings: $settings)
                } label: {
                    Label("On-Device Models", systemImage: "cpu")
                }
                .accessibilityIdentifier("settings.mlx")
                NavigationLink {
                    ImageAnalysisSettingsView(settings: $settings)
                } label: {
                    Label("Image Analysis", systemImage: "viewfinder")
                }
                .accessibilityIdentifier("settings.imageAnalysis")
                NavigationLink {
                    DeveloperSettingsView(settings: $settings)
                } label: {
                    Label("Developer", systemImage: "hammer")
                }
                .accessibilityIdentifier("settings.developer")
            } footer: {
                Text("Fine-tune local models and diagnostics. The defaults are recommended for most people.")
            }
            Section {
                Button("Reset All Settings", role: .destructive) { showingResetConfirmation = true }
                    .accessibilityIdentifier("settings.reset")
            } footer: {
                Text("Restore default preferences and remove the saved web search key. Chats and downloaded models are kept.")
            }
        }
        .navigationTitle("Advanced")
        .navigationBarTitleDisplayMode(.inline)
        .alert("Reset All Settings?", isPresented: $showingResetConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Reset", role: .destructive) { settings.resetToDefaults() }
        } message: {
            Text("This restores all preferences, clears response preferences, and removes the saved web search key. Chats and downloaded models will not be deleted.")
        }
    }
}

private struct MLXSettingsView: View {
    @Binding var settings: AppSettingsDraft

    private var deviceMaximum: Int { MLXDeviceSupportProfile.current.maxContextWindowTokens }
    private var contextTokens: Binding<Double> {
        Binding(
            get: { Double(settings.mlxContextWindowTokens <= 0
                          ? deviceMaximum : min(max(settings.mlxContextWindowTokens, 512), deviceMaximum)) },
            set: { settings.mlxContextWindowTokens = Int($0.rounded()) }
        )
    }
    private var outputTokens: Binding<Double> {
        Binding(get: { Double(max(512, settings.mlxMaxOutputTokens)) },
                set: { settings.mlxMaxOutputTokens = Int($0.rounded()) })
    }

    var body: some View {
        Form {
            Section {
                LabeledContent("Context Window", value: String(localized: "\(Int(contextTokens.wrappedValue)) tokens"))
                Slider(value: contextTokens, in: 512...Double(deviceMaximum), step: 512) {
                    Text("Context Window")
                }
                .accessibilityValue(String(localized: "\(Int(contextTokens.wrappedValue)) tokens"))
            } footer: {
                Text("How much conversation a local MLX model can use. This device supports up to \(deviceMaximum) tokens.")
            }

            Section {
                Toggle("Unlimited Output", isOn: Binding(
                    get: { settings.mlxMaxOutputTokens == 0 },
                    set: { settings.mlxMaxOutputTokens = $0 ? 0 : 1024 }
                ))
                .accessibilityIdentifier("settings.unlimitedOutput")
                if settings.mlxMaxOutputTokens != 0 {
                    LabeledContent("Maximum Output", value: String(localized: "\(settings.mlxMaxOutputTokens) tokens"))
                    Slider(value: outputTokens, in: 512...1024, step: 32) {
                        Text("Maximum Output")
                    }
                    .accessibilityValue(String(localized: "\(settings.mlxMaxOutputTokens) tokens"))
                }
            } footer: {
                Text("Shorter responses usually use less memory and finish sooner. Unlimited removes the output cap.")
            }

            Section {
                Toggle("Reduce Repetition", isOn: Binding(
                    get: { settings.mlxRepetitionPenalty > 1 },
                    set: { settings.mlxRepetitionPenalty = $0 ? 1.1 : 1 }
                ))
                .accessibilityIdentifier("settings.reduceRepetition")
                if settings.mlxRepetitionPenalty > 1 {
                    LabeledContent("Penalty", value: settings.mlxRepetitionPenalty.formatted(.number.precision(.fractionLength(2))))
                    Slider(value: $settings.mlxRepetitionPenalty, in: 1.05...1.5, step: 0.05) {
                        Text("Repetition Penalty")
                    }
                }
            } footer: {
                Text("Discourage repeated words and phrases. Increase gradually if a model loops or echoes itself.")
            }

            Section {
                Toggle("RotorQuant", isOn: $settings.mlxEnableRotorQuant)
                    .accessibilityHint(SettingsSheet.mlxRotorQuantAccessibilityHint)
                    .accessibilityIdentifier("settings.rotorQuant")
                NavigationLink("About RotorQuant") {
                    Form {
                        Section {
                            Text(SettingsSheet.mlxRotorQuantExperimentalMessage)
                        } header: {
                            Text(SettingsSheet.mlxRotorQuantExperimentalTitle)
                        }
                        Section("How It Works") {
                            Text(SettingsSheet.mlxRotorQuantInfoMessage)
                        }
                    }
                    .navigationTitle("RotorQuant")
                    .navigationBarTitleDisplayMode(.inline)
                }
            } header: {
                Text("Experimental")
            } footer: {
                Text("Compresses the MLX conversation cache to reduce memory use. This is an early beta; speed and stability may vary.")
            }
        }
        .navigationTitle("On-Device Models")
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct ImageAnalysisSettingsView: View {
    @Binding var settings: AppSettingsDraft

    var body: some View {
        Form {
            Section {
                LabeledContent("Confidence Threshold", value: settings.visionConfidenceThreshold.formatted(.percent.precision(.fractionLength(0))))
                Slider(value: $settings.visionConfidenceThreshold, in: 0.1...0.9, step: 0.05) {
                    Text("Confidence Threshold")
                }
                .accessibilityValue(settings.visionConfidenceThreshold.formatted(.percent.precision(.fractionLength(0))))
            } footer: {
                Text("Higher confidence reduces false detections but may miss objects. Used only by the Apple Vision fallback when the selected model cannot analyze images directly.")
            }
        }
        .navigationTitle("Image Analysis")
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct DeveloperSettingsView: View {
    @Binding var settings: AppSettingsDraft
    @State private var showingMemoryWarning = false

    var body: some View {
        Form {
            Section {
                Toggle("Show Output Stats", isOn: $settings.developerModeEnabled)
                    .accessibilityIdentifier("settings.outputStats")
            } footer: {
                Text("Add an action beneath assistant messages to view raw output and generation diagnostics.")
            }
            Section {
                Toggle("Disable RAM Precautions", isOn: Binding(
                    get: { settings.disableRAMPrecautions },
                    set: { enabled in
                        if enabled { showingMemoryWarning = true }
                        else { settings.disableRAMPrecautions = false }
                    }
                ))
                .accessibilityIdentifier("settings.disableRAMPrecautions")
            } footer: {
                Text("Allow models and tool calls that exceed this device’s memory limits. This can cause crashes or severe slowdowns.")
            }
        }
        .navigationTitle("Developer")
        .navigationBarTitleDisplayMode(.inline)
        .alert("Disable RAM Precautions?", isPresented: $showingMemoryWarning) {
            Button("Cancel", role: .cancel) {}
            Button("Disable", role: .destructive) { settings.disableRAMPrecautions = true }
        } message: {
            Text("Models that need more memory than your device has can crash the app or make it extremely slow. Only disable these safeguards if you understand the risks.")
        }
    }
}
