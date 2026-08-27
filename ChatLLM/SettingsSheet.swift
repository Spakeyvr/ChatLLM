import SwiftUI

struct SettingsSheet: View {
    static let mlxRotorQuantInfoMessage = String(localized: "Enabled by default for persistent MLX text and image chats, RotorQuant uses IsoQuant block rotations with 3-bit keys, 2-bit values, exact prefill buffering, and per-layer deterministic rotation parameters to reduce memory use. Tool and low-memory turns use safer uncompressed or bounded caches.")
    static let mlxRotorQuantAccessibilityHint = String(localized: "Enabled by default for supported persistent MLX text and image chats. Tool and low-memory turns use safer cache modes.")
    static let mlxRotorQuantExperimentalTitle = String(localized: "RotorQuant Experimental")
    static let mlxRotorQuantExperimentalMessage = String(localized: "RotorQuant is very early and in beta. It can reduce KV-cache memory, but speed and stability are still being tuned.")

    @Binding var settings: AppSettingsDraft

    let hasChats: Bool
    let canDeleteAllExceptCurrent: Bool
    let onDeleteAll: () -> Void
    let onDeleteAllExceptCurrent: () -> Void
    let onExportChats: () -> Void

    let confirmBeforeDeletingChats = true
    private let showCharacterCount = true

    @State private var areChatPreferencesFocused = false
    @State private var tavilyKeyValidation = TavilyKeyValidationModel()

    enum PendingAction: Identifiable {
        case deleteAll
        case deleteAllExceptCurrent
        case resetSettings

        var id: String {
            switch self {
            case .deleteAll: return "deleteAll"
            case .deleteAllExceptCurrent: return "deleteAllExceptCurrent"
            case .resetSettings: return "resetSettings"
            }
        }
    }

    @State var pendingAction: PendingAction?
    @State private var showingRAMPrecautionsWarning = false

    var body: some View {
        NavigationStack {
            Form {
                Section(header: Text("General")) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 6) {
                            Text(String(localized: "Appearance"))
                        }

                        Picker("", selection: $settings.appAppearance) {
                            appearanceOption(localized: "System", icon: "circle.lefthalf.filled", tag: "system")
                            appearanceOption(localized: "Light", icon: "sun.max", tag: "light")
                            appearanceOption(localized: "Dark", icon: "moon", tag: "dark")
                        }
                        .pickerStyle(.segmented)
                        .accessibilityLabel(String(localized: "App Appearance"))
                    }

                    Picker(selection: $settings.appLanguage) {
                        Text(String(localized: "English")).tag("en")
                    } label: {
                        HStack(spacing: 6) {
                            Text(String(localized: "Language"))
                        }
                    }
                    .disabled(true)
                }

                Section(header: Text("Behavior")) {
                    Toggle(isOn: $settings.sendOnReturn) {
                        HStack(spacing: 6) {
                            Text(String(localized: "Send on Return"))
                            InfoButton(
                                title: String(localized: "Send on Return"),
                                message: String(localized: "When enabled, pressing Return sends the message instead of inserting a new line.")
                            )
                        }
                    }
                    .accessibilityHint(String(localized: "Pressing Return sends the message instead of inserting a new line."))

                    Toggle(isOn: $settings.enableHaptics) {
                        HStack(spacing: 6) {
                            Text(String(localized: "Enable Haptics"))
                            InfoButton(
                                title: String(localized: "Enable Haptics"),
                                message: String(localized: "Provide subtle vibration feedback on key actions, when supported by your device.")
                            )
                        }
                    }

                    Toggle(isOn: $settings.reasoningModeDefault) {
                        HStack(spacing: 6) {
                            Text(String(localized: "Reasoning Mode by Default"))
                            InfoButton(
                                title: String(localized: "Reasoning Mode by Default"),
                                message: String(localized: "When enabled, new conversations will start with reasoning mode activated.")
                            )
                        }
                    }

                    NavigationLink {
                        contextAndTokensSettingsView
                    } label: {
                        Text(String(localized: "MLX Settings"))
                    }

                    Toggle(isOn: $settings.mlxEnableRotorQuant) {
                        HStack(spacing: 6) {
                            Text(String(localized: "RotorQuant (MLX Only)"))
                            ExperimentalBadgeButton(
                                title: Self.mlxRotorQuantExperimentalTitle,
                                message: Self.mlxRotorQuantExperimentalMessage
                            )
                            InfoButton(
                                title: String(localized: "RotorQuant (MLX Only)"),
                                message: Self.mlxRotorQuantInfoMessage
                            )
                        }
                    }
                    .accessibilityHint(Self.mlxRotorQuantAccessibilityHint)

                    NavigationLink {
                        devSettingsView
                    } label: {
                        Text(String(localized: "Dev Settings"))
                    }
                }

                Section(
                    header: Text(String(localized: "Display")),
                    footer: Text(String(localized: "Adjust the font size for messages in conversations."))
                ) {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack(spacing: 6) {
                            Label(String(localized: "Message Font Size"), systemImage: "textformat.size")
                            InfoButton(
                                title: String(localized: "Message Font Size"),
                                message: String(localized: "Adjust the size of text in conversation messages. Affects both user and assistant messages.")
                            )
                        }

                        fontSizeSlider
                    }
                    .padding(.vertical, 4)
                }

                Section(
                    header: Text(String(localized: "Object Detection")),
                    footer: Text(String(localized: "Used as a compatibility fallback for image analysis when native model-based vision is unavailable."))
                ) {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack(spacing: 6) {
                            Label(String(localized: "Confidence Threshold"), systemImage: "chart.bar.fill")
                            InfoButton(
                                title: String(localized: "Confidence Threshold"),
                                message: String(localized: "Minimum confidence (10–90%) for Vision fallback object detection. Higher values reduce false positives but may miss valid objects.")
                            )
                        }

                        confidenceThresholdSlider
                    }
                    .padding(.vertical, 4)

                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.caption2)
                            .foregroundStyle(.green)
                        Text("Vision Framework fallback enabled")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section(
                    header: Text(String(localized: "Tavily Search")),
                    footer: Text(String(localized: "Enter your Tavily API key to enable web search capabilities in conversations. Sign up at tavily.com for a free API key."))
                ) {
                    HStack(spacing: 6) {
                        Label(String(localized: "API Key"), systemImage: "key")
                        InfoButton(
                            title: String(localized: "Tavily API Key"),
                            message: String(localized: "Tavily provides AI-optimized search results. Paste your API key here to integrate web search into the assistant's responses. Obtain one from https://tavily.com.")
                        )
                    }

                    SecureField(String(localized: "Enter your Tavily API key"), text: $settings.tavilyApiKey)
                        .textContentType(.password)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .accessibilityLabel(String(localized: "Tavily API Key"))
                        .onChange(of: settings.tavilyApiKey) {
                            tavilyKeyValidation.reset()
                        }

                    HStack {
                        Button {
                            openTavilySignUp()
                        } label: {
                            Label(
                                settings.tavilyApiKey.isEmpty
                                    ? String(localized: "Get API Key")
                                    : String(localized: "Manage Key"),
                                systemImage: "arrow.up.right"
                            )
                        }
                        .buttonStyle(.borderless)

                        Spacer()

                        Button {
                            validateTavilyKey()
                        } label: {
                            if tavilyKeyValidation.status == .checking {
                                ProgressView()
                                    .controlSize(.small)
                            } else {
                                Text(String(localized: "Test Key"))
                            }
                        }
                        .buttonStyle(.bordered)
                        .disabled(
                            settings.tavilyApiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
                            tavilyKeyValidation.status == .checking
                        )
                    }

                    switch tavilyKeyValidation.status {
                    case .idle, .checking:
                        EmptyView()
                    case .valid:
                        Label(String(localized: "Key is valid"), systemImage: "checkmark.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.green)
                    case .invalid(let message):
                        Label(message, systemImage: "exclamationmark.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }

                Section(
                    header: Text(String(localized: "Privacy")),
                    footer: settings.autoDeleteOldChats
                        ? Text("Conversations older than \(settings.autoDeleteDays) days will be automatically deleted.")
                        : Text(String(localized: "Automatically delete conversations after a specified time period."))
                ) {
                    Toggle(isOn: $settings.autoDeleteOldChats) {
                        HStack(spacing: 6) {
                            Text(String(localized: "Auto-Delete Old Chats"))
                            InfoButton(
                                title: String(localized: "Auto-Delete Old Chats"),
                                message: String(localized: "Automatically removes conversations older than the specified number of days.")
                            )
                        }
                    }

                    if settings.autoDeleteOldChats {
                        Picker(selection: $settings.autoDeleteDays) {
                            Text(String(localized: "7 days")).tag(7)
                            Text(String(localized: "14 days")).tag(14)
                            Text(String(localized: "30 days")).tag(30)
                            Text(String(localized: "60 days")).tag(60)
                            Text(String(localized: "90 days")).tag(90)
                        } label: {
                            Text(String(localized: "Delete After"))
                        }
                        .pickerStyle(.menu)
                    }
                }

                Section(
                    header:
                        HStack(spacing: 6) {
                            Text(String(localized: "Chat Preferences"))
                            InfoButton(
                                title: String(localized: "Chat Preferences"),
                                message: String(localized: "Describe how you prefer responses to be written. These preferences are included with your requests in new chats and do not replace the app’s system instructions.")
                            )
                        }
                        .padding(.top, 8),
                    footer: Text(String(localized: "Applied to new chats. Existing conversations keep the preferences they started with."))
                ) {
                    ChatPreferencesEditor(
                        text: $settings.chatPreferences,
                        placeholder: String(localized: "For example: Keep answers concise, use metric units, and explain technical terms…"),
                        isFocused: $areChatPreferencesFocused,
                        minHeight: 140,
                        maxHeight: 240
                    )

                    HStack {
                        if showCharacterCount {
                            let count = settings.chatPreferences.count
                            let counter = String(localized: "characters")
                            Text("\(count) \(counter)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        if !settings.chatPreferences.isEmpty {
                            Button(role: .destructive) {
                                settings.chatPreferences = ""
                            } label: {
                                Label(String(localized: "Clear"), systemImage: "trash")
                                    .font(.caption)
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel(String(localized: "Clear Chat Preferences"))
                        }
                    }
                    .contextMenu {
                        if !settings.chatPreferences.isEmpty {
                            Button {
                                UIPasteboard.general.string = settings.chatPreferences
                            } label: {
                                Label(String(localized: "Copy"), systemImage: "doc.on.doc")
                            }
                        }
                        Button {
                            if let str = UIPasteboard.general.string {
                                settings.chatPreferences = str
                            }
                        } label: {
                            Label(String(localized: "Paste"), systemImage: "doc.on.clipboard")
                        }
                    }
                }

                aboutSection
                dataManagementSection
            }
            .scrollDismissesKeyboard(.interactively)
            .navigationTitle(String(localized: "Settings"))
            .navigationBarTitleDisplayMode(.inline)
        }
        .listSectionSpacing(.compact)
        .alert(item: $pendingAction) { action in
            switch action {
            case .deleteAll:
                return Alert(
                    title: Text(String(localized: "Delete All Chats")),
                    message: Text(String(localized: "This will permanently remove all chats. This action cannot be undone.")),
                    primaryButton: .destructive(Text(String(localized: "Delete All")), action: onDeleteAll),
                    secondaryButton: .cancel()
                )
            case .deleteAllExceptCurrent:
                return Alert(
                    title: Text(String(localized: "Delete All Except Current")),
                    message: Text(String(localized: "This will permanently remove all chats except the current conversation.")),
                    primaryButton: .destructive(Text(String(localized: "Delete")), action: onDeleteAllExceptCurrent),
                    secondaryButton: .cancel()
                )
            case .resetSettings:
                return Alert(
                    title: Text(String(localized: "Reset Settings")),
                    message: Text(String(localized: "Restore appearance, language, editor and behavior settings to defaults and remove custom presets? This cannot be undone.")),
                    primaryButton: .destructive(Text(String(localized: "Reset"))) {
                        resetSettings()
                    },
                    secondaryButton: .cancel()
                )
            }
        }
        .onDisappear {
            tavilyKeyValidation.reset()
        }
    }

    private var effectiveContextWindowTokens: Int {
        let deviceMaximum = MLXDeviceSupportProfile.current.maxContextWindowTokens
        let minimum = 512
        let storedValue = settings.mlxContextWindowTokens

        if storedValue <= 0 {
            return deviceMaximum
        }

        return min(max(storedValue, minimum), deviceMaximum)
    }

    private var contextAndTokensSettingsView: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: 12) {
                    HStack(spacing: 6) {
                        Label(String(localized: "Context Window"), systemImage: "rectangle.compress.vertical")
                        InfoButton(
                            title: String(localized: "Context Window"),
                            message: String(localized: "Caps how much recent conversation is sent to on-device MLX models. The maximum is based on device RAM and this slider cannot exceed that device limit.")
                        )
                    }

                    contextWindowSlider
                }
                .padding(.vertical, 4)

                VStack(alignment: .leading, spacing: 12) {
                    HStack(spacing: 6) {
                        Label(String(localized: "Max Output Tokens"), systemImage: "slider.horizontal.3")
                        InfoButton(
                            title: String(localized: "Max Output Tokens"),
                            message: String(localized: "Limits the maximum number of tokens generated by on-device MLX models. Lower values usually reduce memory use and response time. Turn on Unlimited to remove this cap.")
                        )
                    }

                    maxOutputTokensSlider
                }
                .padding(.vertical, 4)

                VStack(alignment: .leading, spacing: 12) {
                    HStack(spacing: 6) {
                        Label(String(localized: "Repetition Penalty"), systemImage: "text.word.spacing")
                        InfoButton(
                            title: String(localized: "Repetition Penalty"),
                            message: String(localized: "Discourages on-device MLX models from repeating the same tokens too often. Leave it Off to keep the current behavior, or raise it gradually if the model loops or echoes itself.")
                        )
                    }

                    repetitionPenaltySlider
                }
                .padding(.vertical, 4)
            }
        }
        .navigationTitle(String(localized: "Context and tokens"))
        .navigationBarTitleDisplayMode(.inline)
    }

    private var devSettingsView: some View {
        let ramPrecautionsBinding = Binding<Bool>(
            get: { settings.disableRAMPrecautions },
            set: { newValue in
                if newValue {
                    showingRAMPrecautionsWarning = true
                } else {
                    settings.disableRAMPrecautions = false
                }
            }
        )

        return Form {
            Section(
                footer: Text(String(localized: "Disabling RAM precautions can cause crashes or severe slowdowns when a model needs more memory than the device has."))
            ) {
                Toggle(isOn: $settings.developerModeEnabled) {
                    HStack(spacing: 6) {
                        Text(String(localized: "Show Output Stats"))
                        InfoButton(
                            title: String(localized: "Show Output Stats"),
                            message: String(localized: "Shows an extra developer action under assistant messages with raw output and generation diagnostics.")
                        )
                    }
                }
                .accessibilityHint(String(localized: "Shows raw output and generation diagnostics for assistant messages."))

                Toggle(isOn: ramPrecautionsBinding) {
                    HStack(spacing: 6) {
                        Text(String(localized: "Disable RAM Precautions"))
                        InfoButton(
                            title: String(localized: "Disable RAM Precautions"),
                            message: String(localized: "Removes the RAM-based restrictions that block certain models and tool calls on lower-memory devices. Models may crash or run very slowly if the device runs out of memory.")
                        )
                    }
                }
                .accessibilityHint(String(localized: "Removes RAM-based restrictions on models and tool calls. May cause crashes on low-memory devices."))
            }
        }
        .navigationTitle(String(localized: "Dev Settings"))
        .navigationBarTitleDisplayMode(.inline)
        .alert(String(localized: "Disable RAM Precautions?"), isPresented: $showingRAMPrecautionsWarning) {
            Button(String(localized: "Disable"), role: .destructive) {
                settings.disableRAMPrecautions = true
            }
            Button(String(localized: "Cancel"), role: .cancel) {}
        } message: {
            Text(String(localized: "These restrictions exist because models that need more RAM than your device has can crash the app or make it extremely slow. Only disable them if you know what you are doing."))
        }
    }

    @ViewBuilder
    private var fontSizeSlider: some View {
        HStack(spacing: 12) {
            Text("A")
                .font(.caption)
                .foregroundStyle(.primary.opacity(0.6))

            Slider(value: $settings.messageFontSize, in: 12...22, step: 1)
                .tint(.accentColor)
                .shadow(color: .black.opacity(0.15), radius: 2, x: 0, y: 1)
                .accessibilityLabel(String(localized: "Message Font Size"))
                .accessibilityValue("\(Int(settings.messageFontSize)) points")

            Text("A")
                .font(.title3)
                .foregroundStyle(.primary.opacity(0.6))

            fontSizeLabel

            Spacer(minLength: 8)

            Button {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                    settings.messageFontSize = 16.0
                }
            } label: {
                Image(systemName: "arrow.counterclockwise")
                    .foregroundStyle(.gray)
                    .accessibilityLabel(String(localized: "Reset to Default"))
            }
            .buttonStyle(.bordered)
            .controlSize(.mini)
            .help(String(localized: "Reset to 16 pt"))
        }
    }

    @ViewBuilder
    private var fontSizeLabel: some View {
        HStack(spacing: 2) {
            Text("\(Int(settings.messageFontSize))")
                .font(.caption)
            Text(String(localized: "pt"))
                .font(.caption)
        }
        .foregroundStyle(.secondary)
        .frame(minWidth: 40, alignment: .trailing)
        .monospacedDigit()
    }

    @ViewBuilder
    private var confidenceThresholdSlider: some View {
        HStack(spacing: 12) {
            Text("10%")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(minWidth: 30)

            Slider(value: $settings.visionConfidenceThreshold, in: 0.1...0.9, step: 0.05)
                .accessibilityLabel(String(localized: "Confidence Threshold"))
                .accessibilityValue(confidenceAccessibilityValue)

            Text("90%")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(minWidth: 30)

            confidenceValueLabel
        }
    }

    @ViewBuilder
    private var maxOutputTokensSlider: some View {
        let unlimitedBinding = Binding<Bool>(
            get: { settings.mlxMaxOutputTokens == 0 },
            set: { isUnlimited in
                settings.mlxMaxOutputTokens = isUnlimited ? 0 : 1024
            }
        )
        let tokenBinding = Binding<Double>(
            get: { Double(max(512, settings.mlxMaxOutputTokens == 0 ? 1024 : settings.mlxMaxOutputTokens)) },
            set: { settings.mlxMaxOutputTokens = Int($0.rounded()) }
        )

        VStack(alignment: .leading, spacing: 10) {
            Toggle(String(localized: "Unlimited"), isOn: unlimitedBinding)
                .toggleStyle(.switch)

            HStack(spacing: 12) {
                Text("512")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 30)

                Slider(value: tokenBinding, in: 512...1024, step: 32)
                    .disabled(settings.mlxMaxOutputTokens == 0)
                    .accessibilityLabel(String(localized: "Max Output Tokens"))
                    .accessibilityValue(settings.mlxMaxOutputTokens == 0 ? String(localized: "Unlimited") : "\(settings.mlxMaxOutputTokens) tokens")

                Text("1024")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 36)

                HStack(spacing: 2) {
                    Text(settings.mlxMaxOutputTokens == 0 ? String(localized: "Unlimited") : "\(settings.mlxMaxOutputTokens)")
                        .font(.caption)
                    if settings.mlxMaxOutputTokens != 0 {
                        Text(String(localized: "tokens"))
                            .font(.caption)
                    }
                }
                .foregroundStyle(.secondary)
                .frame(minWidth: 72, alignment: .trailing)
                .monospacedDigit()
            }
        }
    }

    @ViewBuilder
    private var repetitionPenaltySlider: some View {
        let offBinding = Binding<Bool>(
            get: { settings.mlxRepetitionPenalty <= 1.0 },
            set: { isOff in
                settings.mlxRepetitionPenalty = isOff ? 1.0 : 1.1
            }
        )
        let penaltyBinding = Binding<Double>(
            get: { settings.mlxRepetitionPenalty <= 1.0 ? 1.1 : settings.mlxRepetitionPenalty },
            set: { settings.mlxRepetitionPenalty = $0 }
        )

        VStack(alignment: .leading, spacing: 10) {
            Toggle(String(localized: "Off"), isOn: offBinding)
                .toggleStyle(.switch)

            HStack(spacing: 12) {
                Text("1.05")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 36)

                Slider(value: penaltyBinding, in: 1.05...1.5, step: 0.05)
                    .disabled(settings.mlxRepetitionPenalty <= 1.0)
                    .accessibilityLabel(String(localized: "Repetition Penalty"))
                    .accessibilityValue(settings.mlxRepetitionPenalty <= 1.0 ? String(localized: "Off") : String(format: "%.2f", settings.mlxRepetitionPenalty))

                Text("1.50")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 36)

                Text(settings.mlxRepetitionPenalty <= 1.0 ? String(localized: "Off") : String(format: "%.2f", settings.mlxRepetitionPenalty))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 56, alignment: .trailing)
                    .monospacedDigit()
            }
        }
    }

    @ViewBuilder
    private var contextWindowSlider: some View {
        let deviceMaximum = MLXDeviceSupportProfile.current.maxContextWindowTokens
        let minimum = 512
        let tokenBinding = Binding<Double>(
            get: { Double(effectiveContextWindowTokens) },
            set: { settings.mlxContextWindowTokens = Int($0.rounded()) }
        )

        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Text("\(minimum)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 40)

                Slider(value: tokenBinding, in: Double(minimum)...Double(deviceMaximum), step: 512)
                    .accessibilityLabel(String(localized: "Context Window"))
                    .accessibilityValue("\(effectiveContextWindowTokens) tokens")

                Text("\(deviceMaximum)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 40)

                HStack(spacing: 2) {
                    Text("\(effectiveContextWindowTokens)")
                        .font(.caption)
                    Text(String(localized: "tokens"))
                        .font(.caption)
                }
                .foregroundStyle(.secondary)
                .frame(minWidth: 96, alignment: .trailing)
                .monospacedDigit()
            }

            Text(String(localized: "Device max: \(deviceMaximum) tokens"))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var confidenceValueLabel: some View {
        Text("\(Int((settings.visionConfidenceThreshold * 100).rounded()))%")
            .font(.caption)
            .foregroundStyle(.secondary)
            .frame(minWidth: 40, alignment: .trailing)
            .monospacedDigit()
    }

    private var confidenceAccessibilityValue: String {
        "\(Int((settings.visionConfidenceThreshold * 100).rounded()))%"
    }

    @ViewBuilder
    private func appearanceOption(localized: String, icon: String, tag: String) -> some View {
        Label(localized, systemImage: icon).tag(tag)
    }

    private func validateTavilyKey() {
        tavilyKeyValidation.validate(settings.tavilyApiKey)
    }

}

private struct InfoButton: View {
    let title: String
    let message: String

    @State private var showingInfo = false

    var body: some View {
        Button {
            showingInfo = true
        } label: {
            Image(systemName: "info.circle")
                .foregroundStyle(.secondary)
                .font(.caption)
        }
        .alert(title, isPresented: $showingInfo) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(message)
        }
        .accessibilityLabel(title)
        .accessibilityHint(message)
    }
}

private struct ExperimentalBadgeButton: View {
    let title: String
    let message: String

    @State private var showingInfo = false

    var body: some View {
        Button {
            showingInfo = true
        } label: {
            Text("(E)")
                .font(.caption.bold())
                .foregroundStyle(.orange)
        }
        .buttonStyle(.plain)
        .alert(title, isPresented: $showingInfo) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(message)
        }
        .accessibilityLabel(title)
        .accessibilityHint(message)
    }
}

private struct ChatPreferencesEditor: View {
    @Binding var text: String
    let placeholder: String
    @Binding var isFocused: Bool
    var minHeight: CGFloat
    var maxHeight: CGFloat? = nil
    @FocusState private var isEditorFocused: Bool

    var body: some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.large, style: .continuous)
                .fill(.thinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.large, style: .continuous)
                        .strokeBorder(isEditorFocused ? Color.accentColor.opacity(0.7) : Color.secondary.opacity(0.15), lineWidth: isEditorFocused ? 1.5 : 1)
                )

            TextEditor(text: $text)
                .focused($isEditorFocused)
                .font(.body)
                .textInputAutocapitalization(.sentences)
                .autocorrectionDisabled(false)
                .scrollContentBackground(.hidden)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .frame(minHeight: minHeight, maxHeight: maxHeight, alignment: .topLeading)

            if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text(placeholder)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 16)
                    .allowsHitTesting(false)
            }
        }
        .contentShape(RoundedRectangle(cornerRadius: DesignTokens.CornerRadius.large, style: .continuous))
        .onTapGesture {
            isEditorFocused = true
        }
        .onChange(of: isEditorFocused) { _, newValue in
            isFocused = newValue
        }
        .onChange(of: isFocused) { _, newValue in
            if isEditorFocused != newValue {
                isEditorFocused = newValue
            }
        }
        .accessibilityElement(children: .contain)
    }
}
