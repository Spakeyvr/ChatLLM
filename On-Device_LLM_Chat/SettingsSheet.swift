import SwiftUI

struct SettingsSheet: View {
    static let mlxKVCacheInfoMessage = String(localized: "Enabled by default for persistent MLX chats, this reduces memory use by quantizing the KV cache to 8-bit after an initial warmup. Devices with less than 12 GB of RAM may automatically switch all MLX turns to a bounded sliding-window cache to avoid crashes. Upstream MLX does not support quantizing rotating KV caches yet, so those low-memory runs skip KV quantization automatically.")
    static let mlxKVCacheAccessibilityHint = String(localized: "Enabled by default for supported persistent MLX chats. Low-memory devices may switch to a bounded sliding-window cache, which skips KV quantization.")

    @Binding var settings: AppSettingsDraft

    let hasChats: Bool
    let canDeleteAllExceptCurrent: Bool
    let onDeleteAll: () -> Void
    let onDeleteAllExceptCurrent: () -> Void
    let onExportChats: () -> Void

    let confirmBeforeDeletingChats = true
    private let showCharacterCount = true
    private let useMonospacedEditors = true

    @State var editingPreset: SystemPromptPreset?
    @State var isEditingSheetPresented = false
    @State private var isDefaultPromptFocused = false

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

    private let builtInPresets: [SystemPromptPreset] = [
        .init(name: String(localized: "Helpful assistant"),
              text: "You are a helpful, concise assistant. Prefer clear explanations and actionable steps."),
        .init(name: String(localized: "Creative writer"),
              text: "You are a creative writing assistant. Use vivid language, varied rhythm, and avoid clichés."),
        .init(name: String(localized: "Code helper"),
              text: "You are a precise coding assistant. Provide runnable Swift examples and explain tradeoffs succinctly."),
        .init(name: String(localized: "Teacher"),
              text: "You are a patient teacher. Break concepts into steps and check for understanding with brief questions.")
    ]

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
                            Label(String(localized: "Send on Return"), systemImage: "return")
                            InfoButton(
                                title: String(localized: "Send on Return"),
                                message: String(localized: "When enabled, pressing Return sends the message instead of inserting a new line.")
                            )
                        }
                    }
                    .accessibilityHint(String(localized: "Pressing Return sends the message instead of inserting a new line."))

                    Toggle(isOn: $settings.enableHaptics) {
                        HStack(spacing: 6) {
                            Label(String(localized: "Enable Haptics"), systemImage: "waveform")
                            InfoButton(
                                title: String(localized: "Enable Haptics"),
                                message: String(localized: "Provide subtle vibration feedback on key actions, when supported by your device.")
                            )
                        }
                    }

                    Toggle(isOn: $settings.reasoningModeDefault) {
                        HStack(spacing: 6) {
                            Label(String(localized: "Reasoning Mode by Default"), systemImage: "brain.head.profile")
                            InfoButton(
                                title: String(localized: "Reasoning Mode by Default"),
                                message: String(localized: "When enabled, new conversations will start with reasoning mode activated.")
                            )
                        }
                    }

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

                    Toggle(isOn: $settings.mlxEnableKVCacheQuantization) {
                        HStack(spacing: 6) {
                            Label(String(localized: "8-Bit KV Cache (MLX Only)"), systemImage: "memorychip")
                            InfoButton(
                                title: String(localized: "8-Bit KV Cache (MLX Only)"),
                                message: Self.mlxKVCacheInfoMessage
                            )
                        }
                    }
                    .accessibilityHint(Self.mlxKVCacheAccessibilityHint)

                    Toggle(isOn: $settings.developerModeEnabled) {
                        HStack(spacing: 6) {
                            Label(String(localized: "Developer Mode"), systemImage: "hammer")
                            InfoButton(
                                title: String(localized: "Developer Mode"),
                                message: String(localized: "Shows an extra developer action under assistant messages with raw output and generation diagnostics.")
                            )
                        }
                    }
                    .accessibilityHint(String(localized: "Shows raw output and generation diagnostics for assistant messages."))
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
                        .autocapitalization(.none)
                        .autocorrectionDisabled()
                        .accessibilityLabel(String(localized: "Tavily API Key"))

                    if !settings.tavilyApiKey.isEmpty {
                        Button {
                            openTavilySignUp()
                        } label: {
                            Label(String(localized: "Change Key"), systemImage: "key.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    } else {
                        Button {
                            openTavilySignUp()
                        } label: {
                            Label(String(localized: "Get API Key"), systemImage: "arrow.up.right")
                                .foregroundStyle(.blue)
                        }
                        .buttonStyle(.plain)
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
                            Label(String(localized: "Auto-Delete Old Chats"), systemImage: "clock.badge.xmark")
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
                            Text(String(localized: "Chat Defaults"))
                            InfoButton(
                                title: String(localized: "Default System Prompt"),
                                message: String(localized: "New chats start with this system prompt. You can change it per chat from the gear button in a conversation.")
                            )
                        }
                        .padding(.top, 8),
                    footer: Text(String(localized: "New chats will start with this system prompt. You can change it per chat from the gear button in the conversation."))
                ) {
                    NativePromptEditor(
                        text: $settings.defaultSystemPrompt,
                        placeholder: String(localized: "Describe the assistant’s default behavior for new chats…"),
                        isFocused: $isDefaultPromptFocused,
                        minHeight: 140,
                        maxHeight: 240,
                        isMonospaced: useMonospacedEditors
                    )

                    HStack {
                        if showCharacterCount {
                            let count = settings.defaultSystemPrompt.count
                            let counter = String(localized: "characters")
                            Text("\(count) \(counter)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Menu {
                            Button {
                                saveCurrentAsPreset()
                            } label: {
                                Label(String(localized: "Save as Preset"), systemImage: "bookmark.badge.plus")
                            }
                            if !settings.defaultSystemPrompt.isEmpty {
                                Button(role: .destructive) {
                                    settings.defaultSystemPrompt = ""
                                } label: {
                                    Label(String(localized: "Clear"), systemImage: "trash")
                                }
                            }
                        } label: {
                            Image(systemName: "ellipsis.circle")
                        }
                        .accessibilityLabel(String(localized: "Default System Prompt Actions"))
                    }
                    .contextMenu {
                        if !settings.defaultSystemPrompt.isEmpty {
                            Button {
                                UIPasteboard.general.string = settings.defaultSystemPrompt
                            } label: {
                                Label(String(localized: "Copy"), systemImage: "doc.on.doc")
                            }
                        }
                        Button {
                            if let str = UIPasteboard.general.string {
                                settings.defaultSystemPrompt = str
                            }
                        } label: {
                            Label(String(localized: "Paste"), systemImage: "doc.on.clipboard")
                        }
                    }
                }

                Section(header: Text(String(localized: "System Prompt Presets"))) {
                    if !builtInPresets.isEmpty {
                        DisclosureGroup(String(localized: "Built‑in")) {
                            ForEach(builtInPresets) { preset in
                                presetRow(preset, isCustom: false)
                            }
                        }
                    }

                    DisclosureGroup(String(localized: "Your Presets")) {
                        if settings.customPresets.isEmpty {
                            Text(String(localized: "No custom presets yet. Tap “New Preset” to create one."))
                                .foregroundStyle(.secondary)
                        } else {
                            EditablePresetList(
                                presets: settings.customPresets,
                                onMove: { from, to in
                                    settings.customPresets.move(fromOffsets: from, toOffset: to)
                                },
                                onDelete: { offsets in
                                    settings.customPresets.remove(atOffsets: offsets)
                                },
                                row: { preset in
                                    presetRow(preset, isCustom: true)
                                }
                            )
                        }

                        Button {
                            editingPreset = SystemPromptPreset(
                                name: "",
                                text: settings.defaultSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
                            )
                            isEditingSheetPresented = true
                        } label: {
                            Label(String(localized: "New Preset"), systemImage: "plus.circle")
                        }
                        .buttonStyle(.glass)
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
        .sheet(isPresented: $isEditingSheetPresented) {
            NavigationStack {
                PresetEditorView(
                    preset: editingPreset ?? SystemPromptPreset(name: "", text: ""),
                    onCancel: { isEditingSheetPresented = false },
                    onSave: { saved in
                        if let idx = settings.customPresets.firstIndex(where: { $0.id == saved.id }) {
                            settings.customPresets[idx] = saved
                        } else {
                            settings.customPresets.insert(saved, at: 0)
                        }
                        isEditingSheetPresented = false
                    }
                )
            }
            .presentationDetents([.medium, .large])
        }
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

    @ViewBuilder
    private func presetRow(_ preset: SystemPromptPreset, isCustom: Bool) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(preset.name.isEmpty ? String(localized: "Untitled Preset") : preset.name)
                    .font(.body)
                Text(preset.text)
                    .font(.caption)
                    .lineLimit(2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                settings.defaultSystemPrompt = preset.text
            } label: {
                Image(systemName: "arrow.down.doc")
                    .accessibilityLabel(String(localized: "Apply preset"))
            }
        }
        .contextMenu {
            Button {
                settings.defaultSystemPrompt = preset.text
            } label: {
                Label(String(localized: "Apply"), systemImage: "arrow.down.doc")
            }

            Button {
                UIPasteboard.general.string = preset.text
            } label: {
                Label(String(localized: "Copy"), systemImage: "doc.on.doc")
            }

            if isCustom {
                Button {
                    var copy = preset
                    copy.id = UUID()
                    copy.createdAt = Date()
                    settings.customPresets.insert(copy, at: 0)
                } label: {
                    Label(String(localized: "Duplicate"), systemImage: "plus.square.on.square")
                }

                Button {
                    editingPreset = preset
                    isEditingSheetPresented = true
                } label: {
                    Label(String(localized: "Edit"), systemImage: "pencil")
                }

                Button(role: .destructive) {
                    deletePreset(preset)
                } label: {
                    Label(String(localized: "Delete"), systemImage: "trash")
                }
            }
        }
        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
            Button {
                settings.defaultSystemPrompt = preset.text
            } label: {
                Label(String(localized: "Apply"), systemImage: "arrow.down.doc")
            }
            .tint(.blue)

            if isCustom {
                Button {
                    var copy = preset
                    copy.id = UUID()
                    copy.createdAt = Date()
                    settings.customPresets.insert(copy, at: 0)
                } label: {
                    Label(String(localized: "Duplicate"), systemImage: "plus.square.on.square")
                }
                .tint(.green)

                Button {
                    editingPreset = preset
                    isEditingSheetPresented = true
                } label: {
                    Label(String(localized: "Edit"), systemImage: "pencil")
                }
                .tint(.orange)

                Button(role: .destructive) {
                    deletePreset(preset)
                } label: {
                    Label(String(localized: "Delete"), systemImage: "trash")
                }
            }
        }
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

private struct EditablePresetList<Row: View>: View {
    var presets: [SystemPromptPreset]
    var onMove: (IndexSet, Int) -> Void
    var onDelete: (IndexSet) -> Void
    var row: (SystemPromptPreset) -> Row

    var body: some View {
        List {
            ForEach(presets) { preset in
                row(preset)
            }
            .onMove(perform: onMove)
            .onDelete(perform: onDelete)
        }
        .frame(minHeight: 44 * CGFloat(max(1, presets.count)))
        .environment(\.editMode, .constant(.active))
        .listStyle(.plain)
    }
}

struct NativePromptEditor: View {
    @Binding var text: String
    let placeholder: String
    @Binding var isFocused: Bool
    var minHeight: CGFloat
    var maxHeight: CGFloat? = nil
    var isMonospaced: Bool = true
    @FocusState private var isEditorFocused: Bool

    var body: some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(.thinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(isEditorFocused ? Color.accentColor.opacity(0.7) : Color.secondary.opacity(0.15), lineWidth: isEditorFocused ? 1.5 : 1)
                )

            TextEditor(text: $text)
                .focused($isEditorFocused)
                .font(isMonospaced ? .body.monospaced() : .body)
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
        .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
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
