import SwiftUI

struct SettingsSheet: View {
    static let mlxKVCacheInfoMessage = String(localized: "Reduces memory use for long on-device MLX chats by quantizing the KV cache to 8-bit after an initial warmup. Tool-enabled runs switch to a quantization-compatible cache strategy when possible and automatically fall back if unsupported. This may slightly change quality or latency, and memory-constrained runs still skip KV quantization.")
    static let mlxKVCacheAccessibilityHint = String(localized: "Reduces memory use for long MLX chats, including tool runs when supported. Memory-constrained runs still skip KV quantization.")

    @Binding var defaultSystemPrompt: String
    @Binding var appAppearance: String   // "system" | "light" | "dark"
    @Binding var appLanguage: String     // "en" | "de" | "es"

    let hasChats: Bool
    let canDeleteAllExceptCurrent: Bool
    let onDeleteAll: () -> Void
    let onDeleteAllExceptCurrent: () -> Void
    let onExportChats: () -> Void

    // Persist custom presets in UserDefaults via AppStorage (per-device)
    @AppStorage("customSystemPromptPresets") var customPresetsData: Data = Data()

    // Tavily API key
    @AppStorage("tavilyApiKey") var tavilyApiKey: String = ""

    // Quality-of-life settings
    @AppStorage("sendOnReturn") var sendOnReturn: Bool = false
    @AppStorage("enableHaptics") var enableHaptics: Bool = true
    @AppStorage("reasoningModeDefault") var reasoningModeDefault: Bool = false
    @AppStorage("messageFontSize") var messageFontSize: Double = 16.0 // 12-22 range
    @AppStorage("mlxMaxOutputTokens") var mlxMaxOutputTokens: Int = 1024
    @AppStorage("mlxEnableKVCacheQuantization") var mlxEnableKVCacheQuantization: Bool = false
    @AppStorage("autoDeleteOldChats") var autoDeleteOldChats: Bool = false
    @AppStorage("autoDeleteDays") var autoDeleteDays: Int = 30 // 7, 14, 30, 60, 90
    @AppStorage("developerModeEnabled") var developerModeEnabled: Bool = false

    // Vision Framework settings
    @AppStorage("visionConfidenceThreshold") var visionConfidenceThreshold: Double = 0.5

    // These are always enabled for best experience
    let confirmBeforeDeletingChats: Bool = true
    private let showCharacterCount: Bool = true
    private let useMonospacedEditors: Bool = true

    // Editor sheet state
    @State var editingPreset: SystemPromptPreset? = nil
    @State var isEditingSheetPresented: Bool = false

    @State private var isDefaultPromptFocused: Bool = false

    // Confirmation alerts
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

    // Built-in presets (read-only)
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

    // Decoded/encoded custom presets
    var customPresets: [SystemPromptPreset] {
        (try? JSONDecoder().decode([SystemPromptPreset].self, from: customPresetsData)) ?? []
    }

    func updateCustomPresets(_ newValue: [SystemPromptPreset]) {
        customPresetsData = (try? JSONEncoder().encode(newValue)) ?? Data()
    }

    var body: some View {
        NavigationStack {
            Form {
                // MARK: General
                Section(header: Text("General")) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 6) {
                            Text(String(localized: "Appearance"))
                        }
                        
                        Picker("", selection: $appAppearance) {
                            appearanceOption(localized: "System", icon: "circle.lefthalf.filled", tag: "system")
                            appearanceOption(localized: "Light", icon: "sun.max", tag: "light")
                            appearanceOption(localized: "Dark", icon: "moon", tag: "dark")
                        }
                        .pickerStyle(.segmented)
                        .accessibilityLabel(String(localized: "App Appearance"))
                    }

                    Picker(selection: $appLanguage) {
                        Text(String(localized: "English")).tag("en")
                    } label: {
                        HStack(spacing: 6) {
                            Text(String(localized: "Language"))
                        }
                    }
                    .disabled(true) // Only English available
                }

                // MARK: Behavior
                Section(header: Text("Behavior")) {
                    Toggle(isOn: $sendOnReturn) {
                        HStack(spacing: 6) {
                            Label(String(localized: "Send on Return"), systemImage: "return")
                            InfoButton(
                                title: String(localized: "Send on Return"),
                                message: String(localized: "When enabled, pressing Return sends the message instead of inserting a new line.")
                            )
                        }
                    }
                    .accessibilityHint(String(localized: "Pressing Return sends the message instead of inserting a new line."))

                    Toggle(isOn: $enableHaptics) {
                        HStack(spacing: 6) {
                            Label(String(localized: "Enable Haptics"), systemImage: "waveform")
                            InfoButton(
                                title: String(localized: "Enable Haptics"),
                                message: String(localized: "Provide subtle vibration feedback on key actions, when supported by your device.")
                            )
                        }
                    }
                    
                    Toggle(isOn: $reasoningModeDefault) {
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
                            Label(String(localized: "Max Output Tokens"), systemImage: "slider.horizontal.3")
                            InfoButton(
                                title: String(localized: "Max Output Tokens"),
                                message: String(localized: "Limits the maximum number of tokens generated by on-device MLX models. Lower values usually reduce memory use and response time.")
                            )
                        }

                        maxOutputTokensSlider
                    }
                    .padding(.vertical, 4)

                    Toggle(isOn: $mlxEnableKVCacheQuantization) {
                        HStack(spacing: 6) {
                            Label(String(localized: "8-Bit KV Cache (MLX Only)"), systemImage: "memorychip")
                            InfoButton(
                                title: String(localized: "8-Bit KV Cache (MLX Only)"),
                                message: Self.mlxKVCacheInfoMessage
                            )
                        }
                    }
                    .accessibilityHint(Self.mlxKVCacheAccessibilityHint)

                    Toggle(isOn: $developerModeEnabled) {
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
                
                // MARK: Display
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
                
                // MARK: Object Detection
                Section(
                    header: Text(String(localized: "Object Detection")),
                    footer: Text(String(localized: "Used as a compatibility fallback for image analysis when native model-based vision is unavailable."))
                ) {
                    // Confidence Threshold slider
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
                    
                    // Status indicator
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.caption2)
                            .foregroundStyle(.green)
                        Text("Vision Framework fallback enabled")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                
                // MARK: Tavily Search
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
                    
                    SecureField(String(localized: "Enter your Tavily API key"), text: $tavilyApiKey)
                        .textContentType(.password)
                        .autocapitalization(.none)
                        .autocorrectionDisabled()
                        .accessibilityLabel(String(localized: "Tavily API Key"))
                    
                    if !tavilyApiKey.isEmpty {
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
                
                // MARK: Privacy
                Section(
                    header: Text(String(localized: "Privacy")),
                    footer: autoDeleteOldChats 
                        ? Text("Conversations older than \(autoDeleteDays) days will be automatically deleted.")
                        : Text(String(localized: "Automatically delete conversations after a specified time period."))
                ) {
                    Toggle(isOn: $autoDeleteOldChats) {
                        HStack(spacing: 6) {
                            Label(String(localized: "Auto-Delete Old Chats"), systemImage: "clock.badge.xmark")
                            InfoButton(
                                title: String(localized: "Auto-Delete Old Chats"),
                                message: String(localized: "Automatically removes conversations older than the specified number of days.")
                            )
                        }
                    }
                    
                    if autoDeleteOldChats {
                        Picker(selection: $autoDeleteDays) {
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

                // MARK: Chat Defaults
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
                        text: $defaultSystemPrompt,
                        placeholder: String(localized: "Describe the assistant’s default behavior for new chats…"),
                        isFocused: $isDefaultPromptFocused,
                        minHeight: 140,
                        maxHeight: 240,
                        isMonospaced: useMonospacedEditors
                    )

                    HStack {
                        if showCharacterCount {
                            let count = defaultSystemPrompt.count
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
                            if !defaultSystemPrompt.isEmpty {
                                Button(role: .destructive) {
                                    defaultSystemPrompt = ""
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
                        if !defaultSystemPrompt.isEmpty {
                            Button {
                                UIPasteboard.general.string = defaultSystemPrompt
                            } label: {
                                Label(String(localized: "Copy"), systemImage: "doc.on.doc")
                            }
                        }
                        Button {
                            if let str = UIPasteboard.general.string {
                                defaultSystemPrompt = str
                            }
                        } label: {
                            Label(String(localized: "Paste"), systemImage: "doc.on.clipboard")
                        }
                    }
                }

                // MARK: Presets
                Section(header: Text(String(localized: "System Prompt Presets"))) {
                    // Built-in (read-only)
                    if !builtInPresets.isEmpty {
                        DisclosureGroup(String(localized: "Built‑in")) {
                            ForEach(builtInPresets) { preset in
                                presetRow(preset, isCustom: false)
                            }
                        }
                    }

                    // Custom (editable)
                    DisclosureGroup(String(localized: "Your Presets")) {
                        if customPresets.isEmpty {
                            Text(String(localized: "No custom presets yet. Tap “New Preset” to create one."))
                                .foregroundStyle(.secondary)
                        } else {
                            EditablePresetList(
                                presets: customPresets,
                                onMove: { from, to in
                                    var items = customPresets
                                    items.move(fromOffsets: from, toOffset: to)
                                    updateCustomPresets(items)
                                },
                                onDelete: { offsets in
                                    var items = customPresets
                                    items.remove(atOffsets: offsets)
                                    updateCustomPresets(items)
                                },
                                row: { preset in
                                    presetRow(preset, isCustom: true)
                                }
                            )
                        }

                        Button {
                            editingPreset = SystemPromptPreset(name: "", text: defaultSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines))
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
                        var items = customPresets
                        if let idx = items.firstIndex(where: { $0.id == saved.id }) {
                            items[idx] = saved
                        } else {
                            items.insert(saved, at: 0)
                        }
                        updateCustomPresets(items)
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
        .onChange(of: tavilyApiKey) { _, _ in
            NotificationCenter.default.post(name: NSNotification.Name("TavilyKeyChanged"), object: nil)
        }
    }

    // MARK: - Rows and actions
    
    @ViewBuilder
    private var fontSizeSlider: some View {
        HStack(spacing: 12) {
            Text("A")
                .font(.caption)
                .foregroundStyle(.primary.opacity(0.6))
            
            Slider(value: $messageFontSize, in: 12...22, step: 1)
                .tint(.accentColor)
                .shadow(color: .black.opacity(0.15), radius: 2, x: 0, y: 1)
                .accessibilityLabel(String(localized: "Message Font Size"))
                .accessibilityValue("\(Int(messageFontSize)) points")
            
            Text("A")
                .font(.title3)
                .foregroundStyle(.primary.opacity(0.6))
            
            fontSizeLabel

            Spacer(minLength: 8)

            Button {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                    messageFontSize = 16.0
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
            Text("\(Int(messageFontSize))")
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
            
            Slider(value: $visionConfidenceThreshold, in: 0.1...0.9, step: 0.05)
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
        let tokenBinding = Binding<Double>(
            get: { Double(mlxMaxOutputTokens) },
            set: { mlxMaxOutputTokens = Int($0.rounded()) }
        )

        HStack(spacing: 12) {
            Text("512")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(minWidth: 30)

            Slider(value: tokenBinding, in: 512...1024, step: 32)
                .accessibilityLabel(String(localized: "Max Output Tokens"))
                .accessibilityValue("\(mlxMaxOutputTokens) tokens")

            Text("1024")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(minWidth: 36)

            HStack(spacing: 2) {
                Text("\(mlxMaxOutputTokens)")
                    .font(.caption)
                Text(String(localized: "tokens"))
                    .font(.caption)
            }
            .foregroundStyle(.secondary)
            .frame(minWidth: 72, alignment: .trailing)
            .monospacedDigit()
        }
    }
    
    @ViewBuilder
    private var confidenceValueLabel: some View {
        Text("\(Int((visionConfidenceThreshold * 100).rounded()))%")
            .font(.caption)
            .foregroundStyle(.secondary)
            .frame(minWidth: 40, alignment: .trailing)
            .monospacedDigit()
    }
    
    private var confidenceAccessibilityValue: String {
        "\(Int((visionConfidenceThreshold * 100).rounded()))%"
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
                defaultSystemPrompt = preset.text
            } label: {
                Image(systemName: "arrow.down.doc")
                    .accessibilityLabel(String(localized: "Apply preset"))
            }
        }
        .contextMenu {
            Button {
                defaultSystemPrompt = preset.text
            } label: {
                Label(String(localized: "Apply"), systemImage: "arrow.down.doc")
            }

            Button {
                // Copy text of preset
                UIPasteboard.general.string = preset.text
            } label: {
                Label(String(localized: "Copy"), systemImage: "doc.on.doc")
            }

            if isCustom {
                Button {
                    // Duplicate preset
                    var items = customPresets
                    var copy = preset
                    copy.id = UUID()
                    copy.createdAt = Date()
                    items.insert(copy, at: 0)
                    updateCustomPresets(items)
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
                defaultSystemPrompt = preset.text
            } label: {
                Label(String(localized: "Apply"), systemImage: "arrow.down.doc")
            }
            .tint(.blue)

            if isCustom {
                Button {
                    // Duplicate preset quickly
                    var items = customPresets
                    var copy = preset
                    copy.id = UUID()
                    copy.createdAt = Date()
                    items.insert(copy, at: 0)
                    updateCustomPresets(items)
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

// MARK: - Models and helpers

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
            Button("OK", role: .cancel) { }
        } message: {
            Text(message)
        }
        .accessibilityLabel(title)
        .accessibilityHint(message)
    }
}

// A tiny wrapper to get reordering + delete with a custom row view.
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
        .environment(\.editMode, .constant(.active)) // always allow drag reordering
        .listStyle(.plain)
    }
}

// MARK: - Native Prompt Editor

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
