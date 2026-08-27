//
//  ContentView.swift
//  ChatLLM
//
//  Created by Nevio on 10/24/25.
//

import SwiftUI
import SwiftData
import FoundationModels
import UIKit

struct ContentView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    @Query(sort: \Conversation.lastUpdated, order: .reverse)
    private var conversations: [Conversation]

    @State private var selection: Conversation?
    @State private var currentViewModel: ChatViewModel?
    @State private var draftConversation: Conversation? = nil
    @State private var preferredCompactColumn = NavigationSplitViewColumn.sidebar

    @State private var searchText: String = ""
    @State private var debouncedSearchText: String = ""
    @State private var searchDebounceTask: Task<Void, Never>?
    @State private var showDeleteAllAlert: Bool = false
    @State private var showSettings: Bool = false
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?
    @State private var isErrorAlertPresented: Bool = false
    @State private var showExportSheet: Bool = false
    @State private var exportURL: URL?
    @FocusState private var isSearchFocused: Bool
    @State private var attachmentCleanupTask: Task<Void, Never>?

    // App-wide preferences
    @AppStorage("chatPreferences") private var chatPreferences: String = ""
    @AppStorage("appAppearance") private var appAppearance: String = "system" // system | light | dark
    @AppStorage("appLanguage") private var appLanguage: String = "en" // en | de | es
    @AppStorage("reasoningModeDefault") private var reasoningModeDefault: Bool = false
    @AppStorage("messageFontSize") private var messageFontSize: Double = 16.0
    @AppStorage("autoDeleteOldChats") private var autoDeleteOldChats: Bool = false
    @AppStorage("autoDeleteDays") private var autoDeleteDays: Int = 30
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding: Bool = false

    private var preferredScheme: ColorScheme? {
        switch appAppearance {
        case "light": return .light
        case "dark": return .dark
        default: return nil
        }
    }

    private var storedChatPreferences: String {
        let defaults = UserDefaults.standard
        if defaults.object(forKey: AppSettingsKeys.chatPreferences) != nil {
            return chatPreferences
        }
        return defaults.string(forKey: AppSettingsKeys.legacyDefaultSystemPrompt) ?? ""
    }

    var body: some View {
        rootContent
            // Apply selected language to entire UI
            .environment(\.locale, Locale(identifier: appLanguage))
            .preferredColorScheme(preferredScheme)
    }

    @ViewBuilder
    private var rootContent: some View {
        if hasCompletedOnboarding {
            mainExperience
                .overlay(alignment: .topLeading) {
                    Color.clear
                        .frame(width: 1, height: 1)
                        .allowsHitTesting(false)
                        .accessibilityIdentifier("content.root")
                }
        } else {
            OnboardingView {
                hasCompletedOnboarding = true
            }
        }
    }

    private var mainExperience: some View {
        splitView
            .onAppear(perform: handleOnAppear)
            .onChange(of: selection, handleSelectionChange)
            .onChange(of: conversations, handleConversationsChange)
            .onChange(of: errorMessage, handleErrorMessageChange)
            .onChange(of: preferredCompactColumn) { _, newColumn in
                // When user presses Back on iPhone from a draft chat, discard the stale draft so
                // the next "New Chat" tap is not blocked by the guard in startDraftChat().
                if newColumn == .sidebar, draftConversation != nil, selection == nil {
                    draftConversation = nil
                    currentViewModel?.cancelGeneration()
                    currentViewModel = nil
                }
            }
            .deleteAllChatsAlert(
                isPresented: $showDeleteAllAlert,
                hasSelection: selection != nil,
                canDeleteAllExceptCurrent: conversations.count > 1,
                onDeleteAll: deleteAllChats,
                onDeleteAllExceptCurrent: deleteAllExceptCurrent
            )
            .errorAlert(
                isPresented: $isErrorAlertPresented,
                message: errorMessage ?? String(localized: "An unknown error occurred."),
                onDismiss: { 
                    isErrorAlertPresented = false
                    errorMessage = nil
                }
            )
            .settingsCover(
                isPresented: $showSettings,
                hasChats: !conversations.isEmpty,
                canDeleteAllExceptCurrent: conversations.count > 1,
                onDeleteAll: { showDeleteAllAlert = true },
                onDeleteAllExceptCurrent: { deleteAllExceptCurrent() },
                onExportChats: { exportAllChats() },
                onDismiss: { showSettings = false }
            )
            .sheet(isPresented: $showExportSheet, onDismiss: cleanupExportFile) {
                if let url = exportURL {
                    ShareSheet(activityItems: [url])
                }
            }
    }

    // Split the NavigationSplitView out of body to reduce complexity
    private var splitView: some View {
        NavigationSplitView(preferredCompactColumn: $preferredCompactColumn) {
            sidebar
        } detail: {
            detailContent
        }
    }
    
    // MARK: - Lifecycle handlers (split out for type-checking simplicity)
    
    private func handleOnAppear() {
        if let demoConversation = seedWebSearchDemoIfNeeded() {
            selection = demoConversation
            preferredCompactColumn = .detail
            return
        }

        // Clean up old chats if auto-delete is enabled
        performAutoDeleteIfNeeded()
        scheduleAttachmentStorageCleanup()
        
        // Only auto-select on regular width (split view visible).
        if horizontalSizeClass == .regular,
           selection == nil,
           let first = conversations.first {
            selection = first
        }
    }

    private func seedWebSearchDemoIfNeeded() -> Conversation? {
        let arguments = ProcessInfo.processInfo.arguments
        guard arguments.contains("-ui-test-web-search-demo") else { return nil }
        if let existing = conversations.first(where: { $0.title == "Web Search Preview" }) {
            return existing
        }

        let includesReasoning = !arguments.contains("-ui-test-web-search-without-reasoning")
        let usesLongReasoning = arguments.contains("-ui-test-long-reasoning-demo")
        let demoReasoning = usesLongReasoning
            ? """
            I need current platform information, so I should check primary sources.
            The first search gives the broad release context. A second, narrower query can confirm the exact SwiftUI API.
            I should compare the terminology used by both sources before drawing a conclusion.
            This newly exposed reasoning must appear immediately when the sheet expands.
            I should also verify that the terminology is consistent across the documentation.
            That comparison rules out a naming mismatch between the two sources.
            The implementation details now line up with the platform overview.
            No additional search is needed before forming the final response.
            Finally, I can summarize the verified platform change concisely.
            """
            : "I need current platform information, so I should check primary sources.\n\n</think>\n\nThe first search gives the broad release context. A second, narrower query can confirm the exact SwiftUI API.\n\n</think>"

        let conversation = Conversation(title: "Web Search Preview", reasoningMode: true)
        let userMessage = Message(
            role: .user,
            text: "What changed in SwiftUI web views this year?",
            order: 0,
            conversation: conversation,
            isFinal: true
        )
        let assistantMessage = Message(
            role: .assistant,
            text: "",
            order: 1,
            conversation: conversation,
            isFinal: true,
            reasoning: includesReasoning
                ? demoReasoning
                : " \n ",
            finalAnswer: "SwiftUI now has a native `WebView` and an observable `WebPage` model.\n\n**The key change is native SwiftUI integration.**\n\nApps can embed and control web content without wrapping `WKWebView` themselves, while Apple’s documentation and release coverage agree on the core API shape.",
            isReasoningMode: true,
            generationBackend: "Preview",
            generationModelName: "UI Fixture"
        )

        let firstResponse = WebSearchResponse(
            query: "SwiftUI WebView iOS 26 changes",
            sources: [
                WebSearchSource(
                    title: "WebKit for SwiftUI",
                    url: "https://developer.apple.com/documentation/webkit/webkit-for-swiftui",
                    snippet: "Apple documents the SwiftUI-native WebView and WebPage APIs for displaying and controlling web content.",
                    score: 0.97,
                    publishedDate: "June 2026"
                ),
                WebSearchSource(
                    title: "What’s new in WebKit and Safari",
                    url: "https://webkit.org/blog/",
                    snippet: "WebKit release notes cover new platform capabilities and behavior across Safari and embedded web views.",
                    score: 0.89
                ),
                WebSearchSource(
                    title: "SwiftUI updates",
                    url: "https://developer.apple.com/swiftui/whats-new/",
                    snippet: "Apple’s SwiftUI overview highlights the newest framework APIs and platform integrations.",
                    score: 0.84
                )
            ],
            responseTimeSeconds: 0.42
        )
        let secondResponse = WebSearchResponse(
            query: "Apple WebPage WebView SwiftUI documentation",
            sources: [
                WebSearchSource(
                    title: "WebView",
                    url: "https://developer.apple.com/documentation/webkit/webview-swift.struct",
                    snippet: "A SwiftUI view that displays web content managed by a WebPage instance.",
                    score: 0.99
                ),
                WebSearchSource(
                    title: "WebPage",
                    url: "https://developer.apple.com/documentation/webkit/webpage",
                    snippet: "An observable object that manages navigation, page state, and interactions for web content.",
                    score: 0.98
                )
            ],
            responseTimeSeconds: 0.31
        )
        let now = Date()
        if includesReasoning {
            assistantMessage.reasoningStartedAt = now.addingTimeInterval(-68)
            assistantMessage.reasoningCompletedAt = now
        }
        assistantMessage.searchInvocations = [
            SearchInvocation(
                query: firstResponse.query,
                results: firstResponse.modelFacingText(),
                response: firstResponse,
                status: .completed,
                anchorStepNumber: 1,
                timestamp: now.addingTimeInterval(-1.1),
                completedAt: now.addingTimeInterval(-0.68)
            ),
            SearchInvocation(
                query: secondResponse.query,
                results: secondResponse.modelFacingText(),
                response: secondResponse,
                status: .completed,
                anchorStepNumber: 2,
                timestamp: now.addingTimeInterval(-0.55),
                completedAt: now.addingTimeInterval(-0.24)
            )
        ]
        conversation.messages = [userMessage, assistantMessage]
        modelContext.insert(conversation)
        try? modelContext.save()
        return conversation
    }
    
    private func handleSelectionChange(_ oldSelection: Conversation?, _ newSelection: Conversation?) {
        _ = oldSelection?.id
        let newID = newSelection?.id
        // Entering draft mode: selection becomes nil but we want to keep the draft vm alive.
        if newID == nil && draftConversation != nil {
            ModelBackendBridge.shared.bindConversation(draftConversation)
            return
        }
        // Navigating to a real conversation discards any open draft.
        if newID != nil { draftConversation = nil }
        ModelBackendBridge.shared.bindConversation(newSelection)
        if let currentVM = currentViewModel, currentVM.conversation.id != newID {
            currentVM.cancelGeneration()
            currentViewModel = nil
        }
    }
    
    private func handleConversationsChange(_ oldConversations: [Conversation], _ newConversations: [Conversation]) {
        // When a draft conversation's first message is saved it gets inserted into the context,
        // which triggers this handler. Detect that moment and promote the draft to a real selection.
        if let draft = draftConversation,
           newConversations.contains(where: { $0.id == draft.id }),
           !oldConversations.contains(where: { $0.id == draft.id }) {
            draftConversation = nil
            selection = draft
            return
        }

        // Ensure selection is still valid after conversations change
        if let currentSelection = selection {
            // Check if current selection still exists by ID (safer than object comparison)
            let selectionExists = newConversations.contains(where: { $0.id == currentSelection.id })
            
            if !selectionExists {
                // Current selection was deleted - clean up view model if it hasn't been already
                if let currentVM = currentViewModel, currentVM.conversation.id == currentSelection.id {
                    currentVM.cancelGeneration()
                    currentViewModel = nil
                }
                
                // Only choose a new selection automatically on regular width.
                if horizontalSizeClass == .regular {
                    selection = newConversations.first
                } else {
                    // In compact, clear selection so we don't auto-push a detail.
                    selection = nil
                }
            }
        } else if selection == nil,
                  horizontalSizeClass == .regular,
                  let first = newConversations.first {
            // Only auto-select the first conversation on regular width.
            selection = first
        }
    }
    
    private func handleErrorMessageChange(_ old: String?, _ new: String?) {
        isErrorAlertPresented = (new != nil)
    }

    private var activeChatViewModel: ChatViewModel? {
        guard let vm = currentViewModel else { return nil }

        if let draft = draftConversation, vm.conversation.id == draft.id {
            return vm
        }

        if let convo = selection, vm.conversation.id == convo.id {
            return vm
        }

        return nil
    }

    private func referencedAttachmentURLs(excludingConversationIDs excludedConversationIDs: Set<UUID> = []) -> Set<URL> {
        Set(
            conversations
                .filter { !excludedConversationIDs.contains($0.id) }
                .flatMap(\.messages)
                .flatMap(\.attachments)
                .map(\.actualFileURL)
        )
    }

    private func attachmentURLs(in conversations: [Conversation]) -> [URL] {
        conversations
            .flatMap(\.messages)
            .flatMap(\.attachments)
            .map(\.actualFileURL)
    }

    private func deleteAttachmentFiles(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        Task {
            for url in urls {
                try? await ImageStore.shared.delete(url: url)
                await ImageStore.shared.deleteInferenceVariant(for: url)
            }
        }
    }

    private func scheduleAttachmentStorageCleanup(excludingConversationIDs excludedConversationIDs: Set<UUID> = []) {
        attachmentCleanupTask?.cancel()
        attachmentCleanupTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }

            let referencedURLs = referencedAttachmentURLs(excludingConversationIDs: excludedConversationIDs)
            try? await ImageStore.shared.cleanupOrphanedFiles(referencedURLs: referencedURLs)
        }
    }
    
    // Extracted to reduce complexity in body
    @ViewBuilder
    private var detailContent: some View {
        if let vm = activeChatViewModel {
            ChatView(viewModel: vm, onNewChat: { startDraftChat() })
        } else if let convo = selection {
            if let vm = currentViewModel, vm.conversation.id == convo.id {
                ChatView(viewModel: vm, onNewChat: { startDraftChat() })
            } else {
                VStack(spacing: 16) {
                    ProgressView()
                    Text(String(localized: "Preparing chat…"))
                        .foregroundStyle(.secondary)
                }
                .task(id: convo.id) {
                    await MainActor.run {
                        currentViewModel?.cancelGeneration()
                        currentViewModel = createViewModel(for: convo)
                    }
                }
            }
        } else if conversations.isEmpty && !isLoading {
            VStack(spacing: 16) {
                Image(systemName: "plus.bubble.right")
                    .font(.system(size: 48))
                    .foregroundStyle(.secondary)
                Text("Start a New Chat")
                    .font(.title2)
                    .fontWeight(.semibold)
                Text("Tap the + button to create your first conversation")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                
                Button("New Chat") {
                    startDraftChat()
                }
                .buttonStyle(.glassProminent)
                .padding(.top)
            }
            .padding()
        } else {
            VStack(spacing: 16) {
                if isLoading {
                    ProgressView()
                        .scaleEffect(1.2)
                    Text("Loading...")
                        .font(.headline)
                        .foregroundStyle(.secondary)
                } else {
                    Image(systemName: "bubble.left.and.bubble.right")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary)
                    Text("Select a Conversation")
                        .font(.title2)
                        .fontWeight(.semibold)
                    Text("Choose a chat from the sidebar to continue")
                        .font(.body)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
            }
            .padding()
        }
    }

    private var filteredConversations: [Conversation] {
        let trimmed = debouncedSearchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return conversations }

        let searchTerms = trimmed.lowercased().components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        guard !searchTerms.isEmpty else { return conversations }

        return conversations.filter { convo in
            let titleLowercased = convo.title.lowercased()
            let titleMatches = searchTerms.allSatisfy { term in
                titleLowercased.range(of: term) != nil
            }
            if titleMatches { return true }

            let searchableText = convo.searchableVisibleText.lowercased()
            return searchTerms.allSatisfy { term in
                searchableText.range(of: term) != nil
            }
        }
    }

    private struct ConversationRecencySection: Identifiable {
        let id: String
        let title: LocalizedStringKey
        let conversations: [Conversation]
    }

    private struct ConversationSectionHeader: View {
        let title: LocalizedStringKey

        var body: some View {
            Text(title)
                .font(.headline.weight(.semibold))
                .foregroundStyle(.secondary)
            .padding(.top, 4)
            .padding(.bottom, 4)
        }
    }

    private var groupedConversations: [ConversationRecencySection] {
        let calendar = Calendar.current
        let now = Date()

        // Evaluate the filter once: when a search term is active it rebuilds every
        // conversation's full visible transcript, so re-deriving it per bucket made
        // sidebar rendering scale with total message history × 5.
        let candidates = filteredConversations
        let sevenDaysAgo = calendar.date(byAdding: .day, value: -7, to: now)
        let thirtyDaysAgo = calendar.date(byAdding: .day, value: -30, to: now)

        let buckets: [(id: String, title: LocalizedStringKey, conversations: [Conversation])] = [
            (
                id: "today",
                title: "Today",
                conversations: candidates.filter { calendar.isDateInToday($0.lastUpdated) }
            ),
            (
                id: "yesterday",
                title: "Yesterday",
                conversations: candidates.filter { calendar.isDateInYesterday($0.lastUpdated) }
            ),
            (
                id: "last7Days",
                title: "Last 7 Days",
                conversations: candidates.filter {
                    guard !calendar.isDateInToday($0.lastUpdated),
                          !calendar.isDateInYesterday($0.lastUpdated) else {
                        return false
                    }
                    guard let sevenDaysAgo else {
                        return false
                    }
                    return $0.lastUpdated >= sevenDaysAgo
                }
            ),
            (
                id: "last30Days",
                title: "Last 30 Days",
                conversations: candidates.filter {
                    guard let sevenDaysAgo, let thirtyDaysAgo else {
                        return false
                    }
                    return $0.lastUpdated < sevenDaysAgo && $0.lastUpdated >= thirtyDaysAgo
                }
            ),
            (
                id: "older",
                title: "Older",
                conversations: candidates.filter {
                    guard let thirtyDaysAgo else {
                        return false
                    }
                    return $0.lastUpdated < thirtyDaysAgo
                }
            )
        ]

        return buckets.compactMap { bucket in
            guard !bucket.conversations.isEmpty else { return nil }
            return ConversationRecencySection(
                id: bucket.id,
                title: bucket.title,
                conversations: bucket.conversations
            )
        }
    }

    @ViewBuilder
    private var sidebar: some View {
        // One pass over the (potentially expensive) search filter per render.
        let sections = groupedConversations

        ZStack(alignment: .bottom) {
            // Main list content — keep selection binding for NavigationSplitView
            List(selection: $selection) {
                if sections.isEmpty && !searchText.isEmpty {
                    // Empty search results state
                    VStack(spacing: 12) {
                        Image(systemName: "magnifyingglass")
                            .font(.largeTitle)
                            .foregroundStyle(.secondary)
                        Text(String(localized: "No matching chats"))
                            .font(.headline)
                        Text(String(localized: "Try adjusting your search terms"))
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 40)
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                } else {
                    ForEach(sections) { section in
                        Section {
                            ForEach(section.conversations, id: \.id) { convo in
                                let conversationID = convo.id

                                ConversationRow(conversation: convo)
                                    .contentShape(Rectangle())
                                    .onTapGesture {
                                        AppHaptics.selectionChanged()
                                        selection = convo
                                    }
                                    .contextMenu {
                                        Button {
                                            if let conversation = conversations.first(where: { $0.id == conversationID }) {
                                                rename(conversation)
                                            }
                                        } label: {
                                            Label(String(localized: "Rename"), systemImage: "pencil")
                                        }

                                        Button(role: .destructive) {
                                            if let conversation = conversations.first(where: { $0.id == conversationID }) {
                                                delete(conversation)
                                            }
                                        } label: {
                                            Label(String(localized: "Delete"), systemImage: "trash")
                                        }
                                    }
                                    .swipeActions(edge: .trailing) {
                                        Button(role: .destructive) {
                                            if let conversation = conversations.first(where: { $0.id == conversationID }) {
                                                delete(conversation)
                                            }
                                        } label: {
                                            Label(String(localized: "Delete"), systemImage: "trash")
                                        }
                                    }
                            }
                        } header: {
                            ConversationSectionHeader(title: section.title)
                        }
                        .listSectionSeparator(.hidden)
                        .listRowBackground(Color.clear)
                    }
                }
            }
            .listStyle(.plain)
            .listSectionSpacing(.compact)
            .listRowSpacing(0)
            .scrollDismissesKeyboard(.interactively)
            // Add safe area inset at bottom to prevent content from being hidden behind floating search bar
            .safeAreaInset(edge: .bottom) {
                // Reserve space for the search bar and its padding
                Color.clear
                    .frame(height: 88) // 44pt search bar + 20pt padding above + 24pt padding below
            }

            // Floating search overlay row with separate action button
            bottomSearchBar
                .padding(.horizontal, 12)
                .padding(.bottom, 24) // lift it off the bottom to float
        }
    }

    private var bottomSearchBar: some View {
        GlassEffectContainer(spacing: 12.0) {
            HStack(alignment: .bottom, spacing: 12) {
                // Search pill with Liquid Glass effect (no extra background tint)
                HStack(spacing: 10) {
                    Image(systemName: isSearchFocused || !searchText.isEmpty ? "magnifyingglass.circle.fill" : "magnifyingglass")
                        .foregroundStyle(isSearchFocused ? .blue : .secondary)
                        .symbolEffect(.bounce, value: isSearchFocused)
                        .background(.clear)

                    DynamicHeightTextEditor(
                        text: $searchText,
                        height: .constant(0), // Not used anymore
                        placeholder: String(localized: "Search chats")
                    )
                    .focused($isSearchFocused)
                    .onChange(of: searchText) { _, newValue in
                        // Debounce search input to avoid excessive filtering
                        searchDebounceTask?.cancel()
                        if newValue.isEmpty {
                            debouncedSearchText = ""
                            return
                        }
                        searchDebounceTask = Task { @MainActor in
                            try? await Task.sleep(for: .milliseconds(300))
                            guard !Task.isCancelled else { return }
                            debouncedSearchText = newValue
                        }
                    }
                    .onSubmit {
                        // Immediate search on submit
                        searchDebounceTask?.cancel()
                        debouncedSearchText = searchText
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                            isSearchFocused = false
                        }
                    }

                    if !searchText.isEmpty {
                        Button {
                            AppHaptics.impact(.light)
                            clearSearch()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                                .frame(width: 32, height: 32)
                                .contentShape(Circle())
                        }
                        .accessibilityLabel(String(localized: "Clear search"))
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 16)
                .frame(minHeight: 52)
                .frame(maxWidth: .infinity, alignment: .leading)
                .glassEffect(.regular.interactive(), in: .capsule)
                .contentShape(.capsule)

                HStack(spacing: 10) {
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "gearshape.fill")
                            .font(.system(size: 20, weight: .semibold))
                            .scaleEffect(isSearchFocused ? 0.9 : 1.0)
                            .frame(width: 50, height: 50)
                            .contentShape(Circle())
                    }
                    .contentShape(Circle())
                    .buttonStyle(.plain)
                    .glassEffect(.regular.interactive(), in: .circle)
                    .accessibilityLabel(String(localized: "Settings"))

                    Button {
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                            startDraftChat()
                        }
                    } label: {
                        Image(systemName: "square.and.pencil")
                            .font(.system(size: 20, weight: .semibold))
                            .scaleEffect(isSearchFocused ? 0.9 : 1.0)
                            .frame(width: 50, height: 50)
                            .contentShape(Circle())
                    }
                    .contentShape(Circle())
                    .buttonStyle(.plain)
                    .glassEffect(.regular.interactive(), in: .circle)
                    .accessibilityLabel(String(localized: "New Chat"))
                }
                .shadow(color: .clear.opacity(0.2), radius: 12, x: 0, y: 6)
            }
        }
        .background(Color.clear)
        .animation(.spring(response: 0.35, dampingFraction: 0.85, blendDuration: 0.2), value: isSearchFocused)
        .accessibilityElement(children: .contain)
    }

    // Pure creator that does not mutate state; we call it from .task or lifecycle hooks.
    private func createViewModel(for conversation: Conversation) -> ChatViewModel {
        // Reuse existing view model if it's for the same conversation
        if let existing = currentViewModel, existing.conversation.id == conversation.id {
            return existing
        }
        
        // Create new view model
        let generator: LLMGenerator = OnDeviceLLMGenerator()
        return ChatViewModel(generator: generator, context: modelContext, conversation: conversation)
    }

    private func startDraftChat() {
        // Only one draft at a time — the button is disabled when a draft is already open.
        guard draftConversation == nil else { return }
        AppHaptics.impact(.medium)
        Task { @MainActor in
            currentViewModel?.cancelGeneration()
            let convo = Conversation(
                title: String(localized: "New Chat"),
                chatPreferences: storedChatPreferences.trimmingCharacters(in: .whitespacesAndNewlines)
            )
            convo.reasoningMode = reasoningModeDefault
            let backendBridge = ModelBackendBridge.shared
            convo.preferredBackendRawValue = backendBridge.selectedBackend.rawValue
            convo.preferredModelID = backendBridge.selectedModelID
            // Set draft BEFORE clearing selection so handleSelectionChange preserves this vm.
            draftConversation = convo
            currentViewModel = createViewModel(for: convo)
            backendBridge.bindConversation(convo)
            selection = nil
            // Push detail column on compact (iPhone) so the draft chat is immediately visible.
            preferredCompactColumn = .detail
            if !searchText.isEmpty {
                clearSearch()
            }
        }
    }

    private func clearSearch() {
        searchDebounceTask?.cancel()
        searchText = ""
        debouncedSearchText = ""
        isSearchFocused = false
    }

    private func rename(_ conversation: Conversation) {
        // Ensure the conversation still exists and is not being modified
        guard conversations.contains(where: { $0.id == conversation.id }),
              currentViewModel?.conversation.id != conversation.id || currentViewModel?.isGenerating != true else { 
            return 
        }
        
        // Localized inline rename via alert-style prompt
        let alert = UIAlertController(
            title: String(localized: "Rename Chat"), 
            message: String(localized: "Enter a new name for this conversation"), 
            preferredStyle: .alert
        )
        
        alert.addTextField { textField in
            textField.text = conversation.title
            textField.placeholder = String(localized: "Conversation Title")
            textField.autocapitalizationType = .words
            textField.returnKeyType = .done
        }
        
        alert.addAction(UIAlertAction(title: String(localized: "Cancel"), style: .cancel))
        alert.addAction(UIAlertAction(title: String(localized: "Save"), style: .default) { _ in
            guard let textField = alert.textFields?.first,
                  let newTitle = textField.text?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !newTitle.isEmpty,
                  newTitle != conversation.title else {
                return
            }
            
            conversation.title = newTitle
            conversation.hasAutoGeneratedTitle = false
            conversation.lastUpdated = Date()
            
            do {
                try self.modelContext.save()
                AppHaptics.notification(.success)
            } catch {
                print("Failed to save renamed conversation: \(error)")
                AppHaptics.notification(.error)
                self.errorMessage = String(localized: "Failed to rename conversation. Please try again.")
            }
        })
        
        // Present alert safely
        DispatchQueue.main.async {
            UIApplication.shared.topMostViewController()?.present(alert, animated: true)
        }
    }

    private func deleteOffsets(_ offsets: IndexSet) {
        // Map the offsets through the filtered collection actually displayed in the List.
        // Capture IDs instead of references to avoid stale object access
        let idsToDelete = offsets.compactMap { index -> UUID? in
            guard filteredConversations.indices.contains(index) else { return nil }
            return filteredConversations[index].id
        }
        
        guard !idsToDelete.isEmpty else { return }
        
        // Find conversations by ID to ensure they still exist
        let toDelete = conversations.filter { convo in
            idsToDelete.contains(convo.id)
        }
        
        guard !toDelete.isEmpty else { return }
        let excludedConversationIDs = Set(toDelete.map(\.id))
        let attachmentURLsToDelete = attachmentURLs(in: toDelete)
        
        // CRITICAL FIX: Eagerly resolve all properties before deletion to avoid SwiftData fault errors
        // Force resolution of message properties that might be accessed during UI updates
        for convo in toDelete {
            _ = convo.messages.map { msg in
                // Eagerly access properties to resolve any faults
                _ = msg.role
                _ = msg.text
                _ = msg.order
            }
        }
        
        AppHaptics.impact(.medium)
        
        // Cancel operations for conversations being deleted - do this BEFORE updating selection
        for convo in toDelete {
            if let currentVM = currentViewModel, currentVM.conversation.id == convo.id {
                currentVM.cancelGeneration()
                // Clear the view model reference immediately
                currentViewModel = nil
            }
        }
        
        // Update selection before deleting
        if let selectedConvo = selection,
           idsToDelete.contains(selectedConvo.id) {
            let remainingConversations = conversations.filter { convo in
                !idsToDelete.contains(convo.id)
            }
            // Only auto-advance selection on regular width to avoid auto-push in compact.
            selection = (horizontalSizeClass == .regular) ? remainingConversations.first : nil
        }
        
        // Delete conversations with animation
        withAnimation {
            for convo in toDelete {
                modelContext.delete(convo)
            }
        }
        
        do {
            try modelContext.save()
            deleteAttachmentFiles(attachmentURLsToDelete)
            scheduleAttachmentStorageCleanup(excludingConversationIDs: excludedConversationIDs)
            AppHaptics.notification(.success)
        } catch {
            print("Failed to save after deleting conversations: \(error)")
            AppHaptics.notification(.error)
            errorMessage = String(localized: "Failed to delete conversations. Please try again.")
        }
    }

    private func delete(_ conversation: Conversation) {
        // CRITICAL FIX: Eagerly resolve all properties before deletion to avoid SwiftData fault errors
        // Capture ID and resolve all message faults to prevent "detached from context" errors
        let conversationID = conversation.id
        
        // Force resolution of message properties that might be accessed during UI updates
        // This prevents fault errors during the deletion animation
        _ = conversation.messages.map { msg in
            // Eagerly access properties to resolve any faults
            _ = msg.role
            _ = msg.text
            _ = msg.order
        }
        
        // Ensure we're not trying to delete a conversation that's already been deleted
        guard conversations.contains(where: { $0.id == conversationID }) else { return }
        let excludedConversationIDs: Set<UUID> = [conversationID]
        let attachmentURLsToDelete = attachmentURLs(in: [conversation])
        
        AppHaptics.impact(.medium)
        
        // Cancel any ongoing operations for this conversation - do this BEFORE updating selection
        if let currentVM = currentViewModel, currentVM.conversation.id == conversationID {
            currentVM.cancelGeneration()
            // Clear the view model reference immediately
            currentViewModel = nil
        }
        
        // Update selection safely - find next available conversation before deleting
        if selection?.id == conversationID {
            let remainingConversations = conversations.filter { $0.id != conversationID }
            // Only auto-advance selection on regular width to avoid auto-push in compact.
            selection = (horizontalSizeClass == .regular) ? remainingConversations.first : nil
        }
        
        // Wrap deletion in animation to ensure smooth UI updates
        withAnimation {
            modelContext.delete(conversation)
        }
        
        do {
            try modelContext.save()
            deleteAttachmentFiles(attachmentURLsToDelete)
            scheduleAttachmentStorageCleanup(excludingConversationIDs: excludedConversationIDs)
            AppHaptics.notification(.success)
        } catch {
            print("Failed to save after deleting conversation: \(error)")
            AppHaptics.notification(.error)
            errorMessage = String(localized: "Failed to delete conversation. Please try again.")
            
            // Try to recover by reloading the context
            do {
                try modelContext.save()
            } catch {
                print("Failed to recover after delete error: \(error)")
            }
        }
    }

    private func deleteAllChats() {
        Task {
            await deleteAllChatsAfterStoppingGeneration()
        }
    }

    private func deleteAllChatsAfterStoppingGeneration() async {
        AppHaptics.impact(.heavy)

        let conversationsToDelete = Array(conversations)
        let conversationIDsToDelete = Set(conversationsToDelete.map(\.id))
        guard await stopGenerationBeforeDeleting(conversationIDs: conversationIDsToDelete) else {
            return
        }

        currentViewModel = nil
        selection = nil

        let excludedConversationIDs = conversationIDsToDelete
        let attachmentURLsToDelete = attachmentURLs(in: conversationsToDelete)
        
        // Delete everything
        for convo in conversationsToDelete {
            modelContext.delete(convo)
        }
        
        do {
            try modelContext.save()
            deleteAttachmentFiles(attachmentURLsToDelete)
            scheduleAttachmentStorageCleanup(excludingConversationIDs: excludedConversationIDs)
            AppHaptics.notification(.success)
        } catch {
            print("Failed to save after deleting all conversations: \(error)")
            AppHaptics.notification(.error)
            errorMessage = String(localized: "Failed to delete all conversations. Please try again.")
        }
    }

    private func deleteAllExceptCurrent() {
        Task {
            await deleteAllExceptCurrentAfterStoppingGeneration()
        }
    }

    private func deleteAllExceptCurrentAfterStoppingGeneration() async {
        AppHaptics.impact(.heavy)
        
        // If there's a selection, keep it; otherwise keep the most recent conversation
        let current = selection ?? conversations.first
        
        guard let current = current else {
            // No conversations at all
            deleteAllChats()
            return
        }
        
        let conversationsToDelete = conversations.filter { $0.id != current.id }
        guard !conversationsToDelete.isEmpty else { return }
        let excludedConversationIDs = Set(conversationsToDelete.map(\.id))
        let attachmentURLsToDelete = attachmentURLs(in: conversationsToDelete)

        guard await stopGenerationBeforeDeleting(conversationIDs: excludedConversationIDs) else {
            return
        }
        
        for convo in conversationsToDelete {
            modelContext.delete(convo)
        }
        
        do {
            try modelContext.save()
            deleteAttachmentFiles(attachmentURLsToDelete)
            scheduleAttachmentStorageCleanup(excludingConversationIDs: excludedConversationIDs)
            AppHaptics.notification(.success)
            // Ensure the kept conversation is selected after deletion
            selection = current
        } catch {
            print("Failed to save after deleting conversations except current: \(error)")
            AppHaptics.notification(.error)
            errorMessage = String(localized: "Failed to delete conversations. Please try again.")
        }
    }

    private func exportAllChats() {
        guard !conversations.isEmpty else { return }
        cleanupExportFile()
        
        // Create export content as JSON
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        dateFormatter.timeStyle = .short
        
        var exportText = "# Chat Export\n"
        exportText += "Exported: \(dateFormatter.string(from: Date()))\n"
        exportText += "Total Conversations: \(conversations.count)\n\n"
        exportText += String(repeating: "=", count: 60) + "\n\n"
        
        for convo in conversations.sorted(by: { $0.lastUpdated > $1.lastUpdated }) {
            exportText += "## \(convo.title)\n"
            exportText += "Last Updated: \(dateFormatter.string(from: convo.lastUpdated))\n"
            if convo.reasoningMode {
                exportText += "Mode: Reasoning\n"
            }
            exportText += "\n"
            
            let sortedMessages = convo.messages.sorted(by: { $0.order < $1.order })
            for message in sortedMessages where message.role != .system {
                let roleLabel = message.role == .user ? "User" : "Assistant"
                exportText += "**\(roleLabel):**\n"
                let visibleText = message.userVisibleText
                if !visibleText.isEmpty {
                    exportText += visibleText + "\n"
                }
                for attachment in message.attachments where attachment.type == .image {
                    exportText += "[Image attachment: \(attachment.fileName)]\n"
                }
                exportText += "\n"
            }
            
            exportText += String(repeating: "-", count: 60) + "\n\n"
        }
        
        // Write to temporary file
        let fileName = "chat-export-\(Date().timeIntervalSince1970).txt"
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
        
        do {
            let exportData = Data(exportText.utf8)
            try exportData.write(to: tempURL, options: [.atomic, .completeFileProtection])
            exportURL = tempURL
            showExportSheet = true
        } catch {
            errorMessage = String(localized: "Failed to export chats. Please try again.")
        }
    }

    private func cleanupExportFile() {
        guard let exportURL else { return }
        try? FileManager.default.removeItem(at: exportURL)
        self.exportURL = nil
    }
    
    private func performAutoDeleteIfNeeded() {
        Task {
            await performAutoDeleteAfterStoppingGenerationIfNeeded()
        }
    }

    private func performAutoDeleteAfterStoppingGenerationIfNeeded() async {
        guard autoDeleteOldChats else { return }
        
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -autoDeleteDays, to: Date()) ?? Date()
        
        let conversationsToDelete = conversations.filter { convo in
            convo.lastUpdated < cutoffDate
        }
        
        guard !conversationsToDelete.isEmpty else { return }
        let excludedConversationIDs = Set(conversationsToDelete.map(\.id))
        let attachmentURLsToDelete = attachmentURLs(in: conversationsToDelete)

        guard await stopGenerationBeforeDeleting(conversationIDs: excludedConversationIDs) else {
            return
        }
        
        // Update selection if current selection is being deleted
        if let selectedConvo = selection,
           conversationsToDelete.contains(where: { $0.id == selectedConvo.id }) {
            let remainingConversations = conversations.filter { convo in
                !conversationsToDelete.contains(where: { $0.id == convo.id })
            }
            selection = (horizontalSizeClass == .regular) ? remainingConversations.first : nil
        }
        
        // Delete old conversations
        for convo in conversationsToDelete {
            modelContext.delete(convo)
        }
        
        do {
            try modelContext.save()
            deleteAttachmentFiles(attachmentURLsToDelete)
            scheduleAttachmentStorageCleanup(excludingConversationIDs: excludedConversationIDs)
            print("Auto-deleted \(conversationsToDelete.count) conversations older than \(autoDeleteDays) days")
        } catch {
            print("Failed to auto-delete old conversations: \(error)")
        }
    }

    private func stopGenerationBeforeDeleting(conversationIDs: Set<UUID>) async -> Bool {
        while let viewModel = currentViewModel,
              conversationIDs.contains(viewModel.conversation.id) {
            guard await viewModel.cancelGenerationAndWait() else {
                errorMessage = String(localized: "Couldn't stop the active response. Please try deleting again.")
                return false
            }
            if currentViewModel === viewModel {
                currentViewModel = nil
            }
        }
        return true
    }
}

// A container that owns temporary copies for Settings, enabling Cancel (discard) and Save (commit).
private struct SettingsSheetContainer: View {
    let hasChats: Bool
    let canDeleteAllExceptCurrent: Bool
    let onDeleteAll: () -> Void
    let onDeleteAllExceptCurrent: () -> Void
    let onExportChats: () -> Void
    let onDismiss: () -> Void

    @State private var draft: AppSettingsDraft

    init(
        hasChats: Bool,
        canDeleteAllExceptCurrent: Bool,
        onDeleteAll: @escaping () -> Void,
        onDeleteAllExceptCurrent: @escaping () -> Void,
        onExportChats: @escaping () -> Void,
        onDismiss: @escaping () -> Void
    ) {
        self.hasChats = hasChats
        self.canDeleteAllExceptCurrent = canDeleteAllExceptCurrent
        self.onDeleteAll = onDeleteAll
        self.onDeleteAllExceptCurrent = onDeleteAllExceptCurrent
        self.onExportChats = onExportChats
        self.onDismiss = onDismiss
        _draft = State(initialValue: AppSettingsDraft.load())
    }

    var body: some View {
        NavigationStack {
            SettingsSheet(
                settings: $draft,
                hasChats: hasChats,
                canDeleteAllExceptCurrent: canDeleteAllExceptCurrent,
                onDeleteAll: onDeleteAll,
                onDeleteAllExceptCurrent: onDeleteAllExceptCurrent,
                onExportChats: onExportChats
            )
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button(String(localized: "Cancel")) {
                        onDismiss()
                    }
                    .fontWeight(.semibold)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button(String(localized: "Save")) {
                        draft.persist()
                        onDismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
        .preferredColorScheme(colorScheme(for: draft.appAppearance))
        .environment(\.locale, Locale(identifier: draft.appLanguage))
    }

    private func colorScheme(for appearance: String) -> ColorScheme? {
        switch appearance {
        case "light": return .light
        case "dark": return .dark
        default: return nil // system
        }
    }
}

// Helper to present UIAlertController in SwiftUI quickly
private extension UIApplication {
    func topMostViewController(base: UIViewController? = nil) -> UIViewController? {
        let baseVC = base ?? connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .flatMap { $0.windows }
            .first { $0.isKeyWindow }?.rootViewController
        if let nav = baseVC as? UINavigationController {
            return topMostViewController(base: nav.visibleViewController)
        }
        if let tab = baseVC as? UITabBarController, let selected = tab.selectedViewController {
            return topMostViewController(base: selected)
        }
        if let presented = baseVC?.presentedViewController {
            return topMostViewController(base: presented)
        }
        return baseVC
    }
}

// MARK: - Share Sheet for exporting

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {
        // No updates needed
    }
}

#Preview {
    let schema = Schema([Conversation.self, Message.self, MessageAttachment.self])
    let container: ModelContainer
    do {
        container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
    } catch {
        fatalError("Failed to create ModelContainer for preview: \(error)")
    }

    return ContentView()
        .modelContainer(container)
}

// MARK: - Dynamic Height Text Editor

private struct DynamicHeightTextEditor: View {
    @Binding var text: String
    @Binding var height: CGFloat
    let placeholder: String
    
    var body: some View {
        TextField(placeholder, text: $text, axis: .vertical)
            .font(.body)
            .lineLimit(1...3)
            .textInputAutocapitalization(.none)
            .disableAutocorrection(true)
            .fixedSize(horizontal: false, vertical: true)
    }
}

// MARK: - Fading edges helper

private struct FadingEdges: View {
    var topHeight: CGFloat?
    var bottomHeight: CGFloat?
    var color: Color = .black.opacity(0.16)

    static func top(height: CGFloat = 14, color: Color = .black.opacity(0.16)) -> some View {
        FadingEdges(topHeight: height, bottomHeight: nil, color: color)
    }

    static func bottom(height: CGFloat = 14, color: Color = .black.opacity(0.16)) -> some View {
        FadingEdges(bottomHeight: height, color: color)
    }

    var body: some View {
        ZStack {
            if let h = topHeight {
                LinearGradient(colors: [color, .clear], startPoint: .top, endPoint: .bottom)
                    .frame(height: h)
                    .frame(maxWidth: .infinity)
            }
            if let h = bottomHeight {
                VStack {
                    Spacer()
                    LinearGradient(colors: [.clear, color], startPoint: .top, endPoint: .bottom)
                        .frame(height: h)
                        .frame(maxWidth: .infinity)
                }
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

// MARK: - Lightweight View helpers to reduce body complexity

private extension View {
    func deleteAllChatsAlert(
        isPresented: Binding<Bool>,
        hasSelection: Bool,
        canDeleteAllExceptCurrent: Bool,
        onDeleteAll: @escaping () -> Void,
        onDeleteAllExceptCurrent: @escaping () -> Void
    ) -> some View {
        alert(String(localized: "Delete previous chats?"), isPresented: isPresented) {
            if hasSelection && canDeleteAllExceptCurrent {
                Button(String(localized: "Delete All Chats Except Current"), role: .destructive) {
                    onDeleteAllExceptCurrent()
                }
            }
            Button(String(localized: "Delete All Chats"), role: .destructive) {
                onDeleteAll()
            }
            Button(String(localized: "Cancel"), role: .cancel) {}
        } message: {
            Text(String(localized: "This will permanently remove your chat history from this device."))
        }
    }
    
    func errorAlert(
        isPresented: Binding<Bool>,
        message: String,
        onDismiss: @escaping () -> Void
    ) -> some View {
        alert(String(localized: "Error"), isPresented: isPresented) {
            Button(String(localized: "OK")) {
                onDismiss()
            }
        } message: {
            Text(message)
        }
    }
    
    func settingsCover(
        isPresented: Binding<Bool>,
        hasChats: Bool,
        canDeleteAllExceptCurrent: Bool,
        onDeleteAll: @escaping () -> Void,
        onDeleteAllExceptCurrent: @escaping () -> Void,
        onExportChats: @escaping () -> Void,
        onDismiss: @escaping () -> Void
    ) -> some View {
        fullScreenCover(isPresented: isPresented) {
            SettingsSheetContainer(
                hasChats: hasChats,
                canDeleteAllExceptCurrent: canDeleteAllExceptCurrent,
                onDeleteAll: onDeleteAll,
                onDeleteAllExceptCurrent: onDeleteAllExceptCurrent,
                onExportChats: onExportChats,
                onDismiss: onDismiss
            )
        }
    }
}
