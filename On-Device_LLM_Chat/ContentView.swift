//
//  ContentView.swift
//  On-Device_LLM_Chat
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

    // App-wide preferences
    @AppStorage("defaultSystemPrompt") private var defaultSystemPrompt: String = ""
    @AppStorage("appAppearance") private var appAppearance: String = "system" // system | light | dark
    @AppStorage("appLanguage") private var appLanguage: String = "en" // en | de | es
    @AppStorage("reasoningModeDefault") private var reasoningModeDefault: Bool = false
    @AppStorage("messageFontSize") private var messageFontSize: Double = 16.0
    @AppStorage("autoDeleteOldChats") private var autoDeleteOldChats: Bool = false
    @AppStorage("autoDeleteDays") private var autoDeleteDays: Int = 30

    private var preferredScheme: ColorScheme? {
        switch appAppearance {
        case "light": return .light
        case "dark": return .dark
        default: return nil
        }
    }

    var body: some View {
        splitView
            // Apply selected language to entire UI
            .environment(\.locale, Locale(identifier: appLanguage))
            .preferredColorScheme(preferredScheme)
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
                initialDefaultSystemPrompt: defaultSystemPrompt,
                initialAppearance: appAppearance,
                initialLanguage: appLanguage,
                hasChats: !conversations.isEmpty,
                canDeleteAllExceptCurrent: conversations.count > 1,
                onDeleteAll: { showDeleteAllAlert = true },
                onDeleteAllExceptCurrent: { deleteAllExceptCurrent() },
                onExportChats: { exportAllChats() },
                onSave: { newPrompt, newAppearance, newLanguage in
                    defaultSystemPrompt = newPrompt
                    appAppearance = newAppearance
                    appLanguage = newLanguage
                    showSettings = false
                },
                onCancel: { showSettings = false }
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
    
    private func handleSelectionChange(_ oldSelection: Conversation?, _ newSelection: Conversation?) {
        _ = oldSelection?.id
        let newID = newSelection?.id
        // Entering draft mode: selection becomes nil but we want to keep the draft vm alive.
        if newID == nil && draftConversation != nil { return }
        // Navigating to a real conversation discards any open draft.
        if newID != nil { draftConversation = nil }
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

    private func scheduleAttachmentStorageCleanup(excludingConversationIDs excludedConversationIDs: Set<UUID> = []) {
        let referencedURLs = referencedAttachmentURLs(excludingConversationIDs: excludedConversationIDs)
        Task {
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
        
        // Pre-compute search terms once
        let searchTerms = trimmed.lowercased().components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        guard !searchTerms.isEmpty else { return conversations }
        
        // Optimize search with early exits and better memory usage
        return conversations.filter { convo in
            // Fast path: check title first (most common match)
            let titleLowercased = convo.title.lowercased()
            
            // Use range(of:) for better performance than localizedCaseInsensitiveContains
            let titleMatches = searchTerms.allSatisfy { term in
                titleLowercased.range(of: term) != nil
            }
            if titleMatches { return true }
            
            // Only search messages if title doesn't match
            // CRITICAL FIX: Safely access message properties to avoid SwiftData fault errors
            // Access properties carefully - SwiftData may return stale data for deleted objects
            let recentMessageSnapshots = convo.messages
                .compactMap { msg -> (order: Int, text: String)? in
                    // Access properties - SwiftData will handle faults internally
                    let role = msg.role
                    let order = msg.order
                    let text = msg.text
                    
                    guard role != .system else { return nil }
                    return (order: order, text: text)
                }
                .sorted(by: { $0.order > $1.order })
                .prefix(5) // Reduced from 10 to 5 for even better performance
            
            // Early exit on first matching message
            for snapshot in recentMessageSnapshots {
                let messageLowercased = snapshot.text.lowercased()
                let matches = searchTerms.allSatisfy { term in
                    messageLowercased.range(of: term) != nil
                }
                if matches {
                    return true
                }
            }
            
            return false
        }
    }

    @ViewBuilder
    private var sidebar: some View {
        ZStack(alignment: .bottom) {
            // Main list content — keep selection binding for NavigationSplitView
            List(selection: $selection) {
                // Conversations
                Section {
                    if filteredConversations.isEmpty && !searchText.isEmpty {
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
                        // Give ForEach an explicit id to avoid inference issues
                        ForEach(filteredConversations, id: \.id) { convo in
                            // Capture the ID to avoid stale reference issues in closures
                            let conversationID = convo.id
                            
                            // No NavigationLink, we drive selection ourselves.
                            ConversationRow(conversation: convo)
                                .contentShape(Rectangle()) // full row tap target
                                .onTapGesture {
                                    // Light haptic for selection
                                    let haptic = UISelectionFeedbackGenerator()
                                    haptic.selectionChanged()
                                    // Manual selection triggers NavigationSplitView navigation (including compact).
                                    selection = convo
                                }
                                .contextMenu {
                                    Button {
                                        // Find conversation by ID to ensure it still exists
                                        if let conversation = conversations.first(where: { $0.id == conversationID }) {
                                            rename(conversation)
                                        }
                                    } label: {
                                        Label(String(localized: "Rename"), systemImage: "pencil")
                                    }

                                    Button(role: .destructive) { 
                                        // Find conversation by ID to ensure it still exists
                                        if let conversation = conversations.first(where: { $0.id == conversationID }) {
                                            delete(conversation)
                                        }
                                    } label: {
                                        Label(String(localized: "Delete"), systemImage: "trash")
                                    }
                                }
                                .swipeActions(edge: HorizontalEdge.trailing) {
                                    Button(role: .destructive) {
                                        // Find conversation by ID to ensure it still exists
                                        if let conversation = conversations.first(where: { $0.id == conversationID }) {
                                            delete(conversation)
                                        }
                                    } label: {
                                        Label(String(localized: "Delete"), systemImage: "trash")
                                    }
                                }
                        }
                        .onDelete(perform: deleteOffsets)
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
        // Pin the Apple Intelligence banner to the top; it won't scroll off-screen.
        .safeAreaInset(edge: .top) {
            SidebarHeader(
                isAvailable: SystemLanguageModel.default.availability == .available,
                openSettings: { showSettings = true },
                hasChats: !conversations.isEmpty
            )
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
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
                            // Light haptic for clearing search
                            let haptic = UIImpactFeedbackGenerator(style: .light)
                            haptic.impactOccurred()
                            searchText = ""
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

                Button {
                    withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                        startDraftChat()
                    }
                } label: {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 22, weight: .semibold))
                        .scaleEffect(isSearchFocused ? 0.9 : 1.0)
                        .frame(width: 54, height: 54)
                        .contentShape(Circle())
                }
                .contentShape(Circle())
                .buttonStyle(.plain)
                .glassEffect(.regular.interactive(), in: .circle)
                .accessibilityLabel(String(localized: "New Chat"))
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
        let haptic = UIImpactFeedbackGenerator(style: .medium)
        haptic.impactOccurred()
        Task { @MainActor in
            currentViewModel?.cancelGeneration()
            let convo = Conversation(title: String(localized: "New Chat"))
            convo.reasoningMode = reasoningModeDefault
            let trimmed = defaultSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                let sys = Message(role: .system, text: trimmed, order: 0, conversation: convo, isFinal: true)
                convo.messages.append(sys)
            }
            // Set draft BEFORE clearing selection so handleSelectionChange preserves this vm.
            draftConversation = convo
            currentViewModel = createViewModel(for: convo)
            selection = nil
            // Push detail column on compact (iPhone) so the draft chat is immediately visible.
            preferredCompactColumn = .detail
            if !searchText.isEmpty {
                searchText = ""
                isSearchFocused = false
            }
        }
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
            conversation.lastUpdated = Date()
            
            do {
                try self.modelContext.save()
                // Success haptic for rename
                let haptic = UINotificationFeedbackGenerator()
                haptic.notificationOccurred(.success)
            } catch {
                print("Failed to save renamed conversation: \(error)")
                // Error haptic
                let haptic = UINotificationFeedbackGenerator()
                haptic.notificationOccurred(.error)
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
        
        // Haptic feedback for deletion
        let haptic = UIImpactFeedbackGenerator(style: .medium)
        haptic.impactOccurred()
        
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
            scheduleAttachmentStorageCleanup(excludingConversationIDs: excludedConversationIDs)
            // Success haptic for deletion
            let successHaptic = UINotificationFeedbackGenerator()
            successHaptic.notificationOccurred(.success)
        } catch {
            print("Failed to save after deleting conversations: \(error)")
            // Error haptic
            let errorHaptic = UINotificationFeedbackGenerator()
            errorHaptic.notificationOccurred(.error)
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
        
        // Haptic feedback for deletion
        let haptic = UIImpactFeedbackGenerator(style: .medium)
        haptic.impactOccurred()
        
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
            scheduleAttachmentStorageCleanup(excludingConversationIDs: excludedConversationIDs)
            // Success haptic for deletion
            let successHaptic = UINotificationFeedbackGenerator()
            successHaptic.notificationOccurred(.success)
        } catch {
            print("Failed to save after deleting conversation: \(error)")
            // Error haptic
            let errorHaptic = UINotificationFeedbackGenerator()
            errorHaptic.notificationOccurred(.error)
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
        // Heavy haptic for bulk delete action
        let haptic = UIImpactFeedbackGenerator(style: .heavy)
        haptic.impactOccurred()
        
        // Cancel any ongoing operations
        currentViewModel?.cancelGeneration()
        currentViewModel = nil
        selection = nil
        
        // Create a copy of the conversations array to avoid modification during iteration
        let conversationsToDelete = Array(conversations)
        let excludedConversationIDs = Set(conversationsToDelete.map(\.id))
        
        // Delete everything
        for convo in conversationsToDelete {
            modelContext.delete(convo)
        }
        
        do {
            try modelContext.save()
            scheduleAttachmentStorageCleanup(excludingConversationIDs: excludedConversationIDs)
            // Success haptic for bulk deletion
            let successHaptic = UINotificationFeedbackGenerator()
            successHaptic.notificationOccurred(.success)
        } catch {
            print("Failed to save after deleting all conversations: \(error)")
            // Error haptic
            let errorHaptic = UINotificationFeedbackGenerator()
            errorHaptic.notificationOccurred(.error)
            errorMessage = String(localized: "Failed to delete all conversations. Please try again.")
        }
    }

    private func deleteAllExceptCurrent() {
        // Heavy haptic for bulk delete action
        let haptic = UIImpactFeedbackGenerator(style: .heavy)
        haptic.impactOccurred()
        
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
        
        // Cancel operations in view model if it's for a conversation being deleted
        if let currentVM = currentViewModel,
           conversationsToDelete.contains(where: { $0.id == currentVM.conversation.id }) {
            currentVM.cancelGeneration()
            currentViewModel = nil
        }
        
        for convo in conversationsToDelete {
            modelContext.delete(convo)
        }
        
        do {
            try modelContext.save()
            scheduleAttachmentStorageCleanup(excludingConversationIDs: excludedConversationIDs)
            // Success haptic for bulk deletion
            let successHaptic = UINotificationFeedbackGenerator()
            successHaptic.notificationOccurred(.success)
            // Ensure the kept conversation is selected after deletion
            selection = current
        } catch {
            print("Failed to save after deleting conversations except current: \(error)")
            // Error haptic
            let errorHaptic = UINotificationFeedbackGenerator()
            errorHaptic.notificationOccurred(.error)
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
                exportText += message.displayText + "\n\n"
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
        guard autoDeleteOldChats else { return }
        
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -autoDeleteDays, to: Date()) ?? Date()
        
        let conversationsToDelete = conversations.filter { convo in
            convo.lastUpdated < cutoffDate
        }
        
        guard !conversationsToDelete.isEmpty else { return }
        let excludedConversationIDs = Set(conversationsToDelete.map(\.id))
        
        // Cancel operations for conversations being deleted
        for convo in conversationsToDelete {
            if let currentVM = currentViewModel, currentVM.conversation.id == convo.id {
                currentVM.cancelGeneration()
                currentViewModel = nil
            }
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
            scheduleAttachmentStorageCleanup(excludingConversationIDs: excludedConversationIDs)
            print("Auto-deleted \(conversationsToDelete.count) conversations older than \(autoDeleteDays) days")
        } catch {
            print("Failed to auto-delete old conversations: \(error)")
        }
    }
}

// A container that owns temporary copies for Settings, enabling Cancel (discard) and Save (commit).
private struct SettingsSheetContainer: View {
    let initialDefaultSystemPrompt: String
    let initialAppearance: String
    let initialLanguage: String
    let hasChats: Bool
    let canDeleteAllExceptCurrent: Bool
    let onDeleteAll: () -> Void
    let onDeleteAllExceptCurrent: () -> Void
    let onExportChats: () -> Void
    let onSave: (String, String, String) -> Void
    let onCancel: () -> Void

    @State private var tempDefaultSystemPrompt: String
    @State private var tempAppearance: String
    @State private var tempLanguage: String

    init(
        initialDefaultSystemPrompt: String,
        initialAppearance: String,
        initialLanguage: String,
        hasChats: Bool,
        canDeleteAllExceptCurrent: Bool,
        onDeleteAll: @escaping () -> Void,
        onDeleteAllExceptCurrent: @escaping () -> Void,
        onExportChats: @escaping () -> Void,
        onSave: @escaping (String, String, String) -> Void,
        onCancel: @escaping () -> Void
    ) {
        self.initialDefaultSystemPrompt = initialDefaultSystemPrompt
        self.initialAppearance = initialAppearance
        self.initialLanguage = initialLanguage
        self.hasChats = hasChats
        self.canDeleteAllExceptCurrent = canDeleteAllExceptCurrent
        self.onDeleteAll = onDeleteAll
        self.onDeleteAllExceptCurrent = onDeleteAllExceptCurrent
        self.onExportChats = onExportChats
        self.onSave = onSave
        self.onCancel = onCancel
        _tempDefaultSystemPrompt = State(initialValue: initialDefaultSystemPrompt)
        _tempAppearance = State(initialValue: initialAppearance)
        _tempLanguage = State(initialValue: initialLanguage)
    }

    var body: some View {
        NavigationStack {
            SettingsSheet(
                defaultSystemPrompt: $tempDefaultSystemPrompt,
                appAppearance: $tempAppearance,
                appLanguage: $tempLanguage,
                hasChats: hasChats,
                canDeleteAllExceptCurrent: canDeleteAllExceptCurrent,
                onDeleteAll: onDeleteAll,
                onDeleteAllExceptCurrent: onDeleteAllExceptCurrent,
                onExportChats: onExportChats
            )
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button(String(localized: "Cancel")) {
                        // Discard changes
                        onCancel()
                    }
                    .fontWeight(.semibold)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button(String(localized: "Save")) {
                        // Commit changes
                        onSave(tempDefaultSystemPrompt, tempAppearance, tempLanguage)
                    }
                    .fontWeight(.semibold)
                }
            }
        }
        // Drive the sheet’s appearance and language from the temporary selections
        .preferredColorScheme(colorScheme(for: tempAppearance))
        .environment(\.locale, Locale(identifier: tempLanguage))
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
        initialDefaultSystemPrompt: String,
        initialAppearance: String,
        initialLanguage: String,
        hasChats: Bool,
        canDeleteAllExceptCurrent: Bool,
        onDeleteAll: @escaping () -> Void,
        onDeleteAllExceptCurrent: @escaping () -> Void,
        onExportChats: @escaping () -> Void,
        onSave: @escaping (String, String, String) -> Void,
        onCancel: @escaping () -> Void
    ) -> some View {
        fullScreenCover(isPresented: isPresented) {
            SettingsSheetContainer(
                initialDefaultSystemPrompt: initialDefaultSystemPrompt,
                initialAppearance: initialAppearance,
                initialLanguage: initialLanguage,
                hasChats: hasChats,
                canDeleteAllExceptCurrent: canDeleteAllExceptCurrent,
                onDeleteAll: onDeleteAll,
                onDeleteAllExceptCurrent: onDeleteAllExceptCurrent,
                onExportChats: onExportChats,
                onSave: onSave,
                onCancel: onCancel
            )
        }
    }
}
