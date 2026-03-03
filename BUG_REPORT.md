# ChatLLM Bug Report
**Date:** 2026-03-03  
**Tester:** ClaudeClaw 🦀 (OpenClaw)  
**Build:** Debug – iPhone 17 Pro Simulator (iOS 26.0)  
**Build Status:** ✅ BUILD SUCCEEDED (no compile errors or warnings)

---

## Screenshot — App Launch State
![Launch Screenshot](chatllm_launch.png)

---

## 🔴 Critical Bugs

### BUG-01 · Error messages persisted as chat content and preview
**File:** `ChatViewModel.swift` → `streamAssistant()`  
**Confirmed by screenshot:** YES

When the model returns an empty response, the full multi-line error message (e.g. *"The model returned an empty response. This sometimes happens with certain requests. Please try: Rephrasing your question…"*) is stored directly in `message.text`. This means:
- The error message appears in the conversation thread as if it were a real assistant reply
- It shows as the chat list **preview text** in the sidebar
- It gets persisted to SwiftData and survives app restarts

**Steps to reproduce:**
1. Launch app — observe existing "Untitled Chat" with error preview
2. Open the chat — the error message is displayed as an assistant message

**Expected:** Error messages should be shown as transient UI overlays (e.g. a toast or inline error badge), NOT stored in `message.text`.

**Suggested fix:** Use a separate `@Published var lastError: String?` on `ChatViewModel` that is shown in the UI but never written to the message model.

---

### BUG-02 · Hardcoded placeholder bundle IDs will break Keychain on real device
**File:** `ChatViewModel.swift`

```swift
let tavilyKeyService = "com.yourapp.chatllm"  // ← placeholder never replaced!
let tavilyKeyAccount = "TavilyAPIKey"
let logger = Logger(subsystem: "com.yourapp.chatllm", ...)
```

The actual bundle ID is `Nevio.On-Device-LLM-Chat`. Keychain items stored under `"com.yourapp.chatllm"` will:
- Fail to match on entitlement-restricted real devices
- Make Tavily API keys invisible across app updates/reinstalls

**Suggested fix:**
```swift
let tavilyKeyService = Bundle.main.bundleIdentifier ?? "com.yourapp.chatllm"
let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "chatllm", category: "ChatViewModel")
```

---

### BUG-03 · `isContextWindowError()` logic never triggers for most error types
**File:** `ChatViewModel.swift` → `isContextWindowError(_:)`

The loop checks `if errorString.contains(keyword)` for words like `"token"`, `"too long"`, `"exceeds"`, `"maximum"` etc. — but inside the loop, the only `return true` path requires the string to contain `"context"` again AND one of `["length", "limit", "window"]`. This means:

- An error message saying *"token limit exceeded"* → returns `false` ❌  
- An error message saying *"input too long"* → returns `false` ❌  
- Only *"context length exceeded"* or *"context window"* would return `true`

**Effect:** Context window errors get displayed as generic errors instead of the helpful multi-step recovery message.

**Suggested fix:** Restructure to check compound conditions independently:
```swift
let hasContextWord = errorString.contains("context")
let hasLimitWord = ["length", "limit", "window", "token", "too long", "exceeds", "maximum"].contains { errorString.contains($0) }
if hasContextWord && hasLimitWord { return true }
if errorString.contains("too long") || errorString.contains("context window") { return true }
```

---

## 🟠 High Priority Issues

### BUG-04 · Web search year detection is outdated (stuck in 2025)
**File:** `ChatViewModel+WebSearch.swift` → `requiresWebSearch(for:)`

```swift
let timeSpecificKeywords = [
    "today", "yesterday", "this week", ... "2024", "2025"  // ← missing 2026!
]
```

Queries like *"what happened in 2026"* or *"latest iOS 26 features"* won't trigger web search even if Tavily is configured, because `"2026"` is not in the list.

**Fix:** Add the current year dynamically:
```swift
let currentYear = Calendar.current.component(.year, from: Date())
var timeSpecificKeywords = ["today", "yesterday", ..., "2024", "2025"]
timeSpecificKeywords.append(String(currentYear))
```

---

### BUG-05 · Deprecated API: `disableAutocorrection(true)` 
**File:** `ContentView.swift` → `DynamicHeightTextEditor`

```swift
.disableAutocorrection(true)  // ← deprecated in iOS 16
```

Should be:
```swift
.autocorrectionDisabled()
```

Will generate deprecation warnings in future Xcode versions.

---

### BUG-06 · MockGenerator used silently as fallback
**File:** `ContentView.swift` → `createViewModel(for:)`

```swift
let generator: LLMGenerator = onDevice.isAvailable() ? onDevice : MockGenerator()
```

If the on-device model isn't available (simulator, unsupported device), the app silently falls back to `MockGenerator` — presumably producing fake/hardcoded responses. **Users have no idea they're talking to a mock.** The `availabilityOverlay()` does show a warning, but only over the chat detail view, not in the sidebar.

**Fix:** Either prominently label mock responses, or disable the send button entirely when in mock mode.

---

### BUG-07 · `saveCount` never resets on early-return path
**File:** `ChatViewModel.swift` → `scheduleCoalescedSave()`

```swift
let timeSinceLastSave = Date().timeIntervalSince(lastSaveTime)
if timeSinceLastSave < 0.15 {
    return  // ← saveCount keeps incrementing!
}
```

After 25 skipped saves (hits `forceSaveThreshold`), `immediateSave()` is called, but if the early-return path is always hit (e.g. save called rapidly in a loop), `saveCount` can overflow past `forceSaveThreshold` without ever saving.

---

## 🟡 Medium Priority Issues

### BUG-08 · Chat title "Untitled Chat" not migrated to "New Chat"
**Confirmed by screenshot:** Chat appears as "Untitled Chat"

New chats are created with `title: String(localized: "New Chat")` in current code, but old records in the database use the old "Untitled Chat" title. There's no migration path.

**Fix:** Add a SwiftData migration or a one-time `onAppear` migration that renames "Untitled Chat" → "New Chat".

---

### BUG-09 · Useless recovery save in `delete()` catch block
**File:** `ContentView.swift` → `delete(_:)`

```swift
} catch {
    errorMessage = "Failed to delete conversation."
    do {
        try modelContext.save()  // ← won't help; same context, same state
    } catch {
        print("Failed to recover after delete error: \(error)")
    }
}
```

Calling `save()` immediately after a failed `save()` with no intervening state change will fail for the exact same reason. Remove the nested recovery attempt.

---

### BUG-10 · `reasoningAvailable` uses fragile string matching
**File:** `ModelBackendBridge.swift` → `reasoningAvailable`

```swift
return modelID.contains("qwen")  // ← any model with "qwen" in name will match
```

A model named `"non-thinking-qwen-lite"` would incorrectly be marked as reasoning-capable. Should check against an explicit allowlist.

---

### BUG-11 · `DynamicHeightTextEditor` has unused `height` binding
**File:** `ContentView.swift`

```swift
DynamicHeightTextEditor(
    text: $searchText,
    height: .constant(0),  // ← comment says "Not used anymore"
```

Dead parameter. Remove it from the struct definition and all call sites to reduce confusion.

---

### BUG-12 · `topMostViewController` uses deprecated `windows` property
**File:** `ContentView.swift`

```swift
.compactMap { $0 as? UIWindowScene }
.flatMap { $0.windows }  // ← deprecated in iOS 15, use windowScene.keyWindow
```

Use `$0.keyWindow` instead of `.windows.first { $0.isKeyWindow }`.

---

## 🟢 Low Priority / UX Observations

### UX-01 · No timestamps in the chat list
The sidebar chat list shows no timestamps on conversations — users can't tell which chat is most recent at a glance without relying on sort order.

### UX-02 · Locale forced to `"en"` on first launch
`@AppStorage("appLanguage") private var appLanguage: String = "en"` forces the entire app to English on first launch, even on German/Spanish devices. Consider defaulting to `Locale.current.language.languageCode?.identifier ?? "en"`.

### UX-03 · No timestamp shown on the "Apple Intelligence" availability banner
The banner at the top of the sidebar is always visible even when Apple Intelligence is fully working. Consider making it collapsible or hiding it after first acknowledgment.

### UX-04 · Model loading state not shown when switching to MLX backend
When a user switches to the MLX backend, there's no clear indication in the chat list sidebar that a model is loading. Only the chat detail view reflects this.

---

## ✅ What Works Correctly
- App builds cleanly with zero errors
- App installs and launches on iPhone 17 Pro Simulator
- Sidebar navigation renders correctly
- Search bar with glass effect renders properly
- New Chat button functional
- Settings gear icon visible in sidebar header
- Apple Intelligence availability detection ("On-device available" correctly reported)
- Swipe-to-delete interactions available on chat rows
- Context menu (Rename/Delete) accessible via long-press
- Haptic feedback wired throughout (selection, create, delete)
- Auto-delete old chats setting present and wired
- Export chats functionality implemented
- Dark/Light/System appearance switching implemented

---

## Summary Table

| ID | Severity | File | Issue |
|---|---|---|---|
| BUG-01 | 🔴 Critical | ChatViewModel.swift | Error messages stored as chat content |
| BUG-02 | 🔴 Critical | ChatViewModel.swift | Hardcoded placeholder bundle IDs |
| BUG-03 | 🔴 Critical | ChatViewModel.swift | `isContextWindowError` logic broken |
| BUG-04 | 🟠 High | ChatViewModel+WebSearch.swift | Year 2026 missing from search triggers |
| BUG-05 | 🟠 High | ContentView.swift | Deprecated `disableAutocorrection` API |
| BUG-06 | 🟠 High | ContentView.swift | MockGenerator used silently |
| BUG-07 | 🟠 High | ChatViewModel.swift | `saveCount` not reset on early-return |
| BUG-08 | 🟡 Medium | SwiftData Model | "Untitled Chat" not migrated |
| BUG-09 | 🟡 Medium | ContentView.swift | Useless recovery save in `delete()` |
| BUG-10 | 🟡 Medium | ModelBackendBridge.swift | Fragile `reasoningAvailable` check |
| BUG-11 | 🟡 Medium | ContentView.swift | Unused `height` binding |
| BUG-12 | 🟡 Medium | ContentView.swift | Deprecated `windows` API |
| UX-01 | 🟢 Low | ContentView.swift | No timestamps in chat list |
| UX-02 | 🟢 Low | ContentView.swift | Locale forced to "en" on first launch |
| UX-03 | 🟢 Low | ContentView.swift | Apple Intelligence banner always visible |
| UX-04 | 🟢 Low | ModelBackendBridge.swift | MLX loading not shown in sidebar |

---

*Report generated by ClaudeClaw 🦀 via static code analysis + simulator launch testing*
