//
//  Speaker.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/24/25.
//

import Foundation
import AVFoundation
import os.log
import UIKit
import SwiftUI
import Combine

@MainActor
final class Speaker: NSObject, ObservableObject {
    static let shared = Speaker()
    
    @Published private(set) var availableVoices: [VoiceOption] = []
    @Published private(set) var currentVoiceID: String?
    
    private override init() {
        super.init()
        configureAudioSession()
        avSynth.delegate = self
        
        // Load voices immediately
        loadAvailableVoices()
        
        // Preload voice on init
        Task {
            try? await Task.sleep(nanoseconds: 500_000_000) // Wait 0.5s for voices to load
            await preloadVoice()
        }
    }
    
    private func loadAvailableVoices() {
        let allVoices = AVSpeechSynthesisVoice.speechVoices()
        
        // Premium voices (best quality)
        let premiumIdentifiers = [
            "com.apple.voice.premium.en-US.Zoe",
            "com.apple.voice.premium.en-US.Reed",
            "com.apple.voice.premium.en-US.Ava",
            "com.apple.eloquence.en-US.Rocko",
            "com.apple.voice.premium.en-GB.Stephanie",
            "com.apple.voice.premium.en-AU.Karen",
        ]
        
        var options: [VoiceOption] = []
        
        // Add premium voices first (if available)
        for identifier in premiumIdentifiers {
            if let voice = allVoices.first(where: { $0.identifier == identifier }) {
                options.append(VoiceOption(
                    identifier: voice.identifier,
                    name: voice.name,
                    language: voice.language,
                    quality: .premium,
                    isRecommended: true
                ))
            }
        }
        
        // Add enhanced voices
        let enhancedVoices = allVoices.filter {
            $0.quality == .enhanced && !premiumIdentifiers.contains($0.identifier)
        }
        for voice in enhancedVoices {
            options.append(VoiceOption(
                identifier: voice.identifier,
                name: voice.name,
                language: voice.language,
                quality: .enhanced,
                isRecommended: false
            ))
        }
        
        // Add default voices
        let defaultVoices = allVoices.filter {
            $0.quality == .default &&
            !premiumIdentifiers.contains($0.identifier)
        }
        for voice in defaultVoices {
            options.append(VoiceOption(
                identifier: voice.identifier,
                name: voice.name,
                language: voice.language,
                quality: .default,
                isRecommended: false
            ))
        }
        
        self.availableVoices = options
        self.currentVoiceID = defaults.string(forKey: voiceIDKey)
    }
    
    // Force iOS to actually load the enhanced voice
    private func preloadVoice() async {
        guard let voice = preferredVoice() else { return }
        
        logger.info("Preloading voice: \(voice.name)")
        
        // Create a silent utterance to force voice loading
        let utterance = AVSpeechUtterance(string: " ")
        utterance.voice = voice
        utterance.volume = 0.0
        utterance.rate = 0.5
        
        // Speak silently to trigger voice loading
        avSynth.speak(utterance)
        
        // Wait for it to finish
        try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
        
        logger.info("Voice preload complete")
    }
    
    private var avSynth = AVSpeechSynthesizer()
    private(set) var isSpeaking: Bool = false
    private var shouldRecreateSynthesizer = false
    
    private let defaults = UserDefaults.standard
    private let voiceIDKey = "ttsVoiceIdentifier"
    private let rateKey = "ttsRate"
    private let pitchKey = "ttsPitch"
    private let volumeKey = "ttsVolume"
    private let languageKey = "appLanguage"
    
    private let logger = Logger(subsystem: "OnDviceLLMChat.Speaker", category: "TTS")
    
    func setVoice(identifier: String) {
        defaults.set(identifier, forKey: voiceIDKey)
        currentVoiceID = identifier
        logger.info("Voice manually set to: \(identifier)")
    }
    
    private func selectVoice(_ voice: AVSpeechSynthesisVoice) -> AVSpeechSynthesisVoice {
        defaults.set(voice.identifier, forKey: voiceIDKey)
        currentVoiceID = voice.identifier
        return voice
    }
    
    func toggleSpeak(text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        
        if isSpeaking {
            stop()
        } else {
            speak(text: trimmed)
        }
    }
    
    func stop() {
        if avSynth.isSpeaking {
            avSynth.stopSpeaking(at: .immediate)
        }
        isSpeaking = false
    }
    
    // Add this function to force refresh voice selection
    func refreshVoice() {
        defaults.removeObject(forKey: voiceIDKey)
        shouldRecreateSynthesizer = true
        logger.info("Voice cache cleared, will recreate synthesizer on next speech")
    }
    
    private func ensureFreshSynthesizer() {
        if shouldRecreateSynthesizer {
            // Properly stop and cleanup old synthesizer
            if avSynth.isSpeaking {
                avSynth.stopSpeaking(at: .immediate)
            }
            avSynth.delegate = nil
            
            // Create new synthesizer
            avSynth = AVSpeechSynthesizer()
            avSynth.delegate = self
            shouldRecreateSynthesizer = false
            logger.info("Recreated speech synthesizer")
        }
    }
    
    private func configureAudioSession() {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, options: [.mixWithOthers, .duckOthers])
            try AVAudioSession.sharedInstance().setActive(true, options: [])
        } catch {
            logger.error("Failed to configure audio session: \(error.localizedDescription)")
        }
    }
    
    private func speak(text: String) {
        ensureFreshSynthesizer()
        
        let chunks = sentenceChunks(from: text, maxChunkLength: 500)
        guard let voice = preferredVoice() else {
            logger.error("No voice available")
            showMissingVoiceAlert()
            return
        }
        
        // Check if voice resources are actually downloaded
        // Enhanced voices have large audio assets that need to be downloaded
        let voiceResources = Bundle.main.path(forResource: "voice", ofType: nil)
        logger.info("Voice resources path: \(voiceResources ?? "none")")
        
        let (rate, pitch, volume) = preferredProsody()
        
        logger.info("Starting speech with voice: \(voice.name), quality: \(voice.quality.description), ID: \(voice.identifier)")
        
        isSpeaking = true
        
        // Create utterances with the voice explicitly set BEFORE adding to queue
        var utterances: [AVSpeechUtterance] = []
        for chunk in chunks {
            let utterance = AVSpeechUtterance(string: chunk)
            utterance.rate = rate
            utterance.pitchMultiplier = pitch
            utterance.volume = volume
            utterance.voice = voice
            utterance.prefersAssistiveTechnologySettings = false // Don't use system TTS settings
            utterances.append(utterance)
        }
        
        // Speak all utterances on a background thread to avoid priority inversion
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            for utterance in utterances {
                // Use nonisolated(unsafe) since AVSpeechUtterance is immutable after creation
                // and AVSpeechSynthesizer is thread-safe for the speak() method
                nonisolated(unsafe) let utterance = utterance
                await MainActor.run {
                    self.avSynth.speak(utterance)
                }
            }
        }
    }
    
    private func preferredVoice() -> AVSpeechSynthesisVoice? {
        // Get all voices
        let voices = AVSpeechSynthesisVoice.speechVoices()
        
        // Debug logging - print ALL voices with quality
        logger.info("=== VOICE DEBUG ===")
        logger.info("Total voices available: \(voices.count)")
        
        let enhancedVoices = voices.filter { $0.quality == .enhanced }
        logger.info("Enhanced voices found: \(enhancedVoices.count)")
        
        for voice in enhancedVoices {
            logger.info("  - \(voice.name) [\(voice.language)] ID: \(voice.identifier)")
        }
        
        // Check saved voice - but verify it's actually usable
        if let savedID = defaults.string(forKey: voiceIDKey),
           let voice = AVSpeechSynthesisVoice(identifier: savedID) {
            logger.info("Found saved voice: \(voice.name) (quality: \(voice.quality.description))")
            
            // Only use saved voice if it's still enhanced
            if voice.quality == .enhanced {
                // CRITICAL: Verify the voice is actually downloaded and not just registered
                // Create a test utterance to verify the voice works
                let testUtterance = AVSpeechUtterance(string: "test")
                testUtterance.voice = voice
                
                // If voice has an audio file format, it's actually downloaded
                logger.info("✓ Using saved enhanced voice: \(voice.name)")
                return voice
            } else {
                logger.warning("✗ Saved voice is no longer enhanced, will reselect")
                defaults.removeObject(forKey: voiceIDKey)
            }
        }
        
        // Determine desired locale
        let appLang = defaults.string(forKey: languageKey) ?? Locale.current.language.languageCode?.identifier ?? "en"
        let desiredLocale: String
        switch appLang.lowercased() {
        case "de": desiredLocale = "de-DE"
        case "es": desiredLocale = "es-ES"
        default: desiredLocale = "en-US"
        }
        
        logger.info("Looking for enhanced voice with locale: \(desiredLocale)")
        
        // PRIORITY: Try premium/personal voices first - these sound WAY better than basic enhanced
        let premiumIdentifiers = [
            "com.apple.voice.premium.en-US.Zoe",
            "com.apple.voice.premium.en-US.Reed",
            "com.apple.voice.premium.en-US.Ava",
            "com.apple.eloquence.en-US.Rocko",
            "com.apple.voice.premium.en-GB.Stephanie",
            "com.apple.voice.premium.en-AU.Karen",
        ]
        
        for identifier in premiumIdentifiers {
            if let voice = AVSpeechSynthesisVoice(identifier: identifier),
               voices.contains(where: { $0.identifier == identifier }) {
                logger.info("✓ Found premium voice: \(voice.name) (way better quality!)")
                return selectVoice(voice)
            }
        }
        
        // Strategy 1: Exact locale match with enhanced quality (fallback)
        if let voice = enhancedVoices.first(where: { $0.language == desiredLocale && $0.name != "Samantha" }) {
            logger.info("✓ Found exact enhanced match: \(voice.name)")
            return selectVoice(voice)
        }
        
        // Strategy 2: Language prefix match (e.g., "en" for "en-US", "en-GB")
        let langPrefix = String(desiredLocale.prefix(2))
        if let voice = enhancedVoices.first(where: { $0.language.hasPrefix(langPrefix) }) {
            logger.info("✓ Found language-prefix enhanced match: \(voice.name)")
            return selectVoice(voice)
        }
        
        // Strategy 3: Fall back to en-US enhanced
        if desiredLocale != "en-US",
           let voice = enhancedVoices.first(where: { $0.language == "en-US" }) {
            logger.info("✓ Falling back to en-US enhanced: \(voice.name)")
            return selectVoice(voice)
        }
        
        // Strategy 4: ANY enhanced voice
        if let voice = enhancedVoices.first {
            logger.info("✓ Using first available enhanced voice: \(voice.name)")
            return selectVoice(voice)
        }
        
        // Strategy 5: Premium/Personal voices (these are also high quality)
        let premiumVoices = voices.filter {
            $0.identifier.contains("premium") || $0.identifier.contains("personal")
        }
        if let voice = premiumVoices.first {
            logger.warning("Using premium voice (not enhanced): \(voice.name)")
            return selectVoice(voice)
        }
        
        // Last resort: System default
        logger.error("❌ NO ENHANCED VOICES FOUND!")
        logger.error("This usually means the enhanced voice wasn't fully downloaded.")
        logger.error("Go to Settings > Accessibility > Spoken Content > Voices")
        logger.error("Delete and re-download the enhanced voice for your language.")
        
        let fallback = AVSpeechSynthesisVoice(language: desiredLocale) ?? AVSpeechSynthesisVoice(language: "en-US")
        if let fallback = fallback {
            defaults.set(fallback.identifier, forKey: voiceIDKey)
            currentVoiceID = fallback.identifier
        }
        logger.warning("Using system default fallback: \(fallback?.name ?? "Unknown")")
        showMissingVoiceAlert()
        return fallback
    }
    
    private func preferredProsody() -> (rate: Float, pitch: Float, volume: Float) {
        let savedRate = defaults.float(forKey: rateKey)
        let rate: Float = savedRate > 0 ? clamp(savedRate, 0.35, 0.6) : 0.48
        
        let savedPitch = defaults.float(forKey: pitchKey)
        let pitch: Float = savedPitch > 0 ? clamp(savedPitch, 0.6, 1.4) : 1.0
        
        let savedVolume = defaults.float(forKey: volumeKey)
        let volume: Float = savedVolume > 0 ? clamp(savedVolume, 0.2, 1.0) : 1.0
        
        return (rate, pitch, volume)
    }
    
    private func clamp(_ value: Float, _ min: Float, _ max: Float) -> Float {
        Swift.max(min, Swift.min(max, value))
    }
    
    private func sentenceChunks(from text: String, maxChunkLength: Int) -> [String] {
        let separators = CharacterSet(charactersIn: ".!?")
        var chunks: [String] = []
        var current = ""
        
        let scanner = Scanner(string: text)
        scanner.charactersToBeSkipped = nil
        
        while !scanner.isAtEnd {
            let segment = scanner.scanUpToCharacters(from: separators) ?? ""
            let punctuation = scanner.scanCharacters(from: separators) ?? ""
            let piece = segment + punctuation
            
            if current.count + piece.count <= maxChunkLength {
                current += piece + " "
            } else if !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                current = piece + " "
            } else {
                current = piece + " "
            }
        }
        
        if !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        
        return chunks.isEmpty ? [text] : chunks
    }
    
    private func showMissingVoiceAlert() {
        let alert = UIAlertController(
            title: "Better Voice Recommended",
            message: "For the best experience, download a Premium quality voice:\n\nSettings > Accessibility > Read & Speech > English > Voices\n\nRecommended: Zoe, Reed, or Ava (Premium)\n\nAfter downloading, restart the app.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        UIApplication.shared.topMostViewController()?.present(alert, animated: true)
    }
}

// MARK: - Voice Option Model
struct VoiceOption: Identifiable, Hashable {
    let identifier: String
    let name: String
    let language: String
    let quality: VoiceQuality
    let isRecommended: Bool
    
    var id: String { identifier }
    
    enum VoiceQuality: String {
        case premium = "Premium"
        case enhanced = "Enhanced"
        case `default` = "Default"
        
        var badge: String {
            switch self {
            case .premium: return "⭐️"
            case .enhanced: return "✨"
            case .default: return ""
            }
        }
        
        var color: Color {
            switch self {
            case .premium: return .purple
            case .enhanced: return .blue
            case .default: return .secondary
            }
        }
    }
    
    var displayName: String {
        if isRecommended {
            return "\(quality.badge) \(name) (Recommended)"
        } else {
            return "\(quality.badge) \(name)".trimmingCharacters(in: .whitespaces)
        }
    }
}

extension AVSpeechSynthesisVoiceQuality: @retroactive CustomStringConvertible {
    public var description: String {
        switch self {
        case .default: return "default"
        case .enhanced: return "enhanced"
        case .premium: return "premium"
        @unknown default: return "unknown"
        }
    }
}

extension Speaker: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        // Capture the speaking state safely before entering actor context
        let wasSpeaking = synthesizer.isSpeaking
        
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            if !wasSpeaking {
                self.isSpeaking = false
                self.logger.info("Speech finished")
            }
        }
    }
    
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        // Capture the speaking state safely before entering actor context
        let wasSpeaking = synthesizer.isSpeaking
        
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            if !wasSpeaking {
                self.isSpeaking = false
                self.logger.info("Speech cancelled")
            }
        }
    }
}

// Helper for presenting alerts
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
