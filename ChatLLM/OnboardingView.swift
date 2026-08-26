//
//  OnboardingView.swift
//  ChatLLM
//
//  First-launch onboarding flow for new users.
//

import SwiftUI

struct OnboardingView: View {
    enum Step: Int, CaseIterable {
        case intro
        case tavily
        case mlx

        var title: String {
            switch self {
            case .intro:
                return "Welcome to ChatLLM"
            case .tavily:
                return "Web Search"
            case .mlx:
                return "Local Models"
            }
        }
    }

    let onFinish: () -> Void

    @ObservedObject private var modelBackendBridge = ModelBackendBridge.shared

    @State private var step: Step = .intro
    @State private var tavilyKey: String = TavilyAPIKeyStore.currentKey() ?? ""
    @State private var selectedModelID: String?

    private var modelManager: MLXModelManager? { modelBackendBridge.modelManager }

    private var selectedModel: MLXModelManager.MLXModelInfo? {
        guard let selectedModelID else { return modelManager?.availableModels.first }
        return modelManager?.model(withID: selectedModelID) ?? modelManager?.availableModels.first
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                header
                progressDots
                content
                Spacer(minLength: 0)
                footer
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .background(backgroundGradient)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Skip") {
                        finishOnboarding()
                    }
                    .accessibilityIdentifier("onboarding.skip")
                }
            }
        }
        .overlay(alignment: .topLeading) {
            Color.clear
                .frame(width: 1, height: 1)
                .allowsHitTesting(false)
                .accessibilityIdentifier("onboarding.root")
        }
        .onAppear {
            if selectedModelID == nil {
                selectedModelID = modelManager?.availableModels.first?.id
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(step.title)
                .font(.title)
                .fontWeight(.bold)
                .foregroundStyle(.primary)
                .accessibilityIdentifier(headerTitleIdentifier)

            Text(stepSubtitle)
                .font(.body)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var progressDots: some View {
        HStack(spacing: 8) {
            ForEach(Step.allCases, id: \.rawValue) { currentStep in
                Capsule()
                    .fill(currentStep == step ? Color.accentColor : Color.secondary.opacity(0.2))
                    .frame(width: currentStep == step ? 28 : 10, height: 10)
                    .animation(.easeInOut(duration: 0.2), value: step)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var content: some View {
        switch step {
        case .intro:
            introStep
        case .tavily:
            tavilyStep
        case .mlx:
            mlxStep
        }
    }

    private var introStep: some View {
        VStack(alignment: .leading, spacing: 18) {

            onboardingCard(
                title: "Start instantly",
                systemImage: "apple.logo",
                description: "The app works right away with Apple Foundation Models when your device supports them."
            )

            onboardingCard(
                title: "Run bigger local models",
                systemImage: "square.stack.3d.down.forward",
                description: "You can optionally download MLX models for fully local chats, reasoning, and native image support."
            )

            onboardingCard(
                title: "Search when needed",
                systemImage: "magnifyingglass.circle",
                description: "Add a Tavily API key if you want the assistant to search the web for current information."
            )
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var tavilyStep: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Tavily is optional")
                .font(.title2)
                .fontWeight(.semibold)

            Text("Paste a Tavily API key if you want web search. You can also skip this and use the rest of the app normally.")
                .foregroundStyle(.secondary)

            SecureField("Enter your Tavily API key", text: $tavilyKey)
                .textContentType(.password)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .padding()
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .accessibilityIdentifier("onboarding.tavily.key")

            Label("Your key is stored in Keychain and sent to Tavily only to authenticate web searches.", systemImage: "lock.shield")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var mlxStep: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Download an MLX model if you want")
                .font(.title2)
                .fontWeight(.semibold)

            Text("This is optional. You can finish onboarding right now, or start a model download and continue using the app while it runs.")
                .foregroundStyle(.secondary)

            if let modelManager {
                ScrollView {
                    VStack(spacing: 12) {
                        ForEach(modelManager.availableModels) { model in
                            modelRow(for: model, modelManager: modelManager)
                        }
                    }
                }
                .frame(maxHeight: 300)

                if let selectedModel {
                    downloadStatus(for: selectedModel, modelManager: modelManager)
                }
            } else {
                LoadingIndicator()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func modelRow(
        for model: MLXModelManager.MLXModelInfo,
        modelManager: MLXModelManager
    ) -> some View {
        Button {
            selectedModelID = model.id
        } label: {
            HStack(alignment: .top, spacing: 12) {
                modelSelectionIcon(for: model)
                modelDescription(for: model, modelManager: modelManager)
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("onboarding.mlx.model.\(model.id)")
    }

    private func modelSelectionIcon(for model: MLXModelManager.MLXModelInfo) -> some View {
        Image(systemName: selectedModelID == model.id ? "checkmark.circle.fill" : "circle")
            .foregroundStyle(selectedModelID == model.id ? Color.accentColor : Color.secondary)
    }

    private func modelDescription(
        for model: MLXModelManager.MLXModelInfo,
        modelManager: MLXModelManager
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(model.displayName)
                    .font(.headline)
                    .foregroundStyle(.primary)
                Spacer()
                Text(model.downloadSizeLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text(model.description)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            if model.isAvailable {
                Label("Already downloaded", systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else if let issue = modelManager.availabilityIssue(for: model) {
                Text(issue)
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    @ViewBuilder
    private func downloadStatus(
        for model: MLXModelManager.MLXModelInfo,
        modelManager: MLXModelManager
    ) -> some View {
        if let activeDownloadModel = modelManager.activeDownloadModel {
            VStack(alignment: .leading, spacing: 8) {
                Text(activeDownloadModel.id == model.id ? "Downloading \(activeDownloadModel.displayName)" : "\(activeDownloadModel.displayName) is downloading")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                ProgressView(value: modelManager.downloadProgress)
                Text("Downloading \(Int(modelManager.downloadProgress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("You can finish onboarding now and keep using the app while this continues.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .accessibilityIdentifier("onboarding.mlx.progress.\(activeDownloadModel.id)")
        } else if model.isAvailable {
            Label("MLX model is installed.", systemImage: "checkmark.circle.fill")
                .font(.subheadline)
                .foregroundStyle(.green)
        } else if let issue = modelManager.availabilityIssue(for: model) {
            Text(issue)
                .font(.caption)
                .foregroundStyle(.orange)
        } else if let error = modelManager.downloadError(for: model.id) {
            Text(error)
                .font(.caption)
                .foregroundStyle(.red)
        } else {
            Button {
                modelManager.startDownload(for: model)
            } label: {
                Label("Download \(model.downloadSizeLabel)", systemImage: "arrow.down.circle.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(modelManager.hasActiveDownload(excluding: model.id))
            .accessibilityIdentifier("onboarding.mlx.download")
        }
    }

    private var footer: some View {
        HStack(spacing: 12) {
            if step != .intro {
                Button("Back") {
                    previousStep()
                }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("onboarding.back")
            }

            Spacer()

            Button(primaryButtonTitle) {
                advance()
            }
            .buttonStyle(.borderedProminent)
            .accessibilityIdentifier(primaryButtonIdentifier)
        }
    }

    private func onboardingCard(title: String, systemImage: String, description: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: systemImage)
                .font(.headline)
            Text(description)
                .foregroundStyle(.secondary)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
    }

    private var backgroundGradient: some View {
        LinearGradient(
            colors: [
                Color(uiColor: .systemBackground),
                Color.accentColor.opacity(0.08),
                Color(uiColor: .secondarySystemBackground)
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }

    private var stepSubtitle: String {
        switch step {
        case .intro:
            return "A quick tour so new users understand what the app can do."
        case .tavily:
            return "Optional setup for web search."
        case .mlx:
            return "Optional setup for larger fully local models."
        }
    }

    private var primaryButtonTitle: String {
        switch step {
        case .intro:
            return "Continue"
        case .tavily:
            return "Continue"
        case .mlx:
            return "Finish"
        }
    }

    private var primaryButtonIdentifier: String {
        switch step {
        case .intro:
            return "onboarding.intro.continue"
        case .tavily:
            return "onboarding.tavily.continue"
        case .mlx:
            return "onboarding.finish"
        }
    }

    private var headerTitleIdentifier: String {
        switch step {
        case .intro:
            return "onboarding.intro.title"
        case .tavily:
            return "onboarding.tavily.title"
        case .mlx:
            return "onboarding.mlx.title"
        }
    }

    private func previousStep() {
        guard let previous = Step(rawValue: step.rawValue - 1) else { return }
        step = previous
    }

    private func advance() {
        switch step {
        case .intro:
            step = .tavily
        case .tavily:
            let trimmedKey = tavilyKey.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedKey.isEmpty {
                TavilyAPIKeyStore.save(trimmedKey)
            } else {
                TavilyAPIKeyStore.clear()
            }
            step = .mlx
        case .mlx:
            finishOnboarding()
        }
    }

    private func finishOnboarding() {
        onFinish()
    }
}
