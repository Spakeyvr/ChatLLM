//
//  MLXModelManager+ToolCalling
//  ChatLLM
//
//  Split out of MLXModelManager.swift as part of the Type+Concern organization.
//

import Foundation
import MLXLMCommon
import OSLog

extension MLXModelManager {
    // MARK: - Tool Calling

    enum ToolTemplateSupport: Equatable {
        case supported
        case unsupported(String)
    }

    struct ToolTemplateInspection {
        let support: ToolTemplateSupport
        let preferredToolCallFormat: ToolCallFormat?
        let usesWrappedXMLToolCalls: Bool
    }
    nonisolated internal static let maxToolInvocationsPerResponse = 6

    nonisolated internal static func shouldDisableTools(after toolResult: String) -> Bool {
        AppWebSearchToolBridge.isSearchLimitToolResponse(toolResult)
    }

    nonisolated internal static func excessiveToolCallToolResponse(maximum: Int) -> String {
        "[tool internal error: limit reached after \(maximum) tool invocations for this response. Do not call tools again. Continue with the available information already available.]"
    }
    nonisolated internal static func wrappedToolResponseContent(_ content: String) -> String {
        "<tool_response>\n\(content)\n</tool_response>"
    }

    nonisolated internal static func toolResponsePromptContent(
        for model: MLXModelInfo,
        toolResult: String
    ) -> String {
        guard prefersUserWrappedToolResponses(for: model) else {
            return toolResult
        }
        return wrappedToolResponseContent(toolResult)
    }

    nonisolated internal static func toolResponsePromptRole(for model: MLXModelInfo) -> Chat.Message.Role {
        prefersUserWrappedToolResponses(for: model) ? .user : .tool
    }

    nonisolated internal static func prefersUserWrappedToolResponses(for model: MLXModelInfo) -> Bool {
        model.id.hasPrefix("qwen3.5-")
    }

    nonisolated internal static func requiresExplicitMessageHistoryForToolLoop(for model: MLXModelInfo) -> Bool {
        prefersUserWrappedToolResponses(for: model)
    }

    nonisolated internal static func normalizedToolArguments(
        _ arguments: [String: any Sendable]
    ) -> [String: any Sendable] {
        arguments.reduce(into: [String: any Sendable]()) { partialResult, entry in
            partialResult[entry.key] = normalizedToolArgumentValue(entry.value)
        }
    }

    nonisolated private static func normalizedToolArgumentValue(_ value: any Sendable) -> any Sendable {
        switch value {
        case let jsonValue as JSONValue:
            return normalized(jsonValue)
        case let string as String:
            return string
        case let bool as Bool:
            return bool
        case let int as Int:
            return int
        case let double as Double:
            return double
        case let array as [JSONValue]:
            return array.map { normalized($0) }
        case let object as [String: JSONValue]:
            return object.mapValues { normalized($0) }
        default:
            return String(describing: value)
        }
    }

    nonisolated private static func normalized(_ value: JSONValue) -> any Sendable {
        switch value {
        case .null:
            return "null"
        case .bool(let bool):
            return bool
        case .int(let int):
            return int
        case .double(let double):
            return double
        case .string(let string):
            return string
        case .array(let array):
            return array.map { normalized($0) }
        case .object(let object):
            return object.mapValues { normalized($0) }
        }
    }
    func applyPreferredToolCallFormatIfNeeded(
        to loaded: ModelContainer,
        for model: MLXModelInfo
    ) async {
        guard let preferredToolCallFormat = preferredToolCallFormat(for: model) else {
            return
        }

        let currentToolCallFormat = await loaded.configuration.toolCallFormat
        guard currentToolCallFormat != preferredToolCallFormat else {
            logger.notice(
                "MLX tool-call format retained: id=\(model.id, privacy: .public) format=\(preferredToolCallFormat.rawValue, privacy: .public)"
            )
            return
        }

        await loaded.update { context in
            context.configuration.toolCallFormat = preferredToolCallFormat
        }
        logger.notice(
            "MLX tool-call format override applied: id=\(model.id, privacy: .public) format=\(preferredToolCallFormat.rawValue, privacy: .public)"
        )
    }

    func validateToolTemplateSupport(for model: MLXModelInfo) throws {
        switch toolTemplateSupport(for: model) {
        case .supported:
            return
        case .unsupported(let message):
            throw GenerationError.unsupportedToolTemplate(message)
        }
    }

    func logToolTemplateSupport(for model: MLXModelInfo) {
        let inspection = toolTemplateInspection(for: model)
        switch inspection.support {
        case .supported:
            logger.notice("MLX tool template support confirmed for \(model.displayName, privacy: .public)")
            if let preferredToolCallFormat = inspection.preferredToolCallFormat {
                logger.notice(
                    "MLX tool template inspection: id=\(model.id, privacy: .public) format=\(preferredToolCallFormat.rawValue, privacy: .public) wrapped_xml=\(inspection.usesWrappedXMLToolCalls, privacy: .public)"
                )
            }
        case .unsupported(let message):
            logger.warning("MLX tool template support unavailable: \(message, privacy: .public)")
        }
    }

    private func toolTemplateSupport(for model: MLXModelInfo) -> ToolTemplateSupport {
        toolTemplateInspection(for: model).support
    }

    private func preferredToolCallFormat(for model: MLXModelInfo) -> ToolCallFormat? {
        toolTemplateInspection(for: model).preferredToolCallFormat
    }

    func usesWrappedXMLToolCalls(for model: MLXModelInfo) -> Bool {
        toolTemplateInspection(for: model).usesWrappedXMLToolCalls
    }

    private func toolTemplateInspection(for model: MLXModelInfo) -> ToolTemplateInspection {
        if let cached = toolTemplateInspectionCache[model.id] {
            return cached
        }

        guard let modelURL = modelDirectoryURL(for: model) else {
            let inspection = ToolTemplateInspection(
                support: .unsupported(
                    "Installed model '\(model.localDirName)' could not be inspected for tool support. Re-download the model package to restore the tokenizer chat template."
                ),
                preferredToolCallFormat: nil,
                usesWrappedXMLToolCalls: false
            )
            toolTemplateInspectionCache[model.id] = inspection
            return inspection
        }

        let candidateFiles = [
            "tokenizer_config.json",
            "tokenizer.json",
            "chat_template.jinja",
            "chat_template.json",
            "processor_config.json",
            "preprocessor_config.json"
        ]

        let combinedContents = candidateFiles.compactMap { fileName -> String? in
            let url = modelURL.appendingPathComponent(fileName)
            guard FileManager.default.fileExists(atPath: url.path),
                  let data = try? Data(contentsOf: url),
                  let text = String(data: data, encoding: .utf8) else {
                return nil
            }
            return text
        }.joined(separator: "\n")

        let lowered = combinedContents.lowercased()
        let hasToolsContext = lowered.contains("tools")
        let hasToolRole = lowered.contains("\"tool\"") ||
            lowered.contains("'tool'") ||
            lowered.contains("role == 'tool'") ||
            lowered.contains("role != 'tool'")
        let hasToolCalls = lowered.contains("tool_call") || lowered.contains("tool_calls")
        let preferredToolCallFormat = Self.inferToolCallFormat(
            packageContents: combinedContents,
            modelType: packageMetadata(for: model)?.modelType
        )
        let usesWrappedXMLToolCalls = Self.usesWrappedXMLToolCallTemplate(packageContents: combinedContents)

        let support: ToolTemplateSupport
        if hasToolsContext && (hasToolRole || hasToolCalls) {
            support = .supported
        } else {
            support = .unsupported(
                "Installed model '\(model.localDirName)' does not expose a tool-aware chat template. Re-download or update this Qwen 3.5 MLX model package to enable MLX tool calling."
            )
        }

        let inspection = ToolTemplateInspection(
            support: support,
            preferredToolCallFormat: preferredToolCallFormat,
            usesWrappedXMLToolCalls: usesWrappedXMLToolCalls
        )
        toolTemplateInspectionCache[model.id] = inspection
        return inspection
    }
    nonisolated internal static func inferToolCallFormat(
        packageContents: String,
        modelType: String?
    ) -> ToolCallFormat? {
        let lowered = packageContents.lowercased()

        if lowered.contains("<function=") && lowered.contains("<parameter=") {
            return .xmlFunction
        }

        let hasJSONToolCallExample =
            lowered.contains("<tool_call>") &&
            lowered.contains("\"name\"") &&
            lowered.contains("\"arguments\"")
        if hasJSONToolCallExample {
            return .json
        }

        if let modelType,
           modelType.lowercased().hasPrefix("qwen3_5"),
           lowered.contains("tool_call") &&
           lowered.contains("function") {
            return .xmlFunction
        }

        return nil
    }

    nonisolated internal static func usesWrappedXMLToolCallTemplate(packageContents: String) -> Bool {
        let lowered = packageContents.lowercased()
        return lowered.contains("<tool_call>") &&
            lowered.contains("<function=") &&
            lowered.contains("<parameter=")
    }
}
