//
//  TavilySearchService.swift
//  ChatLLM
//
//  Created by Nevio on 10/26/25.
//

import Foundation

struct TavilyAutoParameters: Decodable, Sendable {
    let topic: String?
    let searchDepth: String?
    let timeRange: String?

    enum CodingKeys: String, CodingKey {
        case topic
        case searchDepth = "search_depth"
        case timeRange = "time_range"
    }
}

struct TavilySearchResult: Decodable, Identifiable, Sendable {
    var id = UUID()
    let title: String
    let url: String
    let content: String
    let score: Double?
    let publishedDate: String?
    let favicon: String?

    enum CodingKeys: String, CodingKey {
        case title, url, content, score, favicon
        case publishedDate = "published_date"
    }

    nonisolated init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = UUID()
        title = try container.decode(String.self, forKey: .title)
        url = try container.decode(String.self, forKey: .url)
        content = try container.decodeIfPresent(String.self, forKey: .content) ?? ""
        score = try container.decodeIfPresent(Double.self, forKey: .score)
        publishedDate = try container.decodeIfPresent(String.self, forKey: .publishedDate)
        favicon = try container.decodeIfPresent(String.self, forKey: .favicon)
    }
}

struct TavilySearchResponse: Decodable, Sendable {
    let query: String
    let results: [TavilySearchResult]
    let answer: String?
    let autoParameters: TavilyAutoParameters?
    let responseTime: Double?
    let requestID: String?
    let creditsUsed: Int?

    private enum CodingKeys: String, CodingKey {
        case query, results, answer, usage
        case autoParameters = "auto_parameters"
        case responseTime = "response_time"
        case requestID = "request_id"
    }

    private struct Usage: Decodable {
        let credits: Int?
    }

    nonisolated init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        query = try container.decode(String.self, forKey: .query)
        results = try container.decode([TavilySearchResult].self, forKey: .results)
        answer = try container.decodeIfPresent(String.self, forKey: .answer)
        autoParameters = try container.decodeIfPresent(TavilyAutoParameters.self, forKey: .autoParameters)
        responseTime = try container.decodeIfPresent(Double.self, forKey: .responseTime)
        requestID = try container.decodeIfPresent(String.self, forKey: .requestID)
        creditsUsed = try container.decodeIfPresent(Usage.self, forKey: .usage)?.credits
    }
}

private struct TavilyAPIErrorResponse: Decodable, Sendable {
    let detail: String?
    let message: String?
    let error: String?
    let nestedError: NestedError?

    enum CodingKeys: String, CodingKey {
        case detail, message, error
    }

    struct NestedError: Decodable, Sendable {
        let message: String?
        let code: String?
    }

    nonisolated init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        detail = try container.decodeIfPresent(String.self, forKey: .detail)
        message = try container.decodeIfPresent(String.self, forKey: .message)
        error = try? container.decodeIfPresent(String.self, forKey: .error)
        nestedError = try? container.decodeIfPresent(NestedError.self, forKey: .error)
    }
}

enum TavilySearchError: LocalizedError {
    case invalidQuery
    case invalidAPIKey
    case networkError(Error)
    case noResults
    case decodingError(Error)

    var errorDescription: String? {
        switch self {
        case .invalidQuery: "Enter a search query under 400 characters."
        case .invalidAPIKey: "The Tavily API key is missing or invalid."
        case .networkError(let error): "Web search failed: \(error.localizedDescription)"
        case .noResults: "No relevant web results were found."
        case .decodingError: "Tavily returned a response ChatLLM could not read."
        }
    }
}

actor TavilySearchService: WebSearchProviding {
    private struct CacheEntry {
        let value: WebSearchResponse
        let storedAt: Date
        let lifetime: TimeInterval
    }

    nonisolated private static let cacheCapacity = 32
    nonisolated private static let liveCacheTTL: TimeInterval = 60
    nonisolated private static let standardCacheTTL: TimeInterval = 600
    nonisolated private static let retryableStatusCodes: Set<Int> = [429, 500, 502, 503, 504]

    private let apiKey: String
    private let baseURL = URL(string: "https://api.tavily.com/search")!
    private let usageURL = URL(string: "https://api.tavily.com/usage")!
    private let session: URLSession
    private var cache: [String: CacheEntry] = [:]

    init(apiKey: String, session: URLSession = .shared) throws {
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else { throw TavilySearchError.invalidAPIKey }
        self.apiKey = trimmedKey
        self.session = session
    }

    func search(query: String, maxResults: Int = 4, searchDepth: String? = nil) async throws -> WebSearchResponse {
        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedQuery.isEmpty, trimmedQuery.count <= 400 else {
            throw TavilySearchError.invalidQuery
        }

        let resultLimit = min(max(maxResults, 1), 5)
        let resolvedDepth = Self.validSearchDepth(searchDepth)
        let cacheKey = Self.makeCacheKey(query: trimmedQuery, maxResults: resultLimit, searchDepth: resolvedDepth)
        if let cached = cachedResult(for: cacheKey) { return cached }

        var request = URLRequest(url: baseURL, timeoutInterval: 15)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        var body: [String: Any] = [
            "query": trimmedQuery,
            "max_results": resultLimit,
            "include_answer": false,
            "include_raw_content": false,
            "include_favicon": true,
            "include_usage": true,
            "auto_parameters": true
        ]
        if let resolvedDepth { body["search_depth"] = resolvedDepth }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await perform(request)
        try validate(response: response, data: data)

        let decoded: TavilySearchResponse
        do {
            decoded = try JSONDecoder().decode(TavilySearchResponse.self, from: data)
        } catch {
            throw TavilySearchError.decodingError(error)
        }

        let sources = Self.makeSources(from: decoded.results, limit: resultLimit)
        guard !sources.isEmpty else { throw TavilySearchError.noResults }

        let result = WebSearchResponse(
            query: decoded.query.isEmpty ? trimmedQuery : decoded.query,
            sources: sources,
            responseTimeSeconds: decoded.responseTime,
            requestID: decoded.requestID,
            creditsUsed: decoded.creditsUsed
        )
        storeResult(result, for: cacheKey, query: trimmedQuery)
        return result
    }

    func validateAPIKey() async throws {
        var request = URLRequest(url: usageURL, timeoutInterval: 10)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await perform(request, retryCount: 0)
        try validate(response: response, data: data)
    }

    private func perform(_ request: URLRequest, retryCount: Int = 1) async throws -> (Data, URLResponse) {
        var attempt = 0
        while true {
            do {
                let result = try await session.data(for: request)
                if let response = result.1 as? HTTPURLResponse,
                   Self.retryableStatusCodes.contains(response.statusCode),
                   attempt < retryCount {
                    attempt += 1
                    try await Task.sleep(for: .milliseconds(250 * attempt))
                    continue
                }
                return result
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                if attempt < retryCount {
                    attempt += 1
                    try await Task.sleep(for: .milliseconds(250 * attempt))
                    continue
                }
                throw TavilySearchError.networkError(error)
            }
        }
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw TavilySearchError.networkError(
                NSError(domain: "Tavily", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid server response"])
            )
        }
        guard (200...299).contains(httpResponse.statusCode) else {
            if httpResponse.statusCode == 401 || httpResponse.statusCode == 403 {
                throw TavilySearchError.invalidAPIKey
            }
            let message = parseErrorMessage(from: data) ?? "Tavily returned status \(httpResponse.statusCode)."
            throw TavilySearchError.networkError(
                NSError(domain: "Tavily", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: message])
            )
        }
    }

    private func cachedResult(for key: String) -> WebSearchResponse? {
        guard let entry = cache[key] else { return nil }
        guard Date().timeIntervalSince(entry.storedAt) <= entry.lifetime else {
            cache.removeValue(forKey: key)
            return nil
        }
        return entry.value
    }

    private func storeResult(_ value: WebSearchResponse, for key: String, query: String) {
        if cache.count >= Self.cacheCapacity,
           let oldest = cache.min(by: { $0.value.storedAt < $1.value.storedAt })?.key {
            cache.removeValue(forKey: oldest)
        }
        cache[key] = CacheEntry(
            value: value,
            storedAt: Date(),
            lifetime: Self.isLiveQuery(query) ? Self.liveCacheTTL : Self.standardCacheTTL
        )
    }

    nonisolated private static func makeSources(from results: [TavilySearchResult], limit: Int) -> [WebSearchSource] {
        var seenURLs: Set<String> = []
        let ranked = results.sorted { ($0.score ?? 0) > ($1.score ?? 0) }
        let relevant = ranked.filter { ($0.score ?? 1) >= 0.25 }
        let candidates = relevant.isEmpty ? ranked : relevant

        return candidates.compactMap { result in
            guard seenURLs.insert(normalizedURL(result.url)).inserted else { return nil }
            return WebSearchSource(
                title: compactWhitespace(result.title) ?? result.url,
                url: result.url,
                snippet: compactWhitespace(result.content) ?? "",
                score: result.score,
                publishedDate: compactWhitespace(result.publishedDate),
                faviconURL: result.favicon
            )
        }
        .prefix(limit)
        .map { $0 }
    }

    nonisolated private static func normalizedURL(_ value: String) -> String {
        guard var components = URLComponents(string: value) else { return value.lowercased() }
        components.fragment = nil
        if components.path == "/" { components.path = "" }
        return (components.string ?? value).lowercased()
    }

    nonisolated private static func compactWhitespace(_ text: String?) -> String? {
        guard let text else { return nil }
        let compact = text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        return compact.isEmpty ? nil : compact
    }

    nonisolated private static func validSearchDepth(_ value: String?) -> String? {
        guard let value = value?.lowercased() else { return nil }
        return ["basic", "advanced", "fast", "ultra-fast"].contains(value) ? value : nil
    }

    nonisolated private static func makeCacheKey(query: String, maxResults: Int, searchDepth: String?) -> String {
        let normalized = query.lowercased().split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        return [normalized, String(maxResults), searchDepth ?? "auto"].joined(separator: "|")
    }

    nonisolated private static func isLiveQuery(_ query: String) -> Bool {
        let normalized = query.lowercased()
        return [
            "today", "tonight", "current", "currently", "latest", "live", "breaking",
            "price", "score", "weather", "news", "this week", "right now"
        ].contains { normalized.contains($0) }
    }

    private func parseErrorMessage(from data: Data) -> String? {
        guard !data.isEmpty else { return nil }
        if let decoded = try? JSONDecoder().decode(TavilyAPIErrorResponse.self, from: data) {
            return decoded.detail ?? decoded.message ?? decoded.error
                ?? decoded.nestedError?.message ?? decoded.nestedError?.code
        }
        return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
