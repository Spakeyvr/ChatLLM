//
//  TavilySearchService.swift
//  On-Device_LLM_Chat
//
//  Created by Nevio on 10/26/25.
//
//  Service for Tavily Search API (AI-optimized for LLMs)
//

import Foundation

/// Model for a single search result (Tavily's structure)
struct TavilySearchResult: Decodable, Identifiable, Sendable {
    var id = UUID()
    
    let title: String
    let url: String
    let content: String
    let score: Double?  // Relevance score
    
    // NEW: Exclude 'id' from Codable (it's local, not in JSON)
    enum CodingKeys: String, CodingKey {
        case title
        case url
        case content
        case score
    }
}

extension TavilySearchResult {
    nonisolated init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = UUID()
        self.title = try container.decode(String.self, forKey: .title)
        self.url = try container.decode(String.self, forKey: .url)
        self.content = try container.decode(String.self, forKey: .content)
        self.score = try container.decodeIfPresent(Double.self, forKey: .score)
    }
}

/// Model for full response
struct TavilySearchResponse: Decodable, Sendable {
    let query: String
    let results: [TavilySearchResult]
    let answer: String?  // AI-generated summary if enabled
    
    // NEW: Exclude any non-JSON fields if needed; assumes standard snake_case
    enum CodingKeys: String, CodingKey {
        case query
        case results
        case answer
    }
}

extension TavilySearchResponse {
    nonisolated init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.query = try container.decode(String.self, forKey: .query)
        self.results = try container.decode([TavilySearchResult].self, forKey: .results)
        self.answer = try container.decodeIfPresent(String.self, forKey: .answer)
    }
}

/// Errors
enum TavilySearchError: LocalizedError {
    case invalidQuery
    case invalidAPIKey
    case networkError(Error)
    case noResults
    case decodingError(Error)
    
    var errorDescription: String? {
        switch self {
        case .invalidQuery: return "Invalid search query"
        case .invalidAPIKey: return "Invalid or missing Tavily API key"
        case .networkError(let error): return "Network error: \(error.localizedDescription)"
        case .noResults: return "No results found"
        case .decodingError(let error): return "Failed to decode response: \(error.localizedDescription)"
        }
    }
}

/// Service for Tavily searches
actor TavilySearchService {
    private let apiKey: String
    private let baseURL = URL(string: "https://api.tavily.com/search")!
    private let session: URLSession
    
    init(apiKey: String, session: URLSession = .shared) throws {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw TavilySearchError.invalidAPIKey
        }
        self.apiKey = apiKey
        self.session = session
    }
    
    func search(query: String, maxResults: Int = 5, searchDepth: String = "basic") async throws -> String {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw TavilySearchError.invalidQuery
        }
        
        var request = URLRequest(url: baseURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // IMPORTANT: Do NOT ask Tavily for an AI summary. We only want raw citations.
        let body: [String: Any] = [
            "api_key": apiKey,
            "query": query,
            "search_depth": searchDepth,
            "max_results": maxResults,
            "include_answer": false  // Disable AI summary
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw TavilySearchError.networkError(NSError(domain: "Tavily", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response"]))
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let result = try decoder.decode(TavilySearchResponse.self, from: data)
        
        guard !result.results.isEmpty else {
            throw TavilySearchError.noResults
        }
        
        return formatResults(result, query: query)
    }
    
    private func formatResults(_ response: TavilySearchResponse, query: String) -> String {
        // Return a clean, summary-free citation list. Keep it compact and machine-usable.
        var lines: [String] = []
        for (index, result) in response.results.enumerated() {
            // Numbered item
            lines.append("\(index + 1). \(result.title)")
            // Snippet/content
            if !result.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                lines.append(result.content)
            }
            // URL on its own line
            lines.append(result.url)
            // Blank line between entries
            lines.append("")
        }
        // Trim trailing blank lines
        while lines.last?.isEmpty == true { _ = lines.popLast() }
        return lines.joined(separator: "\n")
    }
}

