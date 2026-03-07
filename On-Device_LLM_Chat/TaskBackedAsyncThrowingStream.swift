//
//  TaskBackedAsyncThrowingStream.swift
//  On-Device_LLM_Chat
//

import Foundation

enum TaskBackedAsyncThrowingStream {
    static func make<Element>(
        bufferingPolicy: AsyncThrowingStream<Element, Error>.Continuation.BufferingPolicy = .unbounded,
        startProducer: @escaping @Sendable (AsyncThrowingStream<Element, Error>.Continuation) -> Task<Void, Never>
    ) -> AsyncThrowingStream<Element, Error> {
        AsyncThrowingStream(bufferingPolicy: bufferingPolicy) { continuation in
            let producerTask = startProducer(continuation)
            continuation.onTermination = { @Sendable _ in
                producerTask.cancel()
            }
        }
    }
}
