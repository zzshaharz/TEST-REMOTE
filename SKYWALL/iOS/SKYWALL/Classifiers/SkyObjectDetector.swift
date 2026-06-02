// SkyObjectDetector.swift
// SKYWALL - Autonomous Aerial Detection System
// Vision + CoreML YOLO object detector with mock fallback.

import Vision
import CoreML
import UIKit

// MARK: - Detected Object

struct DetectedObject: Identifiable {
    let id = UUID()
    let label: AerialClass
    let confidence: Float
    let boundingBox: CGRect    // Normalized [0,1] coordinates
    let timestamp: Date
    let trackingID: UUID?

    var isAerial: Bool {
        label != .unknownAerial && confidence > 0.3
    }

    var isDrone: Bool {
        switch label {
        case .droneMultirotor, .droneFixedWing: return true
        default: return false
        }
    }
}

// MARK: - Aerial Class

enum AerialClass: String, Codable, CaseIterable {
    case droneMultirotor  = "drone_multirotor"
    case droneFixedWing   = "drone_fixed_wing"
    case bird             = "bird"
    case aircraft         = "aircraft"
    case helicopter       = "helicopter"
    case balloon          = "balloon"
    case unknownAerial    = "unknown_aerial"

    var threatLevel: ThreatLevel {
        switch self {
        case .droneMultirotor, .droneFixedWing: return .high
        case .helicopter:                       return .medium
        case .aircraft:                         return .low
        default:                                return .none
        }
    }

    var color: UIColor {
        switch self {
        case .droneMultirotor, .droneFixedWing: return .systemRed
        case .helicopter:                        return .systemOrange
        case .aircraft:                          return .systemYellow
        case .bird:                              return .systemGreen
        default:                                 return .systemGray
        }
    }
}

// MARK: - SkyObjectDetector

final class SkyObjectDetector {

    // MARK: - Properties

    private var model: VNCoreMLModel?
    private var useMock = false

    // Processing rate control
    private var lastProcessTime: Date = .distantPast
    private let minProcessInterval: TimeInterval = 1.0 / 30.0  // 30 FPS max

    // Request queue
    private let detectionQueue = DispatchQueue(label: "com.skywall.detection", qos: .userInteractive)

    // Callbacks
    var detectionHandler: (([DetectedObject], CVPixelBuffer, Date) -> Void)?

    // Active tracking rectangle for Vision tracking requests
    private var trackedRect: CGRect?
    private var trackingRequest: VNTrackObjectRequest?

    // MARK: - Initialization

    func loadModel() async {
        guard let modelURL = Bundle.main.url(forResource: "SkyWALLYOLO", withExtension: "mlmodelc") else {
            print("[ObjectDetector] YOLO model not found, using mock.")
            useMock = true
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            let mlModel = try MLModel(contentsOf: modelURL, configuration: config)
            model = try VNCoreMLModel(for: mlModel)
            print("[ObjectDetector] YOLO model loaded.")
        } catch {
            print("[ObjectDetector] Model load failed: \(error). Using mock.")
            useMock = true
        }
    }

    // MARK: - Process Frame

    func process(pixelBuffer: CVPixelBuffer, timestamp: Date) async {
        guard timestamp.timeIntervalSince(lastProcessTime) >= minProcessInterval else { return }
        lastProcessTime = timestamp

        if useMock || model == nil {
            let detections = mockDetect(pixelBuffer: pixelBuffer, timestamp: timestamp)
            await MainActor.run {
                detectionHandler?(detections, pixelBuffer, timestamp)
            }
            return
        }

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            detectionQueue.async { [weak self] in
                guard let self else { cont.resume(); return }
                self.runInference(pixelBuffer: pixelBuffer, timestamp: timestamp) { detections in
                    Task { @MainActor in
                        self.detectionHandler?(detections, pixelBuffer, timestamp)
                    }
                    cont.resume()
                }
            }
        }
    }

    // MARK: - CoreML Inference

    private func runInference(pixelBuffer: CVPixelBuffer, timestamp: Date, completion: @escaping ([DetectedObject]) -> Void) {
        guard let model = model else { completion([]); return }

        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let self else { return }
            if let error {
                print("[ObjectDetector] Inference error: \(error)")
                completion([])
                return
            }

            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                completion([])
                return
            }

            let detections = results.compactMap { obs -> DetectedObject? in
                guard let topLabel = obs.labels.first,
                      let aerialClass = AerialClass(rawValue: topLabel.identifier) else { return nil }
                guard topLabel.confidence > 0.3 else { return nil }

                return DetectedObject(
                    label: aerialClass,
                    confidence: topLabel.confidence,
                    boundingBox: obs.boundingBox,
                    timestamp: timestamp,
                    trackingID: nil
                )
            }
            .sorted { $0.confidence > $1.confidence }

            completion(detections)
        }

        // Configure for image orientation
        request.imageCropAndScaleOption = .scaleFit

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("[ObjectDetector] Handler perform error: \(error)")
            completion([])
        }
    }

    // MARK: - Vision Rectangle Fallback

    func detectRectanglesFallback(pixelBuffer: CVPixelBuffer) async -> [CGRect] {
        await withCheckedContinuation { cont in
            detectionQueue.async {
                let request = VNDetectRectanglesRequest { request, error in
                    guard let results = request.results as? [VNRectangleObservation] else {
                        cont.resume(returning: [])
                        return
                    }
                    // Filter for small, aerial-sized rectangles
                    let rects = results
                        .filter { $0.confidence > 0.5 && $0.boundingBox.width < 0.3 && $0.boundingBox.height < 0.3 }
                        .map { $0.boundingBox }
                    cont.resume(returning: rects)
                }

                request.minimumAspectRatio = 0.5
                request.maximumAspectRatio = 3.0
                request.minimumSize = 0.01
                request.maximumObservations = 5

                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
                try? handler.perform([request])
            }
        }
    }

    // MARK: - Mock Detection

    private var mockFrameCount = 0
    private var mockDroneActive = false
    private var mockDroneBBox = CGRect(x: 0.4, y: 0.3, width: 0.05, height: 0.05)

    private func mockDetect(pixelBuffer: CVPixelBuffer, timestamp: Date) -> [DetectedObject] {
        mockFrameCount += 1

        // Simulate a drone appearing every 300 frames (~10 seconds at 30fps)
        if mockFrameCount % 300 == 0 {
            mockDroneActive = true
            mockDroneBBox = CGRect(
                x: Double.random(in: 0.2...0.7),
                y: Double.random(in: 0.1...0.5),
                width: Double.random(in: 0.03...0.08),
                height: Double.random(in: 0.03...0.08)
            )
        }

        if mockFrameCount % 300 == 150 {
            mockDroneActive = false
        }

        guard mockDroneActive else { return [] }

        // Simulate slight motion
        mockDroneBBox = CGRect(
            x: mockDroneBBox.origin.x + Double.random(in: -0.002...0.002),
            y: mockDroneBBox.origin.y + Double.random(in: -0.002...0.002),
            width: mockDroneBBox.width,
            height: mockDroneBBox.height
        )

        let confidence = Float.random(in: 0.65...0.92)
        return [
            DetectedObject(
                label: .droneMultirotor,
                confidence: confidence,
                boundingBox: mockDroneBBox,
                timestamp: timestamp,
                trackingID: nil
            )
        ]
    }
}
