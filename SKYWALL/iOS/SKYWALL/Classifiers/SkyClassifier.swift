// SkyClassifier.swift
// SKYWALL - Autonomous Aerial Detection System
// Deep classification module - simulates Gemma 4 E4B + LoRA fine-tuned model.
// On-device inference with CoreML; 1-3 second latency by design.

import Foundation
import CoreML
import Vision
import UIKit

// MARK: - Classification Output

struct ClassificationOutput: Codable {
    let droneClass: DroneClass
    let manufacturer: String?
    let model: String?
    let estimatedSize: DroneSize
    let payloadType: PayloadType
    let behaviorPattern: BehaviorPattern
    let operationalMode: OperationalMode
    let estimatedAltitude: Float?       // meters
    let estimatedRange: Float?          // meters from observer
    let confidence: Float
    let rawAnalysis: String
    let timestamp: Date

    enum CodingKeys: String, CodingKey {
        case droneClass, manufacturer, model, estimatedSize
        case payloadType, behaviorPattern, operationalMode
        case estimatedAltitude, estimatedRange, confidence
        case rawAnalysis, timestamp
    }
}

enum DroneClass: String, Codable, CaseIterable {
    case fpvRacer      = "fpv_racer"
    case fpvCinematic  = "fpv_cinematic"
    case djiConsumer   = "dji_consumer"
    case djiPro        = "dji_pro"
    case fixedWing     = "fixed_wing"
    case helicopter    = "helicopter"
    case military      = "military"
    case diy           = "diy_build"
    case unknown       = "unknown"
}

enum DroneSize: String, Codable {
    case micro  = "micro"    // < 250g
    case small  = "small"    // 250g - 2kg
    case medium = "medium"   // 2kg - 10kg
    case large  = "large"    // > 10kg
}

enum PayloadType: String, Codable {
    case none        = "none"
    case camera      = "camera"
    case fpvCamera   = "fpv_camera"
    case thermal     = "thermal"
    case cargo       = "cargo"
    case unknown     = "unknown"
}

enum BehaviorPattern: String, Codable {
    case hovering    = "hovering"
    case surveying   = "surveying"
    case approaching = "approaching"
    case retreating  = "retreating"
    case circling    = "circling"
    case erratic     = "erratic"
    case racing      = "racing"
    case unknown     = "unknown"
}

enum OperationalMode: String, Codable {
    case manual      = "manual"
    case autonomous  = "autonomous"
    case mission     = "mission"
    case unknown     = "unknown"
}

// MARK: - SkyClassifier

final class SkyClassifier {

    // MARK: - Properties

    private var mlModel: MLModel?
    private var useMock = false

    // Classification is expensive - serialize via actor
    private let classificationActor = ClassificationActor()

    // Rate limiting: max 1 classification per 2 seconds per unique target
    private var lastClassificationTime: [UUID: Date] = [:]
    private let minClassificationInterval: TimeInterval = 2.0

    // Callback
    var classificationHandler: ((ClassificationOutput, DetectionEvent) -> Void)?

    // MARK: - Model Loading

    func loadModel() async {
        guard let url = Bundle.main.url(forResource: "SkyWALLClassifier", withExtension: "mlmodelc") else {
            print("[SkyClassifier] Model not found, using mock.")
            useMock = true
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            mlModel = try MLModel(contentsOf: url, configuration: config)
            print("[SkyClassifier] Deep classifier model loaded.")
        } catch {
            print("[SkyClassifier] Model load failed: \(error). Using mock.")
            useMock = true
        }
    }

    // MARK: - Classify

    func classify(pixelBuffer: CVPixelBuffer, detection: DetectedObject, timestamp: Date) async {
        // Rate limit per detection object
        let trackID = detection.trackingID ?? detection.id
        if let last = lastClassificationTime[trackID],
           timestamp.timeIntervalSince(last) < minClassificationInterval {
            return
        }
        lastClassificationTime[trackID] = timestamp

        // Crop the detection region from the pixel buffer
        guard let croppedBuffer = cropPixelBuffer(pixelBuffer, rect: detection.boundingBox) else { return }

        let output: ClassificationOutput
        if useMock || mlModel == nil {
            output = await classificationActor.mockClassify(detection: detection, timestamp: timestamp)
        } else {
            output = await classificationActor.classify(buffer: croppedBuffer, model: mlModel!, detection: detection, timestamp: timestamp)
        }

        // Build DetectionEvent
        let event = buildDetectionEvent(output: output, detection: detection, timestamp: timestamp)

        await MainActor.run {
            classificationHandler?(output, event)
        }
    }

    // MARK: - Pixel Buffer Cropping

    private func cropPixelBuffer(_ buffer: CVPixelBuffer, rect: CGRect) -> CVPixelBuffer? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)

        // Convert normalized rect to pixel coordinates
        let cropX = Int(rect.origin.x * CGFloat(width))
        let cropY = Int(rect.origin.y * CGFloat(height))
        let cropW = max(1, Int(rect.width * CGFloat(width)))
        let cropH = max(1, Int(rect.height * CGFloat(height)))

        // Clamp to buffer bounds
        let clampedX = max(0, min(cropX, width - 1))
        let clampedY = max(0, min(cropY, height - 1))
        let clampedW = max(1, min(cropW, width - clampedX))
        let clampedH = max(1, min(cropH, height - clampedY))

        // Create CIImage and crop
        let ciImage = CIImage(cvPixelBuffer: buffer)
            .cropped(to: CGRect(x: clampedX, y: clampedY, width: clampedW, height: clampedH))
            .transformed(by: CGAffineTransform(translationX: CGFloat(-clampedX), y: CGFloat(-clampedY)))

        // Scale to model input size (224x224)
        let targetSize = CGSize(width: 224, height: 224)
        let scaleX = targetSize.width / CGFloat(clampedW)
        let scaleY = targetSize.height / CGFloat(clampedH)
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        // Render to new pixel buffer
        var outputBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary

        guard CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(targetSize.width),
            Int(targetSize.height),
            kCVPixelFormatType_32BGRA,
            attrs,
            &outputBuffer
        ) == kCVReturnSuccess, let out = outputBuffer else { return nil }

        let context = CIContext(options: [.useSoftwareRenderer: false])
        context.render(scaled, to: out)
        return out
    }

    // MARK: - Build Detection Event

    private func buildDetectionEvent(output: ClassificationOutput, detection: DetectedObject, timestamp: Date) -> DetectionEvent {
        let threat = ThreatClassification(
            droneClass: detection.label,
            confidence: output.confidence,
            bearing: 0,  // Will be filled by direction estimator
            elevation: 0,
            estimatedRange: output.estimatedRange,
            behavior: output.behaviorPattern,
            registrationVisible: false,
            payloadVisible: output.payloadType != .none,
            firstSeen: timestamp,
            lastSeen: timestamp,
            trackPoints: [],
            notes: output.rawAnalysis
        )

        return DetectionEvent(
            id: UUID(),
            timestamp: timestamp,
            classification: threat,
            audioClass: nil,
            bearing: 0,
            elevation: 0,
            confidence: output.confidence,
            location: nil,
            mediaFiles: [],
            nodeID: UIDevice.current.identifierForVendor?.uuidString ?? "unknown",
            meshConfirmed: false,
            boundingBox: detection.boundingBox
        )
    }
}

// MARK: - ClassificationActor (serializes inference)

actor ClassificationActor {

    func classify(buffer: CVPixelBuffer, model: MLModel, detection: DetectedObject, timestamp: Date) async -> ClassificationOutput {
        // Convert pixel buffer to MLFeatureProvider
        do {
            let imageConstraint = model.modelDescription.inputDescriptionsByName.values.first
            guard let _ = imageConstraint else {
                return await mockClassify(detection: detection, timestamp: timestamp)
            }

            // Create feature value from image
            let featureValue = try MLFeatureValue(pixelBuffer: buffer)
            let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
            let features = try MLDictionaryFeatureProvider(dictionary: [inputName: featureValue])

            let start = CFAbsoluteTimeGetCurrent()
            let prediction = try model.prediction(from: features)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            print("[SkyClassifier] Deep inference: \(String(format: "%.0f", elapsed))ms")

            return parseModelOutput(prediction: prediction, detection: detection, timestamp: timestamp)

        } catch {
            print("[SkyClassifier] Inference error: \(error)")
            return await mockClassify(detection: detection, timestamp: timestamp)
        }
    }

    private func parseModelOutput(prediction: MLFeatureProvider, detection: DetectedObject, timestamp: Date) -> ClassificationOutput {
        // Parse model output features
        var droneClass: DroneClass = .unknown
        var confidence: Float = 0.5
        var analysis = ""

        // Try to get class probabilities
        for outputName in prediction.featureNames {
            if let val = prediction.featureValue(for: outputName) {
                if let dict = val.dictionaryValue as? [String: Double] {
                    if let best = dict.max(by: { $0.value < $1.value }),
                       let dc = DroneClass(rawValue: best.key) {
                        droneClass = dc
                        confidence = Float(best.value)
                        analysis = "Model prediction: \(best.key) @ \(String(format: "%.2f", best.value))"
                    }
                }
            }
        }

        return ClassificationOutput(
            droneClass: droneClass,
            manufacturer: nil,
            model: nil,
            estimatedSize: .small,
            payloadType: .camera,
            behaviorPattern: .hovering,
            operationalMode: .unknown,
            estimatedAltitude: nil,
            estimatedRange: nil,
            confidence: confidence,
            rawAnalysis: analysis,
            timestamp: timestamp
        )
    }

    func mockClassify(detection: DetectedObject, timestamp: Date) async -> ClassificationOutput {
        // Simulate ~1.5 second deep analysis
        try? await Task.sleep(nanoseconds: 500_000_000)  // 0.5s for testing

        let droneClass: DroneClass
        let behavior: BehaviorPattern
        let size: DroneSize
        let payload: PayloadType
        let analysis: String

        switch detection.label {
        case .droneMultirotor:
            let classes: [DroneClass] = [.fpvRacer, .djiConsumer, .djiPro, .diy]
            droneClass = classes.randomElement()!
            behavior = [.hovering, .surveying, .approaching, .circling].randomElement()!
            size = droneClass == .fpvRacer ? .micro : .small
            payload = droneClass == .fpvRacer ? .fpvCamera : .camera
            analysis = "Multirotor detected. Frame geometry suggests \(droneClass.rawValue). Propeller disc count: 4. Visible camera gimbal: \(payload == .camera ? "yes" : "no")."

        case .droneFixedWing:
            droneClass = .fixedWing
            behavior = [.surveying, .approaching, .retreating].randomElement()!
            size = .medium
            payload = .camera
            analysis = "Fixed-wing UAV. Wing sweep consistent with commercial survey platform. No visible weaponization."

        case .helicopter:
            droneClass = .helicopter
            behavior = .hovering
            size = .large
            payload = .unknown
            analysis = "Rotary-wing aircraft. Manned or large UAV. Single main rotor configuration."

        case .aircraft:
            droneClass = .unknown
            behavior = [.approaching, .retreating].randomElement()!
            size = .large
            payload = .none
            analysis = "Fixed-wing aircraft. Likely manned. High altitude trajectory."

        default:
            droneClass = .unknown
            behavior = .unknown
            size = .small
            payload = .unknown
            analysis = "Object detected. Insufficient detail for classification."
        }

        let confidence = Float.random(in: 0.72...0.95)

        return ClassificationOutput(
            droneClass: droneClass,
            manufacturer: droneClass == .djiConsumer ? "DJI" : nil,
            model: droneClass == .djiConsumer ? ["Mavic 3", "Mini 4 Pro", "Phantom 4"].randomElement() : nil,
            estimatedSize: size,
            payloadType: payload,
            behaviorPattern: behavior,
            operationalMode: .unknown,
            estimatedAltitude: Float.random(in: 30...150),
            estimatedRange: Float.random(in: 50...500),
            confidence: confidence,
            rawAnalysis: analysis,
            timestamp: timestamp
        )
    }
}
