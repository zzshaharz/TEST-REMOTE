// SkySoundClassifier.swift
// SKYWALL - Autonomous Aerial Detection System
// CoreML-backed sound classifier with mock fallback for development.

import Foundation
import CoreML
import Accelerate

// MARK: - Classification Labels

enum AcousticClass: String, Codable, CaseIterable {
    case fpv5inch        = "fpv_5inch"
    case fpv7inch        = "fpv_7inch"
    case fpv10inch       = "fpv_10inch"
    case djiMavic        = "dji_mavic"
    case djiPhantom      = "dji_phantom"
    case djiInspire      = "dji_inspire"
    case fixedWingSmall  = "fixed_wing_small"
    case fixedWingLarge  = "fixed_wing_large"
    case helicopter      = "helicopter"
    case bird            = "bird"
    case aircraft        = "aircraft"
    case wind            = "wind"
    case traffic         = "traffic"
    case background      = "background"

    var isDroneClass: Bool {
        switch self {
        case .fpv5inch, .fpv7inch, .fpv10inch,
             .djiMavic, .djiPhantom, .djiInspire,
             .fixedWingSmall, .helicopter:
            return true
        default:
            return false
        }
    }

    var threatLevel: ThreatLevel {
        switch self {
        case .fpv5inch, .fpv7inch, .fpv10inch: return .high
        case .djiMavic, .djiPhantom, .djiInspire: return .medium
        case .fixedWingSmall, .helicopter: return .medium
        case .fixedWingLarge, .aircraft: return .low
        default: return .none
        }
    }
}

// MARK: - Confidence Thresholds

struct SoundConfidenceThreshold {
    static let alert: Float    = 0.40
    static let confirm: Float  = 0.70
    static let high: Float     = 0.90
}

// MARK: - AudioClassificationResult

struct AudioClassificationResult {
    let topLabel: AcousticClass
    let confidence: Float
    let allScores: [AcousticClass: Float]
    let timestamp: Date
    let melEnergy: Float    // RMS energy of mel spectrogram
    let isDrone: Bool

    var isAlert: Bool    { confidence >= SoundConfidenceThreshold.alert }
    var isConfirmed: Bool { confidence >= SoundConfidenceThreshold.confirm }
    var isHighConfidence: Bool { confidence >= SoundConfidenceThreshold.high }

    init(topLabel: AcousticClass, confidence: Float, allScores: [AcousticClass: Float], timestamp: Date, melEnergy: Float = 0) {
        self.topLabel = topLabel
        self.confidence = confidence
        self.allScores = allScores
        self.timestamp = timestamp
        self.melEnergy = melEnergy
        self.isDrone = topLabel.isDroneClass && confidence >= SoundConfidenceThreshold.alert
    }
}

// MARK: - SoundClassifier Protocol

protocol SoundClassifying {
    func classify(melSpectrogram: [Float], timestamp: Date) async
}

// MARK: - SkySoundClassifier

final class SkySoundClassifier: SoundClassifying {

    // MARK: - Properties

    private var mlModel: MLModel?
    private var useMockModel = false

    // Inference timing
    private var lastInferenceTime: Date = .distantPast
    private let inferenceIntervalMin: TimeInterval = 0.250  // 250ms minimum
    private let inferenceIntervalMax: TimeInterval = 0.500

    // Smoothing: exponential moving average over last N predictions
    private var predictionHistory: [AudioClassificationResult] = []
    private let historyMaxLength = 5

    // Handler
    var classificationHandler: ((AudioClassificationResult) -> Void)?

    // Processing queue
    private let inferenceQueue = DispatchQueue(label: "com.skywall.soundclassifier", qos: .userInteractive)

    // MARK: - Initialization

    func loadModel() async {
        // Try to load the compiled CoreML model bundle
        guard let modelURL = Bundle.main.url(forResource: "SkyWALLSoundClassifier", withExtension: "mlmodelc") else {
            print("[SoundClassifier] .mlmodelc not found, using mock model.")
            useMockModel = true
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            mlModel = try MLModel(contentsOf: modelURL, configuration: config)
            print("[SoundClassifier] Model loaded from bundle.")
        } catch {
            print("[SoundClassifier] Model load failed (\(error)), using mock.")
            useMockModel = true
        }
    }

    // MARK: - Classify

    func classify(melSpectrogram: [Float], timestamp: Date) async {
        // Rate limiting: max one inference per inferenceIntervalMin
        let now = Date()
        guard now.timeIntervalSince(lastInferenceTime) >= inferenceIntervalMin else { return }
        lastInferenceTime = now

        let result: AudioClassificationResult

        if useMockModel || mlModel == nil {
            result = mockClassify(melSpectrogram: melSpectrogram, timestamp: timestamp)
        } else {
            result = await realClassify(melSpectrogram: melSpectrogram, timestamp: timestamp)
        }

        // Smooth predictions
        let smoothed = smooth(result)

        // Fire handler on main thread
        await MainActor.run {
            classificationHandler?(smoothed)
        }
    }

    // MARK: - Real CoreML Inference

    private func realClassify(melSpectrogram: [Float], timestamp: Date) async -> AudioClassificationResult {
        guard let model = mlModel else { return mockClassify(melSpectrogram: melSpectrogram, timestamp: timestamp) }

        // Compute mel energy for SNR estimation
        let energy = computeRMSEnergy(melSpectrogram)

        // Build MLMultiArray input
        // Expected shape: [1, numMelBins, numFrames]
        let numMels = MelSpectrogramParameters.numMelBins
        let numFrames = melSpectrogram.count / numMels

        guard numFrames > 0 else {
            return AudioClassificationResult(topLabel: .background, confidence: 1.0, allScores: [:], timestamp: timestamp, melEnergy: energy)
        }

        do {
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: numMels), NSNumber(value: numFrames)], dataType: .float32)
            for i in 0..<melSpectrogram.count {
                inputArray[i] = NSNumber(value: melSpectrogram[i])
            }

            let startTime = CFAbsoluteTimeGetCurrent()
            let featureProvider = try MLDictionaryFeatureProvider(dictionary: ["melSpectrogram": inputArray])
            let prediction = try model.prediction(from: featureProvider)
            let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            print("[SoundClassifier] Inference: \(String(format: "%.1f", elapsed))ms")

            // Parse output
            var allScores: [AcousticClass: Float] = [:]
            var topLabel: AcousticClass = .background
            var topScore: Float = 0

            for label in AcousticClass.allCases {
                if let val = prediction.featureValue(for: label.rawValue)?.doubleValue {
                    let score = Float(val)
                    allScores[label] = score
                    if score > topScore {
                        topScore = score
                        topLabel = label
                    }
                }
            }

            return AudioClassificationResult(
                topLabel: topLabel,
                confidence: topScore,
                allScores: allScores,
                timestamp: timestamp,
                melEnergy: energy
            )

        } catch {
            print("[SoundClassifier] Inference error: \(error)")
            return AudioClassificationResult(topLabel: .background, confidence: 1.0, allScores: [:], timestamp: timestamp, melEnergy: energy)
        }
    }

    // MARK: - Mock Model (Development)

    private func mockClassify(melSpectrogram: [Float], timestamp: Date) -> AudioClassificationResult {
        let energy = computeRMSEnergy(melSpectrogram)

        // Simulate realistic background with occasional drone detections
        // Uses energy level and some randomness to create plausible output
        var scores: [AcousticClass: Float] = [:]

        let base: Float = 0.02
        for label in AcousticClass.allCases {
            scores[label] = base
        }

        // Simulate drone detection based on energy patterns
        let highFreqEnergy = computeHighFrequencyEnergy(melSpectrogram)
        let lowFreqEnergy = computeLowFrequencyEnergy(melSpectrogram)

        if highFreqEnergy > 0.3 && lowFreqEnergy > 0.2 {
            // Simulate FPV drone signature
            scores[.fpv5inch] = 0.55 + Float.random(in: -0.05...0.05)
            scores[.background] = 0.15
        } else if lowFreqEnergy > 0.5 {
            // Simulate DJI Mavic (lower frequency)
            scores[.djiMavic] = 0.45 + Float.random(in: -0.05...0.05)
            scores[.background] = 0.2
        } else {
            // Background
            scores[.background] = 0.85 + Float.random(in: -0.05...0.05)
        }

        // Normalize
        let sum = scores.values.reduce(0, +)
        scores = scores.mapValues { $0 / sum }

        let top = scores.max(by: { $0.value < $1.value })!
        return AudioClassificationResult(
            topLabel: top.key,
            confidence: top.value,
            allScores: scores,
            timestamp: timestamp,
            melEnergy: energy
        )
    }

    // MARK: - Prediction Smoothing

    private func smooth(_ result: AudioClassificationResult) -> AudioClassificationResult {
        predictionHistory.append(result)
        if predictionHistory.count > historyMaxLength {
            predictionHistory.removeFirst()
        }

        // Exponential moving average
        var smoothedScores: [AcousticClass: Float] = [:]
        let weights: [Float] = [0.1, 0.15, 0.2, 0.25, 0.3] // Oldest to newest

        for label in AcousticClass.allCases {
            var weightedSum: Float = 0
            var weightSum: Float = 0
            for (i, pred) in predictionHistory.enumerated() {
                let w = i < weights.count ? weights[i] : 0.3
                weightedSum += (pred.allScores[label] ?? 0) * w
                weightSum += w
            }
            smoothedScores[label] = weightSum > 0 ? weightedSum / weightSum : 0
        }

        let top = smoothedScores.max(by: { $0.value < $1.value })!
        return AudioClassificationResult(
            topLabel: top.key,
            confidence: top.value,
            allScores: smoothedScores,
            timestamp: result.timestamp,
            melEnergy: result.melEnergy
        )
    }

    // MARK: - Helpers

    private func computeRMSEnergy(_ mel: [Float]) -> Float {
        guard !mel.isEmpty else { return 0 }
        var sum: Float = 0
        vDSP_svesq(mel, 1, &sum, vDSP_Length(mel.count))
        return sqrt(sum / Float(mel.count))
    }

    private func computeHighFrequencyEnergy(_ mel: [Float]) -> Float {
        let numMels = MelSpectrogramParameters.numMelBins
        guard mel.count >= numMels else { return 0 }
        // Take top 25% of mel bins
        let start = numMels * 3 / 4
        var subMel = [Float](repeating: 0, count: numMels - start)
        for i in start..<numMels {
            subMel[i - start] = mel[i]
        }
        return computeRMSEnergy(subMel)
    }

    private func computeLowFrequencyEnergy(_ mel: [Float]) -> Float {
        let numMels = MelSpectrogramParameters.numMelBins
        guard mel.count >= numMels else { return 0 }
        // Take bottom 25% of mel bins
        let end = numMels / 4
        var subMel = [Float](repeating: 0, count: end)
        for i in 0..<end {
            subMel[i] = mel[i]
        }
        return computeRMSEnergy(subMel)
    }
}
