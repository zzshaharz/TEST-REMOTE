// SkyTracker.swift
// SKYWALL - Autonomous Aerial Detection System
// Object tracking engine: Vision VNTrackObjectRequest + YOLO re-detection + PID controller.

import Vision
import CoreImage
import UIKit
import Accelerate

// MARK: - Track State

struct TrackState {
    let id: UUID
    let boundingBox: CGRect
    let bearing: Float          // degrees, north=0
    let elevation: Float        // degrees above horizon
    let confidence: Float
    let velocity: CGPoint       // pixels/second
    let timestamp: Date
    let isLocked: Bool

    // Predicted position ahead (for aim-ahead)
    let predictedBoundingBox: CGRect
    let predictedBearing: Float
}

// MARK: - Track History Point

struct TrackHistoryPoint {
    let timestamp: Date
    let boundingBox: CGRect
    let bearing: Float
    let elevation: Float
    let confidence: Float
}

// MARK: - PID Controller

struct PIDController {
    var kp: Float
    var ki: Float
    var kd: Float

    private var integral: Float = 0
    private var previousError: Float = 0
    private var lastUpdateTime: Date = Date()

    init(kp: Float, ki: Float, kd: Float) {
        self.kp = kp
        self.ki = ki
        self.kd = kd
    }

    mutating func update(setpoint: Float, measurement: Float) -> Float {
        let now = Date()
        let dt = Float(now.timeIntervalSince(lastUpdateTime))
        lastUpdateTime = now

        guard dt > 0 && dt < 1.0 else {
            previousError = setpoint - measurement
            return 0
        }

        let error = setpoint - measurement

        // Proportional
        let p = kp * error

        // Integral with anti-windup
        integral += error * dt
        integral = max(-100, min(100, integral))  // Clamp
        let i = ki * integral

        // Derivative
        let d = kd * (error - previousError) / dt
        previousError = error

        return p + i + d
    }

    mutating func reset() {
        integral = 0
        previousError = 0
    }
}

// MARK: - SkyTracker

final class SkyTracker {

    // MARK: - Properties

    private(set) var currentTrackState: TrackState?
    private(set) var trackHistory: [TrackHistoryPoint] = []
    private var activeTrackID: UUID?
    private var isTracking = false

    // Vision tracking
    private var trackingRequest: VNTrackObjectRequest?
    private var lastObservation: VNDetectedObjectObservation?
    private let trackingQueue = DispatchQueue(label: "com.skywall.tracker", qos: .userInteractive)

    // Frame rate control
    private var lastTrackTime: Date = .distantPast
    private let maxTrackFPS: Double = 30.0
    private var frameCount = 0

    // PID controllers (pan, tilt)
    private var panPID  = PIDController(kp: 0.8, ki: 0.05, kd: 0.15)
    private var tiltPID = PIDController(kp: 0.8, ki: 0.05, kd: 0.15)

    // Target setpoint (center of frame)
    private let setpointX: Float = 0.5
    private let setpointY: Float = 0.5

    // Current estimated bearing/elevation from gyro/compass
    private var currentPanAngle: Float = 0
    private var currentTiltAngle: Float = 0

    // Aim-ahead prediction window (seconds)
    private let aimAheadTime: Float = 0.15

    // Max track history points
    private let maxHistory = 300

    // Callback
    var trackingHandler: ((TrackState) -> Void)?

    // Re-detection interval: run YOLO every 10 tracking frames
    private let yoloRedetectInterval = 10

    // MARK: - Control

    func startTracking() async {
        isTracking = true
        activeTrackID = UUID()
        trackHistory.removeAll()
        panPID.reset()
        tiltPID.reset()
        print("[Tracker] Tracking started.")
    }

    func pauseTracking() async {
        isTracking = false
        trackingRequest = nil
        lastObservation = nil
        print("[Tracker] Tracking paused.")
    }

    func stopTracking() async {
        await pauseTracking()
        activeTrackID = nil
        currentTrackState = nil
        trackHistory.removeAll()
        print("[Tracker] Tracking stopped.")
    }

    // MARK: - Update with YOLO Detections

    func updateDetections(_ detections: [DetectedObject], pixelBuffer: CVPixelBuffer, timestamp: Date) async {
        guard isTracking else { return }

        frameCount += 1

        // Rate limit
        let now = timestamp
        let interval = now.timeIntervalSince(lastTrackTime)
        guard interval >= 1.0 / maxTrackFPS else { return }
        lastTrackTime = now

        if let observation = lastObservation, frameCount % yoloRedetectInterval != 0 {
            // Use Vision tracking
            await trackWithVision(observation: observation, pixelBuffer: pixelBuffer, timestamp: timestamp)
        } else {
            // Use YOLO detection to establish or re-confirm track
            if let best = detections.filter({ $0.isDrone }).max(by: { $0.confidence < $1.confidence }) {
                await establishTrack(detection: best, pixelBuffer: pixelBuffer, timestamp: timestamp)
            } else if let obs = lastObservation {
                // No YOLO hit, continue Vision tracking
                await trackWithVision(observation: obs, pixelBuffer: pixelBuffer, timestamp: timestamp)
            }
        }
    }

    // MARK: - Establish Track from YOLO Detection

    private func establishTrack(detection: DetectedObject, pixelBuffer: CVPixelBuffer, timestamp: Date) async {
        let observation = VNDetectedObjectObservation(boundingBox: detection.boundingBox)
        lastObservation = observation

        await processTrackUpdate(
            boundingBox: detection.boundingBox,
            confidence: detection.confidence,
            timestamp: timestamp,
            pixelBuffer: pixelBuffer
        )
    }

    // MARK: - Vision KCF/CSRT Tracking

    private func trackWithVision(observation: VNDetectedObjectObservation, pixelBuffer: CVPixelBuffer, timestamp: Date) async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            trackingQueue.async { [weak self] in
                guard let self else { cont.resume(); return }

                let trackReq = VNTrackObjectRequest(detectedObjectObservation: observation) { request, error in
                    defer { cont.resume() }

                    guard error == nil,
                          let results = request.results as? [VNDetectedObjectObservation],
                          let result = results.first else {
                        // Track lost
                        Task { @MainActor in
                            self.handleTrackLost()
                        }
                        return
                    }

                    self.lastObservation = result

                    Task { @MainActor in
                        await self.processTrackUpdate(
                            boundingBox: result.boundingBox,
                            confidence: result.confidence,
                            timestamp: timestamp,
                            pixelBuffer: pixelBuffer
                        )
                    }
                }

                // Use accurate tracking (more CPU but better quality)
                trackReq.trackingLevel = .accurate
                trackReq.isLastFrame = false

                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
                do {
                    try handler.perform([trackReq])
                } catch {
                    print("[Tracker] Vision tracking error: \(error)")
                    cont.resume()
                }
            }
        }
    }

    // MARK: - Process Track Update

    @MainActor
    private func processTrackUpdate(boundingBox: CGRect, confidence: Float, timestamp: Date, pixelBuffer: CVPixelBuffer) async {
        guard let trackID = activeTrackID else { return }

        // Compute velocity from history
        let velocity = computeVelocity(currentBox: boundingBox, timestamp: timestamp)

        // Aim-ahead prediction
        let predicted = CGRect(
            x: boundingBox.origin.x + velocity.x * CGFloat(aimAheadTime),
            y: boundingBox.origin.y + velocity.y * CGFloat(aimAheadTime),
            width: boundingBox.width,
            height: boundingBox.height
        )

        // Convert bounding box center to bearing/elevation
        let center = CGPoint(x: boundingBox.midX, y: boundingBox.midY)
        let bearing  = computeBearing(normalizedX: Float(center.x))
        let elevation = computeElevation(normalizedY: Float(center.y))
        let predictedBearing = computeBearing(normalizedX: Float(predicted.midX))

        // PID output for servo commands
        let panError   = setpointX - Float(center.x)
        let tiltError  = setpointY - Float(center.y)
        let panOutput  = panPID.update(setpoint: setpointX, measurement: Float(center.x))
        let tiltOutput = tiltPID.update(setpoint: setpointY, measurement: Float(center.y))

        // Update current angles based on PID
        currentPanAngle  += panOutput  * 2.0   // 2° per PID unit
        currentTiltAngle += tiltOutput * 2.0
        currentPanAngle  = currentPanAngle.truncatingRemainder(dividingBy: 360)
        currentTiltAngle = max(-35, min(90, currentTiltAngle))

        let state = TrackState(
            id: trackID,
            boundingBox: boundingBox,
            bearing: bearing,
            elevation: elevation,
            confidence: confidence,
            velocity: velocity,
            timestamp: timestamp,
            isLocked: confidence > 0.4,
            predictedBoundingBox: predicted,
            predictedBearing: predictedBearing
        )

        currentTrackState = state

        // Append history
        let historyPoint = TrackHistoryPoint(
            timestamp: timestamp,
            boundingBox: boundingBox,
            bearing: bearing,
            elevation: elevation,
            confidence: confidence
        )
        trackHistory.append(historyPoint)
        if trackHistory.count > maxHistory {
            trackHistory.removeFirst()
        }

        // Fire callback every 100ms (10Hz BLE update rate)
        trackingHandler?(state)
    }

    // MARK: - Track Lost

    private func handleTrackLost() {
        print("[Tracker] Track lost.")
        lastObservation = nil
    }

    // MARK: - Helpers

    private func computeVelocity(currentBox: CGRect, timestamp: Date) -> CGPoint {
        guard let last = trackHistory.last else { return .zero }
        let dt = timestamp.timeIntervalSince(last.timestamp)
        guard dt > 0 else { return .zero }

        let dx = (currentBox.midX - last.boundingBox.midX) / dt
        let dy = (currentBox.midY - last.boundingBox.midY) / dt
        return CGPoint(x: dx, y: dy)
    }

    private func computeBearing(normalizedX: Float) -> Float {
        // Map normalized X [0,1] to relative bearing offset
        // 0.5 = straight ahead, 0 = far left, 1 = far right
        // Assumes ~60° horizontal FOV for telephoto
        let hFOV: Float = 60.0
        return currentPanAngle + (normalizedX - 0.5) * hFOV
    }

    private func computeElevation(normalizedY: Float) -> Float {
        // Map normalized Y [0,1] to elevation
        // 0 = top (high elevation), 1 = bottom (low elevation)
        // Assumes ~45° vertical FOV
        let vFOV: Float = 45.0
        return currentTiltAngle + (0.5 - normalizedY) * vFOV
    }

    // MARK: - Trajectory Prediction

    func predictTrajectory(seconds ahead: Float) -> TrackHistoryPoint? {
        guard trackHistory.count >= 3 else { return nil }

        let recent = Array(trackHistory.suffix(10))
        guard recent.count >= 2 else { return nil }

        // Linear regression on last N points
        let n = Float(recent.count)
        var sumT: Float = 0, sumX: Float = 0, sumY: Float = 0
        var sumT2: Float = 0, sumTX: Float = 0, sumTY: Float = 0

        let t0 = recent.first!.timestamp

        for point in recent {
            let t = Float(point.timestamp.timeIntervalSince(t0))
            let x = Float(point.boundingBox.midX)
            let y = Float(point.boundingBox.midY)
            sumT  += t
            sumX  += x
            sumY  += y
            sumT2 += t * t
            sumTX += t * x
            sumTY += t * y
        }

        let denom = n * sumT2 - sumT * sumT
        guard abs(denom) > 1e-6 else { return nil }

        let slopeX = (n * sumTX - sumT * sumX) / denom
        let slopeY = (n * sumTY - sumT * sumY) / denom
        let interceptX = (sumX - slopeX * sumT) / n
        let interceptY = (sumY - slopeY * sumT) / n

        let lastT = Float(recent.last!.timestamp.timeIntervalSince(t0))
        let futureT = lastT + ahead
        let predX = interceptX + slopeX * futureT
        let predY = interceptY + slopeY * futureT

        let lastBox = recent.last!.boundingBox
        let predBox = CGRect(
            x: CGFloat(predX) - lastBox.width / 2,
            y: CGFloat(predY) - lastBox.height / 2,
            width: lastBox.width,
            height: lastBox.height
        )

        let predBearing = computeBearing(normalizedX: predX)
        let predElevation = computeElevation(normalizedY: predY)

        return TrackHistoryPoint(
            timestamp: Date().addingTimeInterval(TimeInterval(ahead)),
            boundingBox: predBox,
            bearing: predBearing,
            elevation: predElevation,
            confidence: max(0, recent.last!.confidence - ahead * 0.1)
        )
    }
}
