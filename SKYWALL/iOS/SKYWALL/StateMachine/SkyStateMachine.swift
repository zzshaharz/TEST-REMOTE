// SkyStateMachine.swift
// SKYWALL - Autonomous Aerial Detection System
// Thread-safe state machine using Swift actors.

import Foundation
import Combine

// MARK: - SkyMode

enum SkyMode: String, Codable, CaseIterable {
    case sleep    = "SLEEP"
    case alert    = "ALERT"
    case track    = "TRACK"
    case document = "DOCUMENT"
    case patrol   = "PATROL"
}

// MARK: - State Transition Rules

private struct TransitionRule {
    let from: SkyMode
    let to: SkyMode
    let condition: String
}

private let transitionRules: [TransitionRule] = [
    TransitionRule(from: .sleep,    to: .patrol,   condition: "patrol_scheduled"),
    TransitionRule(from: .sleep,    to: .alert,    condition: "audio_alert"),
    TransitionRule(from: .patrol,   to: .alert,    condition: "audio_alert"),
    TransitionRule(from: .patrol,   to: .sleep,    condition: "patrol_complete"),
    TransitionRule(from: .alert,    to: .sleep,    condition: "timeout_no_confirm"),
    TransitionRule(from: .alert,    to: .track,    condition: "visual_lock"),
    TransitionRule(from: .alert,    to: .patrol,   condition: "audio_only_confirmed"),
    TransitionRule(from: .track,    to: .document, condition: "high_confidence"),
    TransitionRule(from: .track,    to: .alert,    condition: "track_lost"),
    TransitionRule(from: .document, to: .track,    condition: "classification_done"),
    TransitionRule(from: .document, to: .alert,    condition: "track_lost"),
    TransitionRule(from: .document, to: .sleep,    condition: "target_gone"),
]

// MARK: - State Machine Events

enum SkyEvent {
    case audioAlert(classification: AudioClassificationResult)
    case audioConfirmed(classification: AudioClassificationResult)
    case audioLost
    case visualLock(detection: DetectedObject)
    case trackLost
    case trackUpdate(state: TrackState)
    case classificationComplete(ThreatClassification)
    case patrolStart
    case patrolComplete
    case forceTransition(SkyMode)
    case timeout
}

// MARK: - State Machine Delegate

protocol SkyStateMachineDelegate: AnyObject {
    func stateMachine(_ sm: SkyStateMachine, didEnter mode: SkyMode, from previous: SkyMode)
    func stateMachine(_ sm: SkyStateMachine, didExit mode: SkyMode)
    func stateMachine(_ sm: SkyStateMachine, didReject event: SkyEvent, reason: String)
}

// MARK: - SkyStateMachine Actor

actor SkyStateMachine {

    // MARK: - Published state (bridged to main actor)
    private(set) var currentMode: SkyMode = .sleep
    private(set) var previousMode: SkyMode = .sleep
    private(set) var lastTransitionTime: Date = Date()
    private(set) var alertCount: Int = 0
    private(set) var trackCount: Int = 0

    // Timeout configuration (seconds)
    private let alertTimeout: TimeInterval   = 30.0
    private let trackTimeout: TimeInterval   = 10.0
    private let documentTimeout: TimeInterval = 60.0
    private let patrolCycleDuration: TimeInterval = 120.0

    private var timeoutTask: Task<Void, Never>?
    private var patrolTask: Task<Void, Never>?

    // Weak references to engines (bridged)
    private weak var audioEngine: SkyAudioEngine?
    private weak var cameraEngine: SkyCameraEngine?
    private weak var tracker: SkyTracker?
    private weak var servoController: SkyServoController?

    weak var delegate: SkyStateMachineDelegate?

    // Combine publisher for UI updates (nonisolated)
    nonisolated let modePublisher = PassthroughSubject<SkyMode, Never>()
    nonisolated let eventPublisher = PassthroughSubject<SkyEvent, Never>()

    // MARK: - Initialization

    func initialize(
        audioEngine: SkyAudioEngine,
        cameraEngine: SkyCameraEngine,
        tracker: SkyTracker,
        servoController: SkyServoController
    ) async {
        self.audioEngine = audioEngine
        self.cameraEngine = cameraEngine
        self.tracker = tracker
        self.servoController = servoController
        print("[StateMachine] Initialized.")
    }

    // MARK: - Event Handling

    func handleAudioClassification(_ result: AudioClassificationResult) async {
        switch currentMode {
        case .sleep, .patrol:
            if result.confidence >= 0.4 {
                await transition(to: .alert)
                await processEvent(.audioAlert(classification: result))
            }
        case .alert:
            if result.confidence >= 0.7 {
                await processEvent(.audioConfirmed(classification: result))
            } else if result.confidence < 0.2 {
                // Signal weakening - refresh timeout
                resetTimeout()
            }
        case .track, .document:
            // Already tracking; audio confirmation is supplemental
            break
        }
    }

    func handleTrackingUpdate(_ state: TrackState) async {
        switch currentMode {
        case .alert:
            if state.isLocked {
                await transition(to: .track)
            }
        case .track:
            if !state.isLocked {
                await processEvent(.trackLost)
            }
            if state.confidence > 0.85 {
                await processEvent(.highConfidenceTrack)
            }
        case .document:
            if !state.isLocked {
                await processEvent(.trackLost)
            }
        default:
            break
        }
    }

    func processEvent(_ event: SkyEvent) async {
        eventPublisher.send(event)

        switch event {
        case .forceTransition(let mode):
            await transition(to: mode)

        case .audioAlert(let classification):
            if currentMode == .sleep || currentMode == .patrol {
                await transition(to: .alert)
            }
            print("[StateMachine] Audio alert: \(classification.topLabel) @ \(String(format: "%.2f", classification.confidence))")

        case .audioConfirmed(let classification):
            print("[StateMachine] Audio confirmed: \(classification.topLabel) @ \(String(format: "%.2f", classification.confidence))")
            if currentMode == .alert {
                alertCount += 1
            }

        case .audioLost:
            if currentMode == .alert {
                startTimeout(duration: alertTimeout)
            }

        case .visualLock(let detection):
            if currentMode == .alert || currentMode == .patrol {
                await transition(to: .track)
                print("[StateMachine] Visual lock on: \(detection.label)")
            }

        case .trackLost:
            if currentMode == .track || currentMode == .document {
                await transition(to: .alert)
                startTimeout(duration: alertTimeout)
            }

        case .trackUpdate:
            resetTimeout()

        case .classificationComplete(let threat):
            if currentMode == .track && threat.confidence > 0.7 {
                await transition(to: .document)
            }

        case .patrolStart:
            if currentMode == .sleep {
                await transition(to: .patrol)
            }

        case .patrolComplete:
            if currentMode == .patrol {
                await transition(to: .sleep)
            }

        case .timeout:
            await handleTimeout()
        }
    }

    // MARK: - Transition Logic

    func transition(to newMode: SkyMode) async {
        guard newMode != currentMode else { return }

        // Validate transition is allowed
        let isValid = isValidTransition(from: currentMode, to: newMode)
        if !isValid {
            print("[StateMachine] Rejected transition \(currentMode) -> \(newMode): no rule.")
            return
        }

        let oldMode = currentMode
        print("[StateMachine] Transition: \(oldMode.rawValue) -> \(newMode.rawValue)")

        // Exit old state
        await onExit(mode: oldMode)
        delegate?.stateMachine(self, didExit: oldMode)

        // Update state
        previousMode = oldMode
        currentMode = newMode
        lastTransitionTime = Date()

        // Enter new state
        await onEnter(mode: newMode, from: oldMode)
        delegate?.stateMachine(self, didEnter: newMode, from: oldMode)

        // Notify UI
        modePublisher.send(newMode)
    }

    private func isValidTransition(from: SkyMode, to: SkyMode) -> Bool {
        // Allow any -> document for forced transitions
        if to == from { return false }
        // Force transitions always allowed
        return transitionRules.contains { $0.from == from && $0.to == to }
            || to == .sleep // Always allowed to go back to sleep
    }

    // MARK: - State Entry/Exit Actions

    private func onEnter(mode: SkyMode, from previous: SkyMode) async {
        cancelTimeout()

        switch mode {
        case .sleep:
            print("[StateMachine] Entering SLEEP: stopping audio/camera.")
            await audioEngine?.setMode(.passive)
            await cameraEngine?.setMode(.sleep)
            servoController?.sendMode(.patrol) // Return to patrol position

        case .alert:
            print("[StateMachine] Entering ALERT: activating wide camera scan.")
            await audioEngine?.setMode(.active)
            await cameraEngine?.setMode(.alert)
            startTimeout(duration: alertTimeout)
            alertCount += 1

        case .track:
            print("[StateMachine] Entering TRACK: telephoto tracking active.")
            await cameraEngine?.setMode(.track)
            await tracker?.startTracking()
            startTimeout(duration: trackTimeout)
            trackCount += 1

        case .document:
            print("[StateMachine] Entering DOCUMENT: 4K capture + classification.")
            await cameraEngine?.setMode(.document)
            startTimeout(duration: documentTimeout)

        case .patrol:
            print("[StateMachine] Entering PATROL: autonomous 360 sweep.")
            await cameraEngine?.setMode(.scan)
            servoController?.sendMode(.patrol)
            startPatrolCycle()
        }
    }

    private func onExit(mode: SkyMode) async {
        switch mode {
        case .track:
            await tracker?.pauseTracking()
        case .document:
            await cameraEngine?.finishDocumentCapture()
        case .patrol:
            cancelPatrol()
        default:
            break
        }
    }

    // MARK: - Timeout Management

    private func startTimeout(duration: TimeInterval) {
        cancelTimeout()
        timeoutTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
            if !Task.isCancelled {
                await self.processEvent(.timeout)
            }
        }
    }

    private func resetTimeout() {
        let mode = currentMode
        switch mode {
        case .alert:    startTimeout(duration: alertTimeout)
        case .track:    startTimeout(duration: trackTimeout)
        case .document: startTimeout(duration: documentTimeout)
        default: break
        }
    }

    private func cancelTimeout() {
        timeoutTask?.cancel()
        timeoutTask = nil
    }

    private func handleTimeout() async {
        print("[StateMachine] Timeout in state: \(currentMode.rawValue)")
        switch currentMode {
        case .alert:
            await transition(to: .sleep)
        case .track:
            await transition(to: .alert)
        case .document:
            await transition(to: .sleep)
        default:
            break
        }
    }

    // MARK: - Patrol Management

    private func startPatrolCycle() {
        cancelPatrol()
        patrolTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(nanoseconds: UInt64(patrolCycleDuration * 1_000_000_000))
            if !Task.isCancelled {
                await self.processEvent(.patrolComplete)
            }
        }
    }

    private func cancelPatrol() {
        patrolTask?.cancel()
        patrolTask = nil
    }

    // MARK: - Status

    var statusDescription: String {
        "[\(currentMode.rawValue)] alerts:\(alertCount) tracks:\(trackCount) since:\(ISO8601DateFormatter().string(from: lastTransitionTime))"
    }
}

// MARK: - Supporting Event Extension (for highConfidenceTrack)

extension SkyStateMachine {
    nonisolated func highConfidenceTrackEvent() -> SkyEvent {
        .classificationComplete(ThreatClassification(
            droneClass: .unknown,
            confidence: 0.9,
            bearing: 0,
            elevation: 0,
            estimatedRange: nil,
            behavior: .hovering,
            registrationVisible: false,
            payloadVisible: false,
            firstSeen: Date(),
            lastSeen: Date(),
            trackPoints: [],
            notes: "Auto high-confidence"
        ))
    }
}

// Extend SkyEvent for internal use
extension SkyEvent {
    static var highConfidenceTrack: SkyEvent {
        .classificationComplete(ThreatClassification(
            droneClass: .unknown,
            confidence: 0.9,
            bearing: 0,
            elevation: 0,
            estimatedRange: nil,
            behavior: .hovering,
            registrationVisible: false,
            payloadVisible: false,
            firstSeen: Date(),
            lastSeen: Date(),
            trackPoints: [],
            notes: "Auto high-confidence track"
        ))
    }
}
