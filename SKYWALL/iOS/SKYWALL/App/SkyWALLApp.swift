// SkyWALLApp.swift
// SKYWALL - Autonomous Aerial Detection System
// iOS 17+ / Xcode 16+
// Entry point: initializes all engines and requests permissions.

import SwiftUI
import AVFoundation
import CoreLocation
import CoreBluetooth
import UserNotifications

@main
struct SkyWALLApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .onAppear {
                    Task {
                        await appState.initialize()
                    }
                }
        }
    }
}

// MARK: - AppState

@MainActor
final class AppState: ObservableObject {
    // Engines
    let audioEngine     = SkyAudioEngine()
    let cameraEngine    = SkyCameraEngine()
    let tracker         = SkyTracker()
    let stateMachine    = SkyStateMachine()

    // Classifiers
    let soundClassifier = SkySoundClassifier()
    let objectDetector  = SkyObjectDetector()
    let deepClassifier  = SkyClassifier()

    // Controllers / Network / Storage
    let servoController = SkyServoController()
    let meshNetwork     = SkyMesh()
    let vault           = SkyVault()
    let directionEst    = SkyDirectionEstimator()

    @Published var isInitialized = false
    @Published var permissionsGranted = false
    @Published var initializationError: String?

    func initialize() async {
        do {
            // 1. Request all required permissions
            let granted = await requestAllPermissions()
            permissionsGranted = granted

            guard granted else {
                initializationError = "Required permissions were denied. Please enable in Settings."
                return
            }

            // 2. Initialize vault first (encryption keys)
            try await vault.initialize()

            // 3. Initialize audio engine and wire up classifier
            try await audioEngine.initialize()
            audioEngine.melSpectrogramHandler = { [weak self] mel, timestamp in
                guard let self else { return }
                Task { @MainActor in
                    await self.soundClassifier.classify(melSpectrogram: mel, timestamp: timestamp)
                }
            }

            // 4. Initialize camera engine
            try await cameraEngine.initialize()

            // 5. Set up object detector with camera frames
            cameraEngine.frameHandler = { [weak self] pixelBuffer, timestamp in
                guard let self else { return }
                Task { @MainActor in
                    await self.objectDetector.process(pixelBuffer: pixelBuffer, timestamp: timestamp)
                }
            }

            // 6. Wire object detector to deep classifier and tracker
            objectDetector.detectionHandler = { [weak self] detections, pixelBuffer, timestamp in
                guard let self else { return }
                Task { @MainActor in
                    await self.tracker.updateDetections(detections, pixelBuffer: pixelBuffer, timestamp: timestamp)

                    if let primaryDetection = detections.first(where: {
                        $0.confidence > 0.6 && ($0.label.contains("drone") || $0.label == "aircraft" || $0.label == "helicopter")
                    }) {
                        await self.deepClassifier.classify(
                            pixelBuffer: pixelBuffer,
                            detection: primaryDetection,
                            timestamp: timestamp
                        )
                    }
                }
            }

            // 7. Wire sound classifier to state machine
            soundClassifier.classificationHandler = { [weak self] result in
                guard let self else { return }
                Task { @MainActor in
                    await self.stateMachine.handleAudioClassification(result)
                }
            }

            // 8. Wire tracker to state machine and servo controller
            tracker.trackingHandler = { [weak self] trackState in
                guard let self else { return }
                Task { @MainActor in
                    await self.stateMachine.handleTrackingUpdate(trackState)
                    self.servoController.sendTrackingCommand(pan: trackState.bearing, tilt: trackState.elevation)
                }
            }

            // 9. Wire deep classifier to vault (persistence)
            deepClassifier.classificationHandler = { [weak self] classification, event in
                guard let self else { return }
                Task { @MainActor in
                    try? await self.vault.saveDetectionEvent(event)
                    await self.meshNetwork.broadcast(event: event)
                }
            }

            // 10. Initialize direction estimator
            directionEst.initialize()

            // 11. Wire direction estimator to audio engine buffers
            audioEngine.multiChannelHandler = { [weak self] buffers, timestamp in
                guard let self else { return }
                Task { @MainActor in
                    self.directionEst.process(channelBuffers: buffers, timestamp: timestamp)
                }
            }

            // 12. Initialize state machine
            await stateMachine.initialize(
                audioEngine: audioEngine,
                cameraEngine: cameraEngine,
                tracker: tracker,
                servoController: servoController
            )

            // 13. Initialize BLE servo controller
            servoController.initialize()

            // 14. Initialize mesh network
            try await meshNetwork.initialize()

            // 15. Start in sleep mode
            await stateMachine.transition(to: .sleep)

            isInitialized = true
            print("[SKYWALL] All systems initialized. Entering sleep mode.")

        } catch {
            initializationError = "Initialization failed: \(error.localizedDescription)"
            print("[SKYWALL] Fatal initialization error: \(error)")
        }
    }

    // MARK: - Permissions

    private func requestAllPermissions() async -> Bool {
        var results: [Bool] = []

        results.append(await requestMicrophonePermission())
        results.append(await requestCameraPermission())
        results.append(await requestLocationPermission())
        results.append(true) // BLE triggers its own dialog on first use

        await requestNotificationPermission()

        return results.allSatisfy { $0 }
    }

    private func requestMicrophonePermission() async -> Bool {
        await withCheckedContinuation { continuation in
            switch AVAudioApplication.shared.recordPermission {
            case .granted:
                continuation.resume(returning: true)
            case .denied:
                continuation.resume(returning: false)
            case .undetermined:
                AVAudioApplication.requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            @unknown default:
                continuation.resume(returning: false)
            }
        }
    }

    private func requestCameraPermission() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            return true
        case .denied, .restricted:
            return false
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        @unknown default:
            return false
        }
    }

    private func requestLocationPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            let manager = LocationPermissionRequester()
            manager.request { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    private func requestNotificationPermission() async {
        let center = UNUserNotificationCenter.current()
        let options: UNAuthorizationOptions = [.alert, .sound, .badge]
        _ = try? await center.requestAuthorization(options: options)
    }
}

// MARK: - LocationPermissionRequester Helper

final class LocationPermissionRequester: NSObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    private var completion: ((Bool) -> Void)?

    func request(completion: @escaping (Bool) -> Void) {
        self.completion = completion
        manager.delegate = self

        let status = manager.authorizationStatus
        switch status {
        case .authorizedAlways, .authorizedWhenInUse:
            completion(true)
        case .denied, .restricted:
            completion(false)
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        @unknown default:
            completion(false)
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        switch manager.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse:
            completion?(true)
        case .denied, .restricted:
            completion?(false)
        default:
            break
        }
        completion = nil
    }
}
