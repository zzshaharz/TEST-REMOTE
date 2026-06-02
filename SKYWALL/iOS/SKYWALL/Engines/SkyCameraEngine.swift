// SkyCameraEngine.swift
// SKYWALL - Autonomous Aerial Detection System
// Full AVFoundation camera engine with mode-based configuration.

import AVFoundation
import UIKit
import CoreImage

// MARK: - Camera Mode

enum CameraMode {
    case sleep      // Camera off / standby
    case scan       // Wide angle, 5 fps
    case alert      // Wide angle, 15-30 fps
    case track      // Telephoto, 30 fps
    case document   // 4K video + highest quality still
}

// MARK: - Capture Outputs

struct CaptureFrame {
    let pixelBuffer: CVPixelBuffer
    let timestamp: Date
    let cameraMode: CameraMode
    let intrinsics: simd_float3x3?  // Camera intrinsic matrix
}

// MARK: - SkyCameraEngine

final class SkyCameraEngine: NSObject {

    // MARK: - Session

    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.skywall.camera.session", qos: .userInteractive)

    // Inputs
    private var wideInput: AVCaptureDeviceInput?
    private var telephotInput: AVCaptureDeviceInput?
    private var activeInput: AVCaptureDeviceInput?

    // Outputs
    private var videoOutput: AVCaptureVideoDataOutput?
    private var photoOutput: AVCapturePhotoOutput?
    private var movieOutput: AVCaptureMovieFileOutput?

    // State
    private(set) var currentMode: CameraMode = .sleep
    private(set) var isRunning = false
    private var isCapturingDocument = false
    private var movieOutputURL: URL?

    // Devices
    private var wideCamera: AVCaptureDevice?
    private var telephotoCamera: AVCaptureDevice?

    // Frame callbacks
    var frameHandler: ((CVPixelBuffer, Date) -> Void)?
    var photoHandler: ((UIImage, Date) -> Void)?

    // MARK: - Initialization

    func initialize() async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            sessionQueue.async { [weak self] in
                guard let self else { cont.resume(); return }
                do {
                    try self.configureSession()
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    private func configureSession() throws {
        captureSession.beginConfiguration()
        defer { captureSession.commitConfiguration() }

        // Select session preset
        captureSession.sessionPreset = .high

        // Discover cameras
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .builtInTelephotoCamera, .builtInUltraWideCamera],
            mediaType: .video,
            position: .back
        )

        for device in discoverySession.devices {
            switch device.deviceType {
            case .builtInWideAngleCamera:
                wideCamera = device
            case .builtInTelephotoCamera:
                telephotoCamera = device
            default:
                break
            }
        }

        guard let wide = wideCamera else {
            throw CameraEngineError.noCamera
        }

        // Configure wide camera for sky imaging
        try configureDevice(wide, for: .alert)

        // Add wide input
        let wideIn = try AVCaptureDeviceInput(device: wide)
        guard captureSession.canAddInput(wideIn) else {
            throw CameraEngineError.cannotAddInput
        }
        captureSession.addInput(wideIn)
        wideInput = wideIn
        activeInput = wideIn

        // Video data output (for ML processing)
        let vidOut = AVCaptureVideoDataOutput()
        vidOut.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        vidOut.alwaysDiscardsLateVideoFrames = true
        vidOut.setSampleBufferDelegate(self, queue: sessionQueue)

        guard captureSession.canAddOutput(vidOut) else {
            throw CameraEngineError.cannotAddOutput
        }
        captureSession.addOutput(vidOut)
        videoOutput = vidOut

        // Configure video connection for portrait orientation
        if let connection = vidOut.connection(with: .video) {
            connection.videoRotationAngle = 90
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .cinematic
            }
        }

        // Photo output
        let photoOut = AVCapturePhotoOutput()
        photoOut.isHighResolutionCaptureEnabled = true
        if captureSession.canAddOutput(photoOut) {
            captureSession.addOutput(photoOut)
            photoOutput = photoOut
        }

        // Movie output
        let movieOut = AVCaptureMovieFileOutput()
        if captureSession.canAddOutput(movieOut) {
            captureSession.addOutput(movieOut)
            movieOutput = movieOut
        }

        print("[CameraEngine] Session configured: wide=\(wide.localizedName) telephoto=\(telephotoCamera?.localizedName ?? "none")")
    }

    // MARK: - Device Configuration

    private func configureDevice(_ device: AVCaptureDevice, for mode: CameraMode) throws {
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }

        switch mode {
        case .sleep:
            break

        case .scan:
            // Low frame rate to save power
            setFrameRate(device: device, fps: 5)
            device.exposureMode = .continuousAutoExposure
            configureExposureForSky(device)

        case .alert:
            setFrameRate(device: device, fps: 15)
            device.exposureMode = .continuousAutoExposure
            configureExposureForSky(device)
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }

        case .track:
            setFrameRate(device: device, fps: 30)
            device.exposureMode = .continuousAutoExposure
            configureExposureForSky(device)
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }

        case .document:
            setFrameRate(device: device, fps: 30)
            device.exposureMode = .continuousAutoExposure
            configureExposureForSky(device)
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
        }
    }

    private func configureExposureForSky(_ device: AVCaptureDevice) {
        // Bias exposure slightly towards sky (negative EV for bright sky)
        if device.isExposurePointOfInterestSupported {
            device.exposurePointOfInterest = CGPoint(x: 0.5, y: 0.3) // Upper portion
        }
        // Slight negative bias for sky
        let targetBias: Float = -0.5
        let clampedBias = min(max(targetBias, device.minExposureTargetBias), device.maxExposureTargetBias)
        device.setExposureTargetBias(clampedBias, completionHandler: nil)
    }

    private func setFrameRate(_ device: AVCaptureDevice, fps: Int) {
        let targetFPS = CMTimeMake(value: 1, timescale: CMTimeScale(fps))
        // Find the active format that supports the desired FPS
        for range in device.activeFormat.videoSupportedFrameRateRanges {
            if Double(fps) >= range.minFrameRate && Double(fps) <= range.maxFrameRate {
                device.activeVideoMinFrameDuration = targetFPS
                device.activeVideoMaxFrameDuration = targetFPS
                return
            }
        }
    }

    // MARK: - Mode Transitions

    func setMode(_ newMode: CameraMode) async {
        guard newMode != currentMode else { return }
        let oldMode = currentMode
        currentMode = newMode
        print("[CameraEngine] Mode: \(oldMode) -> \(newMode)")

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            sessionQueue.async { [weak self] in
                guard let self else { cont.resume(); return }
                self.applyMode(newMode)
                cont.resume()
            }
        }
    }

    private func applyMode(_ mode: CameraMode) {
        captureSession.beginConfiguration()
        defer { captureSession.commitConfiguration() }

        switch mode {
        case .sleep:
            captureSession.stopRunning()
            isRunning = false

        case .scan, .alert:
            if !captureSession.isRunning {
                captureSession.startRunning()
                isRunning = true
            }
            switchToCamera(.wide)
            if let wide = wideCamera {
                try? configureDevice(wide, for: mode)
            }

        case .track:
            if !captureSession.isRunning {
                captureSession.startRunning()
                isRunning = true
            }
            // Switch to telephoto if available, else stay wide
            if telephotoCamera != nil {
                switchToCamera(.telephoto)
                if let tele = telephotoCamera {
                    try? configureDevice(tele, for: mode)
                }
            } else {
                if let wide = wideCamera {
                    try? configureDevice(wide, for: mode)
                }
            }

        case .document:
            if !captureSession.isRunning {
                captureSession.startRunning()
                isRunning = true
            }
            if telephotoCamera != nil {
                switchToCamera(.telephoto)
            }
            // Set 4K session preset
            if captureSession.canSetSessionPreset(.hd4K3840x2160) {
                captureSession.sessionPreset = .hd4K3840x2160
            }
            startMovieCapture()
        }
    }

    private enum CameraType { case wide, telephoto }

    private func switchToCamera(_ type: CameraType) {
        let targetDevice = type == .wide ? wideCamera : telephotoCamera
        guard let target = targetDevice else { return }

        if let currentIn = activeInput {
            captureSession.removeInput(currentIn)
        }

        do {
            let newInput = try AVCaptureDeviceInput(device: target)
            if captureSession.canAddInput(newInput) {
                captureSession.addInput(newInput)
                activeInput = newInput
            }
        } catch {
            print("[CameraEngine] Failed to switch camera: \(error)")
        }
    }

    // MARK: - Document Capture

    private func startMovieCapture() {
        guard let movieOut = movieOutput else { return }
        let docDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileName = "skywall_\(Date().timeIntervalSince1970).mov"
        let url = docDir.appendingPathComponent(fileName)
        movieOut.startRecording(to: url, recordingDelegate: self)
        movieOutputURL = url
        isCapturingDocument = true
        print("[CameraEngine] Movie capture started: \(fileName)")
    }

    func finishDocumentCapture() async {
        guard isCapturingDocument, let movieOut = movieOutput else { return }
        movieOut.stopRecording()
        isCapturingDocument = false
        print("[CameraEngine] Movie capture stopped.")
    }

    // MARK: - Still Photo Capture

    func capturePhoto() {
        guard let photoOut = photoOutput else { return }
        let settings = AVCapturePhotoSettings()
        settings.isHighResolutionPhotoEnabled = true
        if photoOut.availablePhotoCodecTypes.contains(.hevc) {
            // Use HEVC for smaller file sizes
        }
        photoOut.capturePhoto(with: settings, delegate: self)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension SkyCameraEngine: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard currentMode != .sleep else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let timestamp = Date()
        frameHandler?(pixelBuffer, timestamp)
    }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension SkyCameraEngine: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard error == nil, let data = photo.fileDataRepresentation(),
              let image = UIImage(data: data) else { return }
        photoHandler?(image, Date())
    }
}

// MARK: - AVCaptureFileOutputRecordingDelegate

extension SkyCameraEngine: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        if let error {
            print("[CameraEngine] Recording finished with error: \(error)")
        } else {
            print("[CameraEngine] Recording saved to: \(outputFileURL.lastPathComponent)")
        }
    }
}

// MARK: - Errors

enum CameraEngineError: Error, LocalizedError {
    case noCamera
    case cannotAddInput
    case cannotAddOutput
    case sessionPresetUnsupported

    var errorDescription: String? {
        switch self {
        case .noCamera:                    return "No camera device available"
        case .cannotAddInput:              return "Cannot add camera input to session"
        case .cannotAddOutput:             return "Cannot add output to session"
        case .sessionPresetUnsupported:    return "Session preset not supported"
        }
    }
}
