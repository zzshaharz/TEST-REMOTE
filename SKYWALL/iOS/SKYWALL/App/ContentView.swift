// ContentView.swift
// SKYWALL - Autonomous Aerial Detection System
// Main SwiftUI interface: camera feed, detection overlays, mode indicator, log.

import SwiftUI
import AVFoundation
import CoreLocation

// MARK: - ContentView

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var viewModel = ContentViewModel()
    @State private var showSettings = false
    @State private var showLog = false

    var body: some View {
        ZStack {
            // Camera feed background
            CameraPreviewView(session: appState.cameraEngine.captureSession)
                .ignoresSafeArea()

            // Detection overlay (bounding boxes)
            DetectionOverlayView(
                detections: viewModel.currentDetections,
                trackState: viewModel.currentTrackState
            )
            .ignoresSafeArea()

            // HUD layer
            VStack(spacing: 0) {
                TopBarView(
                    mode: viewModel.currentMode,
                    nodeCount: viewModel.connectedNodes,
                    showSettings: $showSettings
                )
                .padding(.top, 50)

                Spacer()

                // Compass bearing display
                if viewModel.currentMode == .alert || viewModel.currentMode == .track {
                    BearingCompassView(
                        bearing: viewModel.currentBearing,
                        confidence: viewModel.bearingConfidence
                    )
                    .padding(.bottom, 12)
                }

                // Threat classification overlay
                if let classification = viewModel.latestClassification {
                    ThreatClassificationView(classification: classification)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 8)
                }

                // Bottom panel: log toggle + servo status
                BottomPanelView(
                    servoController: appState.servoController,
                    showLog: $showLog
                )
                .padding(.bottom, 34)
            }

            // Initialization overlay
            if !appState.isInitialized {
                InitializationOverlayView(error: appState.initializationError)
            }
        }
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showLog) {
            DetectionLogView()
                .environmentObject(appState)
        }
        .sheet(isPresented: $showSettings) {
            SettingsView()
                .environmentObject(appState)
        }
        .onAppear {
            viewModel.bind(to: appState)
        }
    }
}

// MARK: - ContentViewModel

@MainActor
final class ContentViewModel: ObservableObject {
    @Published var currentMode: SkyMode = .sleep
    @Published var currentDetections: [DetectedObject] = []
    @Published var currentTrackState: TrackState?
    @Published var currentBearing: Float = 0
    @Published var bearingConfidence: Float = 0
    @Published var latestClassification: ThreatClassification?
    @Published var connectedNodes: Int = 0
    @Published var recentEvents: [DetectionEvent] = []

    private var appState: AppState?

    func bind(to state: AppState) {
        appState = state

        // Subscribe to state machine mode changes
        Task {
            for await mode in state.stateMachine.modePublisher.values {
                currentMode = mode
            }
        }

        // Subscribe to object detections
        state.objectDetector.detectionHandler = { [weak self] detections, _, _ in
            Task { @MainActor in
                self?.currentDetections = detections
            }
        }

        // Subscribe to tracker updates
        state.tracker.trackingHandler = { [weak self] trackState in
            Task { @MainActor in
                self?.currentTrackState = trackState
                self?.currentBearing = trackState.bearing
            }
        }

        // Subscribe to direction estimator
        state.directionEst.updateHandler = { [weak self] estimate in
            Task { @MainActor in
                self?.currentBearing = estimate.bearingDegrees
                self?.bearingConfidence = estimate.confidence
            }
        }

        // Subscribe to deep classification
        state.deepClassifier.classificationHandler = { [weak self] _, event in
            Task { @MainActor in
                self?.latestClassification = event.classification
                self?.recentEvents.insert(event, at: 0)
                if self?.recentEvents.count ?? 0 > 100 {
                    self?.recentEvents.removeLast()
                }
            }
        }

        // Subscribe to mesh peers
        state.meshNetwork.peerStatusHandler = { [weak self] peers in
            Task { @MainActor in
                self?.connectedNodes = peers.count
            }
        }
    }
}

// MARK: - CameraPreviewView

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}
}

final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

// MARK: - DetectionOverlayView

struct DetectionOverlayView: View {
    let detections: [DetectedObject]
    let trackState: TrackState?

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // YOLO detection boxes
                ForEach(detections) { detection in
                    DetectionBoxView(detection: detection, viewSize: geo.size)
                }

                // Active track highlight
                if let track = trackState, track.isLocked {
                    TrackingReticleView(track: track, viewSize: geo.size)
                }
            }
        }
    }
}

// MARK: - Detection Box

struct DetectionBoxView: View {
    let detection: DetectedObject
    let viewSize: CGSize

    var boxRect: CGRect {
        // Vision coordinates: origin bottom-left, flip Y
        CGRect(
            x: detection.boundingBox.origin.x * viewSize.width,
            y: (1 - detection.boundingBox.origin.y - detection.boundingBox.height) * viewSize.height,
            width: detection.boundingBox.width * viewSize.width,
            height: detection.boundingBox.height * viewSize.height
        )
    }

    var body: some View {
        let rect = boxRect
        let color = Color(uiColor: detection.label.color)

        ZStack(alignment: .topLeading) {
            Rectangle()
                .stroke(color, lineWidth: 2)
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)

            // Label
            Text("\(detection.label.rawValue) \(String(format: "%.0f%%", detection.confidence * 100))")
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundColor(.white)
                .padding(.horizontal, 4)
                .padding(.vertical, 2)
                .background(color.opacity(0.85))
                .cornerRadius(3)
                .position(x: rect.minX + rect.width / 2, y: rect.minY - 10)
        }
    }
}

// MARK: - Tracking Reticle

struct TrackingReticleView: View {
    let track: TrackState
    let viewSize: CGSize

    var centerX: CGFloat { track.boundingBox.midX * viewSize.width }
    var centerY: CGFloat { (1 - track.boundingBox.midY) * viewSize.height }
    var size: CGFloat { max(track.boundingBox.width, track.boundingBox.height) * viewSize.width * 1.5 }

    var body: some View {
        ZStack {
            // Crosshair
            Rectangle()
                .fill(Color.red)
                .frame(width: size, height: 1.5)
            Rectangle()
                .fill(Color.red)
                .frame(width: 1.5, height: size)

            // Corner brackets
            ForEach(0..<4) { corner in
                CornerBracket(corner: corner, size: size * 0.4)
                    .stroke(Color.red, lineWidth: 2)
            }

            // Predicted position
            if track.confidence > 0.5 {
                Circle()
                    .stroke(Color.orange.opacity(0.6), lineWidth: 1.5)
                    .frame(width: size * 0.3, height: size * 0.3)
                    .offset(
                        x: (track.predictedBoundingBox.midX - track.boundingBox.midX) * viewSize.width,
                        y: (track.boundingBox.midY - track.predictedBoundingBox.midY) * viewSize.height
                    )
            }
        }
        .position(x: centerX, y: centerY)
    }
}

struct CornerBracket: Shape {
    let corner: Int  // 0=TL, 1=TR, 2=BL, 3=BR
    let size: CGFloat

    func path(in rect: CGRect) -> Path {
        var p = Path()
        let len = size * 0.25
        let half = size / 2

        let origin: CGPoint
        let h: CGFloat  // horizontal direction
        let v: CGFloat  // vertical direction

        switch corner {
        case 0: origin = CGPoint(x: -half, y: -half); h =  1; v =  1
        case 1: origin = CGPoint(x:  half, y: -half); h = -1; v =  1
        case 2: origin = CGPoint(x: -half, y:  half); h =  1; v = -1
        default: origin = CGPoint(x:  half, y:  half); h = -1; v = -1
        }

        p.move(to: CGPoint(x: origin.x + h * len, y: origin.y))
        p.addLine(to: origin)
        p.addLine(to: CGPoint(x: origin.x, y: origin.y + v * len))
        return p
    }
}

// MARK: - Top Bar

struct TopBarView: View {
    let mode: SkyMode
    let nodeCount: Int
    @Binding var showSettings: Bool

    var modeColor: Color {
        switch mode {
        case .sleep:    return .gray
        case .alert:    return .yellow
        case .track:    return .orange
        case .document: return .red
        case .patrol:   return .cyan
        }
    }

    var body: some View {
        HStack {
            // Mode indicator
            HStack(spacing: 6) {
                Circle()
                    .fill(modeColor)
                    .frame(width: 10, height: 10)
                    .overlay(
                        Circle()
                            .stroke(modeColor.opacity(0.4), lineWidth: 3)
                            .scaleEffect(mode == .track || mode == .document ? 1.8 : 1.0)
                            .opacity(mode == .track || mode == .document ? 0 : 1)
                    )
                    .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true),
                               value: mode == .track)

                Text(mode.rawValue)
                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                    .foregroundColor(modeColor)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.black.opacity(0.7))
            .cornerRadius(8)

            Spacer()

            // SKYWALL logo
            Text("SKYWALL")
                .font(.system(size: 18, weight: .black, design: .monospaced))
                .foregroundColor(.white)
                .shadow(color: .blue, radius: 4)

            Spacer()

            // Node count + settings
            HStack(spacing: 10) {
                HStack(spacing: 4) {
                    Image(systemName: "antenna.radiowaves.left.and.right")
                        .font(.system(size: 12))
                    Text("\(nodeCount)")
                        .font(.system(size: 13, weight: .semibold, design: .monospaced))
                }
                .foregroundColor(nodeCount > 0 ? .green : .gray)
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(Color.black.opacity(0.7))
                .cornerRadius(8)

                Button {
                    showSettings = true
                } label: {
                    Image(systemName: "gearshape.fill")
                        .font(.system(size: 18))
                        .foregroundColor(.white)
                        .padding(8)
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(8)
                }
            }
        }
        .padding(.horizontal, 16)
    }
}

// MARK: - Bearing Compass

struct BearingCompassView: View {
    let bearing: Float
    let confidence: Float

    var body: some View {
        VStack(spacing: 4) {
            ZStack {
                Circle()
                    .stroke(Color.white.opacity(0.3), lineWidth: 1.5)
                    .frame(width: 90, height: 90)

                // Cardinal marks
                ForEach(["N","E","S","W"].indices, id: \.self) { i in
                    let angle = Double(i) * 90.0
                    Text(["N","E","S","W"][i])
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundColor(.white.opacity(0.6))
                        .offset(y: -38)
                        .rotationEffect(.degrees(angle))
                        .rotationEffect(.degrees(-angle))
                }

                // Bearing arrow
                Arrow()
                    .fill(Color.red)
                    .frame(width: 6, height: 35)
                    .offset(y: -17)
                    .rotationEffect(.degrees(Double(bearing)))

                // Center dot
                Circle()
                    .fill(Color.white)
                    .frame(width: 5, height: 5)

                // Confidence ring
                Circle()
                    .trim(from: 0, to: CGFloat(confidence))
                    .stroke(Color.green, lineWidth: 3)
                    .frame(width: 80, height: 80)
                    .rotationEffect(.degrees(-90))
            }

            Text(String(format: "%.0f°  %.0f%%", bearing, confidence * 100))
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundColor(.white)
        }
        .padding(12)
        .background(Color.black.opacity(0.6))
        .cornerRadius(12)
    }
}

struct Arrow: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: rect.midX, y: rect.minY))
        p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.midX, y: rect.maxY * 0.6))
        p.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
        p.closeSubpath()
        return p
    }
}

// MARK: - Threat Classification Overlay

struct ThreatClassificationView: View {
    let classification: ThreatClassification

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    ThreatBadge(level: classification.threatLevel)
                    Text(classification.droneClass.rawValue.uppercased())
                        .font(.system(size: 14, weight: .bold, design: .monospaced))
                        .foregroundColor(.white)
                }

                HStack(spacing: 12) {
                    Label(String(format: "%.0f°", classification.bearing), systemImage: "location.north")
                    Label(String(format: "%.0f°", classification.elevation), systemImage: "arrow.up.right")
                    if let range = classification.estimatedRange {
                        Label(String(format: "%.0fm", range), systemImage: "ruler")
                    }
                }
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.white.opacity(0.8))

                Text(classification.behavior.rawValue.capitalized)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.orange)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text(String(format: "%.0f%%", classification.confidence * 100))
                    .font(.system(size: 22, weight: .black, design: .monospaced))
                    .foregroundColor(.white)
                Text("CONF")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.white.opacity(0.6))
            }
        }
        .padding(12)
        .background(Color.black.opacity(0.75))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(classification.threatLevel == .high ? Color.red : Color.orange, lineWidth: 1.5)
        )
        .cornerRadius(10)
    }
}

struct ThreatBadge: View {
    let level: ThreatLevel

    var color: Color {
        switch level {
        case .none:   return .gray
        case .low:    return .yellow
        case .medium: return .orange
        case .high:   return .red
        }
    }

    var body: some View {
        Text(level.displayName)
            .font(.system(size: 9, weight: .black, design: .monospaced))
            .foregroundColor(.white)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(color)
            .cornerRadius(4)
    }
}

// MARK: - Bottom Panel

struct BottomPanelView: View {
    @ObservedObject var servoController: SkyServoController
    @Binding var showLog: Bool

    var body: some View {
        HStack {
            // Servo status
            HStack(spacing: 6) {
                Image(systemName: "dot.radiowaves.left.and.right")
                    .foregroundColor(servoController.isConnected ? .green : .gray)
                VStack(alignment: .leading, spacing: 1) {
                    Text(servoController.connectionState.displayString)
                        .font(.system(size: 10, design: .monospaced))
                    if servoController.isConnected {
                        Text(String(format: "P%.0f° T%.0f°", servoController.currentPan, servoController.currentTilt))
                            .font(.system(size: 10, design: .monospaced))
                    }
                }
                .foregroundColor(.white.opacity(0.8))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color.black.opacity(0.6))
            .cornerRadius(8)

            Spacer()

            // Log button
            Button {
                showLog = true
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "list.bullet.rectangle")
                    Text("LOG")
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                }
                .foregroundColor(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.6))
                .cornerRadius(8)
            }
        }
        .padding(.horizontal, 16)
    }
}

// MARK: - Detection Log View

struct DetectionLogView: View {
    @EnvironmentObject var appState: AppState
    @State private var events: [DetectionEvent] = []
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            List {
                if events.isEmpty {
                    Text("No detections recorded.")
                        .foregroundColor(.secondary)
                        .italic()
                } else {
                    ForEach(events) { event in
                        DetectionLogRow(event: event)
                    }
                }
            }
            .navigationTitle("Detection Log")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .task {
            events = (try? await appState.vault.loadRecentEvents(limit: 100)) ?? []
        }
    }
}

struct DetectionLogRow: View {
    let event: DetectionEvent

    var body: some View {
        HStack {
            ThreatBadge(level: event.threatLevel)
            VStack(alignment: .leading, spacing: 2) {
                Text(event.classification.droneClass.rawValue)
                    .font(.system(size: 14, weight: .semibold, design: .monospaced))
                Text(event.timestamp.formatted(date: .abbreviated, time: .shortened))
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text(String(format: "%.0f°", event.bearing))
                    .font(.system(size: 13, design: .monospaced))
                Text(String(format: "%.0f%%", event.confidence * 100))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) var dismiss
    @State private var retentionDays = 30
    @State private var servoEnabled = true

    let retentionOptions = [7, 30, 90]

    var body: some View {
        NavigationView {
            Form {
                Section("System") {
                    HStack {
                        Text("Node ID")
                        Spacer()
                        Text(UIDevice.current.identifierForVendor?.uuidString.prefix(8) ?? "N/A")
                            .foregroundColor(.secondary)
                            .font(.system(.caption, design: .monospaced))
                    }
                    HStack {
                        Text("Storage Used")
                        Spacer()
                        Text(formatBytes(appState.vault.databaseSizeBytes()))
                            .foregroundColor(.secondary)
                    }
                }

                Section("Detection") {
                    Picker("Data Retention", selection: $retentionDays) {
                        ForEach(retentionOptions, id: \.self) { days in
                            Text("\(days) days").tag(days)
                        }
                    }
                }

                Section("PTZ Mount") {
                    Toggle("Enable Servo Control", isOn: $servoEnabled)
                    HStack {
                        Text("Connection")
                        Spacer()
                        Text(appState.servoController.connectionState.displayString)
                            .foregroundColor(appState.servoController.isConnected ? .green : .secondary)
                    }
                }

                Section("Mesh Network") {
                    HStack {
                        Text("Connected Nodes")
                        Spacer()
                        Text("\(appState.meshNetwork.connectedPeers.count)")
                    }
                }

                Section("Danger Zone") {
                    Button("Purge All Events", role: .destructive) {
                        Task {
                            await appState.vault.purgeExpiredRecords()
                        }
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .onAppear {
            retentionDays = 30
        }
        .onChange(of: retentionDays) { _, newValue in
            let policy = RetentionPolicy(rawValue: newValue) ?? .thirtyDays
            Task { await appState.vault.setRetentionPolicy(policy) }
        }
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}

// MARK: - Initialization Overlay

struct InitializationOverlayView: View {
    let error: String?

    var body: some View {
        ZStack {
            Color.black.opacity(0.85).ignoresSafeArea()

            VStack(spacing: 24) {
                Image(systemName: "shield.lefthalf.filled")
                    .font(.system(size: 64))
                    .foregroundColor(.blue)

                Text("SKYWALL")
                    .font(.system(size: 36, weight: .black, design: .monospaced))
                    .foregroundColor(.white)

                if let error {
                    VStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                            .font(.title)
                        Text(error)
                            .font(.callout)
                            .foregroundColor(.red)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                    }
                } else {
                    VStack(spacing: 10) {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .tint(.blue)
                        Text("Initializing systems...")
                            .font(.system(size: 14, design: .monospaced))
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
            }
        }
    }
}
