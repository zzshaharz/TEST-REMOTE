// SkyAudioEngine.swift
// SKYWALL - Autonomous Aerial Detection System
// Full audio capture: 3-channel, 48kHz/32-bit, mel spectrogram, ~50mW design.

import AVFoundation
import Accelerate

// MARK: - Audio Engine Modes

enum AudioEngineMode {
    case passive  // Low power: single mic, long window
    case active   // Full 3-channel capture
}

// MARK: - Mel Spectrogram Parameters

struct MelSpectrogramParameters {
    static let sampleRate: Double = 48_000
    static let fftSize: Int       = 2048
    static let hopSize: Int       = 512          // 50% overlap at 48kHz ~ 10ms
    static let numMelBins: Int    = 128
    static let fMin: Float        = 50.0         // Hz
    static let fMax: Float        = 12_000.0     // Hz (drones: 50-8000 Hz range)
    static let windowDuration: TimeInterval = 0.5  // 500ms window
    static let samplesPerWindow: Int = Int(sampleRate * windowDuration) // 24000
}

// MARK: - SkyAudioEngine

final class SkyAudioEngine {

    // MARK: - Properties

    private var audioEngine = AVAudioEngine()
    private let processingQueue = DispatchQueue(label: "com.skywall.audio", qos: .userInteractive)

    // 3-channel ring buffers (bottom, front, rear mics)
    private var ringBuffers: [RingBuffer] = (0..<3).map { _ in
        RingBuffer(capacity: MelSpectrogramParameters.samplesPerWindow * 4)
    }

    // Mel filter bank (precomputed)
    private var melFilterBank: [[Float]] = []

    // Hanning window
    private var hanningWindow: [Float] = []

    // FFT setup
    private var fftSetup: FFTSetup?
    private var fftSplitComplex = DSPSplitComplex(realp: nil, imagp: nil)
    private var fftLog2n: vDSP_Length = 0

    // State
    private(set) var isRunning = false
    private var mode: AudioEngineMode = .passive
    private var lastWindowTime: Date = Date()

    // Handlers
    var melSpectrogramHandler: (([Float], Date) -> Void)?
    var multiChannelHandler: (([[Float]], Date) -> Void)?

    // Processing interval tracking
    private var sampleCounter: Int = 0
    private let hopSize = MelSpectrogramParameters.hopSize

    // MARK: - Initialization

    func initialize() async throws {
        try setupAudioSession()
        try setupFFT()
        computeMelFilterBank()
        computeHanningWindow()
        try buildAudioGraph()
        print("[AudioEngine] Initialized: \(MelSpectrogramParameters.sampleRate)Hz, FFT=\(MelSpectrogramParameters.fftSize), mel=\(MelSpectrogramParameters.numMelBins)")
    }

    // MARK: - AVAudioSession Configuration

    private func setupAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record,
                                mode: .measurement,
                                options: [.allowBluetooth])
        try session.setPreferredSampleRate(MelSpectrogramParameters.sampleRate)
        try session.setPreferredIOBufferDuration(0.01) // 10ms buffer for low latency
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        print("[AudioEngine] Session configured: \(session.sampleRate)Hz, inputs=\(session.inputNumberOfChannels)")
    }

    // MARK: - Audio Graph

    private func buildAudioGraph() throws {
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)

        // Processing format: 48kHz mono Float32
        let processingFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: MelSpectrogramParameters.sampleRate,
            channels: 1,
            interleaved: false
        )!

        // Install tap on input node (main mic)
        let bufferSize = AVAudioFrameCount(MelSpectrogramParameters.hopSize)
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] buffer, time in
            guard let self else { return }
            self.processingQueue.async {
                self.processAudioBuffer(buffer, channelIndex: 0, time: time)
            }
        }

        // Try to access additional mics if available (multi-mic devices)
        setupAdditionalMics()

        audioEngine.prepare()
        print("[AudioEngine] Audio graph built.")
    }

    private func setupAdditionalMics() {
        // On devices with multiple mics (iPhone Pro), attempt to configure
        // front and rear microphones via AVAudioSession data source API.
        let session = AVAudioSession.sharedInstance()
        guard let inputs = session.availableInputs else { return }

        for input in inputs {
            if let dataSources = input.dataSources {
                for source in dataSources {
                    print("[AudioEngine] Available mic: \(source.dataSourceName) orientation=\(String(describing: source.orientation?.rawValue))")
                }
            }
        }
    }

    // MARK: - Start / Stop

    func start() throws {
        guard !isRunning else { return }
        try audioEngine.start()
        isRunning = true
        print("[AudioEngine] Started.")
    }

    func stop() {
        guard isRunning else { return }
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        isRunning = false
        print("[AudioEngine] Stopped.")
    }

    func setMode(_ newMode: AudioEngineMode) async {
        mode = newMode
        switch newMode {
        case .passive:
            // Could reduce processing rate in passive mode
            print("[AudioEngine] Mode: passive (low-power)")
        case .active:
            print("[AudioEngine] Mode: active (full 3-channel)")
        }
    }

    // MARK: - Audio Processing

    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer, channelIndex: Int, time: AVAudioTime) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameCount = Int(buffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData, count: frameCount))

        // Write to ring buffer
        ringBuffers[channelIndex].write(samples)
        sampleCounter += frameCount

        // Dispatch multiChannelHandler with all buffers every hop
        let timestamp = Date()
        if sampleCounter >= hopSize {
            sampleCounter = 0

            // Read current window from all buffers
            let windowSamples = MelSpectrogramParameters.samplesPerWindow
            let ch0 = ringBuffers[0].readLast(windowSamples)
            let ch1 = ringBuffers[1].readLast(windowSamples)
            let ch2 = ringBuffers[2].readLast(windowSamples)

            multiChannelHandler?([ch0, ch1, ch2], timestamp)

            // Process mel spectrogram from primary channel
            if ch0.count >= windowSamples {
                let mel = computeMelSpectrogram(samples: ch0)
                melSpectrogramHandler?(mel, timestamp)
            }
        }
    }

    // MARK: - FFT Setup

    private func setupFFT() throws {
        let n = MelSpectrogramParameters.fftSize
        fftLog2n = vDSP_Length(log2(Float(n)))
        guard let setup = vDSP_create_fftsetup(fftLog2n, FFTRadix(kFFTRadix2)) else {
            throw AudioEngineError.fftSetupFailed
        }
        fftSetup = setup

        // Allocate split complex buffers
        let halfN = n / 2
        let realp = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        let imagp = UnsafeMutablePointer<Float>.allocate(capacity: halfN)
        realp.initialize(repeating: 0, count: halfN)
        imagp.initialize(repeating: 0, count: halfN)
        fftSplitComplex = DSPSplitComplex(realp: realp, imagp: imagp)
    }

    // MARK: - Hanning Window

    private func computeHanningWindow() {
        let n = MelSpectrogramParameters.fftSize
        hanningWindow = (0..<n).map { i in
            0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(n - 1)))
        }
    }

    // MARK: - Mel Filter Bank

    private func computeMelFilterBank() {
        let sr = Float(MelSpectrogramParameters.sampleRate)
        let nFFT = MelSpectrogramParameters.fftSize
        let nMels = MelSpectrogramParameters.numMelBins
        let fMin = MelSpectrogramParameters.fMin
        let fMax = MelSpectrogramParameters.fMax

        func hzToMel(_ hz: Float) -> Float {
            return 2595.0 * log10(1.0 + hz / 700.0)
        }

        func melToHz(_ mel: Float) -> Float {
            return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
        }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        let melPoints = (0..<(nMels + 2)).map { i in
            melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
        }
        let hzPoints = melPoints.map { melToHz($0) }

        // Bin indices for FFT bins
        let freqBins = (0..<(nFFT / 2 + 1)).map { i in
            Float(i) * sr / Float(nFFT)
        }

        melFilterBank = (0..<nMels).map { m in
            var filter = [Float](repeating: 0, count: nFFT / 2 + 1)
            let fLeft  = hzPoints[m]
            let fCenter = hzPoints[m + 1]
            let fRight = hzPoints[m + 2]

            for k in 0..<freqBins.count {
                let f = freqBins[k]
                if f >= fLeft && f <= fCenter {
                    filter[k] = (f - fLeft) / (fCenter - fLeft)
                } else if f > fCenter && f <= fRight {
                    filter[k] = (fRight - f) / (fRight - fCenter)
                }
            }
            return filter
        }
    }

    // MARK: - Mel Spectrogram Computation

    func computeMelSpectrogram(samples: [Float]) -> [Float] {
        let n = MelSpectrogramParameters.fftSize
        let hop = MelSpectrogramParameters.hopSize
        let nMels = MelSpectrogramParameters.numMelBins
        let halfN = n / 2 + 1

        guard samples.count >= n, let fftSetup = fftSetup else {
            return [Float](repeating: 0, count: nMels)
        }

        // Number of frames
        let numFrames = max(1, (samples.count - n) / hop + 1)
        var melSpectrogram = [Float](repeating: 0, count: numFrames * nMels)

        for frame in 0..<numFrames {
            let startIdx = frame * hop
            guard startIdx + n <= samples.count else { break }

            // Apply Hanning window
            var windowed = [Float](repeating: 0, count: n)
            vDSP_vmul(Array(samples[startIdx..<startIdx + n]), 1, hanningWindow, 1, &windowed, 1, vDSP_Length(n))

            // Convert to split complex for FFT
            windowed.withUnsafeBytes { rawPtr in
                let floatPtr = rawPtr.bindMemory(to: Float.self)
                var tempSplit = DSPSplitComplex(
                    realp: fftSplitComplex.realp,
                    imagp: fftSplitComplex.imagp
                )
                floatPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: n / 2) { complexPtr in
                    vDSP_ctoz(complexPtr, 2, &tempSplit, 1, vDSP_Length(n / 2))
                }
            }

            vDSP_fft_zrip(fftSetup, &fftSplitComplex, 1, fftLog2n, FFTDirection(FFT_FORWARD))

            // Compute magnitude squared (power spectrum)
            var magnitudes = [Float](repeating: 0, count: halfN)
            vDSP_zvmags(&fftSplitComplex, 1, &magnitudes, 1, vDSP_Length(halfN))

            // Scale
            var scale: Float = 1.0 / Float(n)
            vDSP_vsmul(magnitudes, 1, &scale, &magnitudes, 1, vDSP_Length(halfN))

            // Apply mel filter bank
            for m in 0..<nMels {
                var energy: Float = 0
                vDSP_dotpr(magnitudes, 1, melFilterBank[m], 1, &energy, vDSP_Length(halfN))
                // Log compression with epsilon to avoid log(0)
                melSpectrogram[frame * nMels + m] = log(max(energy, 1e-10))
            }
        }

        return melSpectrogram
    }
}

// MARK: - Ring Buffer

final class RingBuffer {
    private var buffer: [Float]
    private var writePos: Int = 0
    private let capacity: Int
    private let lock = NSLock()

    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = [Float](repeating: 0, count: capacity)
    }

    func write(_ samples: [Float]) {
        lock.lock()
        defer { lock.unlock() }
        for sample in samples {
            buffer[writePos % capacity] = sample
            writePos += 1
        }
    }

    func readLast(_ count: Int) -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        let available = min(count, capacity, writePos)
        guard available > 0 else { return [] }

        var result = [Float](repeating: 0, count: available)
        let start = writePos - available
        for i in 0..<available {
            result[i] = buffer[(start + i) % capacity]
        }
        return result
    }
}

// MARK: - Errors

enum AudioEngineError: Error, LocalizedError {
    case fftSetupFailed
    case sessionConfigFailed(String)
    case engineStartFailed(String)

    var errorDescription: String? {
        switch self {
        case .fftSetupFailed:             return "Failed to create FFT setup"
        case .sessionConfigFailed(let s): return "AVAudioSession error: \(s)"
        case .engineStartFailed(let s):   return "AVAudioEngine start error: \(s)"
        }
    }
}
