// SkyDirectionEstimator.swift
// SKYWALL - Autonomous Aerial Detection System
// TDOA direction estimation: cross-correlation + delay-and-sum beamforming.

import Foundation
import Accelerate

// MARK: - Direction Estimate

struct DirectionEstimate {
    let bearingDegrees: Float    // 0-360, north=0
    let confidence: Float        // 0-1
    let signalStrength: Float    // dBFS approx
    let timestamp: Date
    let tdoaSamples: (Float, Float, Float)  // TDOA between mic pairs (0-1, 0-2, 1-2)
}

// MARK: - Microphone Geometry

struct MicrophoneArray {
    // Physical positions relative to device center (meters)
    // Approximate positions for iPhone Pro (bottom, front, rear)
    static let positions: [(x: Float, y: Float, z: Float)] = [
        (x:  0.000, y: -0.075, z:  0.000),   // Bottom mic
        (x:  0.000, y:  0.075, z:  0.005),   // Front mic (earpiece area)
        (x:  0.000, y:  0.000, z: -0.005),   // Rear mic
    ]
    static let speedOfSound: Float = 343.0 // m/s at 20°C
    static let sampleRate: Float = Float(MelSpectrogramParameters.sampleRate)
}

// MARK: - SkyDirectionEstimator

final class SkyDirectionEstimator {

    // MARK: - Properties

    private(set) var latestEstimate: DirectionEstimate?
    private var lastProcessTime: Date = .distantPast
    private let updateInterval: TimeInterval = 0.500  // 500ms

    // Beamforming scan resolution
    private let numAzimuths = 72     // 5° steps, 360/5 = 72
    private let numElevations = 19  // -45° to +45° in 5° steps

    // Cross-correlation workspace
    private let maxLag: Int = 128  // max samples lag at 48kHz = 2.7ms ~ 93cm

    var updateHandler: ((DirectionEstimate) -> Void)?

    // MARK: - Initialize

    func initialize() {
        print("[DirectionEst] Initialized. Mic array: \(MicrophoneArray.positions.count) mics")
        precomputeSteeringDelays()
    }

    // Precomputed steering delays for each azimuth/elevation
    private var steeringDelays: [[[Float]]] = []  // [azimuth][elevation][micIndex]

    private func precomputeSteeringDelays() {
        let sr = MicrophoneArray.sampleRate
        let c = MicrophoneArray.speedOfSound

        steeringDelays = (0..<numAzimuths).map { azIdx in
            let azDeg = Float(azIdx) * (360.0 / Float(numAzimuths))
            let azRad = azDeg * .pi / 180.0

            return (0..<numElevations).map { elIdx in
                let elDeg = -45.0 + Float(elIdx) * (90.0 / Float(numElevations - 1))
                let elRad = elDeg * .pi / 180.0

                // Direction vector
                let dx = cos(elRad) * sin(azRad)
                let dy = cos(elRad) * cos(azRad)
                let dz = sin(elRad)

                // Delay for each mic relative to array center
                return MicrophoneArray.positions.map { pos in
                    let dot = pos.x * dx + pos.y * dy + pos.z * dz
                    return dot / c * sr  // delay in samples
                }
            }
        }
    }

    // MARK: - Process Channel Buffers

    func process(channelBuffers: [[Float]], timestamp: Date) {
        guard timestamp.timeIntervalSince(lastProcessTime) >= updateInterval else { return }
        guard channelBuffers.count >= 2 else { return }

        lastProcessTime = timestamp

        let ch0 = channelBuffers[0]
        let ch1 = channelBuffers.count > 1 ? channelBuffers[1] : ch0
        let ch2 = channelBuffers.count > 2 ? channelBuffers[2] : ch0

        // Compute signal energy for SNR gating
        let energy0 = rmsEnergy(ch0)
        guard energy0 > 1e-6 else {
            // Too quiet, no reliable direction
            return
        }

        // TDOA via GCC-PHAT cross-correlation
        let tdoa01 = gccPhat(x: ch0, y: ch1)
        let tdoa02 = gccPhat(x: ch0, y: ch2)
        let tdoa12 = gccPhat(x: ch1, y: ch2)

        // Beamforming: find azimuth with maximum power
        let (bestAz, bestEl, beamPower) = delayAndSumBeamform(ch0: ch0, ch1: ch1, ch2: ch2)

        // Convert to bearing
        let bearing = Float(bestAz) * (360.0 / Float(numAzimuths))
        let elevation = -45.0 + Float(bestEl) * (90.0 / Float(numElevations - 1))

        // Confidence from beam power normalized
        let confidence = min(beamPower / (energy0 * 3.0 + 1e-10), 1.0)

        // Signal strength in dBFS
        let dBFS = 20.0 * log10(max(energy0, 1e-10))

        let estimate = DirectionEstimate(
            bearingDegrees: bearing,
            confidence: confidence,
            signalStrength: dBFS,
            timestamp: timestamp,
            tdoaSamples: (tdoa01, tdoa02, tdoa12)
        )

        latestEstimate = estimate

        DispatchQueue.main.async { [weak self] in
            self?.updateHandler?(estimate)
        }
    }

    // MARK: - GCC-PHAT Cross-Correlation

    /// Generalized Cross-Correlation with Phase Transform.
    /// Returns TDOA in samples (positive = signal arrives at y before x).
    private func gccPhat(x: [Float], y: [Float]) -> Float {
        let n = min(x.count, y.count)
        guard n > maxLag * 2 else { return 0 }

        // Use a window to reduce edge effects
        let windowSize = min(n, 4096)
        let xWin = Array(x.prefix(windowSize))
        let yWin = Array(y.prefix(windowSize))

        // Compute FFT of both
        let fftN = nextPowerOf2(windowSize)
        let xPadded = xWin + [Float](repeating: 0, count: fftN - windowSize)
        let yPadded = yWin + [Float](repeating: 0, count: fftN - windowSize)

        let xFFT = computeFFT(xPadded, n: fftN)
        let yFFT = computeFFT(yPadded, n: fftN)

        // GCC-PHAT: normalize cross-spectrum by magnitude
        let halfN = fftN / 2 + 1
        var crossSpectrum = [Float](repeating: 0, count: halfN * 2) // interleaved re/im

        for k in 0..<halfN {
            let xr = xFFT[2 * k]
            let xi = xFFT[2 * k + 1]
            let yr = yFFT[2 * k]
            let yi = yFFT[2 * k + 1]

            // Cross-power: X * conj(Y)
            let cr = xr * yr + xi * yi
            let ci = xi * yr - xr * yi

            // PHAT weighting: normalize by magnitude
            let mag = sqrt(cr * cr + ci * ci) + 1e-10
            crossSpectrum[2 * k]     = cr / mag
            crossSpectrum[2 * k + 1] = ci / mag
        }

        // IFFT to get correlation
        let corr = computeIFFT(crossSpectrum, n: fftN)

        // Find peak within maxLag range
        var bestLag = 0
        var bestVal: Float = -Float.infinity

        for lag in -maxLag...maxLag {
            let idx = lag >= 0 ? lag : fftN + lag
            if idx < corr.count && corr[idx] > bestVal {
                bestVal = corr[idx]
                bestLag = lag
            }
        }

        return Float(bestLag)
    }

    // MARK: - Delay-and-Sum Beamforming

    private func delayAndSumBeamform(ch0: [Float], ch1: [Float], ch2: [Float]) -> (azimuth: Int, elevation: Int, power: Float) {
        let channels = [ch0, ch1, ch2]
        let len = channels.map { $0.count }.min() ?? 0
        guard len > 0 else { return (0, 0, 0) }

        var bestPower: Float = -Float.infinity
        var bestAz = 0
        var bestEl = 0

        for azIdx in 0..<numAzimuths {
            for elIdx in 0..<numElevations {
                let delays = steeringDelays[azIdx][elIdx]
                var summedPower: Float = 0

                // Sum power of aligned signals
                let evalLen = min(len, 1024)
                var beam = [Float](repeating: 0, count: evalLen)

                for (micIdx, channel) in channels.enumerated() {
                    let delayInt = Int(delays[micIdx])
                    for i in 0..<evalLen {
                        let srcIdx = i + delayInt
                        if srcIdx >= 0 && srcIdx < channel.count {
                            beam[i] += channel[srcIdx]
                        }
                    }
                }

                // Compute power of beamformed output
                vDSP_svesq(beam, 1, &summedPower, vDSP_Length(evalLen))

                if summedPower > bestPower {
                    bestPower = summedPower
                    bestAz = azIdx
                    bestEl = elIdx
                }
            }
        }

        return (bestAz, bestEl, bestPower)
    }

    // MARK: - FFT Helpers

    private func computeFFT(_ input: [Float], n: Int) -> [Float] {
        let log2n = vDSP_Length(log2(Float(n)))
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return [] }
        defer { vDSP_destroy_fftsetup(setup) }

        var realp = [Float](repeating: 0, count: n / 2)
        var imagp = [Float](repeating: 0, count: n / 2)
        var output = DSPSplitComplex(realp: &realp, imagp: &imagp)

        input.withUnsafeBytes { rawPtr in
            rawPtr.bindMemory(to: DSPComplex.self).baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: n / 2) { ptr in
                vDSP_ctoz(ptr, 2, &output, 1, vDSP_Length(n / 2))
            }
        }

        vDSP_fft_zip(setup, &output, 1, log2n, FFTDirection(FFT_FORWARD))

        var result = [Float](repeating: 0, count: n)
        for i in 0..<n/2 {
            result[2 * i]     = realp[i]
            result[2 * i + 1] = imagp[i]
        }
        return result
    }

    private func computeIFFT(_ input: [Float], n: Int) -> [Float] {
        let log2n = vDSP_Length(log2(Float(n)))
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return [] }
        defer { vDSP_destroy_fftsetup(setup) }

        var realp = [Float](repeating: 0, count: n / 2)
        var imagp = [Float](repeating: 0, count: n / 2)

        for i in 0..<n/2 {
            realp[i] = input[2 * i]
            imagp[i] = input[2 * i + 1]
        }

        var split = DSPSplitComplex(realp: &realp, imagp: &imagp)
        vDSP_fft_zip(setup, &split, 1, log2n, FFTDirection(FFT_INVERSE))

        var scale: Float = 1.0 / Float(n)
        var result = [Float](repeating: 0, count: n)
        vDSP_ztoc(&split, 1, UnsafeMutablePointer<DSPComplex>(OpaquePointer(result.withUnsafeMutableBytes { $0.baseAddress! })), 2, vDSP_Length(n / 2))
        vDSP_vsmul(result, 1, &scale, &result, 1, vDSP_Length(n))
        return result
    }

    private func rmsEnergy(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        vDSP_svesq(samples, 1, &sum, vDSP_Length(samples.count))
        return sqrt(sum / Float(samples.count))
    }

    private func nextPowerOf2(_ n: Int) -> Int {
        var p = 1
        while p < n { p <<= 1 }
        return p
    }
}
