// SkyServoController.swift
// SKYWALL - Autonomous Aerial Detection System
// BLE GATT servo controller for ESP32 pan/tilt mount.

import CoreBluetooth
import Foundation

// MARK: - BLE UUIDs (must match ESP32 firmware)

enum SkyBLE {
    static let serviceUUID        = CBUUID(string: "4FAFC201-1FB5-459E-8FCC-C5C9C331914B")
    static let commandCharUUID    = CBUUID(string: "BEB5483E-36E1-4688-B7F5-EA07361B26A8")
    static let statusCharUUID     = CBUUID(string: "BEB5483F-36E1-4688-B7F5-EA07361B26A8")
    static let deviceName         = "SKYWALL_PTZ"
}

// MARK: - Servo Mode

enum ServoMode {
    case patrol   // Autonomous 360° sweep
    case point    // Point to specific angle and hold
    case track    // Follow tracking commands
    case preset   // Named presets (home, park, etc.)
}

// MARK: - Servo Command

struct ServoCommand {
    var pan: Float    // 0-360°
    var tilt: Float   // -35° to +90°
    var speed: Int    // 0-100 (% of max speed)
    var mode: ServoMode

    func toProtocolString() -> String {
        let panInt  = max(0,   min(360, Int(pan.rounded())))
        let tiltInt = max(-35, min(90,  Int(tilt.rounded())))
        return "P\(panInt)T\(tiltInt)\n"
    }
}

// MARK: - SkyServoController

final class SkyServoController: NSObject, ObservableObject {

    // MARK: - BLE Stack

    private var centralManager: CBCentralManager?
    private var peripheral: CBPeripheral?
    private var commandCharacteristic: CBCharacteristic?
    private var statusCharacteristic: CBCharacteristic?

    // MARK: - State

    @Published var isConnected = false
    @Published var connectionState: ConnectionState = .disconnected
    @Published var currentPan: Float = 0
    @Published var currentTilt: Float = 0
    @Published var batteryVoltage: Float = 0

    // MARK: - Timing

    private var lastCommandTime: Date = .distantPast
    private let minCommandInterval: TimeInterval = 1.0 / 30.0  // 30 Hz max

    // Reconnection
    private var reconnectTimer: Timer?
    private var reconnectAttempts = 0
    private let maxReconnectAttempts = 10
    private let reconnectDelay: TimeInterval = 3.0

    // Scan timeout
    private var scanTimeoutTimer: Timer?
    private let scanTimeout: TimeInterval = 10.0

    // PID controllers
    private var panPID  = PIDController(kp: 1.2, ki: 0.05, kd: 0.2)
    private var tiltPID = PIDController(kp: 1.2, ki: 0.05, kd: 0.2)

    // Current setpoints
    private var targetPan: Float = 0
    private var targetTilt: Float = 0

    // Command queue (thread-safe)
    private let commandQueue = DispatchQueue(label: "com.skywall.ble.commands", qos: .userInteractive)
    private var pendingCommand: ServoCommand?

    // MARK: - Initialization

    func initialize() {
        centralManager = CBCentralManager(delegate: self, queue: commandQueue, options: [
            CBCentralManagerOptionShowPowerAlertKey: true
        ])
        print("[ServoController] BLE initialized.")
    }

    // MARK: - Send Commands

    func sendTrackingCommand(pan: Float, tilt: Float) {
        targetPan  = pan
        targetTilt = tilt

        let command = ServoCommand(pan: pan, tilt: tilt, speed: 80, mode: .track)
        enqueueCommand(command)
    }

    func sendMode(_ mode: ServoMode) {
        switch mode {
        case .patrol:
            let command = ServoCommand(pan: 0, tilt: 0, speed: 30, mode: .patrol)
            enqueueCommand(command)
        case .point:
            let command = ServoCommand(pan: targetPan, tilt: targetTilt, speed: 60, mode: .point)
            enqueueCommand(command)
        case .track:
            let command = ServoCommand(pan: targetPan, tilt: targetTilt, speed: 80, mode: .track)
            enqueueCommand(command)
        case .preset:
            // Home position
            let command = ServoCommand(pan: 0, tilt: 0, speed: 50, mode: .preset)
            enqueueCommand(command)
        }
    }

    func pointTo(pan: Float, tilt: Float) {
        targetPan  = pan.truncatingRemainder(dividingBy: 360)
        targetTilt = max(-35, min(90, tilt))
        let command = ServoCommand(pan: targetPan, tilt: targetTilt, speed: 60, mode: .point)
        enqueueCommand(command)
    }

    // MARK: - Command Queue

    private func enqueueCommand(_ command: ServoCommand) {
        commandQueue.async { [weak self] in
            guard let self else { return }
            let now = Date()
            guard now.timeIntervalSince(self.lastCommandTime) >= self.minCommandInterval else {
                self.pendingCommand = command
                return
            }
            self.lastCommandTime = now
            self.sendCommandNow(command)
        }
    }

    private func sendCommandNow(_ command: ServoCommand) {
        guard let char = commandCharacteristic,
              let peripheral = peripheral,
              peripheral.state == .connected else {
            pendingCommand = command
            return
        }

        let str = command.toProtocolString()
        guard let data = str.data(using: .utf8) else { return }

        peripheral.writeValue(data, for: char, type: .withResponse)
        currentPan  = command.pan
        currentTilt = command.tilt
    }

    // MARK: - Scan for Device

    private func startScan() {
        guard let cm = centralManager, cm.state == .poweredOn else { return }
        print("[ServoController] Scanning for \(SkyBLE.deviceName)...")
        connectionState = .scanning

        cm.scanForPeripherals(
            withServices: [SkyBLE.serviceUUID],
            options: [CBCentralManagerScanOptionAllowDuplicatesKey: false]
        )

        // Timeout
        scanTimeoutTimer?.invalidate()
        scanTimeoutTimer = Timer.scheduledTimer(withTimeInterval: scanTimeout, repeats: false) { [weak self] _ in
            self?.handleScanTimeout()
        }
    }

    private func handleScanTimeout() {
        centralManager?.stopScan()
        print("[ServoController] Scan timeout.")
        scheduleReconnect()
    }

    // MARK: - Reconnection

    private func scheduleReconnect() {
        guard reconnectAttempts < maxReconnectAttempts else {
            print("[ServoController] Max reconnect attempts reached. Giving up.")
            connectionState = .failed
            return
        }
        reconnectAttempts += 1
        print("[ServoController] Reconnect attempt \(reconnectAttempts)/\(maxReconnectAttempts) in \(reconnectDelay)s")

        DispatchQueue.main.asyncAfter(deadline: .now() + reconnectDelay) { [weak self] in
            self?.startScan()
        }
    }

    private func resetReconnectCounter() {
        reconnectAttempts = 0
        reconnectTimer?.invalidate()
        reconnectTimer = nil
    }
}

// MARK: - CBCentralManagerDelegate

extension SkyServoController: CBCentralManagerDelegate {

    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            print("[ServoController] BLE powered on. Starting scan.")
            startScan()
        case .poweredOff:
            print("[ServoController] BLE powered off.")
            DispatchQueue.main.async { self.isConnected = false; self.connectionState = .disconnected }
        case .unauthorized:
            print("[ServoController] BLE unauthorized.")
            DispatchQueue.main.async { self.connectionState = .unauthorized }
        case .unsupported:
            print("[ServoController] BLE unsupported on this device.")
        default:
            break
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String: Any], rssi RSSI: NSNumber) {
        let name = peripheral.name ?? advertisementData[CBAdvertisementDataLocalNameKey] as? String ?? ""
        guard name == SkyBLE.deviceName else { return }

        print("[ServoController] Found \(name) RSSI=\(RSSI)")
        central.stopScan()
        scanTimeoutTimer?.invalidate()

        self.peripheral = peripheral
        peripheral.delegate = self
        DispatchQueue.main.async { self.connectionState = .connecting }
        central.connect(peripheral, options: nil)
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        print("[ServoController] Connected to \(peripheral.name ?? "device")")
        resetReconnectCounter()
        DispatchQueue.main.async {
            self.isConnected = true
            self.connectionState = .connected
        }
        peripheral.discoverServices([SkyBLE.serviceUUID])
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        print("[ServoController] Disconnected: \(error?.localizedDescription ?? "no error")")
        DispatchQueue.main.async {
            self.isConnected = false
            self.connectionState = .disconnected
            self.commandCharacteristic = nil
            self.statusCharacteristic = nil
        }
        scheduleReconnect()
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        print("[ServoController] Failed to connect: \(error?.localizedDescription ?? "")")
        scheduleReconnect()
    }
}

// MARK: - CBPeripheralDelegate

extension SkyServoController: CBPeripheralDelegate {

    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard error == nil else { print("[ServoController] Service discovery error: \(error!)"); return }
        guard let services = peripheral.services else { return }

        for service in services where service.uuid == SkyBLE.serviceUUID {
            peripheral.discoverCharacteristics(
                [SkyBLE.commandCharUUID, SkyBLE.statusCharUUID],
                for: service
            )
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard error == nil else { print("[ServoController] Char discovery error: \(error!)"); return }
        guard let chars = service.characteristics else { return }

        for char in chars {
            switch char.uuid {
            case SkyBLE.commandCharUUID:
                commandCharacteristic = char
                print("[ServoController] Command characteristic found.")
            case SkyBLE.statusCharUUID:
                statusCharacteristic = char
                peripheral.setNotifyValue(true, for: char)
                print("[ServoController] Status characteristic found, notifications enabled.")
            default:
                break
            }
        }

        // Send any pending command now that we're connected
        if let pending = pendingCommand {
            sendCommandNow(pending)
            pendingCommand = nil
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard error == nil, characteristic.uuid == SkyBLE.statusCharUUID,
              let data = characteristic.value,
              let str = String(data: data, encoding: .utf8) else { return }

        // Parse status: "S{pan},{tilt},{battery}\n"
        parseStatusString(str)
    }

    func peripheral(_ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?) {
        if let error {
            print("[ServoController] Write error: \(error)")
        }
    }

    private func parseStatusString(_ str: String) {
        // Format: "S{pan},{tilt},{voltage}\n"
        let trimmed = str.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("S") else { return }

        let parts = trimmed.dropFirst().split(separator: ",")
        guard parts.count >= 3 else { return }

        if let pan = Float(parts[0]), let tilt = Float(parts[1]), let voltage = Float(parts[2]) {
            DispatchQueue.main.async { [weak self] in
                self?.currentPan = pan
                self?.currentTilt = tilt
                self?.batteryVoltage = voltage
            }
        }
    }
}

// MARK: - Connection State

enum ConnectionState {
    case disconnected
    case scanning
    case connecting
    case connected
    case failed
    case unauthorized

    var displayString: String {
        switch self {
        case .disconnected:  return "Disconnected"
        case .scanning:      return "Scanning..."
        case .connecting:    return "Connecting..."
        case .connected:     return "Connected"
        case .failed:        return "Connection Failed"
        case .unauthorized:  return "BLE Unauthorized"
        }
    }
}
