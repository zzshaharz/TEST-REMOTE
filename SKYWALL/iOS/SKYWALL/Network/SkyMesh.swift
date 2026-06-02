// SkyMesh.swift
// SKYWALL - Autonomous Aerial Detection System
// Mesh network using MultipeerConnectivity with CryptoKit encryption.

import MultipeerConnectivity
import CryptoKit
import Foundation

// MARK: - Message Types

enum MeshMessageType: String, Codable {
    case alert       = "ALERT"
    case confirm     = "CONFIRM"
    case track       = "TRACK"
    case triangulate = "TRIANGULATE"
    case handoff     = "HANDOFF"
    case heartbeat   = "HEARTBEAT"
    case sync        = "SYNC"
    case ack         = "ACK"
}

// MARK: - Mesh Message

struct MeshMessage: Codable {
    let type: MeshMessageType
    let senderID: String
    let timestamp: Date
    let payload: Data          // Encrypted JSON payload
    let messageID: UUID
    let hopCount: Int
    let signature: Data        // HMAC-SHA256 of type+senderID+timestamp+payload
}

// MARK: - Alert Message

struct AlertMessage: Codable {
    let eventID: UUID
    let senderNodeID: String
    let timestamp: Date
    let bearing: Float
    let elevation: Float
    let confidence: Float
    let droneClass: String
    let location: NodeLocation?
    let tdoaSamples: [Float]?
    var meshConfirmed: Bool = false
}

// MARK: - TDOA Triangulation Data

struct TDOAData: Codable {
    let nodeID: String
    let timestamp: Date
    let bearing: Float
    let tdoaSamples: (Float, Float, Float)
    let location: NodeLocation

    enum CodingKeys: String, CodingKey {
        case nodeID, timestamp, bearing
        case tdoaSample01, tdoaSample02, tdoaSample12
        case location
    }

    init(nodeID: String, timestamp: Date, bearing: Float, tdoaSamples: (Float, Float, Float), location: NodeLocation) {
        self.nodeID = nodeID
        self.timestamp = timestamp
        self.bearing = bearing
        self.tdoaSamples = tdoaSamples
        self.location = location
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(nodeID, forKey: .nodeID)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(bearing, forKey: .bearing)
        try container.encode(tdoaSamples.0, forKey: .tdoaSample01)
        try container.encode(tdoaSamples.1, forKey: .tdoaSample02)
        try container.encode(tdoaSamples.2, forKey: .tdoaSample12)
        try container.encode(location, forKey: .location)
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        nodeID = try container.decode(String.self, forKey: .nodeID)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        bearing = try container.decode(Float.self, forKey: .bearing)
        location = try container.decode(NodeLocation.self, forKey: .location)
        let s01 = try container.decode(Float.self, forKey: .tdoaSample01)
        let s02 = try container.decode(Float.self, forKey: .tdoaSample02)
        let s12 = try container.decode(Float.self, forKey: .tdoaSample12)
        tdoaSamples = (s01, s02, s12)
    }
}

// MARK: - SkyMesh

final class SkyMesh: NSObject {

    // MARK: - Properties

    private var session: MCSession?
    private var advertiser: MCNearbyServiceAdvertiser?
    private var browser: MCNearbyServiceBrowser?
    private var myPeerID: MCPeerID!

    private let serviceType = "skywall-mesh"
    private let nodeID: String

    // Encryption
    private var sharedKey: SymmetricKey
    private let keyStorage = MeshKeyStorage()

    // Connected peers
    private(set) var connectedPeers: [MCPeerID] = []
    private(set) var peerAlerts: [String: AlertMessage] = [:]

    // Message deduplication
    private var receivedMessageIDs = Set<UUID>()
    private let maxMessageIDCache = 500

    // Callbacks
    var alertHandler: ((AlertMessage, String) -> Void)?
    var triangulationHandler: (([TDOAData]) -> Void)?
    var peerStatusHandler: (([MCPeerID]) -> Void)?

    // Pending TDOA data from peers
    private var pendingTDOA: [String: TDOAData] = [:]

    // Encoder/decoder
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        return e
    }()
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    // MARK: - Init

    override init() {
        nodeID = UIDevice.current.identifierForVendor?.uuidString ?? UUID().uuidString
        sharedKey = SymmetricKey(size: .bits256)  // Will be replaced by stored key
        super.init()
    }

    func initialize() async throws {
        // Load or generate pre-shared key
        sharedKey = try await keyStorage.loadOrGenerateKey()

        myPeerID = MCPeerID(displayName: "SKYWALL-\(nodeID.prefix(8))")

        let session = MCSession(peer: myPeerID, securityIdentity: nil, encryptionPreference: .required)
        session.delegate = self
        self.session = session

        let advertiser = MCNearbyServiceAdvertiser(peer: myPeerID, discoveryInfo: nil, serviceType: serviceType)
        advertiser.delegate = self
        self.advertiser = advertiser
        advertiser.startAdvertisingPeer()

        let browser = MCNearbyServiceBrowser(peer: myPeerID, serviceType: serviceType)
        browser.delegate = self
        self.browser = browser
        browser.startBrowsingForPeers()

        print("[SkyMesh] Initialized. Node: \(myPeerID.displayName)")
    }

    // MARK: - Broadcast

    func broadcast(event: DetectionEvent) async {
        guard let session = session, !session.connectedPeers.isEmpty else { return }

        let alert = AlertMessage(
            eventID: event.id,
            senderNodeID: nodeID,
            timestamp: event.timestamp,
            bearing: event.bearing,
            elevation: event.elevation,
            confidence: event.confidence,
            droneClass: event.classification.droneClass.rawValue,
            location: event.location.map { NodeLocation(latitude: $0.coordinate.latitude, longitude: $0.coordinate.longitude, altitude: $0.altitude) },
            tdoaSamples: nil
        )

        await send(type: .alert, payload: alert, to: session.connectedPeers)
    }

    func broadcastTDOA(_ data: TDOAData) async {
        guard let session = session, !session.connectedPeers.isEmpty else { return }
        await send(type: .triangulate, payload: data, to: session.connectedPeers)
    }

    func sendHeartbeat() async {
        guard let session = session, !session.connectedPeers.isEmpty else { return }
        let status = NodeStatus(
            nodeID: nodeID,
            timestamp: Date(),
            mode: "active",
            batteryLevel: UIDevice.current.batteryLevel,
            location: nil
        )
        await send(type: .heartbeat, payload: status, to: session.connectedPeers)
    }

    // MARK: - Generic Send

    private func send<T: Codable>(type: MeshMessageType, payload: T, to peers: [MCPeerID]) async {
        guard let session = session else { return }

        do {
            let payloadData = try encoder.encode(payload)
            let encrypted = try encryptPayload(payloadData)
            let sig = computeHMAC(type: type.rawValue, senderID: nodeID, payloadData: payloadData)

            let message = MeshMessage(
                type: type,
                senderID: nodeID,
                timestamp: Date(),
                payload: encrypted,
                messageID: UUID(),
                hopCount: 0,
                signature: sig
            )

            let messageData = try encoder.encode(message)
            try session.send(messageData, toPeers: peers, with: .reliable)

        } catch {
            print("[SkyMesh] Send error: \(error)")
        }
    }

    // MARK: - Encryption

    private func encryptPayload(_ data: Data) throws -> Data {
        let sealedBox = try AES.GCM.seal(data, using: sharedKey)
        guard let combined = sealedBox.combined else {
            throw MeshError.encryptionFailed
        }
        return combined
    }

    private func decryptPayload(_ data: Data) throws -> Data {
        let sealedBox = try AES.GCM.SealedBox(combined: data)
        return try AES.GCM.open(sealedBox, using: sharedKey)
    }

    // MARK: - HMAC Signing

    private func computeHMAC(type: String, senderID: String, payloadData: Data) -> Data {
        var hmacData = (type + senderID).data(using: .utf8)!
        hmacData.append(payloadData)
        let auth = HMAC<SHA256>.authenticationCode(for: hmacData, using: sharedKey)
        return Data(auth)
    }

    private func verifyHMAC(message: MeshMessage, payloadData: Data) -> Bool {
        let expected = computeHMAC(type: message.type.rawValue, senderID: message.senderID, payloadData: payloadData)
        return expected == message.signature
    }

    // MARK: - Message Handling

    private func handleReceivedData(_ data: Data, from peerID: MCPeerID) {
        do {
            let message = try decoder.decode(MeshMessage.self, from: data)

            // Deduplication
            guard !receivedMessageIDs.contains(message.messageID) else { return }
            receivedMessageIDs.insert(message.messageID)
            if receivedMessageIDs.count > maxMessageIDCache {
                receivedMessageIDs.removeFirst()
            }

            // Decrypt
            let payloadData = try decryptPayload(message.payload)

            // Verify signature
            guard verifyHMAC(message: message, payloadData: payloadData) else {
                print("[SkyMesh] HMAC verification failed from \(peerID.displayName)")
                return
            }

            // Route by type
            switch message.type {
            case .alert:
                let alert = try decoder.decode(AlertMessage.self, from: payloadData)
                handleIncomingAlert(alert, from: peerID.displayName)

            case .triangulate:
                let tdoa = try decoder.decode(TDOAData.self, from: payloadData)
                handleTDOAData(tdoa)

            case .heartbeat:
                let status = try decoder.decode(NodeStatus.self, from: payloadData)
                print("[SkyMesh] Heartbeat from \(status.nodeID): mode=\(status.mode)")

            case .handoff:
                print("[SkyMesh] Handoff request from \(peerID.displayName)")

            case .confirm:
                let alert = try decoder.decode(AlertMessage.self, from: payloadData)
                peerAlerts[message.senderID] = alert

            default:
                break
            }

        } catch {
            print("[SkyMesh] Message handling error: \(error)")
        }
    }

    private func handleIncomingAlert(_ alert: AlertMessage, from peer: String) {
        peerAlerts[alert.senderNodeID] = alert
        DispatchQueue.main.async { [weak self] in
            self?.alertHandler?(alert, peer)
        }
        print("[SkyMesh] Alert from \(peer): \(alert.droneClass) bearing=\(String(format: "%.1f", alert.bearing))° conf=\(String(format: "%.2f", alert.confidence))")
    }

    private func handleTDOAData(_ data: TDOAData) {
        pendingTDOA[data.nodeID] = data

        // If we have 3+ nodes, attempt triangulation
        if pendingTDOA.count >= 3 {
            let tdoaArray = Array(pendingTDOA.values)
            let result = triangulate(tdoaData: tdoaArray)
            print("[SkyMesh] Triangulated position: \(result)")
            pendingTDOA.removeAll()

            DispatchQueue.main.async { [weak self] in
                self?.triangulationHandler?(tdoaArray)
            }
        }
    }

    // MARK: - TDOA Triangulation

    private func triangulate(tdoaData: [TDOAData]) -> String {
        guard tdoaData.count >= 2 else { return "insufficient data" }

        // Hyperbolic triangulation using bearing intersections
        // For 2 nodes: find intersection of bearing lines
        let n1 = tdoaData[0]
        let n2 = tdoaData[1]

        let lat1 = n1.location.latitude
        let lon1 = n1.location.longitude
        let lat2 = n2.location.latitude
        let lon2 = n2.location.longitude

        let b1 = Double(n1.bearing) * .pi / 180.0
        let b2 = Double(n2.bearing) * .pi / 180.0

        // Simple bearing intersection (flat earth approximation for short ranges)
        let dx = lon2 - lon1
        let dy = lat2 - lat1

        let denom = sin(b1) * cos(b2) - cos(b1) * sin(b2)
        guard abs(denom) > 1e-6 else { return "parallel bearings, no intersection" }

        let t = (dy * sin(b2) - dx * cos(b2)) / denom
        let lat = lat1 + t * cos(b1)
        let lon = lon1 + t * sin(b1)

        return String(format: "lat=%.6f lon=%.6f", lat, lon)
    }
}

// MARK: - MCSessionDelegate

extension SkyMesh: MCSessionDelegate {

    func session(_ session: MCSession, peer peerID: MCPeerID, didChange state: MCSessionState) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.connectedPeers = session.connectedPeers
            self.peerStatusHandler?(session.connectedPeers)
        }

        switch state {
        case .connected:
            print("[SkyMesh] Peer connected: \(peerID.displayName)")
        case .notConnected:
            print("[SkyMesh] Peer disconnected: \(peerID.displayName)")
        case .connecting:
            print("[SkyMesh] Peer connecting: \(peerID.displayName)")
        @unknown default:
            break
        }
    }

    func session(_ session: MCSession, didReceive data: Data, fromPeer peerID: MCPeerID) {
        handleReceivedData(data, from: peerID)
    }

    func session(_ session: MCSession, didReceive stream: InputStream, withName streamName: String, fromPeer peerID: MCPeerID) {}

    func session(_ session: MCSession, didStartReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, with progress: Progress) {}

    func session(_ session: MCSession, didFinishReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, at localURL: URL?, withError error: Error?) {}
}

// MARK: - MCNearbyServiceAdvertiserDelegate

extension SkyMesh: MCNearbyServiceAdvertiserDelegate {

    func advertiser(_ advertiser: MCNearbyServiceAdvertiser, didReceiveInvitationFromPeer peerID: MCPeerID, withContext context: Data?, invitationHandler: @escaping (Bool, MCSession?) -> Void) {
        print("[SkyMesh] Received invitation from \(peerID.displayName). Auto-accepting.")
        invitationHandler(true, session)
    }

    func advertiser(_ advertiser: MCNearbyServiceAdvertiser, didNotStartAdvertisingPeer error: Error) {
        print("[SkyMesh] Advertising error: \(error)")
    }
}

// MARK: - MCNearbyServiceBrowserDelegate

extension SkyMesh: MCNearbyServiceBrowserDelegate {

    func browser(_ browser: MCNearbyServiceBrowser, foundPeer peerID: MCPeerID, withDiscoveryInfo info: [String: String]?) {
        print("[SkyMesh] Found peer: \(peerID.displayName). Inviting.")
        guard let session = session else { return }

        // Only invite if not already connected
        guard !session.connectedPeers.contains(peerID) else { return }
        browser.invitePeer(peerID, to: session, withContext: nil, timeout: 10)
    }

    func browser(_ browser: MCNearbyServiceBrowser, lostPeer peerID: MCPeerID) {
        print("[SkyMesh] Lost peer: \(peerID.displayName)")
    }

    func browser(_ browser: MCNearbyServiceBrowser, didNotStartBrowsingForPeers error: Error) {
        print("[SkyMesh] Browse error: \(error)")
    }
}

// MARK: - Key Storage

actor MeshKeyStorage {
    private let keyTag = "com.skywall.mesh.psk"

    func loadOrGenerateKey() async throws -> SymmetricKey {
        // Try loading from Keychain
        if let keyData = loadFromKeychain() {
            print("[MeshKeyStorage] Loaded existing mesh key.")
            return SymmetricKey(data: keyData)
        }

        // Generate new key
        let key = SymmetricKey(size: .bits256)
        let keyData = key.withUnsafeBytes { Data($0) }
        try saveToKeychain(keyData)
        print("[MeshKeyStorage] Generated new mesh key.")
        return key
    }

    private func loadFromKeychain() -> Data? {
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrAccount: keyTag,
            kSecReturnData: true,
            kSecMatchLimit: kSecMatchLimitOne
        ]

        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else { return nil }
        return data
    }

    private func saveToKeychain(_ data: Data) throws {
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrAccount: keyTag,
            kSecValueData: data,
            kSecAttrAccessible: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        ]

        // Delete old if exists
        SecItemDelete(query as CFDictionary)

        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw MeshError.keyStorageFailed(status)
        }
    }
}

// MARK: - Errors

enum MeshError: Error {
    case encryptionFailed
    case decryptionFailed
    case keyStorageFailed(OSStatus)
    case invalidMessage
}
