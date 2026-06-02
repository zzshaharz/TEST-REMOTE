// DetectionEvent.swift
// SKYWALL - Autonomous Aerial Detection System
// All data models: DetectionEvent, ThreatClassification, TrackPoint, NodeStatus, AlertMessage.

import Foundation
import CoreLocation

// MARK: - Threat Level

enum ThreatLevel: Int, Codable, Comparable {
    case none   = 0
    case low    = 1
    case medium = 2
    case high   = 3

    static func < (lhs: ThreatLevel, rhs: ThreatLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    var displayColor: String {
        switch self {
        case .none:   return "gray"
        case .low:    return "yellow"
        case .medium: return "orange"
        case .high:   return "red"
        }
    }

    var displayName: String {
        switch self {
        case .none:   return "NONE"
        case .low:    return "LOW"
        case .medium: return "MEDIUM"
        case .high:   return "HIGH"
        }
    }
}

// MARK: - DetectionEvent

struct DetectionEvent: Identifiable, Codable {
    let id: UUID
    let timestamp: Date

    // Classification
    let classification: ThreatClassification
    let audioClass: AudioClassificationResult?

    // Positional
    let bearing: Float                // degrees, north=0
    let elevation: Float              // degrees above horizon
    let confidence: Float

    // Location of observing node
    let location: CLLocation?

    // Media
    var mediaFiles: [MediaFile]

    // Network
    let nodeID: String
    var meshConfirmed: Bool

    // Visual
    var boundingBox: CGRect?

    // Computed
    var threatLevel: ThreatLevel { classification.droneClass.threatLevel }

    // MARK: - Codable Conformance (CLLocation needs manual handling)

    enum CodingKeys: String, CodingKey {
        case id, timestamp, classification, audioClass
        case bearing, elevation, confidence
        case latitude, longitude, altitude
        case mediaFiles, nodeID, meshConfirmed, boundingBox
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(timestamp, forKey: .timestamp)
        try c.encode(classification, forKey: .classification)
        try c.encodeIfPresent(audioClass, forKey: .audioClass)
        try c.encode(bearing, forKey: .bearing)
        try c.encode(elevation, forKey: .elevation)
        try c.encode(confidence, forKey: .confidence)
        try c.encodeIfPresent(location?.coordinate.latitude, forKey: .latitude)
        try c.encodeIfPresent(location?.coordinate.longitude, forKey: .longitude)
        try c.encodeIfPresent(location?.altitude, forKey: .altitude)
        try c.encode(mediaFiles, forKey: .mediaFiles)
        try c.encode(nodeID, forKey: .nodeID)
        try c.encode(meshConfirmed, forKey: .meshConfirmed)

        if let bbox = boundingBox {
            let bboxDict: [String: Double] = [
                "x": bbox.origin.x, "y": bbox.origin.y,
                "w": bbox.width,    "h": bbox.height
            ]
            try c.encode(bboxDict, forKey: .boundingBox)
        }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id             = try c.decode(UUID.self, forKey: .id)
        timestamp      = try c.decode(Date.self, forKey: .timestamp)
        classification = try c.decode(ThreatClassification.self, forKey: .classification)
        audioClass     = try c.decodeIfPresent(AudioClassificationResult.self, forKey: .audioClass)
        bearing        = try c.decode(Float.self, forKey: .bearing)
        elevation      = try c.decode(Float.self, forKey: .elevation)
        confidence     = try c.decode(Float.self, forKey: .confidence)
        mediaFiles     = try c.decode([MediaFile].self, forKey: .mediaFiles)
        nodeID         = try c.decode(String.self, forKey: .nodeID)
        meshConfirmed  = try c.decode(Bool.self, forKey: .meshConfirmed)

        let lat = try c.decodeIfPresent(Double.self, forKey: .latitude)
        let lon = try c.decodeIfPresent(Double.self, forKey: .longitude)
        let alt = try c.decodeIfPresent(Double.self, forKey: .altitude)
        if let lat, let lon {
            location = CLLocation(
                coordinate: CLLocationCoordinate2D(latitude: lat, longitude: lon),
                altitude: alt ?? 0,
                horizontalAccuracy: 10,
                verticalAccuracy: 10,
                timestamp: timestamp
            )
        } else {
            location = nil
        }

        if let bboxDict = try c.decodeIfPresent([String: Double].self, forKey: .boundingBox) {
            boundingBox = CGRect(
                x: bboxDict["x"] ?? 0,
                y: bboxDict["y"] ?? 0,
                width: bboxDict["w"] ?? 0,
                height: bboxDict["h"] ?? 0
            )
        } else {
            boundingBox = nil
        }
    }

    // Memberwise init
    init(id: UUID, timestamp: Date, classification: ThreatClassification, audioClass: AudioClassificationResult?,
         bearing: Float, elevation: Float, confidence: Float,
         location: CLLocation?, mediaFiles: [MediaFile], nodeID: String, meshConfirmed: Bool, boundingBox: CGRect?) {
        self.id = id
        self.timestamp = timestamp
        self.classification = classification
        self.audioClass = audioClass
        self.bearing = bearing
        self.elevation = elevation
        self.confidence = confidence
        self.location = location
        self.mediaFiles = mediaFiles
        self.nodeID = nodeID
        self.meshConfirmed = meshConfirmed
        self.boundingBox = boundingBox
    }
}

// MARK: - ThreatClassification

struct ThreatClassification: Codable {
    let droneClass: AerialClass
    let confidence: Float
    let bearing: Float
    let elevation: Float
    let estimatedRange: Float?      // meters
    let behavior: BehaviorPattern
    let registrationVisible: Bool
    let payloadVisible: Bool
    let firstSeen: Date
    let lastSeen: Date
    let trackPoints: [TrackPoint]
    let notes: String?

    var threatLevel: ThreatLevel { droneClass.threatLevel }
    var trackDuration: TimeInterval { lastSeen.timeIntervalSince(firstSeen) }
}

// MARK: - TrackPoint

struct TrackPoint: Codable, Identifiable {
    let id: UUID
    let timestamp: Date
    let bearing: Float
    let elevation: Float
    let confidence: Float
    let boundingBox: CodableCGRect

    init(timestamp: Date, bearing: Float, elevation: Float, confidence: Float, boundingBox: CGRect) {
        self.id = UUID()
        self.timestamp = timestamp
        self.bearing = bearing
        self.elevation = elevation
        self.confidence = confidence
        self.boundingBox = CodableCGRect(boundingBox)
    }
}

// MARK: - CodableCGRect

struct CodableCGRect: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double

    init(_ rect: CGRect) {
        x = rect.origin.x
        y = rect.origin.y
        width = rect.width
        height = rect.height
    }

    var cgRect: CGRect { CGRect(x: x, y: y, width: width, height: height) }
}

// MARK: - MediaFile

struct MediaFile: Codable, Identifiable {
    let id: UUID
    let type: MediaFileType
    let encryptedPath: String
    let timestamp: Date
    let fileSizeBytes: Int
    let thumbnailData: Data?

    enum MediaFileType: String, Codable {
        case photo       = "photo"
        case video       = "video"
        case audioClip   = "audio"
        case thumbnail   = "thumbnail"
    }
}

// MARK: - NodeStatus

struct NodeStatus: Codable, Identifiable {
    let id: UUID
    let nodeID: String
    let timestamp: Date
    let mode: String
    let batteryLevel: Float
    let location: NodeLocation?
    let connectedPeers: Int
    let detectionsToday: Int
    let storageUsedBytes: Int64
    let swVersion: String

    init(nodeID: String, timestamp: Date, mode: String, batteryLevel: Float, location: NodeLocation?) {
        self.id = UUID()
        self.nodeID = nodeID
        self.timestamp = timestamp
        self.mode = mode
        self.batteryLevel = batteryLevel
        self.location = location
        self.connectedPeers = 0
        self.detectionsToday = 0
        self.storageUsedBytes = 0
        self.swVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
    }
}

// MARK: - NodeLocation

struct NodeLocation: Codable {
    let latitude: Double
    let longitude: Double
    let altitude: Double

    init(latitude: Double, longitude: Double, altitude: Double = 0) {
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
    }

    var clLocation: CLLocation {
        CLLocation(
            coordinate: CLLocationCoordinate2D(latitude: latitude, longitude: longitude),
            altitude: altitude,
            horizontalAccuracy: 10,
            verticalAccuracy: 10,
            timestamp: Date()
        )
    }

    /// Distance in meters to another NodeLocation
    func distance(to other: NodeLocation) -> Double {
        clLocation.distance(from: other.clLocation)
    }
}

// MARK: - AudioClassificationResult (Codable extension)

extension AudioClassificationResult: Codable {
    enum CodingKeys: String, CodingKey {
        case topLabel, confidence, allScores, timestamp, melEnergy, isDrone
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(topLabel, forKey: .topLabel)
        try c.encode(confidence, forKey: .confidence)
        try c.encode(melEnergy, forKey: .melEnergy)
        try c.encode(isDrone, forKey: .isDrone)
        try c.encode(timestamp, forKey: .timestamp)
        let scores = allScores.map { (key: $0.key.rawValue, value: $0.value) }
        let dict = Dictionary(uniqueKeysWithValues: scores)
        try c.encode(dict, forKey: .allScores)
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let topLabel  = try c.decode(AcousticClass.self, forKey: .topLabel)
        let conf      = try c.decode(Float.self, forKey: .confidence)
        let energy    = try c.decode(Float.self, forKey: .melEnergy)
        let timestamp = try c.decode(Date.self, forKey: .timestamp)
        let rawScores = try c.decode([String: Float].self, forKey: .allScores)
        let allScores = Dictionary(uniqueKeysWithValues: rawScores.compactMap { k, v -> (AcousticClass, Float)? in
            guard let cls = AcousticClass(rawValue: k) else { return nil }
            return (cls, v)
        })

        self.init(topLabel: topLabel, confidence: conf, allScores: allScores, timestamp: timestamp, melEnergy: energy)
    }
}

// MARK: - JSON Helpers

extension DetectionEvent {
    var jsonString: String? {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(self) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

extension NodeStatus {
    var jsonString: String? {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(self) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
