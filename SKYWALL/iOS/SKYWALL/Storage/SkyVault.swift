// SkyVault.swift
// SKYWALL - Autonomous Aerial Detection System
// AES-256-GCM encrypted storage with Secure Enclave key management.
// Uses raw SQLite3 for event persistence.

import Foundation
import CryptoKit
import Security
import SQLite3

// MARK: - Retention Policy

enum RetentionPolicy: Int, Codable {
    case sevenDays   = 7
    case thirtyDays  = 30
    case ninetyDays  = 90
}

// MARK: - SkyVault

actor SkyVault {

    // MARK: - Properties

    private var db: OpaquePointer?
    private var encryptionKey: SymmetricKey?
    private var mediaEncryptionKey: SymmetricKey?

    private let dbPath: String
    private var retentionPolicy: RetentionPolicy = .thirtyDays

    // Encoder
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

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        dbPath = docs.appendingPathComponent("skywall_vault.sqlite3").path
    }

    // MARK: - Initialize

    func initialize() async throws {
        encryptionKey = try await loadOrCreateKey(tag: "com.skywall.vault.events")
        mediaEncryptionKey = try await loadOrCreateKey(tag: "com.skywall.vault.media")
        try openDatabase()
        try createTables()
        await schedulePurge()
        print("[SkyVault] Initialized at \(dbPath)")
    }

    // MARK: - Key Management (Secure Enclave)

    private func loadOrCreateKey(tag: String) async throws -> SymmetricKey {
        // Try Secure Enclave first, fall back to Keychain
        if let existingKey = loadKeyFromKeychain(tag: tag) {
            return existingKey
        }

        let newKey = SymmetricKey(size: .bits256)
        try saveKeyToKeychain(newKey, tag: tag)
        return newKey
    }

    private func loadKeyFromKeychain(tag: String) -> SymmetricKey? {
        let query: [CFString: Any] = [
            kSecClass:       kSecClassGenericPassword,
            kSecAttrAccount: tag,
            kSecReturnData:  true,
            kSecMatchLimit:  kSecMatchLimitOne,
            kSecAttrAccessible: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        ]

        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data, data.count == 32 else {
            return nil
        }
        return SymmetricKey(data: data)
    }

    private func saveKeyToKeychain(_ key: SymmetricKey, tag: String) throws {
        let keyData = key.withUnsafeBytes { Data($0) }

        let deleteQuery: [CFString: Any] = [
            kSecClass:       kSecClassGenericPassword,
            kSecAttrAccount: tag
        ]
        SecItemDelete(deleteQuery as CFDictionary)

        let addQuery: [CFString: Any] = [
            kSecClass:       kSecClassGenericPassword,
            kSecAttrAccount: tag,
            kSecValueData:   keyData,
            kSecAttrAccessible: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        ]

        let status = SecItemAdd(addQuery as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw VaultError.keyStoreFailed(status)
        }
    }

    // MARK: - SQLite Database

    private func openDatabase() throws {
        let result = sqlite3_open_v2(dbPath, &db,
            SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX, nil)
        guard result == SQLITE_OK else {
            throw VaultError.databaseOpenFailed(String(cString: sqlite3_errmsg(db)))
        }

        // Enable WAL mode for better concurrent performance
        exec("PRAGMA journal_mode=WAL")
        exec("PRAGMA synchronous=NORMAL")
        exec("PRAGMA foreign_keys=ON")
        exec("PRAGMA page_size=4096")
    }

    private func createTables() throws {
        let createEvents = """
            CREATE TABLE IF NOT EXISTS detection_events (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                encrypted_data BLOB NOT NULL,
                nonce BLOB NOT NULL,
                confidence REAL NOT NULL,
                bearing REAL,
                node_id TEXT NOT NULL,
                mesh_confirmed INTEGER DEFAULT 0,
                purge_after REAL NOT NULL
            );
        """

        let createMedia = """
            CREATE TABLE IF NOT EXISTS media_files (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                file_type TEXT NOT NULL,
                encrypted_path TEXT NOT NULL,
                nonce BLOB NOT NULL,
                file_size INTEGER,
                FOREIGN KEY(event_id) REFERENCES detection_events(id) ON DELETE CASCADE
            );
        """

        let createIndex = """
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON detection_events(timestamp DESC);
        """

        let createNodeIndex = """
            CREATE INDEX IF NOT EXISTS idx_events_node ON detection_events(node_id);
        """

        try executeStatement(createEvents)
        try executeStatement(createMedia)
        try executeStatement(createIndex)
        try executeStatement(createNodeIndex)
    }

    // MARK: - Save Detection Event

    func saveDetectionEvent(_ event: DetectionEvent) throws {
        guard let key = encryptionKey else { throw VaultError.notInitialized }

        let jsonData = try encoder.encode(event)
        let encrypted = try AES.GCM.seal(jsonData, using: key)
        guard let combined = encrypted.combined else { throw VaultError.encryptionFailed }

        let nonceData = Data(encrypted.nonce)
        let purgeAfter = Date().addingTimeInterval(TimeInterval(retentionPolicy.rawValue * 86400))

        let sql = """
            INSERT OR REPLACE INTO detection_events
            (id, timestamp, encrypted_data, nonce, confidence, bearing, node_id, mesh_confirmed, purge_after)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw VaultError.queryFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, event.id.uuidString, -1, SQLITE_TRANSIENT)
        sqlite3_bind_double(stmt, 2, event.timestamp.timeIntervalSince1970)
        sqlite3_bind_blob(stmt, 3, (combined as NSData).bytes, Int32(combined.count), SQLITE_TRANSIENT)
        sqlite3_bind_blob(stmt, 4, (nonceData as NSData).bytes, Int32(nonceData.count), SQLITE_TRANSIENT)
        sqlite3_bind_double(stmt, 5, Double(event.confidence))
        sqlite3_bind_double(stmt, 6, Double(event.bearing))
        sqlite3_bind_text(stmt, 7, event.nodeID, -1, SQLITE_TRANSIENT)
        sqlite3_bind_int(stmt, 8, event.meshConfirmed ? 1 : 0)
        sqlite3_bind_double(stmt, 9, purgeAfter.timeIntervalSince1970)

        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw VaultError.queryFailed(lastError)
        }
    }

    // MARK: - Load Events

    func loadRecentEvents(limit: Int = 50) throws -> [DetectionEvent] {
        guard let key = encryptionKey else { throw VaultError.notInitialized }

        let sql = """
            SELECT encrypted_data, nonce FROM detection_events
            ORDER BY timestamp DESC LIMIT ?;
        """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw VaultError.queryFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_int(stmt, 1, Int32(limit))

        var events: [DetectionEvent] = []

        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let blobPtr = sqlite3_column_blob(stmt, 0) else { continue }
            let blobSize = Int(sqlite3_column_bytes(stmt, 0))
            let encryptedData = Data(bytes: blobPtr, count: blobSize)

            do {
                let sealedBox = try AES.GCM.SealedBox(combined: encryptedData)
                let decrypted = try AES.GCM.open(sealedBox, using: key)
                let event = try decoder.decode(DetectionEvent.self, from: decrypted)
                events.append(event)
            } catch {
                print("[SkyVault] Failed to decrypt event: \(error)")
            }
        }

        return events
    }

    func loadEvents(since date: Date) throws -> [DetectionEvent] {
        guard let key = encryptionKey else { throw VaultError.notInitialized }

        let sql = """
            SELECT encrypted_data FROM detection_events
            WHERE timestamp >= ?
            ORDER BY timestamp DESC;
        """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw VaultError.queryFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_double(stmt, 1, date.timeIntervalSince1970)

        var events: [DetectionEvent] = []

        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let blobPtr = sqlite3_column_blob(stmt, 0) else { continue }
            let blobSize = Int(sqlite3_column_bytes(stmt, 0))
            let data = Data(bytes: blobPtr, count: blobSize)

            do {
                let sealedBox = try AES.GCM.SealedBox(combined: data)
                let decrypted = try AES.GCM.open(sealedBox, using: key)
                let event = try decoder.decode(DetectionEvent.self, from: decrypted)
                events.append(event)
            } catch {
                continue
            }
        }
        return events
    }

    // MARK: - Save Media File

    func saveMediaFile(data: Data, eventID: UUID, fileType: String) throws -> URL {
        guard let key = mediaEncryptionKey else { throw VaultError.notInitialized }

        let fileID = UUID()
        let encryptedData = try AES.GCM.seal(data, using: key)
        guard let combined = encryptedData.combined else { throw VaultError.encryptionFailed }
        let nonceData = Data(encryptedData.nonce)

        // Write encrypted file to disk
        let mediaDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            .appendingPathComponent("skywall_media", isDirectory: true)
        try? FileManager.default.createDirectory(at: mediaDir, withIntermediateDirectories: true)

        let fileName = "\(fileID.uuidString).\(fileType).enc"
        let fileURL = mediaDir.appendingPathComponent(fileName)
        try combined.write(to: fileURL, options: .atomic)

        // Record in database
        let sql = """
            INSERT INTO media_files (id, event_id, timestamp, file_type, encrypted_path, nonce, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw VaultError.queryFailed(lastError)
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, fileID.uuidString, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 2, eventID.uuidString, -1, SQLITE_TRANSIENT)
        sqlite3_bind_double(stmt, 3, Date().timeIntervalSince1970)
        sqlite3_bind_text(stmt, 4, fileType, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 5, fileName, -1, SQLITE_TRANSIENT)
        sqlite3_bind_blob(stmt, 6, (nonceData as NSData).bytes, Int32(nonceData.count), SQLITE_TRANSIENT)
        sqlite3_bind_int64(stmt, 7, Int64(data.count))

        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw VaultError.queryFailed(lastError)
        }

        return fileURL
    }

    // MARK: - Auto-Purge

    private func schedulePurge() async {
        // Run purge at startup and then every 6 hours
        purgeExpiredRecords()

        // Schedule periodic purge
        Task {
            while true {
                try? await Task.sleep(nanoseconds: 6 * 3600 * 1_000_000_000)
                await self.purgeExpiredRecords()
            }
        }
    }

    func purgeExpiredRecords() {
        let now = Date().timeIntervalSince1970
        exec("DELETE FROM detection_events WHERE purge_after < \(now);")

        // Also clean up orphaned media files
        exec("DELETE FROM media_files WHERE event_id NOT IN (SELECT id FROM detection_events);")

        // Compact database
        exec("VACUUM;")

        print("[SkyVault] Purge complete.")
    }

    func setRetentionPolicy(_ policy: RetentionPolicy) {
        retentionPolicy = policy
    }

    // MARK: - Statistics

    func eventCount() -> Int {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM detection_events;", -1, &stmt, nil) == SQLITE_OK else { return 0 }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
        return Int(sqlite3_column_int(stmt, 0))
    }

    func databaseSizeBytes() -> Int64 {
        let attrs = try? FileManager.default.attributesOfItem(atPath: dbPath)
        return attrs?[.size] as? Int64 ?? 0
    }

    // MARK: - Helpers

    private func executeStatement(_ sql: String) throws {
        var errMsg: UnsafeMutablePointer<CChar>?
        let result = sqlite3_exec(db, sql, nil, nil, &errMsg)
        if result != SQLITE_OK {
            let msg = errMsg.map { String(cString: $0) } ?? "unknown error"
            sqlite3_free(errMsg)
            throw VaultError.queryFailed(msg)
        }
    }

    @discardableResult
    private func exec(_ sql: String) -> Bool {
        return sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK
    }

    private var lastError: String {
        db.map { String(cString: sqlite3_errmsg($0)) } ?? "no db"
    }

    deinit {
        if let db { sqlite3_close(db) }
    }
}

// MARK: - Errors

enum VaultError: Error, LocalizedError {
    case notInitialized
    case encryptionFailed
    case decryptionFailed
    case keyStoreFailed(OSStatus)
    case databaseOpenFailed(String)
    case queryFailed(String)

    var errorDescription: String? {
        switch self {
        case .notInitialized:           return "Vault not initialized"
        case .encryptionFailed:         return "Encryption failed"
        case .decryptionFailed:         return "Decryption failed"
        case .keyStoreFailed(let s):    return "Keychain error: \(s)"
        case .databaseOpenFailed(let s): return "DB open error: \(s)"
        case .queryFailed(let s):       return "SQL error: \(s)"
        }
    }
}
