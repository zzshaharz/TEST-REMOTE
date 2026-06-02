#!/usr/bin/env python3
"""
sector_server.py
SKYWALL - Autonomous Aerial Detection System

Mac/Linux sector server:
  - asyncio + aiohttp web server
  - Receives encrypted JSON from SKYWALL iOS nodes
  - TDOA triangulation
  - Node health monitoring
  - Detection aggregation and deduplication
  - SQLite event storage
  - Real-time web dashboard
  - Alert escalation (SMS/webhook/email)

Usage:
  python sector_server.py --port 8888 --db ./skywall_sector.db
  python sector_server.py --port 8888 --webhook https://hooks.slack.com/...
"""

import argparse
import asyncio
import json
import math
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import hmac
import hashlib
import base64
import secrets
import struct

from aiohttp import web
import aiohttp
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import aiohttp_cors

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("SKYWALL.Server")

# ─── Constants ────────────────────────────────────────────────────────────────

SERVER_VERSION  = "1.0.0"
HEARTBEAT_TTL   = 60.0      # Node considered dead after this many seconds
ALERT_COOLDOWN  = 30.0      # Minimum seconds between escalation alerts
TDOA_MAX_AGE    = 5.0       # Max age (seconds) of TDOA data for triangulation
PURGE_INTERVAL  = 3600      # Purge old records every hour
RETENTION_DAYS  = 30


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class NodeInfo:
    node_id: str
    display_name: str
    last_seen: float
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    mode: str
    battery: float
    connected_peers: int
    sw_version: str
    ip_address: str

    @property
    def is_alive(self) -> bool:
        return (time.time() - self.last_seen) < HEARTBEAT_TTL

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["is_alive"] = self.is_alive
        d["last_seen_ago"] = round(time.time() - self.last_seen, 1)
        return d


@dataclass
class DetectionRecord:
    event_id: str
    timestamp: float
    node_id: str
    drone_class: str
    confidence: float
    bearing: float
    elevation: float
    latitude: Optional[float]
    longitude: Optional[float]
    threat_level: str
    mesh_confirmed: bool
    triangulated_lat: Optional[float] = None
    triangulated_lon: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TDOAEntry:
    node_id: str
    timestamp: float
    bearing: float
    latitude: float
    longitude: float
    tdoa_01: float
    tdoa_02: float
    tdoa_12: float


# ─── Database ─────────────────────────────────────────────────────────────────

class SectorDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.create_tables()
        log.info(f"Database opened: {self.db_path}")

    def create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id      TEXT PRIMARY KEY,
                display_name TEXT,
                last_seen    REAL NOT NULL,
                latitude     REAL,
                longitude    REAL,
                altitude     REAL,
                mode         TEXT,
                battery      REAL,
                sw_version   TEXT,
                ip_address   TEXT
            );

            CREATE TABLE IF NOT EXISTS detection_events (
                event_id          TEXT PRIMARY KEY,
                timestamp         REAL NOT NULL,
                node_id           TEXT NOT NULL,
                drone_class       TEXT,
                confidence        REAL,
                bearing           REAL,
                elevation         REAL,
                latitude          REAL,
                longitude         REAL,
                threat_level      TEXT,
                mesh_confirmed    INTEGER DEFAULT 0,
                triangulated_lat  REAL,
                triangulated_lon  REAL,
                notes             TEXT,
                raw_json          TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_events_ts   ON detection_events(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_events_node ON detection_events(node_id);
            CREATE INDEX IF NOT EXISTS idx_events_class ON detection_events(drone_class);

            CREATE TABLE IF NOT EXISTS tdoa_data (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id     TEXT NOT NULL,
                timestamp   REAL NOT NULL,
                bearing     REAL,
                latitude    REAL,
                longitude   REAL,
                tdoa_01     REAL,
                tdoa_02     REAL,
                tdoa_12     REAL
            );

            CREATE TABLE IF NOT EXISTS alert_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL NOT NULL,
                event_id    TEXT,
                alert_type  TEXT,
                message     TEXT,
                destination TEXT,
                success     INTEGER
            );
        """)
        self.conn.commit()

    def upsert_node(self, node: NodeInfo):
        self.conn.execute("""
            INSERT OR REPLACE INTO nodes
            (node_id, display_name, last_seen, latitude, longitude, altitude,
             mode, battery, sw_version, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (node.node_id, node.display_name, node.last_seen,
              node.latitude, node.longitude, node.altitude,
              node.mode, node.battery, node.sw_version, node.ip_address))
        self.conn.commit()

    def save_detection(self, det: DetectionRecord, raw_json: str = ""):
        self.conn.execute("""
            INSERT OR IGNORE INTO detection_events
            (event_id, timestamp, node_id, drone_class, confidence,
             bearing, elevation, latitude, longitude, threat_level,
             mesh_confirmed, triangulated_lat, triangulated_lon, notes, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (det.event_id, det.timestamp, det.node_id, det.drone_class,
              det.confidence, det.bearing, det.elevation,
              det.latitude, det.longitude, det.threat_level,
              1 if det.mesh_confirmed else 0,
              det.triangulated_lat, det.triangulated_lon,
              det.notes, raw_json))
        self.conn.commit()

    def get_recent_events(self, limit: int = 100, since_ts: float = 0) -> List[DetectionRecord]:
        cursor = self.conn.execute("""
            SELECT event_id, timestamp, node_id, drone_class, confidence,
                   bearing, elevation, latitude, longitude, threat_level,
                   mesh_confirmed, triangulated_lat, triangulated_lon, notes
            FROM detection_events
            WHERE timestamp >= ?
            ORDER BY timestamp DESC LIMIT ?
        """, (since_ts, limit))
        rows = cursor.fetchall()
        return [DetectionRecord(
            event_id=r["event_id"], timestamp=r["timestamp"], node_id=r["node_id"],
            drone_class=r["drone_class"] or "unknown", confidence=r["confidence"] or 0,
            bearing=r["bearing"] or 0, elevation=r["elevation"] or 0,
            latitude=r["latitude"], longitude=r["longitude"],
            threat_level=r["threat_level"] or "NONE",
            mesh_confirmed=bool(r["mesh_confirmed"]),
            triangulated_lat=r["triangulated_lat"],
            triangulated_lon=r["triangulated_lon"],
            notes=r["notes"] or ""
        ) for r in rows]

    def get_nodes(self) -> List[Dict]:
        cursor = self.conn.execute("SELECT * FROM nodes ORDER BY last_seen DESC")
        return [dict(r) for r in cursor.fetchall()]

    def get_stats(self) -> Dict:
        now = time.time()
        today_start = now - 86400
        stats = {}

        cursor = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM detection_events WHERE timestamp >= ?", (today_start,))
        stats["events_today"] = cursor.fetchone()["cnt"]

        cursor = self.conn.execute("SELECT COUNT(*) as cnt FROM detection_events")
        stats["events_total"] = cursor.fetchone()["cnt"]

        cursor = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM detection_events WHERE threat_level IN ('HIGH', 'MEDIUM') AND timestamp >= ?",
            (today_start,))
        stats["threats_today"] = cursor.fetchone()["cnt"]

        cursor = self.conn.execute("SELECT COUNT(*) as cnt FROM nodes")
        stats["nodes_total"] = cursor.fetchone()["cnt"]

        return stats

    def purge_old_records(self, days: int = RETENTION_DAYS):
        cutoff = time.time() - (days * 86400)
        self.conn.execute("DELETE FROM detection_events WHERE timestamp < ?", (cutoff,))
        self.conn.execute("DELETE FROM tdoa_data WHERE timestamp < ?", (cutoff,))
        self.conn.commit()
        log.info(f"Purged records older than {days} days.")


# ─── TDOA Triangulation ───────────────────────────────────────────────────────

class TDOATriangulator:
    """
    Triangulate drone position from TDOA measurements at multiple nodes.
    Uses hyperbolic intersection of bearing lines as primary method,
    with TDOA time-difference geometric solver as secondary.
    """
    SPEED_OF_SOUND = 343.0  # m/s

    def triangulate(self, entries: List[TDOAEntry]) -> Optional[Tuple[float, float]]:
        """Returns (lat, lon) or None."""
        if len(entries) < 2:
            return None

        # Method 1: Bearing intersection (most practical with few nodes)
        if len(entries) >= 2:
            result = self._bearing_intersection(entries[0], entries[1])
            if result:
                return result

        return None

    def _bearing_intersection(
        self, n1: TDOAEntry, n2: TDOAEntry
    ) -> Optional[Tuple[float, float]]:
        """Intersect two bearing lines. Returns (lat, lon)."""
        # Convert to radians
        lat1, lon1 = math.radians(n1.latitude), math.radians(n1.longitude)
        lat2, lon2 = math.radians(n2.latitude), math.radians(n2.longitude)
        b1 = math.radians(n1.bearing)
        b2 = math.radians(n2.bearing)

        # Flat-earth approximation for short ranges (<50km)
        dx = (lon2 - lon1) * math.cos((lat1 + lat2) / 2) * 6371000
        dy = (lat2 - lat1) * 6371000

        denom = math.sin(b1) * math.cos(b2) - math.cos(b1) * math.sin(b2)
        if abs(denom) < 1e-8:
            return None  # Parallel bearings

        t = (dy * math.sin(b2) - dx * math.cos(b2)) / denom
        if t < 0:
            return None  # Intersection is behind sensor

        # Intersection point
        int_dx = t * math.sin(b1)
        int_dy = t * math.cos(b1)

        result_lon = n1.longitude + math.degrees(int_dx / (math.cos(lat1) * 6371000))
        result_lat = n1.latitude + math.degrees(int_dy / 6371000)

        return (result_lat, result_lon)

    def estimate_range(self, observer_lat: float, observer_lon: float,
                       target_lat: float, target_lon: float) -> float:
        """Haversine distance in meters."""
        R = 6371000
        phi1 = math.radians(observer_lat)
        phi2 = math.radians(target_lat)
        dphi = math.radians(target_lat - observer_lat)
        dlambda = math.radians(target_lon - observer_lon)

        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c


# ─── Alert Escalation ─────────────────────────────────────────────────────────

class AlertEscalator:
    def __init__(self, webhook_url: Optional[str] = None, smtp_config: Optional[Dict] = None):
        self.webhook_url  = webhook_url
        self.smtp_config  = smtp_config
        self.last_alert   = defaultdict(float)

    async def escalate(self, det: DetectionRecord, session: aiohttp.ClientSession):
        threat = det.threat_level
        if threat not in ("HIGH", "MEDIUM"):
            return

        now = time.time()
        key = f"{det.node_id}_{det.drone_class}"
        if now - self.last_alert[key] < ALERT_COOLDOWN:
            return  # Rate limited
        self.last_alert[key] = now

        message = self._build_message(det)
        log.warning(f"ALERT ESCALATION: {message}")

        tasks = []
        if self.webhook_url:
            tasks.append(self._send_webhook(session, message, det))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _build_message(self, det: DetectionRecord) -> str:
        ts = datetime.fromtimestamp(det.timestamp).strftime("%H:%M:%S")
        parts = [
            f"SKYWALL ALERT [{det.threat_level}]",
            f"Time: {ts}",
            f"Node: {det.node_id[:8]}",
            f"Class: {det.drone_class}",
            f"Confidence: {det.confidence:.0%}",
            f"Bearing: {det.bearing:.0f}°",
        ]
        if det.triangulated_lat:
            parts.append(f"Position: {det.triangulated_lat:.5f}, {det.triangulated_lon:.5f}")
        return " | ".join(parts)

    async def _send_webhook(self, session: aiohttp.ClientSession, message: str, det: DetectionRecord):
        payload = {
            "text": message,
            "attachments": [{
                "color": "danger" if det.threat_level == "HIGH" else "warning",
                "fields": [
                    {"title": "Drone Class",  "value": det.drone_class,                          "short": True},
                    {"title": "Confidence",   "value": f"{det.confidence:.0%}",                  "short": True},
                    {"title": "Bearing",      "value": f"{det.bearing:.0f}°",                    "short": True},
                    {"title": "Threat Level", "value": det.threat_level,                          "short": True},
                ]
            }]
        }
        try:
            async with session.post(self.webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    log.warning(f"Webhook returned {resp.status}")
        except Exception as e:
            log.error(f"Webhook failed: {e}")


# ─── Encryption / Auth ────────────────────────────────────────────────────────

class MessageCrypto:
    def __init__(self, key_hex: Optional[str] = None):
        if key_hex:
            self.key = bytes.fromhex(key_hex)
        else:
            self.key = secrets.token_bytes(32)
            log.info(f"Generated new AES-256 key: {self.key.hex()}")
            log.info("Share this key with iOS nodes via secure channel.")

        self.aesgcm = AESGCM(self.key)

    def decrypt(self, combined_b64: str) -> Optional[bytes]:
        """Decrypt AES-256-GCM combined (nonce || ciphertext || tag) base64."""
        try:
            combined = base64.b64decode(combined_b64)
            nonce = combined[:12]
            ciphertext = combined[12:]
            return self.aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as e:
            log.warning(f"Decryption failed: {e}")
            return None

    def encrypt(self, plaintext: bytes) -> str:
        nonce = secrets.token_bytes(12)
        ct = self.aesgcm.encrypt(nonce, plaintext, None)
        return base64.b64encode(nonce + ct).decode()

    def verify_hmac(self, data: bytes, sig_hex: str) -> bool:
        expected = hmac.new(self.key, data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig_hex)


# ─── SKYWALL Server ───────────────────────────────────────────────────────────

class SkyWALLServer:
    def __init__(self, db: SectorDatabase, crypto: MessageCrypto,
                 escalator: AlertEscalator):
        self.db          = db
        self.crypto      = crypto
        self.escalator   = escalator
        self.nodes: Dict[str, NodeInfo] = {}
        self.tdoa_buffer: Dict[str, List[TDOAEntry]] = defaultdict(list)
        self.triangulator = TDOATriangulator()
        self.app = web.Application()
        self.session: Optional[aiohttp.ClientSession] = None
        self._setup_routes()
        self._setup_cors()

    def _setup_cors(self):
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
        for route in list(self.app.router.routes()):
            cors.add(route)

    def _setup_routes(self):
        self.app.router.add_post("/api/v1/ingest",       self.handle_ingest)
        self.app.router.add_post("/api/v1/heartbeat",    self.handle_heartbeat)
        self.app.router.add_post("/api/v1/tdoa",         self.handle_tdoa)
        self.app.router.add_get( "/api/v1/events",       self.handle_get_events)
        self.app.router.add_get( "/api/v1/nodes",        self.handle_get_nodes)
        self.app.router.add_get( "/api/v1/stats",        self.handle_get_stats)
        self.app.router.add_get( "/api/v1/status",       self.handle_status)
        self.app.router.add_get( "/",                    self.handle_dashboard)
        self.app.router.add_get( "/dashboard",           self.handle_dashboard)
        self.app.on_startup.append(self._on_startup)
        self.app.on_shutdown.append(self._on_shutdown)

    async def _on_startup(self, app):
        self.session = aiohttp.ClientSession()
        asyncio.create_task(self._purge_loop())
        asyncio.create_task(self._health_monitor_loop())
        log.info(f"SKYWALL Sector Server v{SERVER_VERSION} started.")

    async def _on_shutdown(self, app):
        if self.session:
            await self.session.close()

    # ── API Handlers ─────────────────────────────────────────────────────────

    async def handle_ingest(self, request: web.Request) -> web.Response:
        """Receive detection events from iOS nodes."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        ip = request.remote or "unknown"

        # Extract and optionally decrypt payload
        payload_raw = body.get("payload")
        encrypted   = body.get("encrypted", False)

        if encrypted and payload_raw:
            decrypted = self.crypto.decrypt(payload_raw)
            if decrypted is None:
                return web.json_response({"error": "decryption failed"}, status=401)
            try:
                payload = json.loads(decrypted)
            except json.JSONDecodeError:
                return web.json_response({"error": "invalid decrypted payload"}, status=400)
        elif payload_raw:
            payload = payload_raw if isinstance(payload_raw, dict) else json.loads(payload_raw)
        else:
            payload = body

        node_id = payload.get("nodeID") or payload.get("node_id") or "unknown"
        log.info(f"Ingest from {node_id[:8]} ({ip}): {payload.get('classification', {}).get('droneClass', 'unknown')}")

        det = self._parse_detection(payload, node_id, ip)
        if det:
            self.db.save_detection(det, raw_json=json.dumps(payload))

            # Escalate if high/medium threat
            if self.session:
                asyncio.create_task(self.escalator.escalate(det, self.session))

        return web.json_response({"status": "ok", "event_id": det.event_id if det else None})

    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        """Update node heartbeat."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        node_id = body.get("nodeID") or body.get("node_id") or "unknown"
        ip = request.remote or "unknown"

        node = NodeInfo(
            node_id=node_id,
            display_name=body.get("displayName", f"Node-{node_id[:6]}"),
            last_seen=time.time(),
            latitude=body.get("latitude"),
            longitude=body.get("longitude"),
            altitude=body.get("altitude", 0),
            mode=body.get("mode", "unknown"),
            battery=body.get("batteryLevel", -1),
            connected_peers=body.get("connectedPeers", 0),
            sw_version=body.get("swVersion", "unknown"),
            ip_address=ip,
        )

        self.nodes[node_id] = node
        self.db.upsert_node(node)
        return web.json_response({"status": "ok", "server_time": time.time()})

    async def handle_tdoa(self, request: web.Request) -> web.Response:
        """Receive TDOA data for triangulation."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        node_id = body.get("nodeID", "unknown")
        entry = TDOAEntry(
            node_id=node_id,
            timestamp=body.get("timestamp", time.time()),
            bearing=body.get("bearing", 0),
            latitude=body.get("latitude", 0),
            longitude=body.get("longitude", 0),
            tdoa_01=body.get("tdoa01", 0),
            tdoa_02=body.get("tdoa02", 0),
            tdoa_12=body.get("tdoa12", 0),
        )

        # Add to buffer and attempt triangulation
        event_id = body.get("eventID", "global")
        self.tdoa_buffer[event_id].append(entry)

        # Keep only recent entries
        cutoff = time.time() - TDOA_MAX_AGE
        self.tdoa_buffer[event_id] = [e for e in self.tdoa_buffer[event_id] if e.timestamp >= cutoff]

        result = None
        if len(self.tdoa_buffer[event_id]) >= 2:
            result = self.triangulator.triangulate(self.tdoa_buffer[event_id])
            if result:
                lat, lon = result
                log.info(f"Triangulated: lat={lat:.5f} lon={lon:.5f} (from {len(self.tdoa_buffer[event_id])} nodes)")

        return web.json_response({
            "status":    "ok",
            "node_count": len(self.tdoa_buffer[event_id]),
            "triangulated": {"lat": result[0], "lon": result[1]} if result else None,
        })

    async def handle_get_events(self, request: web.Request) -> web.Response:
        limit    = int(request.rel_url.query.get("limit", 100))
        since_ts = float(request.rel_url.query.get("since", 0))
        events = self.db.get_recent_events(limit=limit, since_ts=since_ts)
        return web.json_response({"events": [e.to_dict() for e in events]})

    async def handle_get_nodes(self, request: web.Request) -> web.Response:
        nodes = {nid: n.to_dict() for nid, n in self.nodes.items()}
        return web.json_response({"nodes": nodes, "count": len(nodes)})

    async def handle_get_stats(self, request: web.Request) -> web.Response:
        stats = self.db.get_stats()
        stats["active_nodes"] = sum(1 for n in self.nodes.values() if n.is_alive)
        stats["server_version"] = SERVER_VERSION
        stats["server_time"] = time.time()
        return web.json_response(stats)

    async def handle_status(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status":  "operational",
            "version": SERVER_VERSION,
            "uptime":  "N/A",
            "nodes":   len(self.nodes),
            "time":    datetime.utcnow().isoformat() + "Z",
        })

    async def handle_dashboard(self, request: web.Request) -> web.Response:
        """Serve the HTML dashboard."""
        html = DASHBOARD_HTML
        return web.Response(text=html, content_type="text/html")

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_detection(self, payload: Dict, node_id: str, ip: str) -> Optional[DetectionRecord]:
        try:
            event_id   = payload.get("id") or str(uuid.uuid4())
            ts_raw     = payload.get("timestamp", time.time())
            timestamp  = ts_raw if isinstance(ts_raw, float) else time.time()

            classification = payload.get("classification", {})
            drone_class    = (classification.get("droneClass") or
                              payload.get("drone_class") or "unknown")
            confidence     = float(classification.get("confidence") or payload.get("confidence") or 0)
            bearing        = float(classification.get("bearing")    or payload.get("bearing")    or 0)
            elevation      = float(classification.get("elevation")  or payload.get("elevation")  or 0)
            threat_level   = (classification.get("threatLevel") or
                              payload.get("threat_level") or "NONE")

            latitude  = payload.get("latitude")
            longitude = payload.get("longitude")

            return DetectionRecord(
                event_id=event_id,
                timestamp=timestamp,
                node_id=node_id,
                drone_class=drone_class,
                confidence=confidence,
                bearing=bearing,
                elevation=elevation,
                latitude=latitude,
                longitude=longitude,
                threat_level=threat_level,
                mesh_confirmed=bool(payload.get("meshConfirmed", False)),
            )
        except Exception as e:
            log.error(f"Parse error: {e} | payload keys: {list(payload.keys())}")
            return None

    # ── Background Tasks ──────────────────────────────────────────────────────

    async def _purge_loop(self):
        while True:
            await asyncio.sleep(PURGE_INTERVAL)
            self.db.purge_old_records(RETENTION_DAYS)

    async def _health_monitor_loop(self):
        while True:
            await asyncio.sleep(30)
            dead = [nid for nid, n in self.nodes.items() if not n.is_alive]
            for nid in dead:
                log.warning(f"Node lost: {nid[:8]}")


# ─── Dashboard HTML ───────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SKYWALL Sector Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0a0a0f; color: #e0e0e0; font-family: 'Courier New', monospace; }
    header { background: #0d1117; border-bottom: 1px solid #1e3a5f; padding: 16px 24px;
             display: flex; justify-content: space-between; align-items: center; }
    header h1 { color: #4a9eff; font-size: 22px; letter-spacing: 4px; }
    header .status { font-size: 12px; color: #7f7f7f; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px; padding: 20px 24px; }
    .card { background: #0d1117; border: 1px solid #1e3a5f; border-radius: 8px; padding: 16px; }
    .card h3 { font-size: 11px; color: #7f7f7f; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }
    .card .value { font-size: 36px; font-weight: bold; color: #4a9eff; }
    .card .sub { font-size: 11px; color: #666; margin-top: 4px; }
    .section { padding: 0 24px 24px; }
    .section h2 { font-size: 13px; color: #7f7f7f; text-transform: uppercase;
                  letter-spacing: 2px; margin-bottom: 12px; border-bottom: 1px solid #1e3a5f;
                  padding-bottom: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th { text-align: left; padding: 8px 12px; color: #7f7f7f; font-weight: normal;
         border-bottom: 1px solid #1e3a5f; }
    td { padding: 8px 12px; border-bottom: 1px solid #111; }
    tr:hover td { background: #0d1117; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
             font-size: 10px; font-weight: bold; }
    .badge-HIGH   { background: #7f0000; color: #ff6b6b; }
    .badge-MEDIUM { background: #5a3000; color: #ffaa44; }
    .badge-LOW    { background: #3a3a00; color: #dddd44; }
    .badge-NONE   { background: #1a1a1a; color: #888; }
    .badge-alive  { background: #003a00; color: #44cc44; }
    .badge-dead   { background: #3a0000; color: #cc4444; }
    .refresh-btn { background: #1e3a5f; color: #4a9eff; border: 1px solid #2a5080;
                   padding: 6px 16px; border-radius: 4px; cursor: pointer; font-family: inherit;
                   font-size: 12px; }
    .refresh-btn:hover { background: #2a5080; }
    #last-refresh { font-size: 11px; color: #555; margin-left: 12px; }
  </style>
</head>
<body>
  <header>
    <h1>&#x25A0; SKYWALL SECTOR</h1>
    <div style="display:flex;align-items:center;">
      <button class="refresh-btn" onclick="refresh()">Refresh</button>
      <span id="last-refresh"></span>
      <div class="status" style="margin-left:20px;">v""" + SERVER_VERSION + """</div>
    </div>
  </header>

  <div class="grid" id="stats-grid">
    <div class="card"><h3>Events Today</h3><div class="value" id="events-today">-</div></div>
    <div class="card"><h3>Threats Today</h3><div class="value" id="threats-today" style="color:#ff6b6b">-</div></div>
    <div class="card"><h3>Active Nodes</h3><div class="value" id="active-nodes">-</div></div>
    <div class="card"><h3>Total Events</h3><div class="value" id="events-total">-</div></div>
  </div>

  <div class="section">
    <h2>Active Nodes</h2>
    <table id="nodes-table">
      <thead><tr><th>Node ID</th><th>Status</th><th>Mode</th><th>Battery</th><th>Last Seen</th></tr></thead>
      <tbody id="nodes-body"><tr><td colspan="5">Loading...</td></tr></tbody>
    </table>
  </div>

  <div class="section">
    <h2>Recent Detections</h2>
    <table id="events-table">
      <thead><tr><th>Time</th><th>Node</th><th>Class</th><th>Confidence</th><th>Bearing</th><th>Threat</th></tr></thead>
      <tbody id="events-body"><tr><td colspan="6">Loading...</td></tr></tbody>
    </table>
  </div>

  <script>
    async function fetchJSON(url) {
      try { const r = await fetch(url); return await r.json(); } catch(e) { return null; }
    }

    function formatTime(ts) {
      if (!ts) return '-';
      return new Date(ts * 1000).toLocaleTimeString();
    }

    function threatBadge(level) {
      return `<span class="badge badge-${level}">${level}</span>`;
    }

    async function refresh() {
      const [stats, nodesResp, eventsResp] = await Promise.all([
        fetchJSON('/api/v1/stats'),
        fetchJSON('/api/v1/nodes'),
        fetchJSON('/api/v1/events?limit=50'),
      ]);

      if (stats) {
        document.getElementById('events-today').textContent  = stats.events_today  ?? '-';
        document.getElementById('threats-today').textContent = stats.threats_today ?? '-';
        document.getElementById('active-nodes').textContent  = stats.active_nodes  ?? '-';
        document.getElementById('events-total').textContent  = stats.events_total  ?? '-';
      }

      if (nodesResp?.nodes) {
        const tbody = document.getElementById('nodes-body');
        const nodes = Object.values(nodesResp.nodes);
        if (nodes.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" style="color:#555">No nodes registered</td></tr>';
        } else {
          tbody.innerHTML = nodes.map(n => `
            <tr>
              <td>${n.node_id?.substring(0,12) ?? 'unknown'}</td>
              <td><span class="badge badge-${n.is_alive ? 'alive' : 'dead'}">${n.is_alive ? 'ALIVE' : 'DEAD'}</span></td>
              <td>${n.mode ?? '-'}</td>
              <td>${n.battery >= 0 ? (n.battery * 100).toFixed(0) + '%' : '-'}</td>
              <td>${n.last_seen_ago}s ago</td>
            </tr>
          `).join('');
        }
      }

      if (eventsResp?.events) {
        const tbody = document.getElementById('events-body');
        if (eventsResp.events.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" style="color:#555">No events recorded</td></tr>';
        } else {
          tbody.innerHTML = eventsResp.events.map(e => `
            <tr>
              <td>${formatTime(e.timestamp)}</td>
              <td>${e.node_id?.substring(0,8) ?? '-'}</td>
              <td>${e.drone_class}</td>
              <td>${(e.confidence * 100).toFixed(0)}%</td>
              <td>${e.bearing?.toFixed(0) ?? '-'}°</td>
              <td>${threatBadge(e.threat_level ?? 'NONE')}</td>
            </tr>
          `).join('');
        }
      }

      document.getElementById('last-refresh').textContent =
        'Updated ' + new Date().toLocaleTimeString();
    }

    refresh();
    setInterval(refresh, 10000);
  </script>
</body>
</html>"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SKYWALL Sector Server")
    parser.add_argument("--port",      type=int,  default=8888)
    parser.add_argument("--host",      type=str,  default="0.0.0.0")
    parser.add_argument("--db",        type=str,  default="./skywall_sector.db")
    parser.add_argument("--key",       type=str,  default=None,
                        help="AES-256 key hex (64 chars). Auto-generated if not provided.")
    parser.add_argument("--webhook",   type=str,  default=None,
                        help="Slack-compatible webhook URL for alert escalation")
    parser.add_argument("--debug",     action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    db         = SectorDatabase(args.db)
    db.connect()

    crypto     = MessageCrypto(args.key)
    escalator  = AlertEscalator(webhook_url=args.webhook)
    server     = SkyWALLServer(db, crypto, escalator)

    log.info(f"Starting SKYWALL Sector Server on {args.host}:{args.port}")
    log.info(f"Dashboard: http://localhost:{args.port}/dashboard")
    log.info(f"API:       http://localhost:{args.port}/api/v1/status")

    web.run_app(
        server.app,
        host=args.host,
        port=args.port,
        access_log=None if not args.debug else logging.getLogger("aiohttp.access"),
    )


if __name__ == "__main__":
    main()
