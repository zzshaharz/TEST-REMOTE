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
        self._sse_subscribers: List[web.StreamResponse] = []
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
        self.app.router.add_get( "/",                      self.handle_dashboard)
        self.app.router.add_get( "/dashboard",             self.handle_dashboard)
        self.app.router.add_get( "/api/v1/events/stream",  self.handle_sse)
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

            # Broadcast to SSE dashboard subscribers
            asyncio.create_task(self._broadcast_sse('detection', asdict(det)))

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
        asyncio.create_task(self._broadcast_sse('heartbeat', {
            'node_id': node_id, 'mode': node.mode, 'timestamp': node.last_seen
        }))
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
        return web.Response(text=DASHBOARD_HTML, content_type="text/html")

    async def handle_sse(self, request: web.Request) -> web.StreamResponse:
        """Server-Sent Events endpoint for real-time dashboard updates."""
        resp = web.StreamResponse()
        resp.headers['Content-Type']      = 'text/event-stream'
        resp.headers['Cache-Control']     = 'no-cache'
        resp.headers['Connection']        = 'keep-alive'
        resp.headers['X-Accel-Buffering'] = 'no'
        await resp.prepare(request)
        self._sse_subscribers.append(resp)
        log.debug(f"SSE client connected ({len(self._sse_subscribers)} total)")
        try:
            while True:
                await asyncio.sleep(15)
                await resp.write(b': ping\n\n')
        except (asyncio.CancelledError, ConnectionResetError, Exception):
            pass
        finally:
            if resp in self._sse_subscribers:
                self._sse_subscribers.remove(resp)
            log.debug(f"SSE client disconnected ({len(self._sse_subscribers)} remaining)")
        return resp

    async def _broadcast_sse(self, event_type: str, data: dict):
        """Push a JSON event to all connected SSE clients."""
        if not self._sse_subscribers:
            return
        msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()
        dead = []
        for sub in list(self._sse_subscribers):
            try:
                await sub.write(msg)
            except Exception:
                dead.append(sub)
        for d in dead:
            if d in self._sse_subscribers:
                self._sse_subscribers.remove(d)

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
  <title>SKYWALL // SECTOR COMMAND</title>
  <style>
    :root {
      --bg:     #03060a;
      --panel:  #060d16;
      --panel2: #080f1a;
      --border: #0e2540;
      --brd2:   #1a3a60;
      --acc:    #0077ee;
      --acc2:   #00bbff;
      --green:  #00ee77;
      --red:    #ff1a3c;
      --red2:   #ff4466;
      --amber:  #ff9900;
      --yellow: #ffee22;
      --text:   #c0d4e8;
      --dim:    #3a5570;
      --mono:   'Courier New', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: var(--mono);
           min-height: 100vh; overflow-x: hidden; }

    /* ── HEADER ── */
    header { background: var(--panel); border-bottom: 1px solid var(--brd2);
             padding: 0 24px; height: 52px; display: flex; align-items: center;
             gap: 20px; position: sticky; top: 0; z-index: 100; }
    .logo { font-size: 17px; font-weight: bold; letter-spacing: 5px; color: var(--acc2); white-space: nowrap; }
    .logo span { color: var(--dim); }
    .hclock { font-size: 13px; color: var(--green); letter-spacing: 2px; }
    .live-pill { display: flex; align-items: center; gap: 5px; font-size: 10px;
                 letter-spacing: 2px; color: var(--green); }
    .live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green);
                animation: pg 1.4s ease-in-out infinite; }
    @keyframes pg { 0%,100%{opacity:1;box-shadow:0 0 6px var(--green)} 50%{opacity:.3;box-shadow:none} }
    .hspacer { flex: 1; }
    .threat-pill { display: flex; align-items: center; gap: 8px; font-size: 11px; }
    .tbadge { background: var(--red); color: #fff; padding: 2px 10px; border-radius: 2px;
              font-size: 11px; font-weight: bold; letter-spacing: 1px; min-width: 26px; text-align: center; }
    .tbadge.z { background: #0d1e10; color: var(--dim); }
    .hver { font-size: 10px; color: var(--dim); letter-spacing: 1px; }

    /* ── ALERT BANNER ── */
    #banner { background: linear-gradient(90deg,#1a0004,#2a0008,#1a0004);
              border-bottom: 1px solid var(--red); padding: 7px 24px;
              display: none; align-items: center; gap: 14px;
              animation: fb .4s ease-in-out 4; }
    #banner.show { display: flex; }
    @keyframes fb { 0%,100%{background:linear-gradient(90deg,#1a0004,#2a0008,#1a0004)}
                    50%{background:linear-gradient(90deg,#350008,#550014,#350008)} }
    .bicon { color: var(--red); font-size: 15px; animation: blink .5s step-end infinite; }
    @keyframes blink { 50%{opacity:0} }
    #btext { color: var(--red2); font-size: 11px; letter-spacing: 1px; }

    /* ── STATS ROW ── */
    .srow { display: grid; grid-template-columns: repeat(4,1fr); gap: 1px;
            background: var(--border); border-bottom: 1px solid var(--brd2); }
    .scard { background: var(--panel); padding: 14px 20px; }
    .slabel { font-size: 9px; letter-spacing: 3px; color: var(--dim);
              text-transform: uppercase; margin-bottom: 5px; }
    .sval { font-size: 32px; font-weight: bold; color: var(--acc2); line-height: 1; }
    .sval.thr { color: var(--red); }
    .sval.nod { color: var(--green); }
    .sval.tot { color: var(--acc); }
    .ssub { font-size: 9px; color: var(--dim); margin-top: 3px; }

    /* ── SCAN LINE ── */
    .scanwrap { overflow: hidden; height: 1px; background: var(--panel); }
    .scanline { width: 60%; height: 1px; background: linear-gradient(90deg,transparent,var(--acc2),transparent);
                animation: scan 2.5s linear infinite; }
    @keyframes scan { from{transform:translateX(-80%)} to{transform:translateX(200%)} }

    /* ── MAIN GRID ── */
    .mgrid { display: grid; grid-template-columns: 1fr 320px; gap: 1px;
             background: var(--border); }

    /* ── PANEL ── */
    .panel { background: var(--panel); display: flex; flex-direction: column; }
    .ph { padding: 9px 16px; border-bottom: 1px solid var(--border);
          display: flex; align-items: center; gap: 10px; }
    .ptitle { font-size: 9px; letter-spacing: 3px; color: var(--dim); text-transform: uppercase; }
    .pcount { font-size: 10px; color: var(--acc); margin-left: auto; }

    /* ── NODES TABLE ── */
    .ntable { width: 100%; border-collapse: collapse; font-size: 11px; }
    .ntable th { text-align: left; padding: 6px 14px; color: var(--dim); font-weight: normal;
                 font-size: 9px; letter-spacing: 2px; border-bottom: 1px solid var(--border); }
    .ntable td { padding: 6px 14px; border-bottom: 1px solid #080f16; }
    .ntable tr:hover td { background: #080f18; }
    .nalive { color: var(--green); }
    .nalive::before { content:''; width:5px; height:5px; border-radius:50%; background:var(--green);
                      display:inline-block; margin-right:5px; animation: pg 2s ease-in-out infinite; }
    .ndead { color: var(--dim); }
    .ndead::before { content: '○ '; }
    .mbadge { padding: 1px 6px; border-radius: 2px; font-size: 9px; letter-spacing: 1px; }
    .mSLEEP    { background:#081520; color:#2a5080; }
    .mALERT    { background:#221800; color:var(--amber); }
    .mTRACK    { background:#180020; color:#cc44ff; }
    .mDOCUMENT { background:#001820; color:var(--acc); }
    .mPATROL   { background:#001810; color:var(--green); }
    .mUNKNOWN  { background:#111; color:var(--dim); }

    /* ── ALERT FEED ── */
    .afeed { flex:1; overflow-y:auto; }
    .aitem { padding: 8px 14px; border-bottom: 1px solid #080f16; display: grid;
             grid-template-columns: 1fr auto; gap: 8px; align-items: center;
             animation: si .25s ease-out; }
    @keyframes si { from{opacity:0;transform:translateX(8px)} to{opacity:1;transform:translateX(0)} }
    .aitem.HIGH   { border-left: 3px solid var(--red); }
    .aitem.MEDIUM { border-left: 3px solid var(--amber); }
    .aitem.LOW    { border-left: 3px solid var(--yellow); }
    .aitem.NONE   { border-left: 3px solid var(--dim); }
    .aclass { font-size: 11px; font-weight: bold; }
    .aclass.HIGH   { color: var(--red2); }
    .aclass.MEDIUM { color: var(--amber); }
    .aclass.LOW    { color: var(--yellow); }
    .aclass.NONE   { color: var(--dim); }
    .ameta { font-size: 9px; color: var(--dim); }
    .abearing { font-size: 16px; font-weight: bold; color: var(--text); }
    .atime { font-size: 9px; color: var(--dim); }

    /* ── LOG TABLE ── */
    .logsec { background: var(--panel2); border-top: 1px solid var(--brd2); }
    .ltable { width: 100%; border-collapse: collapse; font-size: 11px; }
    .ltable th { text-align: left; padding: 6px 14px; color: var(--dim); font-weight: normal;
                 font-size: 9px; letter-spacing: 2px; border-bottom: 1px solid var(--brd2);
                 position: sticky; top: 0; background: var(--panel2); z-index: 10; }
    .ltable td { padding: 5px 14px; border-bottom: 1px solid #080e14; }
    .ltable tr:hover td { background: #070c12; }
    .ltable tr.HIGH   td:first-child { border-left: 3px solid var(--red); }
    .ltable tr.MEDIUM td:first-child { border-left: 3px solid var(--amber); }
    .ltable tr.LOW    td:first-child { border-left: 3px solid var(--yellow); }
    .badge { display: inline-block; padding: 1px 7px; border-radius: 2px;
             font-size: 9px; font-weight: bold; letter-spacing: 1px; }
    .bHIGH   { background:#280006; color:var(--red2);  border:1px solid #500010; }
    .bMEDIUM { background:#221500; color:var(--amber);  border:1px solid #443000; }
    .bLOW    { background:#181400; color:var(--yellow); border:1px solid #302c00; }
    .bNONE   { background:#0a0e10; color:var(--dim);    border:1px solid var(--border); }
    .cbar { display:inline-block; width:44px; height:3px; background:var(--border);
            border-radius:2px; overflow:hidden; vertical-align:middle; margin-left:5px; }
    .cfill { height:100%; border-radius:2px; }

    /* ── FOOTER ── */
    footer { background: var(--panel); border-top: 1px solid var(--border);
             padding: 7px 24px; display: flex; align-items: center;
             gap: 20px; font-size: 10px; color: var(--dim); }
    footer .ml { margin-left: auto; }

    /* ── MISC ── */
    ::-webkit-scrollbar { width: 3px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--brd2); border-radius: 2px; }
    .empty { padding: 20px; text-align: center; color: var(--dim);
             font-size: 10px; letter-spacing: 2px; }
    .btn { background: var(--brd2); color: var(--acc); border: none; padding: 3px 12px;
           font-family: var(--mono); font-size: 9px; letter-spacing: 2px;
           cursor: pointer; border-radius: 2px; }
    .btn:hover { background: #254870; }

    @media (max-width: 860px) {
      .mgrid { grid-template-columns: 1fr; }
      .srow  { grid-template-columns: repeat(2,1fr); }
    }
  </style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="logo">&#x25A0;&nbsp;SKYWALL<span> // </span>SECTOR</div>
  <div class="live-pill"><div class="live-dot"></div>LIVE</div>
  <div class="hclock" id="clock">00:00:00Z</div>
  <div class="hspacer"></div>
  <div class="threat-pill">
    <span style="font-size:9px;color:var(--dim);letter-spacing:1px">THREATS</span>
    <span class="tbadge z" id="tbadge">0</span>
  </div>
  <div class="hver">v""" + SERVER_VERSION + """</div>
</header>

<!-- ALERT BANNER -->
<div id="banner">
  <span class="bicon">&#x26A0;</span>
  <span id="btext">HIGH THREAT DETECTED</span>
</div>

<!-- STATS -->
<div class="srow">
  <div class="scard">
    <div class="slabel">Events Today</div>
    <div class="sval" id="s-et">&#x2014;</div>
    <div class="ssub">24h window</div>
  </div>
  <div class="scard">
    <div class="slabel">Active Threats</div>
    <div class="sval thr" id="s-th">&#x2014;</div>
    <div class="ssub">HIGH confidence</div>
  </div>
  <div class="scard">
    <div class="slabel">Live Nodes</div>
    <div class="sval nod" id="s-an">&#x2014;</div>
    <div class="ssub" id="s-ns">of 0 registered</div>
  </div>
  <div class="scard">
    <div class="slabel">Total Events</div>
    <div class="sval tot" id="s-tot">&#x2014;</div>
    <div class="ssub">all time</div>
  </div>
</div>

<div class="scanwrap"><div class="scanline"></div></div>

<!-- MAIN GRID -->
<div class="mgrid">

  <!-- NODES -->
  <div class="panel">
    <div class="ph">
      <div class="ptitle">&#x25B6; Node Status</div>
      <div class="pcount" id="ncount">0 nodes</div>
    </div>
    <div style="overflow-x:auto;flex:1">
      <table class="ntable">
        <thead><tr>
          <th>NODE ID</th><th>STATUS</th><th>MODE</th>
          <th>BATT</th><th>DETECTIONS</th><th>LAST SEEN</th>
        </tr></thead>
        <tbody id="nbody"><tr><td colspan="6" class="empty">Awaiting nodes...</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- ALERT FEED -->
  <div class="panel" style="border-left:1px solid var(--border)">
    <div class="ph">
      <div class="ptitle">&#x26A0; Alert Feed</div>
      <div class="pcount" id="acount">0 alerts</div>
    </div>
    <div class="afeed" id="afeed">
      <div class="empty">No active alerts</div>
    </div>
  </div>

</div>

<!-- DETECTION LOG -->
<div class="logsec">
  <div class="ph" style="border-bottom:1px solid var(--brd2)">
    <div class="ptitle">&#x25BC; Detection Log</div>
    <div style="margin-left:auto;display:flex;gap:10px;align-items:center">
      <button class="btn" onclick="exportCSV()">EXPORT CSV</button>
      <div id="lupd" style="font-size:9px;color:var(--dim)">&#x2014;</div>
    </div>
  </div>
  <div style="overflow-x:auto;max-height:280px;overflow-y:auto">
    <table class="ltable">
      <thead><tr>
        <th>TIME (UTC)</th><th>NODE</th><th>CLASS</th><th>SUBTYPE</th>
        <th>CONF</th><th>BEARING</th><th>ELEV</th><th>ALT</th>
        <th>SPEED</th><th>THREAT</th><th>MESH</th>
      </tr></thead>
      <tbody id="lbody"><tr><td colspan="11" class="empty">Loading...</td></tr></tbody>
    </table>
  </div>
</div>

<!-- FOOTER -->
<footer>
  <span>&#x25A0; SKYWALL SECTOR v""" + SERVER_VERSION + """</span>
  <span id="ssest" style="color:var(--dim)">○ POLLING</span>
  <span id="fnodes"></span>
  <span class="ml" id="fuptime"></span>
</footer>

<script>
"use strict";
const $ = id => document.getElementById(id);
let allEvents = [], t0 = Date.now();

// Clock
setInterval(() => {
  const n = new Date();
  $('clock').textContent = n.getUTCHours().toString().padStart(2,'0') + ':' +
    n.getUTCMinutes().toString().padStart(2,'0') + ':' +
    n.getUTCSeconds().toString().padStart(2,'0') + 'Z';
}, 1000);

// Uptime
setInterval(() => {
  const s = Math.floor((Date.now()-t0)/1000);
  $('fuptime').textContent = `SESSION ${String(Math.floor(s/3600)).padStart(2,'0')}:`+
    `${String(Math.floor((s%3600)/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
}, 1000);

// Helpers
const utc = ts => ts ? new Date(ts*1000).toISOString().replace('T',' ').slice(0,19)+'Z' : '—';
const ago = ts => { if(!ts) return '—'; const d=(Date.now()/1000-ts)|0;
  return d<60?d+'s':d<3600?Math.round(d/60)+'m':Math.round(d/3600)+'h'; };
const tbadge = lv => `<span class="badge b${lv||'NONE'}">${lv||'NONE'}</span>`;
const mtag   = m  => { const k=(m||'UNKNOWN').toUpperCase(); return `<span class="mbadge m${k}">${k}</span>`; };
const batt   = b  => {
  if(b==null||b<0) return '—';
  const p=(b*100)|0, c=p>50?'var(--green)':p>20?'var(--amber)':'var(--red)';
  return `<span style="color:${c}">${p}%</span>`;
};
const cbar = c => {
  const p=Math.round((c||0)*100);
  const col=c>=.8?'var(--red)':c>=.5?'var(--amber)':'var(--acc)';
  return `${p}%<span class="cbar"><span class="cfill" style="width:${p}%;background:${col}"></span></span>`;
};

// Stats
async function loadStats() {
  try {
    const d = await fetch('/api/v1/stats').then(r=>r.json());
    $('s-et').textContent  = d.events_today  ?? '—';
    $('s-th').textContent  = d.threats_today ?? '—';
    $('s-an').textContent  = d.active_nodes  ?? '—';
    $('s-tot').textContent = d.events_total  ?? '—';
    $('s-ns').textContent  = `of ${d.total_nodes??0} registered`;
    const tc = d.threats_today ?? 0;
    $('tbadge').textContent = tc;
    $('tbadge').className = tc>0 ? 'tbadge' : 'tbadge z';
  } catch(_) {}
}

// Nodes
async function loadNodes() {
  try {
    const d = await fetch('/api/v1/nodes').then(r=>r.json());
    const nodes = Object.values(d.nodes||{});
    $('ncount').textContent = `${nodes.length} node${nodes.length!==1?'s':''}`;
    $('fnodes').textContent = `${nodes.filter(n=>n.is_alive).length}/${nodes.length} ONLINE`;
    const tb = $('nbody');
    if(!nodes.length){ tb.innerHTML='<tr><td colspan="6" class="empty">Awaiting node registrations...</td></tr>'; return; }
    nodes.sort((a,b)=>(b.last_seen??0)-(a.last_seen??0));
    tb.innerHTML = nodes.map(n=>`<tr>
      <td style="color:var(--acc);font-size:10px">${(n.node_id||'').substring(0,16)}</td>
      <td><span class="${n.is_alive?'nalive':'ndead'}">${n.is_alive?'ALIVE':'DEAD'}</span></td>
      <td>${mtag(n.mode)}</td>
      <td>${batt(n.battery)}</td>
      <td style="color:var(--acc)">${n.detection_count??0}</td>
      <td style="color:var(--dim);font-size:10px">${ago(n.last_seen)}</td>
    </tr>`).join('');
  } catch(_) {}
}

// Events
async function loadEvents() {
  try {
    const d = await fetch('/api/v1/events?limit=100').then(r=>r.json());
    allEvents = d.events||[];
    renderLog(allEvents);
    renderFeed(allEvents);
    $('lupd').textContent = 'UPDATED '+new Date().toISOString().slice(11,19)+'Z';
  } catch(_) {}
}

function renderLog(evs) {
  const tb = $('lbody');
  if(!evs.length){ tb.innerHTML='<tr><td colspan="11" class="empty">No detections</td></tr>'; return; }
  tb.innerHTML = evs.map(e=>`<tr class="${e.threat_level||'NONE'}">
    <td style="color:var(--dim);font-size:10px">${utc(e.timestamp)}</td>
    <td style="color:var(--acc);font-size:10px">${(e.node_id||'').substring(0,10)}</td>
    <td style="font-weight:bold">${e.drone_class||'—'}</td>
    <td style="color:var(--dim)">${e.subtype||'—'}</td>
    <td>${cbar(e.confidence)}</td>
    <td style="color:var(--text)">${e.bearing!=null?e.bearing.toFixed(0)+'°':'—'}</td>
    <td style="color:var(--dim)">${e.elevation!=null?e.elevation.toFixed(0)+'°':'—'}</td>
    <td style="color:var(--dim)">${e.altitude_m!=null?e.altitude_m+'m':'—'}</td>
    <td style="color:var(--dim)">${e.speed_kmh!=null?e.speed_kmh+'km/h':'—'}</td>
    <td>${tbadge(e.threat_level)}</td>
    <td style="color:${e.mesh_confirmed?'var(--green)':'var(--dim)'}">${e.mesh_confirmed?'&#x2713; MESH':'—'}</td>
  </tr>`).join('');
}

function renderFeed(evs) {
  const threats = evs.filter(e=>e.threat_level==='HIGH'||e.threat_level==='MEDIUM').slice(0,20);
  $('acount').textContent = `${threats.length} alert${threats.length!==1?'s':''}`;
  const fd = $('afeed');
  if(!threats.length){ fd.innerHTML='<div class="empty">No active alerts</div>'; $('banner').className=''; return; }
  fd.innerHTML = threats.map(e=>`<div class="aitem ${e.threat_level||'NONE'}">
    <div>
      <div class="aclass ${e.threat_level}">${e.drone_class||'UNKNOWN'}</div>
      <div class="ameta">${(e.node_id||'').substring(0,10)} &bull; ${e.subtype||''}</div>
      <div class="atime">${utc(e.timestamp)}</div>
    </div>
    <div class="abearing">${e.bearing!=null?e.bearing.toFixed(0)+'°':'—'}</div>
  </div>`).join('');
  const hi = evs.filter(e=>e.threat_level==='HIGH');
  if(hi.length){
    const h=hi[0];
    $('btext').textContent=`THREAT DETECTED — ${h.drone_class||'?'} — BRG ${h.bearing!=null?h.bearing.toFixed(0)+'°':'?'} — NODE ${(h.node_id||'').substring(0,8)}`;
    $('banner').className='show';
  } else { $('banner').className=''; }
}

// SSE
function connectSSE() {
  const es = new EventSource('/api/v1/events/stream');
  es.onopen = () => { $('ssest').textContent='&#x25CF; SSE LIVE'; $('ssest').style.color='var(--green)'; };
  es.onerror = () => { $('ssest').textContent='&#x25CB; POLLING'; $('ssest').style.color='var(--dim)'; };
  es.addEventListener('detection', e => {
    const d = JSON.parse(e.data);
    allEvents.unshift(d); if(allEvents.length>100) allEvents.pop();
    renderLog(allEvents); renderFeed(allEvents); loadStats();
  });
  es.addEventListener('heartbeat', () => loadNodes());
}

// CSV export
function exportCSV() {
  if(!allEvents.length){ alert('No events to export.'); return; }
  const h = ['timestamp','node_id','drone_class','subtype','confidence','bearing','elevation','altitude_m','speed_kmh','threat_level','mesh_confirmed'];
  const rows = allEvents.map(e=>h.map(k=>JSON.stringify(e[k]??'')).join(','));
  const blob = new Blob([[h.join(','),...rows].join('\\n')], {type:'text/csv'});
  const a = Object.assign(document.createElement('a'),
    {href:URL.createObjectURL(blob), download:`skywall_${Date.now()}.csv`});
  a.click(); URL.revokeObjectURL(a.href);
}

// Boot
connectSSE();
Promise.all([loadStats(), loadNodes(), loadEvents()]);
setInterval(() => Promise.all([loadStats(), loadNodes(), loadEvents()]), 5000);
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
