/**
 * pan_tilt_controller.ino
 * SKYWALL - Autonomous Aerial Detection System
 * ESP32 BLE GATT server for pan/tilt PTZ mount.
 *
 * Hardware:
 *   - ESP32 (WROOM-32 or equivalent)
 *   - 2x MG996R servo (or similar high-torque)
 *   - Pan servo: GPIO 18  (0-360°, continuous rotation or standard)
 *   - Tilt servo: GPIO 19 (-35° to +90°)
 *   - Status LED: GPIO 2 (built-in)
 *   - Battery voltage divider: GPIO 34 (ADC1_CH6)
 *   - Optional limit switches: GPIO 22 (pan home), GPIO 23 (tilt home)
 *
 * BLE Protocol:
 *   Service UUID: 4FAFC201-1FB5-459E-8FCC-C5C9C331914B
 *   Command Char: BEB5483E-36E1-4688-B7F5-EA07361B26A8 (WRITE WITH RESPONSE)
 *   Status Char:  BEB5483F-36E1-4688-B7F5-EA07361B26A8 (NOTIFY)
 *
 *   Command format: "P{pan}T{tilt}\n"  (pan=0-360, tilt=-35 to +90)
 *   Status format:  "S{pan},{tilt},{voltage_mv}\n"
 *
 * Special commands:
 *   "HOME\n"    - Move to home position (pan=0, tilt=0)
 *   "PATROL\n"  - Start/stop autonomous patrol sweep
 *   "STOP\n"    - Emergency stop, hold current position
 */

#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <ESP32Servo.h>
#include <esp_task_wdt.h>

// ─── Pin Definitions ─────────────────────────────────────────────────────────
#define PIN_SERVO_PAN       18
#define PIN_SERVO_TILT      19
#define PIN_STATUS_LED      2
#define PIN_BATTERY_ADC     34
#define PIN_LIMIT_PAN_HOME  22
#define PIN_LIMIT_TILT_HOME 23

// ─── BLE UUIDs (must match iOS app) ──────────────────────────────────────────
#define SERVICE_UUID        "4FAFC201-1FB5-459E-8FCC-C5C9C331914B"
#define COMMAND_CHAR_UUID   "BEB5483E-36E1-4688-B7F5-EA07361B26A8"
#define STATUS_CHAR_UUID    "BEB5483F-36E1-4688-B7F5-EA07361B26A8"
#define DEVICE_NAME         "SKYWALL_PTZ"

// ─── Servo Limits ─────────────────────────────────────────────────────────────
#define PAN_MIN_DEG         0
#define PAN_MAX_DEG         360
#define TILT_MIN_DEG        (-35)
#define TILT_MAX_DEG        90
#define TILT_HOME_DEG       0
#define PAN_HOME_DEG        0

// MG996R pulse widths (microseconds)
#define PAN_PULSE_MIN_US    544
#define PAN_PULSE_MAX_US    2400
#define TILT_PULSE_MIN_US   544
#define TILT_PULSE_MAX_US   2400

// ─── Motion Parameters ────────────────────────────────────────────────────────
#define SERVO_UPDATE_HZ     50              // 50 Hz servo update rate
#define BLE_STATUS_HZ       5               // 5 Hz status notification rate
#define SMOOTH_STEPS        20              // Interpolation steps per move
#define SMOOTH_STEP_MS      (1000 / SERVO_UPDATE_HZ)

// Patrol parameters
#define PATROL_SPEED_DPS    15.0f           // degrees per second during patrol
#define PATROL_TILT_DEG     15.0f           // patrol tilt angle
#define PATROL_PAUSE_MS     500             // pause at each end of sweep

// ─── Watchdog ─────────────────────────────────────────────────────────────────
#define WDT_TIMEOUT_SEC     10

// ─── Globals ─────────────────────────────────────────────────────────────────
Servo servoPan;
Servo servoTilt;

BLEServer*         bleServer      = nullptr;
BLECharacteristic* cmdChar        = nullptr;
BLECharacteristic* statusChar     = nullptr;
bool               bleConnected   = false;
bool               bleAdvertising = false;

// Current positions (degrees)
float currentPanDeg  = PAN_HOME_DEG;
float currentTiltDeg = TILT_HOME_DEG;

// Target positions
float targetPanDeg   = PAN_HOME_DEG;
float targetTiltDeg  = TILT_HOME_DEG;

// Smooth motion state
float smoothPanStep  = 0.0f;
float smoothTiltStep = 0.0f;

// Patrol state
bool  patrolActive        = false;
float patrolDirection     = 1.0f;   // +1 = CW, -1 = CCW
unsigned long patrolPauseUntil = 0;

// Timing
unsigned long lastServoUpdate  = 0;
unsigned long lastStatusUpdate = 0;
unsigned long lastCommandTime  = 0;
const unsigned long CMD_TIMEOUT_MS = 5000;  // 5s without command → idle

// Battery
float batteryVoltage = 0.0f;
unsigned long lastBatteryRead = 0;
const unsigned long BATTERY_READ_MS = 2000;

// LED state
uint8_t ledPattern = 0;
uint8_t ledStep    = 0;
unsigned long lastLedUpdate = 0;

// ─── LED Pattern Codes ────────────────────────────────────────────────────────
#define LED_BOOT        0   // Fast blink
#define LED_ADVERTISING 1   // Slow blink
#define LED_CONNECTED   2   // Solid
#define LED_TRACKING    3   // Double blink
#define LED_PATROL      4   // Triple blink
#define LED_ERROR       5   // Rapid blink

// ─── BLE Callbacks ───────────────────────────────────────────────────────────
class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* srv) override {
        bleConnected = true;
        ledPattern   = LED_CONNECTED;
        Serial.println("[BLE] Client connected.");
        // Stop advertising when connected (single client)
        BLEDevice::stopAdvertising();
        bleAdvertising = false;
    }

    void onDisconnect(BLEServer* srv) override {
        bleConnected = false;
        ledPattern   = LED_ADVERTISING;
        Serial.println("[BLE] Client disconnected. Restarting advertising.");
        // Restart advertising for reconnection
        BLEDevice::startAdvertising();
        bleAdvertising = true;
    }
};

class CommandCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* chr) override {
        std::string raw = chr->getValue();
        if (raw.empty()) return;

        String cmd = String(raw.c_str());
        cmd.trim();
        Serial.print("[CMD] Received: ");
        Serial.println(cmd);

        lastCommandTime = millis();
        handleCommand(cmd);
    }
};

// ─── Command Parser ───────────────────────────────────────────────────────────
void handleCommand(const String& cmd) {
    if (cmd == "HOME") {
        setTarget(PAN_HOME_DEG, TILT_HOME_DEG);
        patrolActive = false;
        return;
    }

    if (cmd == "PATROL") {
        patrolActive = !patrolActive;
        ledPattern   = patrolActive ? LED_PATROL : LED_CONNECTED;
        Serial.print("[Patrol] ");
        Serial.println(patrolActive ? "STARTED" : "STOPPED");
        return;
    }

    if (cmd == "STOP") {
        patrolActive = false;
        targetPanDeg  = currentPanDeg;
        targetTiltDeg = currentTiltDeg;
        ledPattern    = LED_CONNECTED;
        Serial.println("[STOP] Emergency stop.");
        return;
    }

    // Parse "P{pan}T{tilt}" format
    if (cmd.startsWith("P")) {
        int tIdx = cmd.indexOf('T');
        if (tIdx > 0) {
            float pan  = cmd.substring(1, tIdx).toFloat();
            float tilt = cmd.substring(tIdx + 1).toFloat();

            // Clamp to valid range
            pan  = constrain(pan,  PAN_MIN_DEG,  PAN_MAX_DEG);
            tilt = constrain(tilt, TILT_MIN_DEG, TILT_MAX_DEG);

            setTarget(pan, tilt);
            patrolActive = false;
            ledPattern = LED_TRACKING;
        }
    }
}

// ─── Set Target Position ──────────────────────────────────────────────────────
void setTarget(float pan, float tilt) {
    targetPanDeg  = pan;
    targetTiltDeg = tilt;

    // Pre-compute smooth motion increments
    smoothPanStep  = (targetPanDeg  - currentPanDeg)  / SMOOTH_STEPS;
    smoothTiltStep = (targetTiltDeg - currentTiltDeg) / SMOOTH_STEPS;
}

// ─── Degree to Microseconds Conversion ───────────────────────────────────────
int panDegToUs(float deg) {
    // Standard servo: 0° = 544µs, 180° = 2400µs
    // For continuous rotation servo, center is ~1500µs (stop)
    // This implementation assumes geared standard servo for 0-360
    // Adjust per your hardware
    deg = fmod(deg, 360.0f);
    if (deg < 0) deg += 360.0f;
    // Map 0-360 to pulse range
    return (int)map((long)(deg * 10), 0, 3600,
                    PAN_PULSE_MIN_US, PAN_PULSE_MAX_US);
}

int tiltDegToUs(float deg) {
    deg = constrain(deg, TILT_MIN_DEG, TILT_MAX_DEG);
    // Map TILT_MIN_DEG..TILT_MAX_DEG to pulse range
    float normalized = (deg - TILT_MIN_DEG) / (TILT_MAX_DEG - TILT_MIN_DEG);
    return (int)(TILT_PULSE_MIN_US + normalized * (TILT_PULSE_MAX_US - TILT_PULSE_MIN_US));
}

// ─── Apply Current Positions to Servos ───────────────────────────────────────
void applyServos() {
    servoPan.writeMicroseconds(panDegToUs(currentPanDeg));
    servoTilt.writeMicroseconds(tiltDegToUs(currentTiltDeg));
}

// ─── Smooth Motion Update (called at SERVO_UPDATE_HZ) ─────────────────────────
void updateServoPositions() {
    if (patrolActive) {
        updatePatrol();
        return;
    }

    // Smooth interpolation towards target
    float panError  = targetPanDeg  - currentPanDeg;
    float tiltError = targetTiltDeg - currentTiltDeg;

    // Handle wrap-around for pan (take shortest path)
    if (panError > 180.0f)  panError -= 360.0f;
    if (panError < -180.0f) panError += 360.0f;

    const float deadband = 0.5f;  // degrees

    if (fabs(panError) > deadband) {
        float step = constrain(panError * 0.15f, -5.0f, 5.0f);
        currentPanDeg += step;
        if (currentPanDeg < 0)     currentPanDeg += 360.0f;
        if (currentPanDeg >= 360.0f) currentPanDeg -= 360.0f;
    }

    if (fabs(tiltError) > deadband) {
        float step = constrain(tiltError * 0.15f, -5.0f, 5.0f);
        currentTiltDeg = constrain(currentTiltDeg + step, TILT_MIN_DEG, TILT_MAX_DEG);
    }

    applyServos();
}

// ─── Patrol Mode ──────────────────────────────────────────────────────────────
void updatePatrol() {
    unsigned long now = millis();
    if (now < patrolPauseUntil) return;  // Pausing at end of sweep

    // Advance patrol pan
    float stepDeg = (PATROL_SPEED_DPS / SERVO_UPDATE_HZ) * patrolDirection;
    currentPanDeg += stepDeg;

    // Wrap and reverse at limits
    if (currentPanDeg >= PAN_MAX_DEG) {
        currentPanDeg = PAN_MAX_DEG;
        patrolDirection = -1.0f;
        patrolPauseUntil = now + PATROL_PAUSE_MS;
    } else if (currentPanDeg <= PAN_MIN_DEG) {
        currentPanDeg = PAN_MIN_DEG;
        patrolDirection = 1.0f;
        patrolPauseUntil = now + PATROL_PAUSE_MS;
    }

    currentTiltDeg = PATROL_TILT_DEG;  // Fixed patrol tilt
    applyServos();
}

// ─── Battery Voltage Reading ──────────────────────────────────────────────────
void readBattery() {
    // Voltage divider: R1=10k, R2=10k → V_adc = V_bat / 2
    // ADC reference: 3.3V, 12-bit (0-4095)
    int raw = analogRead(PIN_BATTERY_ADC);
    float adcVoltage = (raw / 4095.0f) * 3.3f;
    batteryVoltage = adcVoltage * 2.0f;  // Account for divider
    // Add calibration offset if needed
}

// ─── BLE Status Notification ──────────────────────────────────────────────────
void sendStatusNotification() {
    if (!bleConnected || statusChar == nullptr) return;

    char buf[64];
    snprintf(buf, sizeof(buf), "S%.1f,%.1f,%.3f\n",
             currentPanDeg, currentTiltDeg, batteryVoltage);

    statusChar->setValue((uint8_t*)buf, strlen(buf));
    statusChar->notify();
}

// ─── LED Update ───────────────────────────────────────────────────────────────
void updateLED() {
    unsigned long now = millis();
    if (now - lastLedUpdate < 100) return;
    lastLedUpdate = now;

    ledStep++;

    switch (ledPattern) {
        case LED_BOOT:
            // Fast blink: 100ms on/off
            digitalWrite(PIN_STATUS_LED, (ledStep % 2 == 0) ? HIGH : LOW);
            break;
        case LED_ADVERTISING:
            // Slow blink: 500ms on, 500ms off
            digitalWrite(PIN_STATUS_LED, (ledStep % 10 < 5) ? HIGH : LOW);
            break;
        case LED_CONNECTED:
            // Solid on
            digitalWrite(PIN_STATUS_LED, HIGH);
            break;
        case LED_TRACKING:
            // Double blink: on-off-on-off-pause
            {
                uint8_t phase = ledStep % 20;
                bool on = (phase == 0 || phase == 1 || phase == 4 || phase == 5);
                digitalWrite(PIN_STATUS_LED, on ? HIGH : LOW);
            }
            break;
        case LED_PATROL:
            // Triple blink
            {
                uint8_t phase = ledStep % 30;
                bool on = (phase < 2 || (phase >= 4 && phase < 6) || (phase >= 8 && phase < 10));
                digitalWrite(PIN_STATUS_LED, on ? HIGH : LOW);
            }
            break;
        case LED_ERROR:
            // Very fast blink
            digitalWrite(PIN_STATUS_LED, (ledStep % 2 == 0) ? HIGH : LOW);
            break;
    }
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n[SKYWALL PTZ] Booting...");

    // Watchdog
    esp_task_wdt_init(WDT_TIMEOUT_SEC, true);
    esp_task_wdt_add(nullptr);

    // GPIO setup
    pinMode(PIN_STATUS_LED, OUTPUT);
    pinMode(PIN_LIMIT_PAN_HOME, INPUT_PULLUP);
    pinMode(PIN_LIMIT_TILT_HOME, INPUT_PULLUP);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);  // For full 0-3.3V range

    ledPattern = LED_BOOT;

    // Servo init
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    servoPan.setPeriodHertz(50);
    servoTilt.setPeriodHertz(50);
    servoPan.attach(PIN_SERVO_PAN, PAN_PULSE_MIN_US, PAN_PULSE_MAX_US);
    servoTilt.attach(PIN_SERVO_TILT, TILT_PULSE_MIN_US, TILT_PULSE_MAX_US);

    // Move to home position on boot
    currentPanDeg  = PAN_HOME_DEG;
    currentTiltDeg = TILT_HOME_DEG;
    targetPanDeg   = PAN_HOME_DEG;
    targetTiltDeg  = TILT_HOME_DEG;
    applyServos();
    delay(1000);  // Allow servos to reach home

    // BLE setup
    BLEDevice::init(DEVICE_NAME);
    BLEDevice::setMTU(128);

    bleServer = BLEDevice::createServer();
    bleServer->setCallbacks(new ServerCallbacks());

    BLEService* service = bleServer->createService(SERVICE_UUID);

    // Command characteristic (write)
    cmdChar = service->createCharacteristic(
        COMMAND_CHAR_UUID,
        BLECharacteristic::PROPERTY_WRITE |
        BLECharacteristic::PROPERTY_WRITE_NR
    );
    cmdChar->setCallbacks(new CommandCallbacks());
    cmdChar->setValue("READY");

    // Status characteristic (notify)
    statusChar = service->createCharacteristic(
        STATUS_CHAR_UUID,
        BLECharacteristic::PROPERTY_NOTIFY |
        BLECharacteristic::PROPERTY_READ
    );
    statusChar->addDescriptor(new BLE2902());
    statusChar->setValue("S0.0,0.0,0.000");

    service->start();

    // Advertising
    BLEAdvertising* adv = BLEDevice::getAdvertising();
    adv->addServiceUUID(SERVICE_UUID);
    adv->setScanResponse(true);
    adv->setMinPreferred(0x06);  // Helps iOS connection
    adv->setMinPreferred(0x12);
    BLEDevice::startAdvertising();
    bleAdvertising = true;

    ledPattern = LED_ADVERTISING;
    Serial.println("[SKYWALL PTZ] BLE advertising as: " DEVICE_NAME);
    Serial.printf("[SKYWALL PTZ] Service: %s\n", SERVICE_UUID);
    Serial.println("[SKYWALL PTZ] Ready.");
}

// ─── Main Loop ────────────────────────────────────────────────────────────────
void loop() {
    unsigned long now = millis();

    // Pat watchdog
    esp_task_wdt_reset();

    // ── Servo update at SERVO_UPDATE_HZ ────────────────────────────────────
    if (now - lastServoUpdate >= (1000 / SERVO_UPDATE_HZ)) {
        lastServoUpdate = now;
        updateServoPositions();
    }

    // ── BLE status notification at BLE_STATUS_HZ ───────────────────────────
    if (now - lastStatusUpdate >= (1000 / BLE_STATUS_HZ)) {
        lastStatusUpdate = now;
        sendStatusNotification();
    }

    // ── Battery reading every 2 seconds ────────────────────────────────────
    if (now - lastBatteryRead >= BATTERY_READ_MS) {
        lastBatteryRead = now;
        readBattery();
    }

    // ── LED update ─────────────────────────────────────────────────────────
    updateLED();

    // ── Command timeout: if no command received in 5s, enter idle patrol ──
    if (bleConnected && (now - lastCommandTime > CMD_TIMEOUT_MS) &&
        !patrolActive && lastCommandTime > 0) {
        patrolActive = true;
        ledPattern   = LED_PATROL;
        Serial.println("[Timeout] No command received. Starting patrol.");
        lastCommandTime = now;  // Prevent re-triggering immediately
    }

    // ── Handle limit switches (if installed) ───────────────────────────────
    if (digitalRead(PIN_LIMIT_PAN_HOME) == LOW) {
        currentPanDeg = PAN_HOME_DEG;
    }
    if (digitalRead(PIN_LIMIT_TILT_HOME) == LOW) {
        currentTiltDeg = TILT_HOME_DEG;
    }

    // Small yield to allow BLE stack time
    delay(1);
}
