##
## SKYWALL — Autonomous Aerial Detection System
## Build & Development Makefile
##

PYTHON    := python3
PIP       := $(PYTHON) -m pip
PORT      := 8888
DB        := ./Server/skywall_sector.db
WEBHOOK   ?=

# Colors
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[32m
BLUE  := \033[34m
CYAN  := \033[36m

.PHONY: all help ios server training clean deps check

##─────────────────────────────────────────────────────────────
## DEFAULT
##─────────────────────────────────────────────────────────────

all: help

help:
	@echo "$(BOLD)$(CYAN)SKYWALL Build System$(RESET)"
	@echo ""
	@echo "$(BOLD)iOS App:$(RESET)"
	@echo "  make ios           Generate Xcode project and open (requires xcodegen)"
	@echo "  make ios-generate  Generate Xcode project only"
	@echo "  make ios-clean     Remove generated Xcode project"
	@echo ""
	@echo "$(BOLD)Sector Server:$(RESET)"
	@echo "  make server        Install deps + start sector server"
	@echo "  make server-deps   Install server Python dependencies only"
	@echo "  make server-run    Start sector server (deps must be installed)"
	@echo ""
	@echo "$(BOLD)Training:$(RESET)"
	@echo "  make train-sound   Train sound classifier (PyTorch → CoreML)"
	@echo "  make train-yolo    Train YOLO object detector (Ultralytics → CoreML)"
	@echo "  make train-lora    Fine-tune LoRA adapter (Unsloth + QLoRA)"
	@echo "  make train-all     Run all training pipelines"
	@echo ""
	@echo "$(BOLD)Utility:$(RESET)"
	@echo "  make check         Verify Python environment"
	@echo "  make clean         Remove generated files and caches"
	@echo ""

##─────────────────────────────────────────────────────────────
## iOS
##─────────────────────────────────────────────────────────────

ios: ios-generate
	@echo "$(GREEN)Opening Xcode...$(RESET)"
	open iOS/SKYWALL.xcodeproj

ios-generate:
	@echo "$(BLUE)→ Generating Xcode project with XcodeGen...$(RESET)"
	@if ! which xcodegen > /dev/null 2>&1; then \
		echo "$(BOLD)XcodeGen not found. Install with: brew install xcodegen$(RESET)"; \
		exit 1; \
	fi
	cd iOS && xcodegen generate
	@echo "$(GREEN)✓ SKYWALL.xcodeproj generated$(RESET)"

ios-clean:
	rm -rf iOS/SKYWALL.xcodeproj
	rm -rf iOS/DerivedData
	rm -rf iOS/.build
	@echo "$(GREEN)✓ iOS build artifacts removed$(RESET)"

ios-entitlements:
	@echo "Creating entitlements file..."
	@mkdir -p iOS/SKYWALL
	@cat > iOS/SKYWALL/SKYWALL.entitlements << 'EOF'
	<?xml version="1.0" encoding="UTF-8"?>
	<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
	<plist version="1.0"><dict>
	    <key>com.apple.developer.networking.multicast</key><true/>
	    <key>com.apple.developer.networking.wifi-info</key><true/>
	</dict></plist>
	EOF

##─────────────────────────────────────────────────────────────
## SERVER
##─────────────────────────────────────────────────────────────

server: server-deps server-run

server-deps:
	@echo "$(BLUE)→ Installing server dependencies...$(RESET)"
	cd Server && $(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Server dependencies installed$(RESET)"

server-run:
	@echo "$(BLUE)→ Starting SKYWALL Sector Server on port $(PORT)...$(RESET)"
	@echo "$(CYAN)   Dashboard: http://localhost:$(PORT)/$(RESET)"
	@if [ -n "$(WEBHOOK)" ]; then \
		cd Server && $(PYTHON) sector_server.py --port $(PORT) --db $(DB) --webhook $(WEBHOOK); \
	else \
		cd Server && $(PYTHON) sector_server.py --port $(PORT) --db $(DB); \
	fi

server-debug:
	cd Server && $(PYTHON) sector_server.py --port $(PORT) --db $(DB) --debug

##─────────────────────────────────────────────────────────────
## TRAINING
##─────────────────────────────────────────────────────────────

train-all: train-sound train-yolo train-lora

train-sound:
	@echo "$(BLUE)→ Training sound classifier...$(RESET)"
	cd Training/sound_classifier && $(PIP) install -r requirements.txt -q
	cd Training/sound_classifier && $(PYTHON) train_sound_classifier.py
	@echo "$(GREEN)✓ Sound classifier training complete$(RESET)"

train-yolo:
	@echo "$(BLUE)→ Training YOLO object detector...$(RESET)"
	cd Training/yolo_trainer && $(PIP) install -r requirements.txt -q
	cd Training/yolo_trainer && $(PYTHON) train_yolo.py
	@echo "$(GREEN)✓ YOLO training complete$(RESET)"

train-lora:
	@echo "$(BLUE)→ Fine-tuning LoRA adapter...$(RESET)"
	cd Training/lora_trainer && $(PIP) install -r requirements.txt -q
	cd Training/lora_trainer && $(PYTHON) train_lora.py
	@echo "$(GREEN)✓ LoRA fine-tuning complete$(RESET)"

##─────────────────────────────────────────────────────────────
## UTILITY
##─────────────────────────────────────────────────────────────

check:
	@echo "$(BLUE)→ Environment check...$(RESET)"
	@$(PYTHON) --version
	@$(PYTHON) -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null || echo "  PyTorch: not installed"
	@$(PYTHON) -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)" 2>/dev/null || echo "  Ultralytics: not installed"
	@$(PYTHON) -c "import coremltools; print('CoreMLTools:', coremltools.__version__)" 2>/dev/null || echo "  CoreMLTools: not installed"
	@$(PYTHON) -c "import aiohttp; print('aiohttp:', aiohttp.__version__)" 2>/dev/null || echo "  aiohttp: not installed"
	@which xcodegen > /dev/null 2>&1 && echo "  XcodeGen: $(shell xcodegen version)" || echo "  XcodeGen: not installed (brew install xcodegen)"
	@echo "$(GREEN)✓ Check complete$(RESET)"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	rm -rf Training/sound_classifier/runs/
	rm -rf Training/yolo_trainer/runs/
	rm -rf Training/lora_trainer/outputs/
	@echo "$(GREEN)✓ Cleaned$(RESET)"
