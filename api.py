"""
Damage.ai — FastAPI Assessment Endpoint
========================================
REST API layer for the Car Damage Assessment system.
Accepts vehicle images, runs CV detection + policy engine,
returns structured JSON with decision trace.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoint:
    POST /assess  — upload image, get assessment result
    GET  /health  — health check
    GET  /policy  — current policy configuration
"""

import os
import sys
import uuid
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("damage_ai_api")

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import existing modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import existing project modules (graceful fallback if not available)
# ---------------------------------------------------------------------------
try:
    from car_damage_detector import CarDamageDetector
    CV_AVAILABLE = True
    logger.info("CV detection module loaded successfully")
except ImportError:
    CV_AVAILABLE = False
    logger.warning("CV detection module not found — using demo detection")

try:
    from agentic.decision_agent import make_decision
    AGENT_AVAILABLE = True
    logger.info("Decision agent loaded successfully")
except ImportError:
    AGENT_AVAILABLE = False
    logger.warning("Decision agent not found — using demo decisioning")

# ---------------------------------------------------------------------------
# Response models (Pydantic)
# ---------------------------------------------------------------------------

class DamageDetection(BaseModel):
    """Single detected damage instance."""
    damage_type: str = Field(..., description="Type of damage: scratch, dent, crack, shattered_glass, flat_tire, broken_lamp")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    severity: str = Field(..., description="Severity level: minor, moderate, severe")
    location: str = Field("unknown", description="Approximate location on vehicle")
    bbox: list = Field(default_factory=list, description="Bounding box [x1, y1, x2, y2]")
    area_percentage: float = Field(0.0, description="Percentage of image area affected")
    estimated_cost: int = Field(0, description="Estimated repair cost in USD")
    bbox: list = Field(default_factory=list, description="Bounding box [x1, y1, x2, y2]")
    area_percentage: float = Field(0.0, description="Percentage of image area affected")
    estimated_cost: int = Field(0, description="Estimated repair cost in USD")


class DecisionTrace(BaseModel):
    """Explains why a specific decision was made."""
    rule_applied: str = Field(..., description="Which policy rule triggered the decision")
    threshold: str = Field(..., description="Threshold that was evaluated")
    evidence: str = Field(..., description="SOP or policy evidence supporting the decision")


class AssessmentResult(BaseModel):
    """Full assessment response."""
    assessment_id: str = Field(..., description="Unique assessment identifier")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    
    # Detection results
    damages_detected: list[DamageDetection] = Field(default_factory=list)
    total_damages: int = Field(0, description="Number of damages detected")
    
    # Decision
    decision: str = Field(..., description="AUTO_APPROVE | HUMAN_REVIEW | ESCALATE")
    decision_confidence: float = Field(..., ge=0.0, le=1.0)
    decision_trace: list[DecisionTrace] = Field(default_factory=list)
    
    # Metadata
    model_version: str = Field("demo-v0.1", description="CV model version used")
    policy_version: str = Field("v1.0", description="Policy version applied")
    cv_backend: str = Field("demo", description="CV backend: demo | yolov8 | custom")
    human_review_required: bool = Field(False)


class HealthResponse(BaseModel):
    status: str
    version: str
    cv_available: bool
    agent_available: bool
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Demo detection (fallback when real CV module is not loaded)
# ---------------------------------------------------------------------------

def demo_detect_damage(image_bytes: bytes) -> list[dict]:
    """
    Demo detection — returns synthetic results based on image size.
    This will be replaced by real YOLOv8 inference in Phase 2.
    """
    import hashlib
    
    # Use image hash to generate deterministic but varied results
    img_hash = hashlib.md5(image_bytes).hexdigest()
    hash_int = int(img_hash[:8], 16)
    
    damage_types = [
        ("scratch", "minor", "rear_bumper"),
        ("dent", "moderate", "front_door_left"),
        ("crack", "severe", "windshield"),
        ("scratch", "minor", "hood"),
        ("dent", "moderate", "rear_quarter_panel"),
        ("shattered_glass", "severe", "headlight_left"),
    ]
    
    # Select 1-3 damages based on image hash
    num_damages = (hash_int % 3) + 1
    results = []
    
    for i in range(num_damages):
        idx = (hash_int + i * 7) % len(damage_types)
        dtype, severity, location = damage_types[idx]
        confidence = round(0.65 + (((hash_int >> (i * 4)) % 30) / 100), 2)
        confidence = min(confidence, 0.98)
        
        results.append({
            "damage_type": dtype,
            "confidence": confidence,
            "severity": severity,
            "location": location,
        })
    
    return results


def demo_make_decision(damages: list[dict]) -> dict:
    """
    Demo decision agent — applies simple rules.
    This will be replaced by the real policy engine in Phase 2.
    """
    if not damages:
        return {
            "decision": "AUTO_APPROVE",
            "confidence": 0.95,
            "trace": [{
                "rule_applied": "no_damage_detected",
                "threshold": "damages == 0",
                "evidence": "SOP: No visible damage → auto-approve"
            }],
            "human_review_required": False,
        }
    
    # Check for severe damage
    severe = [d for d in damages if d.get("severity") == "severe"]
    moderate = [d for d in damages if d.get("severity") == "moderate"]
    
    if severe:
        return {
            "decision": "ESCALATE",
            "confidence": 0.90,
            "trace": [{
                "rule_applied": "severe_damage_escalation",
                "threshold": "severity == severe",
                "evidence": f"SOP: {len(severe)} severe damage(s) detected → escalate to senior assessor"
            }],
            "human_review_required": True,
        }
    
    if len(moderate) >= 2:
        return {
            "decision": "HUMAN_REVIEW",
            "confidence": 0.80,
            "trace": [{
                "rule_applied": "multiple_moderate_review",
                "threshold": "moderate_count >= 2",
                "evidence": f"Policy: {len(moderate)} moderate damages → requires human review"
            }],
            "human_review_required": True,
        }
    
    if moderate:
        return {
            "decision": "HUMAN_REVIEW",
            "confidence": 0.75,
            "trace": [{
                "rule_applied": "moderate_damage_review",
                "threshold": "severity == moderate",
                "evidence": "Policy: Moderate damage detected → human review recommended"
            }],
            "human_review_required": True,
        }
    
    # Only minor damages
    return {
        "decision": "AUTO_APPROVE",
        "confidence": 0.85,
        "trace": [{
            "rule_applied": "minor_only_approve",
            "threshold": "all(severity == minor)",
            "evidence": f"Policy: {len(damages)} minor damage(s) only → auto-approve within threshold"
        }],
        "human_review_required": False,
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

APP_START_TIME = time.time()

app = FastAPI(
    title="Damage.ai — Vehicle Damage Assessment API",
    description=(
        "AI-powered vehicle damage assessment system. "
        "Combines Computer Vision detection with deterministic policy-driven "
        "decisioning and full decision trace for auditability."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for development, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint. Returns system status and module availability.
    """
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        cv_available=CV_AVAILABLE,
        agent_available=AGENT_AVAILABLE,
        uptime_seconds=round(time.time() - APP_START_TIME, 1),
    )


@app.post("/assess", response_model=AssessmentResult, tags=["Assessment"])
async def assess_damage(
    image: UploadFile = File(..., description="Vehicle image (JPEG/PNG)"),
):
    """
    **Main assessment endpoint.**
    
    Upload a vehicle image → get structured damage assessment with:
    - Detected damages (type, severity, location, confidence)
    - Policy-based decision (AUTO_APPROVE / HUMAN_REVIEW / ESCALATE)
    - Full decision trace explaining *why* the decision was made
    
    The response includes all information needed for audit compliance.
    """
    start_time = time.time()
    assessment_id = str(uuid.uuid4())[:12]
    
    # Validate file type
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {image.content_type}. Use JPEG, PNG, or WebP."
        )
    
    # Read image bytes
    image_bytes = await image.read()
    if len(image_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Image file is too small or empty.")
    if len(image_bytes) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="Image file exceeds 20MB limit.")
    
    logger.info(f"[{assessment_id}] Processing image: {image.filename} ({len(image_bytes)} bytes)")
    
    # Step 1: Damage detection
    if CV_AVAILABLE:
        try:
            from PIL import Image as PILImage
            import io
            detector = CarDamageDetector(confidence_threshold=0.25)
            pil_img = PILImage.open(io.BytesIO(image_bytes))
            result = detector.detect_damage(pil_img)
            raw_detections_raw = result.get('damages', [])
            # Normalize keys for API response
            raw_detections = []
            for det in raw_detections_raw:
                raw_detections.append({
                    'damage_type': det.get('type', det.get('damage_type', 'unknown')),
                    'confidence': det.get('confidence', 0.5),
                    'severity': det.get('severity', 'minor'),
                    'location': det.get('location', 'unknown'),
                    'bbox': det.get('bbox', []),
                    'area_percentage': det.get('area_percentage', 0),
                    'estimated_cost': det.get('estimated_cost', 0),
                })
            cv_backend = "model-backed"
        except Exception as e:
            logger.error(f"[{assessment_id}] CV detection failed: {e}, falling back to demo")
            raw_detections = demo_detect_damage(image_bytes)
            cv_backend = "demo-fallback"
    else:
        raw_detections = demo_detect_damage(image_bytes)
        cv_backend = "demo"
    
    # Normalize detections
    damages = [
        DamageDetection(
            damage_type=d.get("damage_type", "unknown"),
            confidence=d.get("confidence", 0.5),
            severity=d.get("severity", "minor"),
            location=d.get("location", "unknown"),
            bbox=d.get("bbox", []),
            area_percentage=d.get("area_percentage", 0.0),
            estimated_cost=d.get("estimated_cost", 0),
        )
        for d in raw_detections
    ]
    
    # Step 2: Decision agent
    if AGENT_AVAILABLE:
        try:
            decision_result = make_decision(raw_detections)
            policy_version = "v1.0-production"
        except Exception as e:
            logger.error(f"[{assessment_id}] Decision agent failed: {e}, falling back to demo")
            decision_result = demo_make_decision(raw_detections)
            policy_version = "v1.0-demo-fallback"
    else:
        decision_result = demo_make_decision(raw_detections)
        policy_version = "v1.0-demo"
    
    # Build trace
    traces = [
        DecisionTrace(
            rule_applied=t.get("rule_applied", "unknown"),
            threshold=t.get("threshold", ""),
            evidence=t.get("evidence", ""),
        )
        for t in decision_result.get("trace", [])
    ]
    
    processing_time = int((time.time() - start_time) * 1000)
    
    logger.info(
        f"[{assessment_id}] Done: {len(damages)} damages, "
        f"decision={decision_result['decision']}, "
        f"time={processing_time}ms"
    )
    
    return AssessmentResult(
        assessment_id=assessment_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        processing_time_ms=processing_time,
        damages_detected=damages,
        total_damages=len(damages),
        decision=decision_result["decision"],
        decision_confidence=decision_result.get("confidence", 0.5),
        decision_trace=traces,
        model_version="demo-v0.1" if cv_backend == "demo" else "yolov8-v0.1",
        policy_version=policy_version,
        cv_backend=cv_backend,
        human_review_required=decision_result.get("human_review_required", False),
    )


@app.get("/policy", tags=["Policy"])
async def get_policy():
    """
    Returns the current policy configuration.
    Shows what rules are active and how decisions are made.
    """
    policy_path = PROJECT_ROOT / "policies"
    policies = {}
    
    if policy_path.exists():
        for f in policy_path.glob("*.yaml"):
            policies[f.stem] = str(f)
        for f in policy_path.glob("*.yml"):
            policies[f.stem] = str(f)
    
    return {
        "policy_version": "v1.0",
        "policy_files": policies,
        "decision_types": ["AUTO_APPROVE", "HUMAN_REVIEW", "ESCALATE"],
        "severity_levels": ["minor", "moderate", "severe"],
        "rules_summary": {
            "no_damage": "AUTO_APPROVE",
            "minor_only": "AUTO_APPROVE",
            "any_moderate": "HUMAN_REVIEW",
            "multiple_moderate": "HUMAN_REVIEW",
            "any_severe": "ESCALATE",
        }
    }


# ---------------------------------------------------------------------------
# Run with: python api.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
