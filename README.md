# 🚗 Car Damage Assessment AI

[![CI](https://github.com/artemxdata/Car-Damage-Assessment-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/artemxdata/Car-Damage-Assessment-AI/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/CV-YOLOv8-FF6F61.svg)](https://docs.ultralytics.com/)

> High-trust vehicle damage assessment system combining **trained YOLOv8 Computer Vision**, deterministic policy-driven decisioning, human-in-the-loop governance, and optional LLM guidance.  
> Designed for **insurance, fleet management, car rental, and automotive damage workflows**, where explainability, auditability, and human control are critical.

### 💼 Why this matters commercially

This system demonstrates how AI-assisted decisioning can:

* **Reduce operator workload** by auto-approving low-risk, well-defined cases
* **Standardize decisions** across teams, regions, and partners using explicit policies
* **Accelerate triage and escalation**, improving customer response times without sacrificing trust or control

---

## 📊 Model performance

The CV model is trained on the [CarDD dataset](https://cardd-ustc.github.io/) — the first public large-scale dataset for car damage detection (4,000+ images, 9,000+ annotations).

| Metric | Value |
|--------|-------|
| **mAP@50** | **0.581** |
| **mAP@50-95** | **0.409** |
| **Precision** | **0.675** |
| **Recall** | **0.534** |
| Model | YOLOv8n (nano) |
| Image size | 640×640 |
| Training epochs | 50 |
| Model size | 5.9 MB |

**10 damage classes:** crack, crash, dent, dislocated part, glass shatter, lamp broken, no part, rub, scratch, tire flat

Training notebook: [`notebooks/train_yolov8_cardd.ipynb`](notebooks/train_yolov8_cardd.ipynb) — fully reproducible on Google Colab (T4 GPU, ~20 min).

---

## 🧠 Problem

Vehicle damage intake and triage remains slow, inconsistent, and expensive:

* Manual inspections do not scale
* Decisions vary across operators and regions
* Escalation rules are implicit and poorly documented
* Auditability and explainability are often missing
* Humans are either overloaded or bypassed entirely

This system shows how **AI-assisted decisioning** can standardize assessment **without removing human control**.

## 🎯 What this project demonstrates

* ✅ **Trained YOLOv8 model** on real damage data (CarDD, 10 classes, mAP@50: 0.58)
* ✅ Deterministic, policy-driven decisioning (`AUTO_APPROVE`, `HUMAN_REVIEW`, `ESCALATE`)
* ✅ Explainability by design (Decision Trace + SOP evidence)
* ✅ Human-in-the-loop governance (override + audit log)
* ✅ Optional LLM guidance (non-critical, fully disableable)
* ✅ Production-style Streamlit UX with strong demo value
* ✅ **REST API** for integration into existing claims pipelines
* ✅ **Automated test suite** (22 tests) with CI/CD pipeline

## 🏗 High-Level Architecture

```
[ Vehicle Image ]
        |
        v
[ YOLOv8 Detection ]
 (trained on CarDD — 10 damage classes)
        |
        v
[ Normalized Damage Signal ]
        |
        v
[ Decision Agent ]
   ├─ Rules & thresholds
   ├─ Policy (YAML)
   └─ SOP evidence (Markdown)
        |
        v
[ Decision Output ]
   ├─ AUTO_APPROVE
   ├─ HUMAN_REVIEW
   └─ ESCALATE
        |
        v
[ Human Override ]
 (optional, always auditable)
        |
        v
[ REST API / Streamlit UI ]
 (integration or demo)
```

## 🧭 Decision philosophy

* Decisions are **deterministic by default**
* Policies and thresholds are **explicit and versioned**
* Every decision produces a **traceable explanation**
* Humans can override any outcome
* Overrides are treated as **first-class governance events**

## 🤖 Why the LLM is optional

* Core decisions **do not rely on generative AI**
* LLM is used only for:
  + operator guidance
  + repair explanations
  + UX storytelling
* Disabling the LLM **does not affect correctness**
* System remains safe for regulated environments

---

## 🔌 REST API

The system includes a **FastAPI-based REST API** for programmatic access — enabling integration into existing claims pipelines, fleet management systems, or third-party platforms.

### Quick start (API)

```bash
pip install -r requirements-api.txt
python api.py
# API running at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/assess` | Upload image → get damage assessment with decision trace |
| `GET` | `/health` | System health check and module availability |
| `GET` | `/policy` | Current policy configuration and decision rules |
| `GET` | `/docs` | Interactive Swagger UI documentation |

### Example: assess vehicle damage via curl

```bash
curl -X POST http://localhost:8000/assess \
  -F "image=@photo_of_car.jpg" \
  | python -m json.tool
```

### Example response

```json
{
  "assessment_id": "a1b2c3d4e5f6",
  "timestamp": "2026-05-17T17:30:00+00:00",
  "processing_time_ms": 120,
  "damages_detected": [
    {
      "damage_type": "scratch",
      "confidence": 0.87,
      "severity": "minor",
      "location": "rear_bumper"
    },
    {
      "damage_type": "dent",
      "confidence": 0.73,
      "severity": "moderate",
      "location": "front_door_left"
    }
  ],
  "total_damages": 2,
  "decision": "HUMAN_REVIEW",
  "decision_confidence": 0.75,
  "decision_trace": [
    {
      "rule_applied": "moderate_damage_review",
      "threshold": "severity == moderate",
      "evidence": "Policy: Moderate damage detected → human review recommended"
    }
  ],
  "model_version": "yolov8n-cardd-v1",
  "policy_version": "v1.0",
  "cv_backend": "yolov8",
  "human_review_required": true
}
```

### Example: health check

```bash
curl http://localhost:8000/health | python -m json.tool
```

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "cv_available": true,
  "agent_available": true,
  "uptime_seconds": 12.3
}
```

---

## 🖥 Demo flow (Streamlit UI)

1. Upload vehicle image
2. YOLOv8 detects visible damages (10 classes)
3. Normalize detections into a damage signal
4. Decision Agent evaluates policies and thresholds
5. Decision Trace explains *why* the outcome was chosen
6. Operator may override the decision (logged)
7. Repair Strategy Simulator & Before/After Preview provide UX "wow"

## 🖼 UI walkthrough (screenshots)

Screenshots are in: `docs/screenshots/`  
Recommended order:

1. `01_app_overview_dashboard.png` — main dashboard
2. `02_image_upload_input.png` — image upload
3. `03_damage_detection_results.png` — CV detections
4. `04_agent_decision_human_review.png` — decision trace & human review
5. `05_before_after_damage_visualization.png` — before/after preview
6. `06_assessment_summary_and_analytics.png` — analytics & charts
7. `07_assessment_report_and_export.png` — report & export

## 🛠 Technology stack

* **UI**: Streamlit
* **API**: FastAPI + Uvicorn
* **Computer Vision**: YOLOv8n (trained on CarDD, 10 damage classes)
* **Decisioning**: rule-based agent + policy YAML
* **Policies / SOPs**: Markdown + YAML
* **Retrieval (optional)**: lightweight KB lookup
* **LLM (optional)**: guidance only (no decision authority)
* **Visualization**: Plotly
* **Testing**: pytest (22 tests)
* **CI/CD**: GitHub Actions
* **Runtime**: Python 3.11+
* **Deployment**: Docker & Docker Compose
* GPU dependencies are **not required** (CPU inference supported)

## 📦 Dependency strategy

Separated dependency layers:

* `requirements.txt` — local / full environment
* `requirements-api.txt` — REST API layer (FastAPI, Uvicorn)
* `requirements-dev.txt` — dev utilities
* `requirements-llm.txt` — optional LLM integration
* `requirements.docker.txt` — minimal runtime deps (Docker)

This keeps Docker images small and predictable.

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test group
pytest tests/test_api.py::TestHealth -v
pytest tests/test_api.py::TestAssessment -v
pytest tests/test_api.py::TestValidation -v
```

Tests cover:
* API health and status reporting
* Image upload and assessment flow
* Response structure and decision trace presence
* Input validation (file type, size limits)
* Policy endpoint and OpenAPI docs

## 🚀 Quick start (local, no Docker)

```bash
git clone https://github.com/artemxdata/Car-Damage-Assessment-AI.git
cd Car-Damage-Assessment-AI

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -r requirements.txt
pip install -r requirements-api.txt

# Option 1: Streamlit UI
streamlit run app.py
# Open: http://localhost:8501

# Option 2: REST API
python api.py
# Open: http://localhost:8000/docs
```

## 🐳 Docker (recommended)

**Build & run**

```bash
docker build -t car-damage-ai:cpu .  
docker run --rm -p 8501:8501 car-damage-ai:cpu
```

## Docker Compose

```bash
docker-compose up --build
```

## 🧠 Runtime vs Source Architecture (Important Note)

**The Docker setup intentionally runs a minimal runtime image.**

**Core system intelligence — including:**

* agentic decision logic
* policy evaluation (YAML)
* SOP evidence (Markdown)
* decision trace and human override mechanisms

—is part of the source code and is fully executed inside the container at runtime.

**Development tooling, experimentation utilities, and optional LLM integrations are intentionally kept outside the runtime image to keep deployments:**

* lightweight
* deterministic
* production-aligned

**This separation mirrors real-world enterprise deployment practices, where runtime environments remain minimal while decision logic stays explicit, traceable, and auditable.**

---

**Services:**

* app — Streamlit UI + decision engine

**Ports:**

* 8501 — Web UI (Streamlit)
* 8000 — REST API (FastAPI)

---

⚙ **Configuration**

**Environment variables (optional):**  
LLM\_BASE\_URL=  
LLM\_API\_KEY=  
LLM\_MODEL=  
CONFIDENCE\_THRESHOLD=0.5

LLM can be fully disabled without breaking the system.

---

## 📁 Project structure

```
Car-Damage-Assessment-AI/  
├── app.py                    # Streamlit UI
├── api.py                    # FastAPI REST endpoint
├── car_damage_detector.py    # CV detection module
├── utils.py                  # Shared utilities
├── agentic/                  # Decision agent logic
├── policies/                 # YAML policy definitions
├── knowledge/                # SOP evidence (Markdown)
├── models/
│   ├── best.pt               # YOLOv8n trained weights (CarDD)
│   └── training_config.json  # Training metrics & config
├── notebooks/
│   └── train_yolov8_cardd.ipynb  # Training notebook (Colab)
├── data/                     # Sample images
├── docs/                     # Documentation & screenshots
├── outputs/                  # Detection output images
├── tests/
│   └── test_api.py           # API endpoint tests (22 tests)
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt          # Core dependencies
├── requirements-api.txt      # API layer dependencies
├── requirements-dev.txt      # Dev utilities
├── requirements-llm.txt      # Optional LLM deps
├── requirements.docker.txt   # Minimal Docker deps
└── README.md
```

## 🧪 What this is (and is not)

**This is:**

* a working vehicle damage detection system with a trained model
* a decision-centric architecture with full auditability
* a strong product & UX prototype
* an integration-ready system (REST API)

**This is NOT:**

* a production insurance system (requires domain-specific fine-tuning)
* a replacement for human judgment

---

## 📈 Future directions

* Improve model accuracy with larger datasets and YOLOv8m/l
* Multi-image / video ingestion
* Policy versioning & analytics dashboard
* Audit log persistence (database-backed)
* PDF / claims system export
* Landing page & pilot program
* Additional verticals (property damage, construction QC)

---

## 📄 License

MIT License

---

## 👤 Author

Artem (@artemxdata) — AI / Agentic Systems Engineering  
Focused on high-trust, explainable AI systems
