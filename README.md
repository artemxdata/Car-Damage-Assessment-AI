# 🚗 Car Damage Assessment AI — Internal POC

> **Internal Proof of Concept**  
> Designed as an internal demo for **insurance, fleet management, and automotive damage decision workflows**, where explainability, auditability, and human control are critical.  
> High-trust vehicle damage assessment system combining Computer Vision, deterministic policy-driven decisioning, human-in-the-loop governance, and optional LLM guidance.

### 💼 Why this matters commercially
This system demonstrates how AI-assisted decisioning can:
- **Reduce operator workload** by auto-approving low-risk, well-defined cases
- **Standardize decisions** across teams, regions, and partners using explicit policies
- **Accelerate triage and escalation**, improving customer response times without sacrificing trust or control

## 🧠 Problem
Vehicle damage intake and triage remains slow, inconsistent, and expensive:
- Manual inspections do not scale
- Decisions vary across operators and regions
- Escalation rules are implicit and poorly documented
- Auditability and explainability are often missing
- Humans are either overloaded or bypassed entirely

This POC shows how **AI-assisted decisioning** can standardize assessment **without removing human control**.

## 🎯 What this POC demonstrates
- ✅ Computer Vision damage detection (demo / model-backed, YOLO-compatible interface)
- ✅ Deterministic, policy-driven decisioning (`AUTO_APPROVE`, `HUMAN_REVIEW`, `ESCALATE`)
- ✅ Explainability by design (Decision Trace + SOP evidence)
- ✅ Human-in-the-loop governance (override + audit log)
- ✅ Optional LLM guidance (non-critical, fully disableable)
- ✅ Production-style Streamlit UX with strong demo value (“wow” moments)

## 🏗 High-Level Architecture

```text
[ Vehicle Image ]
        |
        v
[ CV Detection ]
 (demo / model-backed)
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
```

## 🧭 Decision philosophy
- Decisions are **deterministic by default**
- Policies and thresholds are **explicit and versioned**
- Every decision produces a **traceable explanation**
- Humans can override any outcome
- Overrides are treated as **first-class governance events**

## 🤖 Why the LLM is optional
- Core decisions **do not rely on generative AI**
- LLM is used only for:
  - operator guidance
  - repair explanations
  - UX storytelling
- Disabling the LLM **does not affect correctness**
- System remains safe for regulated environments

## 🖥 Demo flow
1. Upload vehicle image
2. Detect visible damages (demo or CV-backed)
3. Normalize detections into a damage signal
4. Decision Agent evaluates policies and thresholds
5. Decision Trace explains *why* the outcome was chosen
6. Operator may override the decision (logged)
7. Repair Strategy Simulator & Before/After Preview provide UX “wow”

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
- **UI**: Streamlit
- **Computer Vision**: OpenCV
- **Decisioning**: rule-based agent + policy YAML
- **Policies / SOPs**: Markdown + YAML
- **Retrieval (optional)**: lightweight KB lookup
- **LLM (optional)**: guidance only (no decision authority)
- **Visualization**: Plotly
- **Runtime**: Python 3.12
- **Deployment**: Docker & Docker Compose  
- GPU dependencies are **not required**

## 📦 Dependency strategy
Separated dependency layers:
- `requirements.txt` — local / full environment
- `requirements.dev.txt` — dev utilities
- `requirements.docker.txt` — minimal runtime deps (Docker)

This keeps Docker images small and predictable.

## 🚀 Quick start (local, no Docker)
```bash
git clone https://github.com/artemxdata/Car-Damage-Assessment-AI.git
cd Car-Damage-Assessment-AI

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
# Open: http://localhost:8501
```

##  🐳 **Docker (recommended)**

**Build & run**  
```bash
docker build -t car-damage-ai:cpu .  
docker run --rm -p 8501:8501 car-damage-ai:cpu  
```
##  **Docker Compose**  
```bash
docker-compose up --build  
```
## 🧠 Runtime vs Source Architecture (Important Note)

**The Docker setup intentionally runs a minimal runtime image.**

**Core system intelligence — including:**

- agentic decision logic
- policy evaluation (YAML)
- SOP evidence (Markdown)
- decision trace and human override mechanisms

—is part of the source code and is fully executed inside the container at runtime.

**Development tooling, experimentation utilities, and optional LLM integrations are intentionally kept outside the runtime image to keep deployments:**

- lightweight
- deterministic
- production-aligned

**This separation mirrors real-world enterprise deployment practices, where runtime environments remain minimal while decision logic stays explicit, traceable, and auditable.**

---

**Services:**  
- app — Streamlit UI + decision engine  

**Ports:**  
- 8501 — Web UI  

---

⚙ **Configuration**

**Environment variables (optional):**  
LLM_BASE_URL=  
LLM_API_KEY=  
LLM_MODEL=  
CONFIDENCE_THRESHOLD=0.5  

LLM can be fully disabled without breaking the system.

---

## 📁 **Project structure**

```bash
Car-Damage-Assessment-AI/  
├── app.py  
├── agentic/  
├── policies/  
├── knowledge/  
├── docs/  
│   └── screenshots/  
├── models/  
├── data/  
├── outputs/  
├── Dockerfile  
├── docker-compose.yml  
├── requirements.txt  
├── requirements.dev.txt  
├── requirements.docker.txt  
└── README.md  
```

##  🧪 **What this is (and is not)**

**This is:**  
- a serious internal POC  
- a decision-centric architecture demo  
- a strong product & UX prototype  

**This is NOT:**  
- a production insurance system  
- a fully trained CV model  
- a replacement for human judgment  

---

##  📈 **Future directions**

- API-first architecture  
- model-backed CV inference  
- multi-image / video ingestion  
- policy versioning & analytics  
- audit log persistence  
- PDF / claims system export  

---

##  📄 **License**

MIT License  

---

##  👤 **Author**

Artem (@artemxdata) — AI / Agentic Systems Engineering  
Focused on high-trust, explainable AI systems
