# Hackathon Win Checklist

## Phase 1 — Setup (Person A on Mac)
- [ ] Clone both repos (`functiongemma-hackathon` + `cactus`)
- [ ] Run `cd cactus && source ./setup && cd ..`
- [ ] Run `cactus build --python`
- [ ] Run `cactus download google/functiongemma-270m-it --reconvert`
- [ ] Run `cactus auth` with key from cactuscompute.com
- [ ] Run `pip install google-genai`
- [ ] Set `export GEMINI_API_KEY="..."` from Google AI Studio
- [ ] Run `python main.py` — confirm no errors
- [ ] Run `python benchmark.py` — record baseline score

---

## Phase 2 — Core Algorithm (Person B writes → Person A tests)
- [x] Add `classify_query()` — pre-routes hard (multi-call) cases directly to cloud
- [x] Add `validate_local_result()` — rejects bad local outputs before returning
- [x] Rewrite `generate_hybrid()` with 3-layer strategy
- [x] Add `tool_rag_top_k=2` to `cactus_complete` call
- [x] Fix deprecated `gemini-2.0-flash` → `gemini-2.0-flash-001`
- [ ] Person A: run `python benchmark.py` and report scores per difficulty tier

---

## Phase 3 — Iterate on Thresholds (based on benchmark feedback)
- [ ] Check easy F1 — if < 0.90, lower confidence threshold in `validate_local_result` for easy
- [ ] Check medium F1 — if < 0.80, tune the `medium_local` confidence threshold (currently 0.70)
- [ ] Check hard F1 — if < 0.75, review `classify_query` signals (add more multi-action keywords)
- [ ] Check on-device ratio — if too low for easy/medium, trust local more aggressively
- [ ] Re-run benchmark after each tweak, compare total score

### Threshold tuning guide
| Scenario | Fix |
|----------|-----|
| Easy cases going to cloud | Lower `confidence < 0.55` to `0.45` in `validate_local_result` |
| Medium cases going to cloud too often | Lower medium threshold from `0.70` to `0.60` |
| Hard cases staying local (wrong) | Add more signals to `classify_query` |
| Cloud being called for obvious single-action queries | Raise `signal_count >= 1` requirement in hard detection |

---

## Phase 4 — Submit to Leaderboard
- [ ] Score beats baseline (~40-50%) — submit first version
- [ ] Run `python submit.py --team "YourTeamName" --location "YourCity"`
- [ ] Check live leaderboard: https://cactusevals.ngrok.app
- [ ] Submit again after each improvement (max 1x per hour)
- [ ] **Target: Top 3 on leaderboard before building UI**

---

## Phase 5 — UI / Product Demo (only after Phase 4 target hit)

### App idea options (pick one):
- [ ] **Option A — Smart Mobile Assistant UI**: A chat-style web app where you type commands (e.g. "Text Emma and check weather") and it shows which calls were made, from which source (on-device/cloud), and how fast. Visualizes the routing decision in real time.
- [ ] **Option B — Voice-to-Action Demo** (Rubric 3 bonus): Record audio → Whisper transcribes locally → `generate_hybrid` routes → show executed actions. Run entirely on-device for single actions.
- [ ] **Option C — Dashboard**: Live benchmark runner with charts showing F1, speed, on-device ratio per difficulty tier as you tune thresholds.

### If building Option A or B:
- [ ] `pip install sounddevice scipy` (for voice)
- [ ] `cactus download whisper-small` (for voice)
- [ ] Build simple UI (Streamlit or plain HTML+Flask)
- [ ] Wire up `generate_hybrid` as the backend
- [ ] Show routing decision and timing visually
- [ ] Test end-to-end with live demo cases

---

## Phase 6 — Qualitative Judging Prep
- [ ] **Rubric 1** (routing algorithm): Prepare 2-min explanation of 3-layer strategy — pre-routing, tool RAG, output validation
- [ ] **Rubric 2** (end-to-end product): Have a live product demo ready that calls real functions
- [ ] **Rubric 3** (voice): If time permits, show voice → transcribe → tool call pipeline
- [ ] Rehearse demo with a hard multi-call example (e.g. "Text Bob hi and check weather in London")

---

## Key Numbers to Hit
| Metric | Target |
|--------|--------|
| Easy F1 | ≥ 0.95 |
| Medium F1 | ≥ 0.85 |
| Hard F1 | ≥ 0.80 |
| On-device ratio | ≥ 0.60 |
| Avg time (easy/medium) | < 500ms |
| Total score | > 0.75 |

---

## Quick Reference
```bash
python benchmark.py          # test score
python submit.py --team "X" --location "Y"   # submit (1x per hour)
python main.py               # quick sanity check
```
Leaderboard: https://cactusevals.ngrok.app
