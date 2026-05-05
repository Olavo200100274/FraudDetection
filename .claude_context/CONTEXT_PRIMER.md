# CONTEXT_PRIMER.md
**Purpose**: Paste this as the FIRST message in a new Claude Code session to reconstruct full project context.  
**Recommended model**: Claude Sonnet 4.6 (1M context) for SoA/thesis writing.  
**Switch to**: Opus 4.7 (1M) only for critical decisions (structure review, argument validation).

---

## How a new session works

The new Claude Code session will have **two layers of context** automatically:

1. **Auto-memory** (loaded automatically): files at `C:\Users\olavo\.claude\projects\d--Faculdade-Mestrado-em-Inform-tica-2-ano-Disserta--o-Fraud-Detection\memory\` — already include `user_profile.md`, `project_overview.md`, `feedback_dataset_names.md`, `project_methodology_review.md`, `project_thesis_quality_and_framing.md`, `MEMORY.md`. These contain the high-level project context AND the agreed framing direction from Maryam.

2. **`.claude_context/`** (the prompt below loads): granular structured files I created. These reinforce auto-memory and add specifics (code paths, table cells, chapter status).

The prompt below tells the new Claude to READ the `.claude_context/` files BEFORE doing anything. Auto-memory loads silently anyway.

---

## ===== PASTE BELOW INTO NEW SESSION =====

You are continuing a long-running MSc dissertation project. Auto-memory has already loaded high-level context. Before doing anything else, read these 4 files in order to load operational detail:

1. `.claude_context/PROJECT_STATE.md` — ML pipeline, datasets, models, all experimental results, code structure
2. `.claude_context/THESIS_STATE.md` — Overleaf chapter status, thesis contributions, Maryam's directives
3. `.claude_context/REFERENCES_STATE.md` — All 41 BibTeX entries, reference folder structure, citation counts
4. `.claude_context/PLAN.md` — The 16-step sequential plan; check current position

After reading all 4 files, briefly confirm (in 5-6 lines) that you understand:
- What the project is (fraud detection MSc thesis, datasets ULB + BAF, 6 models, 7 strategies)
- What's done (Methodology ✅, Results ✅, references curated ✅)
- What's next (current step in PLAN.md)
- The advisor's central framing (Maryam Abbasi: "show why simple LightGBM beats sophisticated FT-Transformer")
- The article target (IEEE TPAMI, Part 2 — after thesis)

Then immediately proceed with the current step from PLAN.md without waiting for further instructions.

**Working directory**: `d:\Faculdade\Mestrado em Informática\2 ano\Dissertação\Fraud Detection`

**Key locations**:
- Thesis chapters: `Overleaf/Chapters/` (next target: `2-State of the Art.tex`)
- BibTeX file: `Overleaf/references.bib` (41 entries)
- PDF library: `Overleaf/references/General/` + `BAF/` + `ULB/`
- Code: `src/` (main.py, main_transformer.py, models/, attention_analysis.py)
- Results: `results/baf_base/` + `results/ulb_2013/` (raw artefacts)
- Pre-generated figures: `results_thesis/figures/baf/` + `results_thesis/figures/ulb/`
- Context files (THIS FOLDER): `.claude_context/` — UPDATE these as work progresses

**Maintenance rule**: After every significant step completed, update:
- `.claude_context/PLAN.md` (mark step ✅, update "current position")
- `.claude_context/THESIS_STATE.md` (chapter status when chapters are written)
- `.claude_context/REFERENCES_STATE.md` (if new BibTeX entries are added)

Git: commit and push after each major milestone. Never amend commits without asking.

## ===== END OF PASTE =====

---

## Practical session tips

- **First action after the new session reads the files**: it should ask you to confirm the SoA structure proposed in `THESIS_STATE.md` — that's Step 2 of the plan
- **If the new session forgets the framing**: point it back to `THESIS_STATE.md` "Advisor directives" section
- **If it suggests scope creep** (e.g., "let's also test Periodic Embeddings"): point to `PROJECT_STATE.md` "Foggy Vision" finding and `THESIS_STATE.md` advisor directive — the experimental work is DONE
- **For writing the SoA**: Sonnet 4.6 (1M) is ideal — fast, capable, cheaper per turn
- **For final review/argument validation**: switch to Opus 4.7 (1M)
