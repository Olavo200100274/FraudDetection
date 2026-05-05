# CONTEXT_PRIMER.md
**Purpose**: Paste this as the FIRST message in a new Claude Code session to reconstruct full project context.  
**Recommended model**: Claude Sonnet 4.6 (1M context) for SoA/thesis writing.  
**Switch to**: Opus 4.7 (1M) only for critical decisions (structure review, argument validation).

---

## ===== PASTE BELOW INTO NEW SESSION =====

You are continuing a long-running MSc dissertation project. Before doing anything else, read these 4 files in order:

1. `.claude_context/PROJECT_STATE.md` — ML pipeline, datasets, models, all experimental results
2. `.claude_context/THESIS_STATE.md` — Overleaf chapter status, thesis contributions, advisor directives
3. `.claude_context/REFERENCES_STATE.md` — All 41 BibTeX entries, reference folder structure, what's curated
4. `.claude_context/PLAN.md` — The 16-step sequential plan; check current position

After reading all 4 files, confirm you understand:
- What the project is (fraud detection MSc thesis, datasets ULB + BAF, 6 models, 7 strategies)
- What's done (Methodology ✅, Results ✅, references curated ✅)
- What's next (current step in PLAN.md)
- The advisor's central framing (Maryam Abbasi: "show why simple LightGBM beats sophisticated FT-Transformer")
- The article target (IEEE TPAMI, after thesis)

Then immediately proceed with the current step from PLAN.md without waiting for further instructions.

**Working directory**: `d:\Faculdade\Mestrado em Informática\2 ano\Dissertação\Fraud Detection`  
**Key files**:
- Thesis: `Overleaf/Chapters/` (2-State of the Art.tex is main target)
- References: `Overleaf/references.bib` (41 entries) + `Overleaf/references/General/` + `BAF/` + `ULB/`
- Code: `src/` (main.py, main_transformer.py, models/, attention_analysis.py)
- Results: `results/baf_base/` + `results/ulb_2013/`
- Context files: `.claude_context/` (this folder — update PLAN.md when steps complete)

**Rule**: After every significant step completed, update `.claude_context/PLAN.md` to reflect new current position and `.claude_context/THESIS_STATE.md` if chapter status changes.

## ===== END OF PASTE =====
