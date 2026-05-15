# CONTEXT_PRIMER.md
**Purpose**: Paste this as the FIRST message in a new Claude Code session to reconstruct full project context.
**Last updated**: 2026-05-14 (after Article 2 deep-review pass — commit `6c25574`)
**Recommended model**: Claude Sonnet 4.6 (1M context) for routine work and follow-up edits. Switch to **Opus 4.7** (1M) for critical decisions (Maryam-feedback integration, structural rewrites, argument validation).

---

## How a new session works

The new Claude Code session will have **two layers of context** automatically:

1. **Auto-memory** (loaded automatically): files at `C:\Users\olavo\.claude\projects\d--Faculdade-Mestrado-em-Inform-tica-2-ano-Disserta--o-Fraud-Detection\memory\` — already include `user_profile.md`, `project_overview.md`, `feedback_dataset_names.md`, `project_methodology_review.md`, `project_thesis_quality_and_framing.md`, `MEMORY.md`. These contain the high-level project context AND the agreed framing direction.

2. **`.claude_context/`** (the prompt below loads): granular structured files. These reinforce auto-memory and add specifics (chapter status, results, commits, current position).

The prompt below tells the new Claude to READ the `.claude_context/` files BEFORE doing anything. Auto-memory loads silently anyway.

---

## ===== PASTE BELOW INTO NEW SESSION =====

You are continuing a long-running MSc dissertation project. Auto-memory has already loaded high-level context. Before doing anything else, read these 4 files in order to load operational detail:

1. `.claude_context/PROJECT_STATE.md` — ML pipeline, datasets, models, all experimental results, code structure
2. `.claude_context/THESIS_STATE.md` — Overleaf chapter status, thesis contributions, advisor directives, Articles 1 & 2 status
3. `.claude_context/REFERENCES_STATE.md` — BibTeX entries per artefact (thesis / Article 1 / Article 2)
4. `.claude_context/PLAN.md` — The 20-step sequential plan; check current position

After reading all 4 files, briefly confirm (in 6–8 lines) that you understand:
- What the project is (MSc thesis on fraud detection under extreme imbalance, ULB + BAF, 6 models, 7 imbalance strategies, 4 threshold strategies)
- **State of the thesis**: COMPLETE (all 7 chapters; LaTeX layout polished with minipage approach; commit history through `6c25574`)
- **State of Article 1**: COMPLETE first draft (Expert Systems with Applications target; "Threshold Selection as a Critical Design Choice..."); awaiting Maryam review (Step 14)
- **State of Article 2**: COMPLETE reframed draft (Applied Soft Computing target; "Why Simple Tree-Based Models Suffice... via SHAP–Attention Convergence and Cross-Domain Transfer"); awaiting Maryam review (Step 20)
- **What's next**: depends on Maryam's feedback. If no feedback yet, the project is in a "wait" state. Do NOT start new structural work on Articles 1 or 2 without explicit instruction from the user.
- **Advisor directive (Maryam Abbasi, Discord 2026-05-04)**: "show why a simple LightGBM outperforms the transformer." After data-honesty review, this was refined to "show why simple tree-based models suffice" — the 5-pillar argument now lives in Article 2 §5.5.

Then ask the user what they want to work on. Do not start writing or editing without instruction — both articles are awaiting review and any change risks pre-empting reviewer feedback.

**Working directory**: `d:\Faculdade\Mestrado em Informática\2 ano\Dissertação\Fraud Detection`

**Key locations**:
- Thesis chapters: `Overleaf/Chapters/` (all complete)
- Thesis BibTeX: `Overleaf/references.bib` (43 entries)
- Article 1 (Paper 1 — practitioner / threshold-centric, ESWA): `Article 1/main.tex` + `Article 1/cas-refs.bib` (47 entries) + `Article 1/figs/`
- Article 2 (Paper 2 — mechanistic / "Simple Trees Suffice", ASC): `Article 2/main.tex` + `Article 2/cas-refs.bib` (49 entries) + `Article 2/figs/`
- Code: `src/` (main.py, attention_analysis.py, generate_pr_curves_threshold.py, generate_crossdomain_chart.py, generate_smote_asymmetry_chart.py, generate_convergent_validity_chart.py, ...)
- Results: `results/baf_base/` + `results/ulb_2013/` (raw artefacts; tracked via Git LFS)
- Pre-generated figures (thesis): `Overleaf/figures/baf/` + `Overleaf/figures/ulb/`
- Context files (THIS FOLDER): `.claude_context/` — **UPDATE these whenever a step is closed**

**Maintenance rule**: After every significant step completed, update:
- `.claude_context/PLAN.md` (mark step ✅, update "current position")
- `.claude_context/THESIS_STATE.md` (chapter or article status when content changes)
- `.claude_context/REFERENCES_STATE.md` (if BibTeX entries are added or removed)
- `.claude_context/PROJECT_STATE.md` (if new experimental data or scripts are introduced)
- Memory files at `~/.claude/projects/<this-project>/memory/` (only the few that need a project-state hook, e.g. `project_overview.md`)

**Git policy**: commit and push after each milestone. NEVER amend commits without asking. NEVER force-push to main. Use Co-Authored-By trailer in commits when the user requests one.

## ===== END OF PASTE =====

---

## Current state snapshot (2026-05-14)

### Repository
- Remote: `https://github.com/Olavo200100274/FraudDetection.git`
- Branch: `main`
- Last commit: `6c25574` — *Article 2 deep-review pass: fix 7 factual inconsistencies*
- Commit history (recent → older):
  - `6c25574` Article 2 deep-review pass: fix 7 factual inconsistencies
  - `1782c46` Step 19.5: Reframe Article 2 to "Simple Trees Suffice" + venue switch
  - `9b2d912` Steps 15–19: Article 2 full first draft (mechanistic paper)
  - `c5646e8` Article 1: pre-submission tweaks (decision framework framing + Future Work)
  - `bd02145` Article 1: align section 4 prose with new PR-curves figure
- Git LFS active for: `*.joblib`, `*.pt`, `*.npy`, `datasets/*.csv`

### What's done
- Steps 1–8: thesis writing **complete**
- Steps 9–13.5: Article 1 draft + polish **complete**; submitted to Maryam for review (Step 14 pending)
- Steps 15–19: Article 2 first draft **complete**
- Step 19.5: Article 2 reframe + figure overhaul + deep-review pass **complete** (Step 20 review pending)

### What's pending
- Step 14: Maryam reviews Article 1 → iterate → submit to Expert Systems with Applications
- Step 20: Maryam reviews Article 2 → iterate → submit to Applied Soft Computing
- (Both reviews are out of our hands — the project is in a "wait" state until feedback comes back)

### What is OUT of scope without explicit instruction
- Rewriting any section of Article 1 or Article 2 before Maryam's review (risks pre-empting her feedback)
- Adding new experimental runs (all experiments are complete; new data only if a reviewer demands it)
- Re-running the SHAP / attention analysis (artefacts in `results/` are the authoritative source)
- Touching the thesis chapters (frozen as of `6c25574`; the thesis is the basis for both papers)

### What MIGHT come up during the wait
- Maryam's feedback on Article 1 → integrate revisions, regenerate PDF, commit
- Maryam's feedback on Article 2 → same
- A request to copy updated `main.tex` / `cas-refs.bib` / `figs/` to Overleaf for the user to compile
- An external review (e.g. `/ultrareview`) requested by the user — read the latest commit and respond to suggestions
- The user wanting to switch focus to a related side-project (e.g. preparing slides for the defence, drafting a cover letter for ESWA, writing a short blog post on "Foggy Vision")

---

## Practical session tips

- **Default first action after the new session reads the files**: confirm understanding (6–8 lines) and then ASK the user what they want to do. Do not assume the user wants to keep writing.
- **If the user provides Maryam-feedback content**: read the relevant article carefully BEFORE editing; integrate revisions surgically with `Edit` (preserve existing structure unless feedback explicitly calls for rewrite).
- **If the user asks for a review of the article**: refer to the deep-review checklist used in commit `6c25574` (numerical claims vs data sources; figure captions vs actual figures; internal consistency Abstract ↔ §1 ↔ §5 ↔ §6; bib coverage).
- **If the user starts a new artefact (new article, new chapter, new analysis)**: that is a new sequence of steps — extend `PLAN.md` to a new Part rather than overloading Part 3.
- **For Maryam-feedback integration**: Opus 4.7 — argument validation matters more than throughput. For routine edits / Overleaf-fix passes / status-file updates: Sonnet 4.6.
- **Don't trust round numbers without checking**: the Article 2 deep-review caught an inflated "40× training advantage" claim that was actually ~22×. Always cross-check headline figures against the source tables/JSON.
- **Variant descriptions warning**: the BAF Variants I–V have specific definitions in Jesus et al. 2022 (group-size disparity, prevalence disparity, separability disparity, training bias). Do NOT invent per-variant descriptions; defer to the original paper.
