## Purpose of this file

This file holds a **revision prompt** for performing a writing-only pass on a journal
article. It is **not** part of the standard session bootstrap (`PROJECT_STATE.md`,
`THESIS_STATE.md`, `REFERENCES_STATE.md`, `PLAN.md`). It is only to be used when the
**user explicitly requests** an article-revision pass (e.g. *"revê o Article 1 com o
prompt da Maryam"*, *"faz a revisão de escrita do Article 2"*).

## How to use

1. The user must ask explicitly for the revision pass and name the target article.
2. Read the article's `main.tex`.
3. **Keep the original intact**: write the revised version to a new file (e.g.
   `main_revised.tex`) so the user can compare side by side and rename later.
4. Apply the prompt below verbatim — it is the prompt Maryam Abbasi authored.

## Per-article adaptation needed

The prompt below was written for **Article 1** (ESWA, threshold-centric). When applied
to another article, two sections must be adapted to that article's content:

- **Section 3 (repeated ideas to deduplicate)** — currently lists Article-1-specific
  ideas (fixed $\tau{=}0.5$ fails, threshold dominates, SMOTE = SMOTE+Tomek,
  threshold is first-order). For **Article 2** (ASC, "Simple Trees Suffice") the
  candidate over-repeated ideas to deduplicate would instead be: the 5-pillar argument
  in §5.5, Foggy Vision (H = 0.976), SHAP–Attention convergent validity, the ~20×
  training / ~37× tuning compute advantage.
- **Section 6 (conceptual framing)** — currently reframes Article 1 around
  *score-space behavior under extreme imbalance / probability compression /
  operating-point instability*. For **Article 2** the equivalent reframe would centre on
  *mechanistic evidence for tree-based sufficiency / convergent validity across
  architecturally distinct attribution estimators / dataset-ceiling vs. architectural
  headroom*.
- **Venue line at the top** — change "Expert Systems with Applications" to the target
  article's venue (Article 2: Applied Soft Computing).

The remaining sections (1, 2, 4, 5, 7–10) are generic writing-quality guidance and
apply unchanged to any article.

---

## Prompt (verbatim, as authored by Maryam)

You are revising a journal paper intended for Expert Systems with Applications (Elsevier).

Your task is NOT to change the experiments, tables, metrics, results, citations, figures, methodology, numerical values, or scientific claims.

Your task is ONLY to improve the WRITING, SCIENTIFIC NARRATIVE, and PRESENTATION QUALITY of the manuscript while preserving all technical content and LaTeX compatibility.

Important constraints:
- Do NOT invent new experiments.
- Do NOT add fake citations.
- Do NOT modify reported results or numbers.
- Do NOT remove any important methodological details.
- Preserve all LaTeX commands, labels, references, citations, equations, and tables.
- Keep the same section structure unless explicitly improving flow within sections.
- Maintain an academic tone suitable for Expert Systems with Applications.
- Return improved LaTeX-ready text.

Main revision goals:

1. Reduce “thesis-like” writing
The manuscript currently reads partially like a thesis chapter or technical report.
Rewrite sections into smoother scientific prose with stronger narrative flow.

2. Reduce excessive enumeration and list-style writing
Avoid too many:
- “First, Second, Third”
- repeated bullet-style transitions
- over-structured prose

Convert overly segmented passages into integrated analytical discussion where appropriate.

3. Reduce repetition
The following ideas are repeated too many times across the paper:
- fixed threshold τ=0.5 fails under imbalance
- threshold selection dominates model selection
- SMOTE and SMOTE+Tomek are identical
- threshold selection is first-order

Keep these findings, but avoid restating them repeatedly in nearly identical wording across:
- Introduction
- Results
- Discussion
- Conclusion

4. Improve scientific maturity of the prose
Avoid overly rhetorical or dramatic wording such as:
- “catastrophic”
- “destroying recall”
- “detect nothing”

Replace with more journal-appropriate scientific language such as:
- “severely degrades recall”
- “operationally unsuitable”
- “substantially reduces detection performance”

5. Increase mechanistic interpretation
The Discussion section currently partially repeats the Results section.

Rewrite the Discussion to focus more on:
- WHY the observed behavior occurs
- score compression under imbalance
- probability distribution effects
- operational implications
- architecture-specific behavior
- threshold-dependent operating geometry

and less on simply restating tables.

6. Improve conceptual framing
The paper should be framed not merely as:
“threshold tuning matters”

but more as:
- score-space behavior under extreme imbalance
- probability compression
- operating-point instability
- interaction between imbalance and threshold geometry

Without changing the actual scientific claims.

7. Improve transitions between paragraphs
Add smoother logical transitions between:
- related work subsections
- methodology subsections
- results interpretation paragraphs
- discussion subsections

8. Tighten the Introduction
The Introduction is slightly too dense and repetitive.
Reduce redundancy while preserving:
- motivation
- methodological gap
- contributions

Make the opening more engaging and concise.

9. Tighten the Conclusion
The Conclusion currently repeats too much from the Results and Discussion.

Rewrite it into:
- concise synthesis,
- broader implications,
- limitations,
- future directions,
with less repetition of numerical findings.

10. Preserve clarity
Even after tightening and improving flow:
- maintain technical precision,
- keep explicit methodological rigor,
- preserve readability.

Most important:
This is a WRITING and SCIENTIFIC POSITIONING revision, NOT a methodological rewrite.

Return revised LaTeX-ready text section by section.
Do not summarize.
Do not explain what you changed.
Only output the improved text.