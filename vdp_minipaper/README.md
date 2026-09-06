# Van der Pol PINN mini-paper

Overleaf-ready LaTeX source for the mini-paper drawn from the joint Notion record
("Physics-informed neural networks on the van der Pol oscillator — the joint record",
rebuilt 2026-07-16). Every number traces to that page and, through it, to the seeded
scripts and saved results of github.com/GeorgeWilliam1999/Van_Der_Pole (repository
state `981bdeb`).

## Use with Overleaf

Upload `vdp_minipaper.zip` (built next to this folder's contents) via
**New Project → Upload Project**. Compiles with pdfLaTeX; bibliography via BibTeX
(`natbib`, `unsrtnat`). No exotic packages: geometry, amsmath, graphicx, booktabs,
caption, natbib, xcolor, hyperref. 30 pages as of 2026-07-18.

Revision history: 2026-07-16 second pass made all calculation details verbose
(worked metric example, reference-verification numbers, Riemann-sum causal
weights, quadrature derivation of Runge–Kutta, order/A-stability definitions,
parameter counts, success-rate and label-cost arithmetic). 2026-07-18 third
pass rewrote the prose in George's personal style (per his MSc/case-study
samples): first-person active voice for design decisions, a roadmap paragraph
closing the introduction, procedures as numbered lists (reference
verification, causal training algorithm, hypotheses H1/H2, scheme guardrails,
selection protocol, order/A-stability), rhetorical-question transitions, and
five dry footnotes. Results statements remain neutral throughout. A fourth
pass the same day removed all remaining em-dashes and residual stock
phrasing from the prose (verified: numeric content byte-identical to the
previous revision), and folded in George's own edits (\tableofcontents,
plain author line). Now 31 pages. 2026-08-19 fifth pass added the classical explicit RK6 versus implicit Gauss–Legendre comparison (Figures 4–5, Table 2); 35 pages. **2026-09-06 sixth pass: the failure study folded in** (four new subsections at the end of the continuous-time route: the supervised control, the family of spurious solutions and its 1/T price, eight objectives trained to convergence, density and horizon for pseudo-time stepping; 11 new figures, two new tables, three new references; abstract, contributions, diagnosis, density, route summary, discussion, future work and conclusion revised to match). Every pre-existing number was re-verified against the repository result files that day (all match to the printed precision; figures byte-identical to their committed sources) and the new numbers trace to the Notion write-up 3d25d544-b9d9-81fe and commits c00c312 / e56b168 / b4ce164. Now 48 pages. George's steer of the same day: this document is the van der Pol thesis chapter, to be condensed into a paper later, so nothing was trimmed.

## Contents

- `main.tex` — the paper (single file).
- `references.bib` — 12 entries.
- `figures/` — 37 PNGs, copied from the study `figures/` folders of the repository
  (24 at state `981bdeb`, the classical pair at `dacac09`, and the 11 failure-study figures
  `fig_control_*`, `fig_arms_*`, `fig_axes_*` from commits `b4ce164`, `c00c312` and `e56b168`).

## What this revision covers (2026-07-16, superseding the 2026-07-13 draft)

- The agreed capped error metric ρ = min(1, ‖y−ŷ‖²/‖y‖²) throughout (Eq. 3), with
  median-led reporting; all whiskers defined as min–max over seeds.
- Continuous route complete: horizon sweep, causal weighting + 4× budget extension,
  **network size × starting state (600 runs)**, **collocation density (72 runs, 64×
  range)**, **collocation placement (24 runs)** — all resource axes measured null;
  success ordered by the starting state (12.7% at two periods at the 1% bar, 0% at four).
- Discrete route complete: stage sweep vs exact scheme, capped-metric table and test
  set, **architecture grid + ten-seed replication (optimisation floor, Spearman
  +0.974)**, **the seed-selection protocol and the selected 4×32 network** (six periods
  at 2.0e-3, flat to thirty), **Route A vs Route B at 10×**, and the
  **data-supervised baseline** (physics within 2.1× of perfect labels at zero data cost).
- Future-work list pruned: the collocation study and the floor attribution proposed in
  the 2026-07-13 draft are now results sections, not proposals.

## Before submission (TODOs left in the source)

- Affiliation in `\author` (marked TODO).
- Acknowledgements block (commented out).
- Venue-specific class/format if required.

## Provenance note

The paper cites no code and quotes no commit hashes in the body, per design; the Data
Availability section links the repository. The source-page corrections of 2026-07-13
(the source work's true reported accuracy; its own capacity attribution and appendix
error floor) are incorporated — do not reintroduce the older phrasing when editing.
