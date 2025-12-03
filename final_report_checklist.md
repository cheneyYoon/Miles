# Final Report Completion Checklist
**Quick Reference for Implementation Plan | Track Progress Here**

---

## Phase 1: Content Audit & Asset Preparation ☐

### Content Mapping
- [ ] Audit project_proposal.md for reusable sections
- [ ] Audit mid_report.md for results and analysis
- [ ] Identify missing content gaps
- [ ] Create content reuse matrix

### Figure & Table Extraction
- [ ] Run `python scripts/generate_report_figures.py`
- [ ] Extract Figure 1: System Architecture diagram
- [ ] Extract Figure 2: Training curves (loss + AUROC)
- [ ] Extract Figure 3: Confusion matrix
- [ ] Check notebooks for data distribution visualizations
- [ ] Create Figure 4: Baseline vs Primary comparison (optional)
- [ ] Verify all figures are 300+ DPI
- [ ] Export Table 1: Dataset statistics (from mid-report)
- [ ] Export Table 2: Baseline results (from mid-report)
- [ ] Export Table 3: Primary model results (from mid-report)

### LaTeX Setup
- [ ] Obtain APS360 LaTeX template
- [ ] Set up Overleaf project or local LaTeX environment
- [ ] Create 9-section document structure
- [ ] Configure BibTeX bibliography
- [ ] Test compilation with dummy content

### Page Budget Planning
- [ ] Allocate page limits per section (see implementation plan)
- [ ] Set up page tracking workflow
- [ ] Identify "nice-to-have" content for potential cuts

---

## Phase 2: Critical Content Development ☐

### 2.1 New Data Evaluation (HIGHEST PRIORITY - 10 points)
- [ ] Choose evaluation strategy (temporal split recommended)
- [ ] Load full dataset and check date/platform columns
- [ ] Create new data split (last 20% by date or separate platform)
- [ ] Load best model checkpoint from `experiments/checkpoints/`
- [ ] Run evaluation on new data
- [ ] Generate results table comparing val vs. new data
- [ ] Write 2-3 paragraphs analyzing:
  - [ ] Performance comparison (metrics held up or degraded?)
  - [ ] Generalization success/failure modes
  - [ ] Surprising findings or insights
- [ ] Document methodology clearly (must show data was "new")

**Expected Deliverable:**
```
Table 4: Evaluation on New Data (Temporal Split)
Metric      | Validation | New Data | Change
AUROC       | 0.855      | 0.XXX    | ±X.XX
Accuracy    | 0.802      | 0.XXX    | ±X.XX
F1 Score    | 0.72       | 0.XXX    | ±X.XX
```

### 2.2 Discussion Section Expansion
- [ ] Write "Performance Assessment" (150 words)
  - [ ] Quantify results vs. targets
  - [ ] Explain why model succeeds
- [ ] Write "Surprising/Interesting Findings" (200 words)
  - [ ] Feature ablation insights
  - [ ] Temporal patterns discovered
  - [ ] Failure mode analysis
- [ ] Write "Lessons Learned" (150 words)
  - [ ] Feature normalization critical importance
  - [ ] Loss balancing challenges
  - [ ] Training optimizations (mixed precision, early stopping)
- [ ] Write "Limitations & Future Work" (150 words)
  - [ ] Visual features not utilized (ResNet implemented but not trained)
  - [ ] English-only constraint
  - [ ] Temporal drift requiring retraining
  - [ ] SHAP interpretability missing
- [ ] Write "Broader Implications" (100 words)
  - [ ] Algorithm transparency
  - [ ] Creator equity considerations
- [ ] Total: 600-800 words, aim for "exceeds expectations" (8/8)

### 2.3 Ethical Considerations Refinement
- [ ] Expand "Data Ethics" paragraph
  - [ ] No PII, public data only
  - [ ] ToS compliance
  - [ ] Dataset bias (English, Western)
- [ ] Expand "Model Limitations" paragraph
  - [ ] Temporal drift
  - [ ] Class imbalance effects
  - [ ] Missing modalities (visual/audio)
- [ ] Add "Negative Use Cases" paragraph
  - [ ] Clickbait optimization risk
  - [ ] Content homogenization risk
  - [ ] Mitigation strategies
- [ ] Add "Bias Analysis" paragraph
  - [ ] Popularity bias (engagement_rate dominance)
  - [ ] Cold start problem for new creators
  - [ ] Recommendations for fairness
- [ ] Total: ~300 words, 2 paragraphs

---

## Phase 3: Document Assembly & Refinement ☐

### 3.1 LaTeX Content Integration

#### Preamble & Setup
- [ ] Configure document class and packages
- [ ] Set margins (1 inch all sides)
- [ ] Add title, author, contact info

#### Section 1: Introduction
- [ ] Import from project_proposal.md
- [ ] Expand to ~400 words
- [ ] Add forward reference to results
- [ ] Verify ML justification is clear

#### Section 2: Illustration
- [ ] Insert Figure 1 (architecture diagram)
- [ ] Write detailed caption (50-100 words)
- [ ] Cross-reference in introduction

#### Section 3: Background & Related Work
- [ ] Import 5 papers from project_proposal.md
- [ ] Tighten to ~500 words
- [ ] Verify all citations in BibTeX
- [ ] Connect to your specific approach

#### Section 4: Data Processing
- [ ] Import from mid_report.md (lines 30-63)
- [ ] Insert Table 1 (dataset statistics)
- [ ] Insert data distribution figure
- [ ] Verify reproducibility statements
- [ ] Target: 0.7 pages

#### Section 5: Architecture
- [ ] Merge proposal + mid-report architecture descriptions
- [ ] Reference Figure 1
- [ ] Include parameter count (110.5M total, 109.5M trainable)
- [ ] Specify loss function (0.95 CE + 0.05 MSE)
- [ ] Target: 0.5 pages

#### Section 6: Baseline Model
- [ ] Import baseline description from proposal
- [ ] Insert Table 2 (baseline results AUROC 0.488)
- [ ] Emphasize failure justifies deep learning
- [ ] Target: 0.3 pages

#### Section 7: Results
- [ ] **Subsection 7.1: Quantitative Results**
  - [ ] Insert Table 3 (model comparison)
  - [ ] Insert Figure 2 (training curves)
  - [ ] Highlight key metrics (AUROC 0.855, +14% vs target)
  - [ ] Target: 0.4 pages

- [ ] **Subsection 7.2: Qualitative Results**
  - [ ] Import success/failure analysis from mid-report
  - [ ] Insert Figure 3 (confusion matrix)
  - [ ] Add 2-3 specific example predictions
  - [ ] Target: 0.4 pages

- [ ] **Subsection 7.3: Evaluation on New Data**
  - [ ] Insert Table 4 (new data results)
  - [ ] Add methodology paragraph
  - [ ] Add analysis paragraph
  - [ ] Target: 0.3 pages

#### Section 8: Discussion
- [ ] Insert Phase 2.2 expanded discussion (600-800 words)
- [ ] Add cross-references to figures/tables
- [ ] Structure with clear flow
- [ ] Target: 0.6 pages

#### Section 9: Ethical Considerations
- [ ] Insert Phase 2.3 refined content (300 words)
- [ ] Connect to Discussion points
- [ ] Target: 0.3 pages

#### References
- [ ] Merge project_proposal + mid_report bibliographies
- [ ] Add new citations (e.g., temporal evaluation papers)
- [ ] Verify format consistency (IEEEtran or author-year)

### 3.2 Figure Integration & Formatting
- [ ] All figures compile correctly
- [ ] All figures have descriptive captions
- [ ] All figures referenced in text
- [ ] Consistent visual style (colors, fonts, size)
- [ ] All tables use booktabs package (\toprule, \midrule, \bottomrule)
- [ ] Number formatting consistent (0.855 not .855)

### 3.3 Page Limit Enforcement
- [ ] Check page count (target: ≤4.0 pages excluding refs)
- [ ] If over, apply compression strategies:
  - [ ] Tighten prose (remove filler, passive voice)
  - [ ] Use LaTeX space-saving tricks (compact sections, small captions)
  - [ ] Prioritize content (cut lowest-value details)
  - [ ] Last resort: remove one figure or abbreviate Discussion
- [ ] Re-check page count after each compression pass
- [ ] Final verification: exactly 4 pages or less

---

## Phase 4: Review & Polish ☐

### 4.1 Technical Accuracy Audit
- [ ] AUROC 0.855 consistent across all mentions
- [ ] All figure references point to correct figures
- [ ] All table references point to correct tables
- [ ] Equations formatted correctly (loss function, engagement_velocity)
- [ ] Hyperparameters consistent (lr=2e-5, batch=32, epochs=15→stopped at 7)
- [ ] No contradictions (dataset size, splits, etc.)
- [ ] Cross-reference chain complete (Intro→Methods→Results→Discussion)

### 4.2 Rubric Compliance Check
Use the grading matrix in implementation plan:
- [ ] Introduction (2 pts): Clear goal, motivation, ML justification?
- [ ] Illustration (2 pts): Architecture diagram clear and accessible?
- [ ] Background (2 pts): 5+ papers cited with context?
- [ ] Data Processing (4 pts): Sources + cleaning + stats + examples?
- [ ] Architecture (4 pts): Reproducible description?
- [ ] Baseline (4 pts): Clear description + comparison?
- [ ] Quantitative Results (4 pts): Insightful metrics?
- [ ] Qualitative Results (4 pts): Sample outputs + failure analysis?
- [ ] New Data Eval (10 pts): Truly new data + strong performance?
- [ ] Discussion (8 pts): Insightful + specific + surprising findings?
- [ ] Ethics (2 pts): Thoughtful consideration of bias/limits?
- [ ] Difficulty/Quality (6 pts): Challenging problem + exceeds expectations?
- [ ] Grammar (8 pts): Clear, concise, error-free?
- [ ] **Self-graded total: ___/64 (target: 57+ for A-range)**

### 4.3 Grammar & Clarity Pass
- [ ] Run spell check (aspell or Grammarly)
- [ ] Run grammar check (LanguageTool or Grammarly)
- [ ] Run LaTeX linter (lacheck, chktex)
- [ ] Verify consistent tense (past for methods/results, present for discussion)
- [ ] Verify consistent voice (first-person "we" or passive)
- [ ] Define all acronyms on first use
- [ ] Check number formatting consistency
- [ ] Read entire paper aloud for flow

### 4.4 Pre-Submission Checklist
- [ ] PDF compiles without errors
- [ ] All figures render correctly (no broken images)
- [ ] Page count ≤4 pages (excluding references)
- [ ] References formatted correctly per template
- [ ] All citations have bibliography entries
- [ ] No TODO comments or placeholders
- [ ] File named correctly per submission instructions
- [ ] Metadata correct (student number, email, links)
- [ ] Backup saved (Overleaf history + local git)
- [ ] Final PDF downloaded and verified

### Optional: Peer Review
- [ ] Ask classmate to read intro + conclusion (2-min test)
- [ ] Confirm main contribution is clear
- [ ] Check for confusing jargon or unexplained concepts
- [ ] Incorporate feedback

### Final Submission
- [ ] Submit to Quercus before deadline
- [ ] Verify submission uploaded correctly
- [ ] Save submission confirmation screenshot
- [ ] Commit to git: `git commit -m "Final report submitted - AUROC 0.855"`
- [ ] Celebrate 🎉

---

## Quick Status Check

**Phase Completion:**
- Phase 1 (Asset Prep): ☐ Not Started | ☐ In Progress | ☐ Complete
- Phase 2 (Critical Content): ☐ Not Started | ☐ In Progress | ☐ Complete
- Phase 3 (Assembly): ☐ Not Started | ☐ In Progress | ☐ Complete
- Phase 4 (Review): ☐ Not Started | ☐ In Progress | ☐ Complete

**Critical Path Items (Must Complete):**
- [ ] New Data Evaluation (10 pts) - Highest priority
- [ ] Discussion Section (8 pts) - Second priority
- [ ] Page limit ≤4 pages - Hard constraint
- [ ] All figures included and referenced

**Estimated Time Remaining:** ___ hours (see implementation plan timelines)

**Target Submission Date:** ____________

**Current Blockers:** _________________________________

---

## Emergency Quick Reference

**If running out of time:**
1. Prioritize new data evaluation (10 points at stake)
2. Use existing test set as "new data" if temporal split fails
3. Reduce Discussion to 400 words (accept 6/8 instead of 8/8)
4. Use hand-drawn architecture diagram (acceptable per rubric)
5. Accept 4.2 pages (minor penalty better than missing content)

**Minimum viable product = 6 hours:**
- Phase 1: 1.5 hours (setup + assets)
- Phase 2.1: 30 min (new data eval with existing test set)
- Phase 2.2-2.3: 1 hour (discussion + ethics)
- Phase 3: 2 hours (LaTeX integration)
- Phase 4: 1 hour (review + polish)
- **Expected grade: 52-56/64 (81-87%, B+ to A-)**

**Full execution = 8-12 hours:**
- Follow implementation plan phases completely
- **Expected grade: 58-61/64 (90-95%, A to A+)**

---

**Last Updated:** 2025-12-03
**Document Version:** 1.0
**Use this checklist to track progress systematically through the implementation plan**
