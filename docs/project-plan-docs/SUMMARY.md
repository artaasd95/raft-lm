# Documentation Summary

This document explains what's in `docs/project-plan-docs/` and how it all fits together.

---

## What We've Built

A **complete operational framework** for research and development in Raft-LM. This isn't just documentation—it's a **research operating system** that ensures:

1. **Reproducibility**: Every experiment produces traceable artifacts
2. **Rigor**: Statistical testing, multiple seeds, baseline comparisons
3. **Efficiency**: Checklists prevent mistakes, workflows guide decisions
4. **Transparency**: Document both successes and failures

---

## The Documentation Structure

### 🚀 Getting Started

**`00-START-HERE.md`**
- First month playbook
- The core research loop (Formulate → Implement → Train → Evaluate → Decide)
- Quick answers to common questions
- Initial scope guidance
- What "good progress" looks like

**`README.md`**
- Navigation hub for all docs
- Quick task-based navigation
- Philosophy and principles
- Learning path for new contributors

**`QUICK-REFERENCE.md`**
- One-page summary of everything
- Checklists, decision trees, quick commands
- Print this and keep it visible
- Quick lookup for common tasks

### 📋 Operational Guides

**`01-RD-PHASES.md`**
- Long-term roadmap broken into phases (0-5)
- Clear "done when" criteria for each phase
- Priority guidance and decision framework
- Phase transition checklist
- Method evolution protocol (add/keep/modify/remove)

**`02-CHECKLISTS.md`**
- Copy-paste checklists for common tasks
- Training models, implementing losses/metrics
- Running experiments, evaluating performance
- Before committing code
- Preparing results for publication

**`03-ADD-A-MODULE.md`**
- Step-by-step guide for adding components
- Covers loss functions, risk metrics, training methods
- Testing and validation steps
- Integration into experiment framework
- Decision-making framework (keep/modify/remove)

**`04-RESEARCH-WORKFLOW.md`**
- How to turn questions into reproducible experiments
- Formulating hypotheses
- Designing experiment matrices
- Running and analyzing experiments
- Statistical testing and visualization
- Decision-making and documentation

**`05-EXPERIMENT-REVIEW.md`**
- Quality gates: when are results trustworthy?
- Three review levels (sanity, research-grade, publication-ready)
- Review checklist and decision tree
- Statistical testing guide
- Red flags and when to re-run

**`06-PERFORMANCE-PROTOCOL.md`**
- How to measure, track, and optimize performance
- Baseline protocol and regression detection
- Common bottlenecks and fixes
- Performance budgets and guidelines
- Optimization workflow

---

## How It All Fits Together

### The Research Loop (Core Workflow)

```
┌─────────────────────────────────────────────────────────┐
│  1. Formulate   →  2. Implement  →  3. Train            │
│       ↑                                    ↓             │
│  5. Decide      ←  4. Evaluate   ←─────────┘             │
└─────────────────────────────────────────────────────────┘
```

Each step has supporting documentation:

1. **Formulate**: `04-RESEARCH-WORKFLOW.md` helps design experiments
2. **Implement**: `03-ADD-A-MODULE.md` guides component development
3. **Train**: Run experiments following `02-CHECKLISTS.md`
4. **Evaluate**: `05-EXPERIMENT-REVIEW.md` defines quality standards
5. **Decide**: Make keep/modify/remove decisions based on data

### The Phase System (Long-Term Planning)

```
Phase 0 (Foundation) → Phase 1 (Baselines) → Phase 2 (Loss Functions) → 
Phase 3 (Policy Dev) → Phase 4 (Evaluation) → Phase 5 (Extension)
```

Each phase in `01-RD-PHASES.md` has:
- Clear scope and timeline
- "Done when" criteria (checkboxes)
- Deliverables list
- Key metrics (KPIs)
- Decision points

### The Quality System (Ensuring Trustworthiness)

```
Level 1 (Sanity) → Level 2 (Research-Grade) → Level 3 (Publication-Ready)
```

Defined in `05-EXPERIMENT-REVIEW.md`:
- **Level 1**: Runs without crashing, metrics plausible
- **Level 2**: Multiple seeds, baseline comparison, statistical validity
- **Level 3**: Publication-quality figures, reproducibility tested, code documented

---

## Key Design Principles

### 1. **Research-First, Not Demo-First**
- Every claim backed by statistical tests
- Multiple seeds required (not single lucky runs)
- Document failures (negative results matter)
- Reproducibility is mandatory, not optional

### 2. **Experimental and Iterative**
- Methods are subject to change based on evidence
- Add → Test → Evaluate → Decide (keep/modify/remove)
- No sacred cows (remove what doesn't work)
- Document all decisions with supporting data

### 3. **Practical and Scannable**
- Heavy use of tables, checklists, code examples
- Quick reference cards
- Decision trees for common questions
- "How to verify" sections

### 4. **Living and Evolving**
- Designed to be updated as you learn
- Each guide includes maintenance guidance
- Principle: document hard-won lessons
- Improve based on experience

---

## Usage Patterns

### Pattern 1: Starting the Project (Week 1)
1. Read `00-START-HERE.md`
2. Set up environment
3. Run first training experiment (even just baseline)
4. Use checklist from `02-CHECKLISTS.md`
5. Produce first experiment folder
6. Review using `05-EXPERIMENT-REVIEW.md` (Level 1)

### Pattern 2: Adding a New Loss Function
1. Read `03-ADD-A-MODULE.md`
2. Implement loss with tests
3. Use checklist from `02-CHECKLISTS.md` → "Implementing a Loss Function"
4. Run comparative experiments (`04-RESEARCH-WORKFLOW.md`)
5. Perform statistical analysis
6. Make decision: keep/modify/remove
7. Document in research note

### Pattern 3: Running a Research Study
1. Read `04-RESEARCH-WORKFLOW.md`
2. Formulate hypothesis
3. Design experiment matrix (vary one thing, multiple seeds)
4. Create configs for all conditions
5. Run experiments
6. Aggregate and analyze results
7. Statistical testing
8. Review using `05-EXPERIMENT-REVIEW.md` (aim for Level 2-3)
9. Write research note with decision

### Pattern 4: Planning Next Steps
1. Check current phase in `01-RD-PHASES.md`
2. Review "done when" criteria
3. Assess progress on deliverables
4. Use priority matrix to choose next task
5. Follow relevant guide for that task

---

## What Makes This Different from Typical ML Projects

Most ML repos have:
- A README
- Maybe some API docs
- Maybe a paper

Raft-LM has:
- **Operational guides**: How to actually do the work
- **Quality gates**: When is work "done"?
- **Checklists**: Prevent common mistakes
- **Decision frameworks**: Keep/modify/remove based on data
- **Statistical rigor**: Multiple seeds, significance tests, effect sizes
- **Provenance tracking**: Every result is traceable
- **Performance tracking**: Detect regressions
- **Phase system**: Long-term planning with clear milestones

This transforms Raft-LM from "an LLM training project" into **a research platform**.

---

## Core Concepts

### Experiment Folder Structure
Every training run produces a complete artifact set:
- `config.json`: Complete configuration
- `environment.json`: Software/hardware details
- `metrics.json`: Training metrics
- `evaluation.json`: Test set results
- `training_log.txt`: Full log
- `checkpoints/`: Model checkpoints
- `artifacts/`: Plots and analyses

**Why**: Enables reproducibility and fair comparison.

### Multiple Seeds Requirement
Minimum 3 seeds for any comparative claim.

**Why**: Single runs can be lucky/unlucky. Variance matters.

### Statistical Testing
Always use statistical tests when claiming "improvement":
- t-test for comparing two methods
- Effect size (Cohen's d) for practical significance
- Report both p-values AND effect sizes

**Why**: "Better" without statistical backing is just anecdote.

### Keep/Modify/Remove Decisions
Every experimental method goes through evaluation:
- **Keep**: Shows significant, meaningful, stable improvement
- **Modify**: Shows promise but needs tuning
- **Remove**: No improvement after 3 iterations

**Why**: Prevents accumulation of dead code and failed methods.

### Research Notes
Document all experiments, especially failures:
- Question, hypothesis, method, results
- Statistical analysis
- Interpretation and limitations
- Decision and next steps
- Reproducibility information

**Why**: Learn from both successes and failures.

---

## Success Metrics

You'll know this documentation system is working when:

- ✅ New contributors can start contributing quickly (< 1 week)
- ✅ Experiments are consistently reproducible
- ✅ Decisions are data-driven (not gut feeling)
- ✅ Mistakes are caught early (via checklists)
- ✅ Performance regressions are detected quickly
- ✅ Research notes are easy to write (process is clear)
- ✅ You can return after months and know what to do
- ✅ Failed methods are documented (not just successes)

---

## Maintenance and Evolution

### When to Update These Docs

- **Found a common pitfall?** Add it to the relevant guide
- **Discovered a better workflow?** Update the guide
- **Need a new checklist?** Add it to `02-CHECKLISTS.md`
- **Phase criteria changed?** Update `01-RD-PHASES.md`
- **New best practice?** Document it

### How to Keep Docs Useful

1. **Keep them scannable**: Use tables, lists, checklists
2. **Keep them practical**: Include code examples and commands
3. **Keep them current**: Update based on experience
4. **Keep them concise**: No unnecessary detail

### Principle

**"If you had to figure something out the hard way, document it so the next person (or future you) doesn't have to."**

---

## Comparison to Other Frameworks

### vs TailWarp (inspiration)
- **TailWarp**: GPU-accelerated Monte Carlo for financial risk
- **Raft-LM**: Training LLMs for risk understanding and decision-making
- **Shared**: Research-first philosophy, rigorous validation, experiment tracking
- **Different**: LLM training vs numerical simulation, different metrics and methods

### vs Standard ML Projects
- **Standard**: Train model, tune hyperparameters, report best result
- **Raft-LM**: Hypothesis → Multiple methods → Statistical comparison → Decision
- **Key difference**: Experimental rigor and method evaluation

### vs Production ML Projects
- **Production**: Optimize for deployment, monitoring, serving
- **Raft-LM**: Optimize for research insights, reproducibility, method comparison
- **Note**: Raft-LM can inform production, but research comes first

---

## Common Questions

### "Isn't this over-engineering?"
No. This is research-grade engineering. Without this rigor:
- Can't trust results (single seed might be lucky)
- Can't build on results (not reproducible)
- Can't make good decisions (no statistical backing)
- Can't collaborate effectively (no shared process)

### "Do I need to follow everything?"
For quick prototypes: Use the core loop and basic checklists.
For decisions: Need Level 2 review (multiple seeds, statistical tests).
For publications: Need Level 3 review (full rigor).

### "Can I customize the process?"
Yes! These docs are templates. Adapt to your needs. But keep the core principles:
- Multiple seeds
- Statistical testing
- Baseline comparison
- Decision documentation

### "What if I don't have time?"
Then you don't have time to waste on unreliable results. This process actually saves time by:
- Catching mistakes early
- Avoiding false positives
- Building on solid foundations
- Preventing rework

---

## Quick Links

- **New to the repo?** → `00-START-HERE.md`
- **Need a checklist?** → `02-CHECKLISTS.md`
- **Adding a module?** → `03-ADD-A-MODULE.md`
- **Running experiments?** → `04-RESEARCH-WORKFLOW.md`
- **Checking quality?** → `05-EXPERIMENT-REVIEW.md`
- **Tracking performance?** → `06-PERFORMANCE-PROTOCOL.md`
- **Planning next steps?** → `01-RD-PHASES.md`
- **Need quick reference?** → `QUICK-REFERENCE.md`

---

## Final Thoughts

This documentation framework is **infrastructure**, just like the code. It:

- Reduces cognitive load (checklists tell you what to do)
- Prevents mistakes (multiple seeds, statistical tests)
- Enables collaboration (clear workflows, shared standards)
- Supports long-term work (phase system prevents chaos)
- Builds trust in results (reproducibility, rigor)

**Invest in maintaining it**, and it will pay dividends throughout the project lifecycle.

---

## Remember

**This is a research operating system. Use it, maintain it, improve it.**

The goal is not perfect process adherence—it's reliable, reproducible, trustworthy research that advances the field of risk-aware LLMs.

