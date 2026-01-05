# Raft-LM Project Plan Documentation

This folder is the **operating manual** for research and development in Raft-LM: checklists, phase gates, and workflow guides.

---

## 🚀 Start here

**New to the repo?** Read these in order:

1. **`00-START-HERE.md`** — First month playbook, core workflow, quick answers
2. **`01-RD-PHASES.md`** — Research phases with clear gate criteria
3. **`QUICK-REFERENCE.md`** — One-page summary of everything

---

## 📋 Use these guides as needed

| Guide | When to use it |
|-------|----------------|
| **`02-CHECKLISTS.md`** | Every time you add a component or run experiments |
| **`03-ADD-A-MODULE.md`** | Step-by-step: how to add loss functions, metrics, training methods |
| **`04-RESEARCH-WORKFLOW.md`** | When running experiments to answer research questions |
| **`05-EXPERIMENT-REVIEW.md`** | To decide if results are trustworthy/publishable |
| **`06-PERFORMANCE-PROTOCOL.md`** | To track and optimize training/inference performance |

---

## 🎯 Quick navigation by task

### "I want to add a new loss function"
1. Read `03-ADD-A-MODULE.md` (Step-by-step guide)
2. Use checklist from `02-CHECKLISTS.md` → "Implementing a Loss Function"
3. Run comparative experiments (`04-RESEARCH-WORKFLOW.md`)
4. Review using `05-EXPERIMENT-REVIEW.md`

### "I want to answer a research question"
1. Read `04-RESEARCH-WORKFLOW.md` (How to design experiments)
2. Create experiment configs
3. Run experiments with multiple seeds
4. Analyze statistically
5. Review using `05-EXPERIMENT-REVIEW.md`

### "I want to know what to build next"
1. Read `01-RD-PHASES.md` (Phase gates and priorities)
2. Check current phase "done when" criteria
3. Pick next task from phase scope

### "I want to verify my results are trustworthy"
1. Read `05-EXPERIMENT-REVIEW.md` (Review levels)
2. Use checklists from `02-CHECKLISTS.md`
3. Run statistical tests
4. Document in research notes

### "I want to check performance"
1. Read `06-PERFORMANCE-PROTOCOL.md`
2. Measure training time and memory
3. Compare to baseline
4. Document in performance log

---

## 🔄 The core research loop

```
┌─────────────────────────────────────────────────────────┐
│  1. Formulate   →  2. Implement  →  3. Train            │
│       ↑                                    ↓             │
│  5. Decide      ←  4. Evaluate   ←─────────┘             │
└─────────────────────────────────────────────────────────┘
```

**Never skip steps 3-5.** They're what make this research-grade.

---

## 💡 Philosophy

- **Research-first**: Correctness and reproducibility over speed of iteration
- **Experimental**: Try methods, evaluate rigorously, keep what works
- **Measurable**: All claims backed by data and statistical tests
- **Iterative**: Each experiment builds on previous learnings
- **Transparent**: Document both successes and failures

---

## 📊 What makes this framework different

This is not just another LLM fine-tuning repo. Raft-LM is focused on:

1. **Risk Understanding**: Training models to understand and make decisions based on risk, not just maximize accuracy
2. **Multi-Method Evaluation**: Systematically comparing different approaches (loss functions, training methods, metrics)
3. **Research Rigor**: Statistical testing, multiple seeds, reproducibility tracking
4. **Iterative Development**: Add, test, evaluate, decide (keep/modify/remove)
5. **Comprehensive Documentation**: Every experiment tracked, every decision documented

**Goal**: Build a framework where you can trust the results and build upon them.

---

## 🛠️ How to maintain these docs

These docs are **living documents**. Improve them as you learn:

- Found a common pitfall? Add it to the relevant guide
- Discovered a better workflow? Update the guide
- Need a new checklist? Add it to `02-CHECKLISTS.md`
- Keep docs **scannable** — use tables, lists, examples

**Principle**: If you figured something out the hard way, document it so others (or future you) don't have to.

---

## 📝 Document Structure Overview

### Foundation
- **`00-START-HERE.md`**: Entry point, first month playbook
- **`01-RD-PHASES.md`**: Long-term roadmap, phase gates, priorities

### Operational Guides
- **`02-CHECKLISTS.md`**: Copy-paste checklists for common tasks
- **`03-ADD-A-MODULE.md`**: How to add components (losses, metrics, methods)
- **`04-RESEARCH-WORKFLOW.md`**: How to design and run experiments
- **`05-EXPERIMENT-REVIEW.md`**: Quality gates, when to trust results
- **`06-PERFORMANCE-PROTOCOL.md`**: Tracking and optimizing performance

### Quick Reference
- **`QUICK-REFERENCE.md`**: One-page summary of everything
- **`SUMMARY.md`**: Detailed overview of documentation structure

---

## 🎓 Learning Path

### Week 1: Getting Started
```
[ ] Read 00-START-HERE.md
[ ] Skim 01-RD-PHASES.md
[ ] Set up environment
[ ] Run first training experiment
```

### Week 2: First Real Experiment
```
[ ] Read 04-RESEARCH-WORKFLOW.md
[ ] Design comparative experiment
[ ] Use checklists from 02-CHECKLISTS.md
[ ] Run experiments with multiple seeds
```

### Week 3: Analysis & Decision
```
[ ] Read 05-EXPERIMENT-REVIEW.md
[ ] Perform statistical analysis
[ ] Make keep/modify/remove decision
[ ] Document in research note
```

### Week 4: Iteration
```
[ ] Implement next component
[ ] Use 03-ADD-A-MODULE.md as guide
[ ] Track performance (06-PERFORMANCE-PROTOCOL.md)
[ ] Review phase progress (01-RD-PHASES.md)
```

---

## 🚦 Quality Gates

Every component should pass through:

1. **Implementation**: Code works, tested on toy examples
2. **Integration**: Added to experiment framework, configurable
3. **Evaluation**: Compared to baseline with multiple seeds
4. **Analysis**: Statistical tests, effect sizes, visualization
5. **Decision**: Keep/modify/remove with documented rationale
6. **Documentation**: Code docs, usage examples, research notes

**Don't skip gates.** They catch mistakes early and build trust in results.

---

## 🔗 Quick Links

- **New to the repo?** → `00-START-HERE.md`
- **Need a checklist?** → `02-CHECKLISTS.md`
- **Adding a component?** → `03-ADD-A-MODULE.md`
- **Running experiments?** → `04-RESEARCH-WORKFLOW.md`
- **Checking quality?** → `05-EXPERIMENT-REVIEW.md`
- **Tracking performance?** → `06-PERFORMANCE-PROTOCOL.md`
- **Planning next steps?** → `01-RD-PHASES.md`
- **Need quick reference?** → `QUICK-REFERENCE.md`

---

## 💬 Feedback & Improvements

These docs should be:
- **Practical**: Immediately usable
- **Scannable**: Easy to find what you need
- **Complete**: Cover common scenarios
- **Concise**: No unnecessary detail

If something is unclear, confusing, or missing, improve it. These docs serve the research, not the other way around.

---

## Remember

**This is a research operating system.** 

Use it to:
- Make better decisions (data-driven)
- Avoid common mistakes (checklists)
- Build trust in results (reproducibility)
- Move faster over time (stable foundation)

Invest in the process, reap the benefits in results.

