# Research Notes

Documentation of research findings, experiments, and decisions.

## Structure

One markdown file per research question/experiment series.

## Research Note Template

```markdown
# Research Note: [Title]

## Question
[What you're testing]

## Hypothesis
[What you expect and why]

## Method
- Models: [which models]
- Data: [which data]
- Conditions: [what varies]
- Seeds: [how many]

## Results
| Metric | Baseline | New Method | Change |
|--------|----------|------------|--------|
| Acc    | 0.85±0.01| 0.89±0.01  | +4.7%  |

Statistical test: t(4)=5.2, p=0.006, d=1.2

## Decision
✅ Keep / 🔄 Modify / ❌ Remove
[Reasoning]
```

## Guidelines

- Document all major experiments
- Include statistical tests
- Record decisions with rationale
- Link to experiment result folders
- Update as experiments evolve

