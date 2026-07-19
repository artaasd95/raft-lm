# Custom rewards

Extend `BaseReward` and register by name or use YAML `custom_module`.

## Subclass

```python
from src.rewards.base import BaseReward
from src.domain.trajectory import RewardBatch
import numpy as np

class MyReward(BaseReward):
    name = "my_reward"

    def compute(self, batch):
        vals = np.asarray(batch.get("scores", [0.0]), dtype=np.float32)
        return RewardBatch(values=vals)
```

## Composite YAML

```yaml
reward:
  name: composite
  components:
    - name: task_accuracy
      weight: 1.0
    - name: risk_cvar
      weight: 0.3
      params:
        alpha: 0.05
```

Wire risk metrics from `src/metrics/risk_metrics.py` and tools from `src/tools/risk_tools.py`.
