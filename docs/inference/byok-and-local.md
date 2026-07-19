# BYOK and local models

Configure LLM providers via YAML under `configs/llm_*.yaml`.

| Provider | Config |
|----------|--------|
| Mock | `configs/llm_mock.yaml` |
| Ollama | `configs/llm_ollama.yaml` |
| vLLM | `configs/llm_vllm.yaml` |
| Custom / LiteLLM | `configs/llm_custom.yaml` |

Factory: `src/llm_integration/factory.py`

See [llm-integration.md](../llm-integration.md) for full BYOK wiring.
