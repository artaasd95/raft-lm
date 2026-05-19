.PHONY: test benchmark demo

test:
	pytest

benchmark:
	python scripts/run_benchmark.py --mode stub --pipeline standard_rag

benchmark-compare:
	python scripts/run_benchmark.py --mode stub --pipeline both

demo:
	streamlit run src/demo/streamlit_app.py
