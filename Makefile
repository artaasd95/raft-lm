.PHONY: test benchmark demo

test:
	pytest

benchmark:
	python -m src.evals.benchmark_runner

demo:
	streamlit run src/demo/streamlit_app.py
