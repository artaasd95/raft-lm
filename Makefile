.PHONY: test benchmark benchmark-raft benchmark-compare benchmark-smoke demo

test:
	pytest

benchmark:
	python scripts/run_benchmark.py --mode stub --pipeline standard_rag

benchmark-raft:
	python scripts/run_benchmark.py --mode stub --pipeline raft_lm

benchmark-compare:
	python scripts/run_benchmark.py --mode stub --pipeline both

benchmark-smoke:
	python scripts/run_benchmark.py --mode smoke --pipeline both --questions-limit 1

demo:
	streamlit run src/demo/streamlit_app.py
