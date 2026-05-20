"""Unit tests for benchmark schema."""

from src.evals.benchmark_schema import (
    CitationRecord,
    ComparisonReport,
    RagasScores,
    SCHEMA_VERSION,
    backfill_ragas_scores,
    export_json_schema,
    new_comparison_report,
)


def test_new_comparison_report_has_version():
    report = new_comparison_report("financial_policy_v1")
    assert report.schema_version == SCHEMA_VERSION
    assert report.corpus_id == "financial_policy_v1"
    assert report.standard.pipeline_name == "standard_rag"
    assert report.raft_lm.pipeline_name == "raft_lm"


def test_comparison_report_to_json_roundtrip_keys():
    report = new_comparison_report("financial_policy_v1")
    report.standard.ragas = RagasScores(context_precision=0.8, faithfulness=0.7)
    data = report.to_dict()
    assert data["standard"]["ragas"]["context_precision"] == 0.8


def test_export_json_schema_required_fields():
    schema = export_json_schema()
    assert "required" in schema
    assert "schema_version" in schema["required"]


def test_citation_record_fields():
    cite = CitationRecord(
        chunk_id="policy_capital::chunk_0",
        doc_id="policy_capital",
        excerpt="CET1",
        score=0.9,
    )
    assert cite.chunk_id.endswith("chunk_0")


def test_backfill_ragas_scores_fills_empty_slots():
    report = new_comparison_report("financial_policy_v1")
    report.standard.ragas = RagasScores(context_precision=0.0, faithfulness=0.0)
    report.raft_lm.ragas = RagasScores(context_precision=0.0, faithfulness=0.0)
    filled = backfill_ragas_scores(
        report,
        standard=RagasScores(context_precision=0.72, faithfulness=0.68),
        raft_lm=RagasScores(context_precision=0.81, faithfulness=0.77),
    )
    assert filled.standard.ragas.context_precision == 0.72
    assert filled.raft_lm.ragas.faithfulness == 0.77
    assert filled.chart_standard_values == [0.72, 0.68]
