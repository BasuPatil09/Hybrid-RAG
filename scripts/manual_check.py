"""
    python scripts/manual_check.py ingest
    python scripts/manual_check.py query "your question here"
    python scripts/manual_check.py ask "your question here"
    python scripts/manual_check.py eval 
"""

from __future__ import annotations
import sys
from pathlib import Path
import chromadb
from rag.config import get_settings
from rag.eval.golden_qa import load_golden_questions
from rag.eval.runner import run_eval_suite, summarize
from rag.generation.confidence import get_completeness_judge
from rag.generation.generator import get_generator
from rag.generation.verification import get_citation_verifier
from rag.indexing.embeddings import get_embedder
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.vector_store import ChromaVectorStore
from rag.ingestion.chunking import get_chunker
from rag.ingestion.loaders import DocumentLoaderRegistry
from rag.pipeline import answer_question, retrieve, run_ingestion
from rag.retrieval.reranker import CrossEncoderReranker, load_cross_encoder

_GOLDEN_QA_PATH = (
    Path(__file__).resolve().parent.parent / "eval" / "golden_qa_draft.csv"
)

def _build_embedder_and_store():
    settings = get_settings()
    embedder = get_embedder(
        settings.embedding_provider,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        device=settings.embedding_device,
    )
    if hasattr(embedder, "device"):
        print(f"Local embedding model running on: {embedder.device}")
    vector_store = ChromaVectorStore(
        chromadb.PersistentClient(path=str(settings.chroma_persist_dir)),
        settings.chroma_collection_name,
    )
    return settings, embedder, vector_store

def _sparse_index_from_store(vector_store: ChromaVectorStore) -> SparseIndex:
    sparse_index = SparseIndex()
    sparse_index.rebuild(vector_store.get_all_chunks())
    return sparse_index

def _retrieve_and_rerank(text, settings, embedder, vector_store, sparse_index, top_n):
    fused = retrieve(
        text,
        embedder,
        vector_store,
        sparse_index,
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        rrf_k=settings.rrf_k,
        rrf_dense_weight=settings.rrf_dense_weight,
        rrf_sparse_weight=settings.rrf_sparse_weight,
    )
    reranker = CrossEncoderReranker(load_cross_encoder(settings.reranker_model_name))
    return fused, reranker.rerank(text, fused[:20], top_n=top_n)

def ingest() -> None:
    settings, embedder, vector_store = _build_embedder_and_store()
    registry = DocumentLoaderRegistry(pdf_min_page_chars=settings.pdf_min_page_chars)
    chunker = get_chunker(
        settings.chunking_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        header_levels=settings.header_levels,
    )
    sparse_index = SparseIndex()

    print(
        f"Ingesting {settings.corpus_dir} via local embeddings -- takes a few minutes..."
    )
    report = run_ingestion(
        settings.corpus_dir,
        registry,
        chunker,
        embedder,
        vector_store,
        sparse_index,
        dedup_threshold=settings.dedup_similarity_threshold,
        checkpoint_every=settings.ingestion_checkpoint_every,
    )
    print(report.model_dump_json(indent=2))
    if report.stopped_early:
        print(
            f"\nStopped early: {report.stopped_early}\n"
            f"Everything embedded before the failure is already saved to {settings.chroma_persist_dir} "
            f"({report.chunks_indexed} chunks so far). Could be a transient network issue (just re-run) --"
            f"either way, already-indexed chunks won't be re-embedded, so re-running the exact same command "
            f"costs no wasted work."
        )


def query(text: str, top_n: int | None = None) -> None:
    settings, embedder, vector_store = _build_embedder_and_store()
    if vector_store.count() == 0:
        raise SystemExit(
            "Nothing indexed yet -- run `python scripts/manual_check.py ingest` first."
        )

    top_n = top_n or settings.rerank_top_n
    sparse_index = _sparse_index_from_store(vector_store)
    fused, reranked = _retrieve_and_rerank(
        text, settings, embedder, vector_store, sparse_index, top_n
    )

    print(f"--- {len(fused)} fused results for {text!r} (RRF, pre-rerank) ---\n")
    for r in fused[:10]:
        print(f"score={r.score:.4f} [{r.chunk.id}]")
        print(f"  {r.chunk.content[:200]!r}\n")

    print(
        f"--- reranked top {top_n} (cross-encoder: {settings.reranker_model_name}) ---\n"
    )
    for r in reranked:
        print(f"score={r.score:.4f} [{r.chunk.id}]")
        print(f"  {r.chunk.content[:200]!r}\n")


def _build_generation_components(settings):
    if not settings.groq_api_key:
        raise SystemExit("Set GROQ_API_KEY in .env first.")

    reranker = CrossEncoderReranker(load_cross_encoder(settings.reranker_model_name))
    generator = get_generator(
        settings.generation_provider,
        api_key=settings.groq_api_key,
        model=settings.generation_model,
    )
    verifier = get_citation_verifier(
        settings.generation_provider,
        api_key=settings.groq_api_key,
        model=settings.generation_model,
    )
    completeness_judge = get_completeness_judge(
        settings.generation_provider,
        api_key=settings.groq_api_key,
        model=settings.generation_model,
    )
    return reranker, generator, verifier, completeness_judge


def ask(text: str, top_n: int | None = None) -> None:
    settings, embedder, vector_store = _build_embedder_and_store()
    if vector_store.count() == 0:
        raise SystemExit(
            "Nothing indexed yet -- run `python scripts/manual_check.py ingest` first."
        )

    top_n = top_n or settings.rerank_top_n
    sparse_index = _sparse_index_from_store(vector_store)
    reranker, generator, verifier, completeness_judge = _build_generation_components(
        settings
    )

    answered = answer_question(
        text,
        embedder,
        vector_store,
        sparse_index,
        reranker,
        generator,
        verifier,
        completeness_judge,
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        rrf_k=settings.rrf_k,
        rrf_dense_weight=settings.rrf_dense_weight,
        rrf_sparse_weight=settings.rrf_sparse_weight,
        top_n=top_n,
    )
    result, verified, confidence = (
        answered.answer,
        answered.verified,
        answered.confidence,
    )
    chunks = [r.chunk for r in answered.reranked]

    print(f"--- answer for {text!r} ---\n")
    print(result.answer)
    print()

    if result.is_insufficient:
        print(
            "(model explicitly declined to answer -- context did not support one; this is the intended behavior, not a failure)"
        )
    else:
        print("Source excerpts provided (marked ones were actually cited):")
        for i, c in enumerate(chunks, start=1):
            marker = "  <- cited" if c.id in result.cited_chunk_ids else ""
            print(f"  [{i}] {c.id}{marker}")

        print(
            "\n--- citation verification (separate LLM-as-judge call per claim) ---\n"
        )
        for v in verified.claim_verdicts:
            status = "SUPPORTED" if v.supported else "NOT SUPPORTED"
            print(f"[{status}] {v.claim}")
            print(f"  sources: {', '.join(v.chunk_ids)}")
            print(f"  judge: {v.explanation}\n")
        print(
            f"citation accuracy: {verified.citation_accuracy:.0%} ({sum(1 for v in verified.claim_verdicts if v.supported)}/{len(verified.claim_verdicts)} claims supported)"
        )

    print("\n--- confidence score ---\n")
    print(
        f"retrieval confidence:  {confidence.retrieval_confidence:.0%}  (from reranker scores of cited chunks)"
    )
    print(
        f"citation coverage:     {confidence.citation_coverage:.0%}  (from verified claims -- see reliability caveat below)"
    )
    print(
        f"answer completeness:   {confidence.answer_completeness:.0%}  ({confidence.completeness_explanation})"
    )
    print(f"composite:             {confidence.composite:.0%}")
    print(
        "\nNote: citation coverage comes from an LLM-as-judge -- treat it as directional, not exact."
    )


def eval_suite(golden_qa_path: Path = _GOLDEN_QA_PATH) -> None:
    settings, embedder, vector_store = _build_embedder_and_store()
    if vector_store.count() == 0:
        raise SystemExit(
            "Nothing indexed yet -- run `python scripts/manual_check.py ingest` first."
        )

    questions = load_golden_questions(golden_qa_path)
    if not questions:
        raise SystemExit(
            f"No approved questions found in {golden_qa_path}. Mark at least one row's `approved` "
            f"column 'yes' first -- unreviewed drafts don't count as ground truth."
        )
    print(
        f"Running eval suite: {len(questions)} approved questions from {golden_qa_path}\n"
    )

    sparse_index = _sparse_index_from_store(vector_store)
    reranker, generator, verifier, completeness_judge = _build_generation_components(
        settings
    )

    results = run_eval_suite(
        questions,
        embedder,
        vector_store,
        sparse_index,
        reranker,
        generator,
        verifier,
        completeness_judge,
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        rrf_k=settings.rrf_k,
        rrf_dense_weight=settings.rrf_dense_weight,
        rrf_sparse_weight=settings.rrf_sparse_weight,
        rerank_top_n=settings.rerank_top_n,
    )

    for r in results:
        mark = "OK" if r.behavior_matched else "MISMATCH"
        print(
            f"[{mark}] {r.question_id} ({r.category}): expected={r.expected_behavior} actual_insufficient={r.actual_is_insufficient}"
        )
        if not r.behavior_matched:
            print(f"  answer: {r.answer[:200]!r}")
        if not r.actual_is_insufficient:
            print(
                f"  citation_accuracy={r.citation_accuracy:.0%}  composite_confidence={r.composite_confidence:.0%}"
            )
        print()

    summary = summarize(results)
    print("=== summary ===")
    print(f"total questions:              {summary.total_questions}")
    print(f"behavior match rate:          {summary.behavior_match_rate:.0%}")
    for cat, rate in sorted(summary.behavior_match_rate_by_category.items()):
        print(f"  {cat}: {rate:.0%}")
    print(
        f"mean citation accuracy:       {summary.mean_citation_accuracy:.0%}  (answered questions only)"
    )
    print(f"mean retrieval confidence:    {summary.mean_retrieval_confidence:.0%}")
    print(f"mean composite confidence:    {summary.mean_composite_confidence:.0%}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("ingest", "query", "ask", "eval"):
        raise SystemExit(__doc__)
    command = sys.argv[1]
    if command == "ingest":
        ingest()
    elif command == "eval":
        eval_suite()
    elif len(sys.argv) < 3:
        raise SystemExit(
            f'Usage: python scripts/manual_check.py {command} "your question"'
        )
    elif command == "query":
        query(sys.argv[2])
    else:
        ask(sys.argv[2])
