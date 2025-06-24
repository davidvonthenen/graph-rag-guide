#!/usr/bin/env python3
"""
cache_benchmark.py ― high-resolution latency benchmark for dual-memory
Graph-based RAG

Scenarios per iteration
-----------------------
1. LT         – direct query against the long-term Neo4j store
2. ST-cold    – cache flushed, promotion LT → ST
3. ST-warm    – second question answered *only* from ST

Each scenario reports
    • total    (ms, three decimals)
    • retrieval (Cypher + context assembly, ms)
    • generation (LLM, ms – optional)

Use --no-llm to exclude the expensive generation step.

Example
-------
python cache_benchmark.py \
    --question1 "Tell me about the connection between Ernie Wise and Vodafone." \
    --question2 "Tell me something about the personal life of Ernie Wise?" \
    --runs 20 --no-llm
"""

import argparse
import statistics
import time
from typing import List, Tuple

import cache_cypher_query as ccq

###############################################################################
# Utility helpers – nanosecond-precision timers
###############################################################################


def _now_ns() -> int:
    return time.perf_counter_ns()


def _ns_to_ms(ns: int) -> float:
    """Convert nanoseconds to milliseconds with micro-second precision."""
    return ns / 1_000_000.0


def _avg(lst):
    return _ns_to_ms(statistics.mean(lst)) if lst else 0.0


def flush_short_term(driver) -> None:
    """Remove every node & relationship from the short-term Neo4j instance."""
    with driver.session() as sess:
        sess.run("MATCH (n) DETACH DELETE n")


def _build_context_block(paras: List[dict]) -> str:
    """Format paragraph snippets exactly as in ccq.ask() for fairness."""
    ctx = ""
    for p in paras:
        snippet = p["text"][:350].replace("\n", " ")
        ctx += (
            f"\n---\nDoc: {p['title']} | Para #{p['idx']} "
            f"| Matches: {p['matchingEntities']}\n{snippet}…"
        )
    return ctx


###############################################################################
# Timing wrappers
###############################################################################


def _time_generation(llm, question: str, context: str, use_llm: bool) -> int:
    """Return generation latency in *nanoseconds* (0 if skipped)."""
    if not use_llm:
        return 0
    start = _now_ns()
    ccq.generate_answer(llm, question, context)
    return _now_ns() - start


def timed_long_term(
    question: str,
    long_driver,
    llm,
    nlp,
    use_llm: bool,
) -> Tuple[int, int, int]:
    """Return (total_ns, retrieval_ns, generation_ns) for LT path."""
    t0 = _now_ns()

    ent_pairs = ccq.extract_entities(nlp, question)
    if not ent_pairs:
        return 0, 0, 0

    r0 = _now_ns()
    with long_driver.session() as sess:
        paras = ccq.fetch_paragraphs(sess, ent_pairs)
    retrieval = _now_ns() - r0

    generation = _time_generation(
        llm, question, _build_context_block(paras), use_llm
    )
    return _now_ns() - t0, retrieval, generation


def timed_short_term_cold(
    question: str,
    short_driver,
    long_driver,
    llm,
    nlp,
    use_llm: bool,
) -> Tuple[int, int, int]:
    """
    Cold path: ccq.ask() handles promotion. We patch generate_answer for timing
    or stubbing depending on use_llm.
    """
    total_start = _now_ns()
    real_gen = ccq.generate_answer
    gen_start_ns = None

    if use_llm:

        def _timed_gen(*args, **kwargs):
            nonlocal gen_start_ns
            gen_start_ns = _now_ns()
            return real_gen(*args, **kwargs)

        ccq.generate_answer = _timed_gen
    else:

        def _stub_gen(*_a, **_kw):
            return None

        ccq.generate_answer = _stub_gen

    try:
        ccq.ask(llm, nlp, question, short_driver, long_driver)
    finally:
        ccq.generate_answer = real_gen

    total_ns = _now_ns() - total_start

    if use_llm and gen_start_ns:
        generation_ns = _now_ns() - gen_start_ns
        retrieval_ns = total_ns - generation_ns
    else:
        generation_ns = 0
        retrieval_ns = total_ns
    return total_ns, retrieval_ns, generation_ns


def timed_short_term_warm(
    question: str,
    short_driver,
    llm,
    nlp,
    use_llm: bool,
) -> Tuple[int, int, int]:
    """Warm path: ST only."""
    t0 = _now_ns()

    ent_pairs = ccq.extract_entities(nlp, question)
    if not ent_pairs:
        return 0, 0, 0

    r0 = _now_ns()
    with short_driver.session() as sess:
        paras = ccq.fetch_paragraphs(sess, ent_pairs)
    retrieval = _now_ns() - r0

    generation = _time_generation(
        llm, question, _build_context_block(paras), use_llm
    )
    return _now_ns() - t0, retrieval, generation


###############################################################################
# Benchmark orchestrator
###############################################################################


def run_benchmark(q1: str, q2: str, runs: int, use_llm: bool) -> None:
    short_driver = ccq.connect_short()
    long_driver = ccq.connect_long()

    # Ensure indexes
    with long_driver.session() as s:
        ccq.create_indexes(s)
    with short_driver.session() as s:
        ccq.create_indexes(s)

    llm = ccq.load_llm() if use_llm else None
    nlp = ccq.load_spacy()

    # Accumulators (store *nanoseconds*)
    lt_tot, lt_ret, lt_gen = [], [], []
    cold_tot, cold_ret, cold_gen = [], [], []
    warm_tot, warm_ret, warm_gen = [], [], []

    for i in range(1, runs + 1):
        print(f"\n=== Iteration {i}/{runs} ===")

        # Long-term baseline
        flush_short_term(short_driver)
        ccq._seen_entities.clear()
        t, r, g = timed_long_term(q1, long_driver, llm, nlp, use_llm)
        lt_tot.append(t); lt_ret.append(r); lt_gen.append(g)
        print(f"[LT]       total:{_ns_to_ms(t):9.3f} ms  "
              f"ret:{_ns_to_ms(r):9.3f} ms  gen:{_ns_to_ms(g):9.3f} ms")

        # Cold ST
        flush_short_term(short_driver)
        ccq._seen_entities.clear()
        t, r, g = timed_short_term_cold(q1, short_driver, long_driver,
                                        llm, nlp, use_llm)
        cold_tot.append(t); cold_ret.append(r); cold_gen.append(g)
        print(f"[ST-cold]  total:{_ns_to_ms(t):9.3f} ms  "
              f"ret:{_ns_to_ms(r):9.3f} ms  gen:{_ns_to_ms(g):9.3f} ms")

        # Warm ST
        t, r, g = timed_short_term_warm(q2, short_driver, llm, nlp, use_llm)
        warm_tot.append(t); warm_ret.append(r); warm_gen.append(g)
        print(f"[ST-warm]  total:{_ns_to_ms(t):9.3f} ms  "
              f"ret:{_ns_to_ms(r):9.3f} ms  gen:{_ns_to_ms(g):9.3f} ms")

    # Averages
    print("\n──────── Average over", runs, "runs ────────")
    print("               total (ms)   retrieval   generation")
    print("Long-term   : {:10.3f}   {:9.3f}   {:10.3f}"
          .format(_avg(lt_tot), _avg(lt_ret), _avg(lt_gen)))
    print("ST-cold     : {:10.3f}   {:9.3f}   {:10.3f}"
          .format(_avg(cold_tot), _avg(cold_ret), _avg(cold_gen)))
    print("ST-warm     : {:10.3f}   {:9.3f}   {:10.3f}"
          .format(_avg(warm_tot), _avg(warm_ret), _avg(warm_gen)))

    print("\nSpeed-ups (based on *retrieval* only):")
    print("  LT  → ST-cold : {:.2f}×".format(_avg(lt_ret) / _avg(cold_ret)))
    print("  LT  → ST-warm : {:.2f}×".format(_avg(lt_ret) / _avg(warm_ret)))
    print("  ST-cold → warm: {:.2f}×".format(_avg(cold_ret) / _avg(warm_ret)))
    print("────────────────────────────────────────")


###############################################################################
# CLI
###############################################################################


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--question1",
        default="Tell me about the connection between Ernie Wise and Vodafone.",
        help="First (cold) question.",
    )
    p.add_argument(
        "--question2",
        default="Tell me something about the personal life of Ernie Wise?",
        help="Second (warm) question.",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=1000,
        help="Number of benchmark iterations.",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help="Enable answer generation to measure with LLM latency.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(args.question1, args.question2, args.runs, use_llm=args.llm)
