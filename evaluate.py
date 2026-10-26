# ============================================================
# EVALUATION
#
# Part 1 - Retrieval and classification (fast, no LLM needed)
#   Compares four retrieval set-ups on the same questions:
#     BM25 only | Dense (FAISS) only | Hybrid (RRF) | Hybrid + category
#   Metrics: Hit@1, Hit@3, MRR@10 (chunk level), classifier accuracy.
#
# Part 2 - Answers (optional, needs Ollama running)
#   Generates an answer for every question with the full system and
#   saves them to a CSV with automatic checks + empty columns for
#   manual scoring (correct / grounded).
#
# Usage (from the project folder, venv active):
#     python evaluate.py                 # part 1 only
#     python evaluate.py --generate      # part 1 + part 2
#     LLM_MODEL=qwen2.5:7b python evaluate.py --generate
# ============================================================

import contextlib
import csv
import io
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import rag_engine
from ticket_classifier import classify_ticket, load_classifier


PROJECT_DIR = Path(__file__).resolve().parent

QUESTIONS_PATH = (
    PROJECT_DIR / "data" / "evaluation" / "eval_questions_v2.csv"
)

RESULTS_DIR = PROJECT_DIR / "data" / "evaluation" / "results"

TOP_K = 10

METHODS = [
    "BM25 only",
    "Dense only",
    "Hybrid (RRF)",
    "Hybrid + category",
]


# ============================================================
# 1. LOAD
# ============================================================

def quiet(function, *args, **kwargs):
    """
    Run a rag_engine function without its progress prints.
    """

    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def load_questions():

    with open(QUESTIONS_PATH, encoding="utf-8") as file:
        questions = list(csv.DictReader(file))

    for question in questions:
        question["relevant"] = [
            chunk_id.strip()
            for chunk_id in question["relevant_chunks"].split(";")
            if chunk_id.strip()
        ]

    return questions


# ============================================================
# 2. RETRIEVAL METHODS (each returns a ranked list of chunk IDs)
# ============================================================

def rank_bm25(question, chunks, bm25_data):

    results = rag_engine.bm25_search(
        query=question,
        tokenized_docs=bm25_data[0],
        doc_lengths=bm25_data[1],
        avg_doc_length=bm25_data[2],
        document_frequencies=bm25_data[3],
        top_k=TOP_K,
    )

    return [chunks[idx]["chunk_id"] for idx, _ in results]


def rank_dense(question, index, chunks, embedding_model):

    vector = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    _, indices = index.search(vector, TOP_K)

    return [chunks[int(idx)]["chunk_id"] for idx in indices[0] if idx != -1]


def rank_hybrid(
    question,
    index,
    chunks,
    embedding_model,
    bm25_data,
    category=None,
    confidence=None,
):

    # The full system refuses out-of-scope categories (no documents).
    if rag_engine.is_out_of_scope(category, confidence):
        return []

    results = quiet(
        rag_engine.hybrid_search,
        question=question,
        index=index,
        chunks=chunks,
        embedding_model=embedding_model,
        bm25_data=bm25_data,
        top_k=TOP_K,
        predicted_category=category,
        classification_confidence=confidence,
    )

    return [result["chunk"]["chunk_id"] for result in results]


# ============================================================
# 3. METRICS
# ============================================================

def first_relevant_rank(ranked_ids, relevant):

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return rank

    return None


def summarize(rows):

    if not rows:
        return {"n": 0, "Hit@1": 0, "Hit@3": 0, "MRR@10": 0}

    ranks = [row["rank"] for row in rows]

    return {
        "n": len(rows),
        "Hit@1": sum(rank == 1 for rank in ranks) / len(ranks),
        "Hit@3": sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks),
        "MRR@10": sum(1 / rank if rank else 0 for rank in ranks) / len(ranks),
    }


def print_table(title, rows_by_method):

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'Method':<22}{'n':>5}{'Hit@1':>12}{'Hit@3':>12}{'MRR@10':>12}")
    print("-" * 72)

    for method in METHODS:
        summary = summarize(rows_by_method.get(method, []))
        print(
            f"{method:<22}{summary['n']:>5}"
            f"{summary['Hit@1']:>12.1%}"
            f"{summary['Hit@3']:>12.1%}"
            f"{summary['MRR@10']:>12.3f}"
        )


# ============================================================
# 4. PART 1 - RETRIEVAL + CLASSIFICATION
# ============================================================

def evaluate_retrieval(questions, kb, classifier):

    index, chunks, embedding_model, bm25_data = kb
    tokenizer, model = classifier

    known_ids = {chunk["chunk_id"] for chunk in chunks}

    missing = sorted({
        chunk_id
        for question in questions
        for chunk_id in question["relevant"]
        if chunk_id not in known_ids
    })

    if missing:
        print(
            "\nWARNING: these labelled chunk IDs are not in the index "
            "(did the documents change?):"
        )
        print("  " + ", ".join(missing))

    detail_rows = []
    rows_by_method = {method: [] for method in METHODS}
    classification_rows = []

    for question in questions:

        text = question["question"]

        classification = classify_ticket(text, tokenizer, model)
        question["predicted_category"] = classification["category"]
        question["confidence"] = classification["confidence"]

        if question["expected_category"]:
            classification_rows.append(
                classification["category"] == question["expected_category"]
            )

        if not question["relevant"]:
            continue

        rankings = {
            "BM25 only": rank_bm25(text, chunks, bm25_data),
            "Dense only": rank_dense(text, index, chunks, embedding_model),
            "Hybrid (RRF)": rank_hybrid(
                text, index, chunks, embedding_model, bm25_data
            ),
            "Hybrid + category": rank_hybrid(
                text, index, chunks, embedding_model, bm25_data,
                category=classification["category"],
                confidence=classification["confidence"],
            ),
        }

        for method, ranked_ids in rankings.items():

            rank = first_relevant_rank(ranked_ids, question["relevant"])

            row = {
                "question_id": question["question_id"],
                "language": question["language"],
                "type": question["type"],
                "method": method,
                "rank": rank,
                "top3": ";".join(ranked_ids[:3]),
            }

            rows_by_method[method].append(row)
            detail_rows.append(row)

    # ---------------- printed report ----------------

    print_table("RETRIEVAL - ALL IN-SCOPE QUESTIONS", rows_by_method)

    for language in ["english", "italian"]:
        print_table(
            f"RETRIEVAL - {language.upper()}",
            {
                method: [r for r in rows if r["language"] == language]
                for method, rows in rows_by_method.items()
            },
        )

    for question_type in ["specific", "vague"]:
        print_table(
            f"RETRIEVAL - {question_type.upper()} QUESTIONS",
            {
                method: [r for r in rows if r["type"] == question_type]
                for method, rows in rows_by_method.items()
            },
        )

    print("\n" + "=" * 72)
    print("CLASSIFIER")
    print("=" * 72)

    if classification_rows:
        accuracy = sum(classification_rows) / len(classification_rows)
        print(
            f"Accuracy on in-scope questions: {accuracy:.1%} "
            f"({sum(classification_rows)}/{len(classification_rows)})"
        )

    print("\nMisclassified or out-of-scope questions:")

    for question in questions:

        wrong = (
            question["expected_category"]
            and question["predicted_category"] != question["expected_category"]
        )

        if wrong or question["type"] == "out_of_scope" or not question["expected_category"]:
            print(
                f"  {question['question_id']:<26} expected: "
                f"{question['expected_category'] or '-':<10} predicted: "
                f"{question['predicted_category']} "
                f"({question['confidence']:.0%})"
            )

    out_of_scope = [q for q in questions if q["type"] == "out_of_scope"]

    if out_of_scope:

        refused = [
            q for q in out_of_scope
            if rag_engine.is_out_of_scope(q["predicted_category"], q["confidence"])
        ]

        print(
            f"\nOut-of-scope questions refused (no documents used): "
            f"{len(refused)}/{len(out_of_scope)}"
        )

    wrongly_refused = [
        q for q in questions
        if q["type"] != "out_of_scope"
        and rag_engine.is_out_of_scope(q["predicted_category"], q["confidence"])
    ]

    print(f"In-scope questions wrongly refused: {len(wrongly_refused)}")

    for q in wrongly_refused:
        print(f"  {q['question_id']:<26} predicted: {q['predicted_category']}")

    print("\nRetrieval failures for the full system (Hybrid + category, not in top 3):")

    for row in rows_by_method["Hybrid + category"]:
        if row["rank"] is None or row["rank"] > 3:
            print(f"  {row['question_id']:<26} got: {row['top3']}")

    # ---------------- CSV files ----------------

    with open(RESULTS_DIR / "retrieval_details.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    with open(RESULTS_DIR / "retrieval_summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["subset", "method", "n", "Hit@1", "Hit@3", "MRR@10"])

        subsets = {
            "all": lambda r: True,
            "english": lambda r: r["language"] == "english",
            "italian": lambda r: r["language"] == "italian",
            "specific": lambda r: r["type"] == "specific",
            "vague": lambda r: r["type"] == "vague",
        }

        for subset, keep in subsets.items():
            for method in METHODS:
                summary = summarize([r for r in rows_by_method[method] if keep(r)])
                writer.writerow([
                    subset, method, summary["n"],
                    round(summary["Hit@1"], 3),
                    round(summary["Hit@3"], 3),
                    round(summary["MRR@10"], 3),
                ])


# ============================================================
# 5. PART 2 - ANSWERS
# ============================================================

def automatic_checks(answer, language, expected_language, retrieved_count):

    headers = rag_engine.SECTION_HEADERS[language]

    continue_header = (
        "SE IL PROBLEMA PERSISTE"
        if language == "Italian"
        else "IF THE PROBLEM CONTINUES"
    )

    has_sections = all(
        header in answer
        for header in headers
        if header != "POSSIBLE CAUSES"
    )

    steps = re.findall(r"^\s*[1-9]\.\s.*$", answer, flags=re.MULTILINE)

    steps_cited = sum("[Source" in step for step in steps)

    cited = {int(n) for n in re.findall(r"\[Source (\d+)\]", answer)}

    invalid_citations = sorted(n for n in cited if n < 1 or n > retrieved_count)

    # Compare with the TRUE language of the question, not the
    # language the system detected (that would hide detection errors).
    answer_language = rag_engine.detect_answer_language(answer)

    return {
        "has_all_sections": has_sections,
        "n_steps": len(steps),
        "steps_with_citation": f"{steps_cited}/{len(steps)}",
        "invalid_citations": ";".join(map(str, invalid_citations)),
        "language_ok": (
            answer_language == expected_language
            and language == expected_language
        ),
        "asks_question": "?" in answer.split(continue_header)[-1],
    }


def evaluate_answers(questions, kb):

    index, chunks, embedding_model, bm25_data = kb

    rows = []

    for number, question in enumerate(questions, start=1):

        print(f"  [{number}/{len(questions)}] {question['question_id']}", flush=True)

        start = time.perf_counter()

        result = quiet(
            rag_engine.generate_rag_answer,
            question=question["question"],
            index=index,
            chunks=chunks,
            embedding_model=embedding_model,
            bm25_data=bm25_data,
            top_k=3,
            predicted_category=question["predicted_category"],
            classification_confidence=question["confidence"],
        )

        retrieved_ids = [
            source["chunk"]["chunk_id"] for source in result["sources"]
        ]

        checks = automatic_checks(
            result["answer"],
            result["language"],
            question["language"].capitalize(),
            len(retrieved_ids),
        )

        rows.append({
            "question_id": question["question_id"],
            "language": question["language"],
            "type": question["type"],
            "question": question["question"],
            "predicted_category": question["predicted_category"],
            "confidence": question["confidence"],
            "retrieved_chunks": ";".join(retrieved_ids),
            "relevant_in_top3": any(c in question["relevant"] for c in retrieved_ids),
            "refused": bool(result.get("out_of_scope")),
            "answer": result["answer"],
            **checks,
            "response_time_s": round(time.perf_counter() - start, 2),
            # Fill these in by hand: 0 = no, 1 = partly, 2 = yes
            "correct_manual": "",
            "grounded_manual": "",
            "notes": "",
        })

    model_name = rag_engine.LLM_MODEL.replace(":", "-")

    output_path = RESULTS_DIR / f"answers_{model_name}.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    in_scope = [row for row in rows if row["type"] != "out_of_scope"]

    print("\n" + "=" * 72)
    print(f"ANSWERS - {rag_engine.LLM_MODEL}")
    print("=" * 72)
    answered = [row for row in rows if not row["refused"]]

    print(f"Refused (out of scope): {len(rows) - len(answered)}/{len(rows)}")
    print(f"All sections present:   {sum(r['has_all_sections'] for r in answered)}/{len(answered)} answered")
    print(f"Correct language:       {sum(r['language_ok'] for r in rows)}/{len(rows)}")

    total_steps = sum(r["n_steps"] for r in in_scope)
    cited_steps = sum(int(r["steps_with_citation"].split("/")[0]) for r in in_scope)

    if total_steps:
        print(f"Steps with a citation:  {cited_steps}/{total_steps} ({cited_steps / total_steps:.0%})")

    print(f"Invalid citations:      {sum(bool(r['invalid_citations']) for r in rows)} answers")

    vague = [r for r in rows if r["type"] == "vague"]

    if vague:
        print(f"Vague -> asks question: {sum(r['asks_question'] for r in vague)}/{len(vague)}")

    out_of_scope = [r for r in rows if r["type"] == "out_of_scope"]

    if out_of_scope:
        print("\nOut-of-scope questions (should say: no documentation / contact IT):")
        for row in out_of_scope:
            print(f"  {row['question_id']:<22} -> {row['answer'][:110]!r}")

    average_time = sum(r["response_time_s"] for r in rows) / len(rows)
    print(f"\nAverage response time:  {average_time:.1f} s")
    print(f"\nSaved: {output_path.relative_to(PROJECT_DIR)}")
    print("Open it and fill in correct_manual and grounded_manual (0/1/2).")


# ============================================================
# 6. MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Evaluation started {datetime.now():%Y-%m-%d %H:%M}")

    questions = load_questions()

    print(f"Questions: {len(questions)}")

    index, chunks, embedding_model = quiet(rag_engine.load_knowledge_base)
    bm25_data = rag_engine.prepare_bm25(chunks)
    kb = (index, chunks, embedding_model, bm25_data)

    print(f"Indexed chunks: {len(chunks)}")

    classifier = quiet(load_classifier)

    evaluate_retrieval(questions, kb, classifier)

    if "--generate" in sys.argv:
        print(f"\nGenerating answers with {rag_engine.LLM_MODEL} ...")
        evaluate_answers(questions, kb)

    print(f"\nResults saved in {RESULTS_DIR.relative_to(PROJECT_DIR)}/")


if __name__ == "__main__":
    main()
