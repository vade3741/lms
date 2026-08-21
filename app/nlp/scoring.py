import re
# pyrefly: ignore [missing-import]
import numpy as np

from app.rag.rag_engine import get_embedding


def clean_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if denominator == 0:
        return 0

    return float(np.dot(vec1, vec2) / denominator)


def keyword_match_score(answer, expected_keywords):
    if not expected_keywords:
        return 0, []

    answer_clean = clean_text(answer)

    matched_keywords = []
    missing_keywords = []

    for keyword in expected_keywords:
        keyword_clean = clean_text(keyword)

        if keyword_clean and keyword_clean in answer_clean:
            matched_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    score = int((len(matched_keywords) / len(expected_keywords)) * 100)

    return score, missing_keywords


def semantic_similarity_score(candidate_answer, ideal_answer):
    if not candidate_answer or not ideal_answer:
        return 0

    try:
        answer_embedding = get_embedding(candidate_answer)
        ideal_embedding = get_embedding(ideal_answer)

        similarity = cosine_similarity(answer_embedding, ideal_embedding)

        score = int(max(0, min(similarity * 100, 100)))

        return score
    except Exception:
        return 0


def answer_quality_score(answer):
    if not answer:
        return 0

    words = answer.split()
    word_count = len(words)

    if word_count < 20:
        return 40

    if 20 <= word_count < 50:
        return 65

    if 50 <= word_count <= 150:
        return 90

    if 150 < word_count <= 250:
        return 80

    return 65


def calculate_hybrid_score(
    candidate_answer,
    ideal_answer,
    expected_keywords,
    llm_score
):
    keyword_score, missing_keywords = keyword_match_score(
        candidate_answer,
        expected_keywords
    )

    semantic_score = semantic_similarity_score(
        candidate_answer,
        ideal_answer
    )

    quality_score = answer_quality_score(candidate_answer)

    final_score = (
        llm_score * 0.50 +
        keyword_score * 0.20 +
        semantic_score * 0.20 +
        quality_score * 0.10
    )

    final_score = int(round(final_score))

    return {
        "final_score": final_score,
        "llm_score": llm_score,
        "keyword_score": keyword_score,
        "semantic_score": semantic_score,
        "answer_quality_score": quality_score,
        "missing_keywords": missing_keywords
    }