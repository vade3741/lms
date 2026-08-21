import json
from collections import Counter, defaultdict


def safe_json_loads(value):
    try:
        return json.loads(value)
    except Exception:
        return []


def extract_weak_areas_from_history(rows):
    """
    rows format from get_interviews():
    id, role, question, answer, score, technical_feedback,
    communication_feedback, missing_points, improved_answer,
    final_suggestion, created_at
    """

    weak_counter = Counter()
    role_scores = defaultdict(list)
    low_score_questions = []
    total_scores = []

    for row in rows:
        interview_id = row[0]
        role = row[1]
        question = row[2]
        answer = row[3]
        score = row[4]
        technical_feedback = row[5]
        communication_feedback = row[6]
        missing_points = row[7]
        improved_answer = row[8]
        final_suggestion = row[9]
        created_at = row[10]

        try:
            score = int(score)
        except Exception:
            score = 0

        total_scores.append(score)
        role_scores[role].append(score)

        points = safe_json_loads(missing_points)

        if isinstance(points, list):
            for point in points:
                clean_point = str(point).strip()
                if clean_point:
                    weak_counter[clean_point] += 1
        else:
            clean_point = str(missing_points).strip()
            if clean_point:
                weak_counter[clean_point] += 1

        if score < 70:
            low_score_questions.append({
                "role": role,
                "question": question,
                "score": score,
                "technical_feedback": technical_feedback,
                "final_suggestion": final_suggestion,
                "created_at": created_at
            })

    average_score = 0
    if total_scores:
        average_score = round(sum(total_scores) / len(total_scores), 2)

    role_average_scores = {}

    for role, scores in role_scores.items():
        if scores:
            role_average_scores[role] = round(sum(scores) / len(scores), 2)

    top_weak_areas = weak_counter.most_common(10)

    return {
        "average_score": average_score,
        "role_average_scores": role_average_scores,
        "top_weak_areas": top_weak_areas,
        "low_score_questions": low_score_questions
    }


def generate_learning_plan(weak_areas):
    plan = []

    for weak_area, count in weak_areas:
        topic = str(weak_area)

        plan.append({
            "weak_area": topic,
            "frequency": count,
            "study_action": f"Revise and practice: {topic}",
            "practice_task": f"Write 2 interview answers and solve 3 examples related to {topic}.",
            "priority": "High" if count >= 2 else "Medium"
        })

    return plan


def get_performance_level(score):
    if score >= 85:
        return "Strong"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Needs Improvement"
    return "Weak"