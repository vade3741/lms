def _truncate(text, limit):
  if not text:
    return ""

  text = str(text).strip()
  if len(text) <= limit:
    return text

  return text[:limit].rstrip() + "..."


def generate_questions_prompt(
  role,
  resume,
  job_description,
  rag_context="",
  resume_limit=700,
  job_description_limit=700,
  rag_context_limit=800
):
    return f"""
Generate 2 short interview questions for {role}.
Use the resume, job description, and RAG context below.

Resume: {_truncate(resume, resume_limit)}

Job Description: {_truncate(job_description, job_description_limit)}

RAG Context: {_truncate(rag_context, rag_context_limit)}

Return compact valid JSON only in this exact shape:
{{"questions":[{{"id":1,"type":"resume_based","difficulty":"easy","question":"..."}},{{"id":2,"type":"technical","difficulty":"medium","question":"..."}}]}}

Rules:
- Keep every question under 15 words.
- Output JSON only.
"""


def evaluate_answer_prompt(
  role,
  question,
  answer,
  rag_context="",
  rag_context_limit=1200,
  answer_limit=1200
):
    return f"""
You are an interview evaluator.

Evaluate the candidate's answer using the question, answer, and relevant resume/JD context.

Target Role:
{role}

Relevant Resume/JD Context from RAG:
{_truncate(rag_context, rag_context_limit)}

Interview Question:
{question}

Candidate Answer:
{_truncate(answer, answer_limit)}

Return only valid JSON in this format:

{{
  "score": 75,
  "technical_feedback": "feedback here",
  "communication_feedback": "feedback here",
  "missing_points": ["point 1", "point 2"],
  "improved_answer": "better answer here",
  "ideal_answer": "ideal answer here",
  "expected_keywords": ["keyword 1", "keyword 2", "keyword 3"],
  "final_suggestion": "suggestion here"
}}

Rules:
- Score must be between 0 and 100.
- expected_keywords must contain 5 to 10 important technical keywords.
- ideal_answer should be concise, ideally 4 to 6 sentences.
- Check if the answer is relevant to the role.
- Check if the answer matches resume/JD context.
- Be honest but beginner-friendly.
- Do not write markdown.
- Do not write explanation outside JSON.
"""