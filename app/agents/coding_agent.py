def coding_question_prompt(role, resume, job_description, rag_context=""):
    return f"""
You are a coding round interviewer.

Generate exactly 3 coding interview questions for the candidate.

Role:
{role}

Resume:
{resume}

Job Description:
{job_description}

Relevant RAG Context:
{rag_context}

Return only valid JSON:

{{
  "questions": [
    {{
      "id": 1,
      "type": "coding",
      "difficulty": "medium",
      "question": "coding problem here",
      "test_cases": "assert function_name(args) == expected_output"
    }}
  ]
}}

Rules:
- Questions should be solvable in Python.
- Include problem statement clearly.
- Include input/output expectation if needed.
- Do not provide solution.
- Do not write markdown.
- `test_cases` must be valid Python assert statements that test the candidate's solution.
"""