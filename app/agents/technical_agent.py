def technical_question_prompt(role, resume, job_description, rag_context=""):
    return f"""
You are a technical interviewer.

Generate exactly 5 technical interview questions.

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
      "type": "technical",
      "difficulty": "medium",
      "question": "question here"
    }}
  ]
}}

Rules:
- Ask role-specific technical questions.
- Use resume and JD context.
- Include practical scenario-based questions.
- Do not provide answers.
- Do not write markdown.
"""