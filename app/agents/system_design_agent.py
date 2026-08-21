def system_design_question_prompt(role, resume, job_description, rag_context=""):
    return f"""
You are a system design interviewer.

Generate exactly 3 system design questions.

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
      "type": "system_design",
      "difficulty": "medium",
      "question": "question here"
    }}
  ]
}}

Rules:
- Ask architecture and scalability questions.
- Prefer questions related to backend, APIs, databases, AI systems, or RAG if relevant.
- Do not provide answers.
- Do not write markdown.
"""