def hr_question_prompt(role, resume, job_description, rag_context=""):
    return f"""
You are an HR interviewer.

Generate exactly 3 HR interview questions for this candidate.

Role:
{role}

Resume Context:
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
      "type": "hr",
      "difficulty": "easy",
      "question": "question here"
    }}
  ]
}}

Rules:
- Ask about introduction, motivation, teamwork, strengths, weaknesses, and career goals.
- Include case-based behavioral scenarios and thinking ability questions to test their decision-making.
- The questions MUST be highly tailored and specific to the candidate's Resume Context and Job Description.
- Do not ask technical coding questions.
- Do not write markdown.
"""