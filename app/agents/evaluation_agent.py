def agent_evaluation_prompt(role, round_type, question, answer, rag_context="", execution_output=""):
    prompt = f"""
You are an interview evaluation agent.

Evaluate the candidate's answer.

Role:
{role}

Round Type:
{round_type}

Question:
{question}

Candidate Answer:
{answer}
"""
    if execution_output:
        prompt += f"""
Code Execution Output:
{execution_output}
"""
    prompt += f"""
Relevant RAG Context:
{rag_context}

Return only valid JSON:

{{
  "score": 75,
  "technical_feedback": "feedback here",
  "communication_feedback": "feedback here",
  "missing_points": ["point 1", "point 2"],
  "improved_answer": "improved answer here",
  "ideal_answer": "ideal answer here",
  "expected_keywords": ["keyword1", "keyword2", "keyword3"],
  "final_suggestion": "suggestion here"
}}

Rules:
- Score must be between 0 and 100.
- For HR round, focus more on communication and clarity.
- For technical round, focus on correctness.
- For coding round, focus on logic, edge cases, and complexity.
- For system design round, focus on architecture, scalability, database, API design, and trade-offs.
- Do not write markdown.
"""
    return prompt