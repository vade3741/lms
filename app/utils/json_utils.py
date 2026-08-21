import json
import re


def extract_json(text):
    if not text:
        return None

    normalized = str(text).strip().lower()
    if normalized.startswith("error:") or "ollama timed out" in normalized or "ollama is not running" in normalized:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group()
            try:
                return json.loads(json_str)
            except Exception:
                # Sanitize common LLM JSON syntax errors (control chars / newlines)
                clean_str = re.sub(r'[\r\n\t]', ' ', json_str)
                return json.loads(clean_str)
    except Exception:
        pass

    return None


def get_fallback_questions(agent_type="Technical Agent", role="Software Engineer"):
    agent_str = str(agent_type).lower() if agent_type else ""
    role_str = str(role) if role else "Software Developer"

    if "coding" in agent_str:
        return [
            {
                "id": 1,
                "type": "coding",
                "difficulty": "easy",
                "question": "Write a Python function `two_sum(nums, target)` that returns the indices of the two numbers in `nums` that add up to `target`.",
                "test_cases": "assert two_sum([2, 7, 11, 15], 9) == [0, 1]"
            },
            {
                "id": 2,
                "type": "coding",
                "difficulty": "medium",
                "question": "Write a Python function `is_palindrome(s)` that returns True if string `s` is a palindrome (ignoring non-alphanumeric characters and case), else False.",
                "test_cases": "assert is_palindrome('A man, a plan, a canal: Panama') == True"
            },
            {
                "id": 3,
                "type": "coding",
                "difficulty": "medium",
                "question": "Write a Python function `find_duplicates(nums)` that returns a sorted list of all duplicate elements in `nums`.",
                "test_cases": "assert sorted(find_duplicates([1, 2, 3, 2, 4, 3])) == [2, 3]"
            }
        ]
    elif "hr" in agent_str:
        return [
            {
                "id": 1,
                "type": "hr",
                "difficulty": "easy",
                "question": f"Tell me about yourself and your key technical accomplishments relevant to the {role_str} role.",
                "test_cases": ""
            },
            {
                "id": 2,
                "type": "hr",
                "difficulty": "medium",
                "question": "Describe a challenging situation or project deadline conflict and how you successfully navigated it.",
                "test_cases": ""
            },
            {
                "id": 3,
                "type": "hr",
                "difficulty": "medium",
                "question": f"Where do you see yourself professionally in 3-5 years, and how does this {role_str} position align with your career goals?",
                "test_cases": ""
            }
        ]
    elif "system" in agent_str or "design" in agent_str:
        return [
            {
                "id": 1,
                "type": "system_design",
                "difficulty": "medium",
                "question": "Design a high-level URL Shortening service (like Bitly). Detail your database choice, API design, and hash collision strategy.",
                "test_cases": ""
            },
            {
                "id": 2,
                "type": "system_design",
                "difficulty": "hard",
                "question": "How would you design a real-time push notification system capable of sending millions of events per minute reliably?",
                "test_cases": ""
            },
            {
                "id": 3,
                "type": "system_design",
                "difficulty": "medium",
                "question": "Explain how Load Balancers, DB Replication, and Caching (e.g., Redis) scale a high-traffic web application.",
                "test_cases": ""
            }
        ]
    else:
        return [
            {
                "id": 1,
                "type": "technical",
                "difficulty": "medium",
                "question": f"What are the core technical concepts, design patterns, and best practices essential for a {role_str}?",
                "test_cases": ""
            },
            {
                "id": 2,
                "type": "technical",
                "difficulty": "medium",
                "question": "Explain memory management, garbage collection, or concurrency model in your primary programming language.",
                "test_cases": ""
            },
            {
                "id": 3,
                "type": "technical",
                "difficulty": "medium",
                "question": "How do you design, test, and debug high-performance microservices or backend APIs?",
                "test_cases": ""
            }
        ]


def extract_questions_fallback(text):
    if not text:
        return []

    normalized = str(text).strip().lower()
    if normalized.startswith("error:") or "ollama timed out" in normalized or "ollama is not running" in normalized:
        return []

    questions = []

    pattern = r'"question"\s*:\s*"([^"]+)'
    matches = re.findall(pattern, text)

    for index, question in enumerate(matches, start=1):
        questions.append({
            "id": index,
            "type": "general",
            "difficulty": "medium",
            "question": question.strip()
        })

    if questions:
        return questions

    lines = text.splitlines()

    for line in lines:
        line = line.strip()

        match = re.match(r"^\d+[\.\)]\s*(.+)", line)
        if match:
            questions.append({
                "id": len(questions) + 1,
                "type": "general",
                "difficulty": "medium",
                "question": match.group(1).strip()
            })

    return questions


def get_questions_from_response(response_text, agent_type=None, role=None):
    data = extract_json(response_text)

    if data:
        questions = data.get("questions", [])

        clean_questions = []

        for item in questions:
            question_text = item.get("question", "").strip()

            if question_text:
                clean_questions.append({
                    "id": item.get("id", len(clean_questions) + 1),
                    "type": item.get("type", "general"),
                    "difficulty": item.get("difficulty", "medium"),
                    "question": question_text,
                    "test_cases": item.get("test_cases", "")
                })

        if clean_questions:
            return clean_questions

    fallback_parsed = extract_questions_fallback(response_text)
    if fallback_parsed:
        return fallback_parsed

    return get_fallback_questions(agent_type=agent_type, role=role)