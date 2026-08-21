from app.agents.hr_agent import hr_question_prompt
from app.agents.technical_agent import technical_question_prompt
from app.agents.coding_agent import coding_question_prompt
from app.agents.system_design_agent import system_design_question_prompt
from app.agents.evaluation_agent import agent_evaluation_prompt


def get_question_prompt_by_round(
    agent_type,
    role,
    resume,
    job_description,
    rag_context=""
):
    if agent_type in ["HR Round", "HR Agent"]:
        return hr_question_prompt(role, resume, job_description, rag_context)

    if agent_type in ["Technical Round", "Technical Agent"]:
        return technical_question_prompt(role, resume, job_description, rag_context)

    if agent_type in ["Coding Round", "Coding Agent"]:
        return coding_question_prompt(role, resume, job_description, rag_context)

    if agent_type in ["System Design Round", "System Design Agent"]:
        return system_design_question_prompt(role, resume, job_description, rag_context)

    return technical_question_prompt(role, resume, job_description, rag_context)


def get_evaluation_prompt_by_round(
    agent_type,
    role,
    question,
    answer,
    rag_context="",
    execution_output=""
):
    return agent_evaluation_prompt(
        role=role,
        round_type=agent_type,
        question=question,
        answer=answer,
        rag_context=rag_context,
        execution_output=execution_output
    )