from app.report.pdf_report import generate_interview_pdf, generate_full_report_pdf

def generate_report_pdf(
    role,
    question,
    answer,
    score,
    technical_feedback,
    communication_feedback,
    missing_points,
    improved_answer,
    final_suggestion,
    created_at
):
    """
    Report Agent function to generate the final interview PDF.
    """
    return generate_interview_pdf(
        role=role,
        question=question,
        answer=answer,
        score=score,
        technical_feedback=technical_feedback,
        communication_feedback=communication_feedback,
        missing_points=missing_points,
        improved_answer=improved_answer,
        final_suggestion=final_suggestion,
        created_at=created_at
    )

def generate_full_report_pdf_agent(
    role,
    average_score,
    results_list,
    created_at
):
    """
    Report Agent function to generate the final full interview PDF.
    """
    return generate_full_report_pdf(
        role=role,
        average_score=average_score,
        results_list=results_list,
        created_at=created_at
    )
