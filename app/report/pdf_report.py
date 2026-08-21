from io import BytesIO

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)


def create_score_chart(score):
    buffer = BytesIO()

    labels = ["Score", "Remaining"]
    values = [score, max(0, 100 - score)]

    plt.figure(figsize=(4, 4))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Interview Score")
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    plt.close()

    buffer.seek(0)
    return buffer


def safe_text(value):
    if value is None:
        return ""
    return str(value).replace("\n", "<br/>")


def generate_interview_pdf(
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
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=16
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=10,
        spaceAfter=6
    )

    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14
    )

    story = []

    story.append(Paragraph("AI Interview Assistant Report", title_style))
    story.append(Spacer(1, 10))

    summary_data = [
        ["Role", safe_text(role)],
        ["Score", f"{score}/100"],
        ["Date", safe_text(created_at)]
    ]

    summary_table = Table(summary_data, colWidths=[1.5 * inch, 4.8 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 1, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 14))

    chart_buffer = create_score_chart(int(score or 0))
    story.append(Image(chart_buffer, width=3 * inch, height=3 * inch))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Interview Question", heading_style))
    story.append(Paragraph(safe_text(question), normal_style))

    story.append(Paragraph("Candidate Answer", heading_style))
    story.append(Paragraph(safe_text(answer), normal_style))

    story.append(Paragraph("Technical Feedback", heading_style))
    story.append(Paragraph(safe_text(technical_feedback), normal_style))

    story.append(Paragraph("Communication Feedback", heading_style))
    story.append(Paragraph(safe_text(communication_feedback), normal_style))

    story.append(Paragraph("Missing Points", heading_style))
    story.append(Paragraph(safe_text(missing_points), normal_style))

    story.append(Paragraph("Improved Answer", heading_style))
    story.append(Paragraph(safe_text(improved_answer), normal_style))

    story.append(Paragraph("Final Suggestion", heading_style))
    story.append(Paragraph(safe_text(final_suggestion), normal_style))

    doc.build(story)

    buffer.seek(0)
    return buffer

def generate_full_report_pdf(
    role,
    average_score,
    results_list,
    created_at
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=16
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=10,
        spaceAfter=6
    )
    
    agent_heading_style = ParagraphStyle(
        "AgentHeading",
        parent=styles["Heading1"],
        fontSize=16,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.darkblue
    )

    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14
    )

    story = []

    story.append(Paragraph("Full AI Interview Report", title_style))
    story.append(Spacer(1, 10))

    summary_data = [
        ["Role", safe_text(role)],
        ["Average Score", f"{average_score:.2f}/100"],
        ["Date", safe_text(created_at)]
    ]

    summary_table = Table(summary_data, colWidths=[1.5 * inch, 4.8 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 1, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 14))

    chart_buffer = create_score_chart(int(average_score or 0))
    story.append(Image(chart_buffer, width=3 * inch, height=3 * inch))
    story.append(Spacer(1, 14))

    for res in results_list:
        story.append(Paragraph(f"Round: {res.get('agent', 'Unknown')}", agent_heading_style))
        story.append(Paragraph(f"Score: {res.get('final_score', 0)}/100", normal_style))
        
        story.append(Paragraph("Interview Question", heading_style))
        story.append(Paragraph(safe_text(res.get("question", "")), normal_style))

        story.append(Paragraph("Candidate Answer", heading_style))
        story.append(Paragraph(safe_text(res.get("candidate_answer", "")), normal_style))

        story.append(Paragraph("Technical Feedback", heading_style))
        story.append(Paragraph(safe_text(res.get("technical_feedback", "")), normal_style))

        story.append(Paragraph("Communication Feedback", heading_style))
        story.append(Paragraph(safe_text(res.get("communication_feedback", "")), normal_style))

        story.append(Paragraph("Improved Answer", heading_style))
        story.append(Paragraph(safe_text(res.get("improved_answer", "")), normal_style))

        story.append(Paragraph("Final Suggestion", heading_style))
        story.append(Paragraph(safe_text(res.get("final_suggestion", "")), normal_style))
        
        story.append(Spacer(1, 20))

    doc.build(story)

    buffer.seek(0)
    return buffer
