import sys
from pathlib import Path
import json

import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from app.analytics.weak_area_tracker import (
    extract_weak_areas_from_history,
    generate_learning_plan,
    get_performance_level
)
from app.agents.report_agent import generate_report_pdf, generate_full_report_pdf_agent
from app.speech.transcriber import save_audio_file, transcribe_audio
from app.nlp.scoring import calculate_hybrid_score
from app.config import get_settings
from app.rag.rag_engine import (
    PDF_DIRECTORY,
    build_rag_index,
    retrieve_context,
    retrieve_study_context
)
from app.database.db import (
    get_practice_history,
    get_study_documents,
    get_study_notes,
    get_interviews,
    init_db,
    save_interview
)
from app.llm.ollama_client import ask_ollama, check_ollama_status
from app.agents.orchestrator import (
    get_question_prompt_by_round,
    get_evaluation_prompt_by_round
)
from app.utils.json_utils import extract_json, get_questions_from_response
from app.utils.file_reader import extract_text_from_file
from app.execution.code_runner import run_python_code
from app.speech.tts import render_tts_button, text_to_speech_bytes
from app.study_workflow import (
    add_pdf_files,
    add_manual_question,
    evaluate_study_answer,
    generate_pdf_notes,
    generate_pdf_questions,
    generate_web_questions,
    get_question_bank,
    get_saved_web_sources,
    import_web_source,
    remove_pdf,
    search_public_questions,
    sync_pdf_library
)


init_db()

st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("AI Interview Assistant")
st.write(
    "Free local AI interview practice app using Streamlit, Ollama, SQLite, file upload, and RAG."
)


if "raw_questions" not in st.session_state:
    st.session_state.raw_questions = ""

if "questions_list" not in st.session_state:
    st.session_state.questions_list = []

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""

if "interview_mode" not in st.session_state:
    st.session_state.interview_mode = "Practice Specific Round"

if "current_stage_index" not in st.session_state:
    st.session_state.current_stage_index = 0

if "full_interview_results" not in st.session_state:
    st.session_state.full_interview_results = []

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

with st.sidebar:
    st.header("Local AI setup")
    if (
        "ollama_status" not in st.session_state
        or st.button("Refresh Ollama model status")
    ):
        st.session_state.ollama_status = check_ollama_status()

    ollama_status = st.session_state.ollama_status
    if not ollama_status["service_available"]:
        st.error(ollama_status["error"])
    else:
        chat_label = (
            "available" if ollama_status["chat_model_available"] else "missing"
        )
        embedding_label = (
            "available"
            if ollama_status["embedding_model_available"]
            else "missing"
        )
        st.write(
            f"Chat model `{ollama_status['chat_model']}`: **{chat_label}**"
        )
        st.write(
            f"Embedding model `{ollama_status['embedding_model']}`: "
            f"**{embedding_label}**"
        )
        if not (
            ollama_status["chat_model_available"]
            and ollama_status["embedding_model_available"]
        ):
            settings = get_settings()
            missing_models = []
            if not ollama_status["chat_model_available"]:
                missing_models.append(settings.chat_model)
            if not ollama_status["embedding_model_available"]:
                missing_models.append(settings.embedding_model)
            st.caption("Install a missing model with:")
            st.code("\n".join(f"ollama pull {model}" for model in missing_models))

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Start Interview",
    "Interview History",
    "Dashboard",
    "Learning Plan",
    "Study & Revision"
])


with tab1:
    st.header("Start New Interview")

    st.session_state.interview_mode = st.radio(
        "Select Interview Mode",
        ["Practice Specific Round", "Full Mock Interview"],
        horizontal=True
    )
    
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        role = st.text_input(
            "Target Role",
            placeholder="Example: Python Developer, AI/ML Engineer, Full Stack Developer"
        )

    with col2:
        if st.session_state.interview_mode == "Practice Specific Round":
            selected_agent = st.selectbox(
                "Select Interview Agent",
                ["HR Agent", "Technical Agent", "Coding Agent", "System Design Agent"]
            )
        else:
            selected_agent = "Full Mock Interview"
            st.info("The AI will guide you through HR, Technical, Coding, and System Design rounds automatically.")

    st.subheader("Resume Input")

    resume_input_mode = st.radio(
        "Choose resume input method",
        ["Upload Resume File", "Paste Resume Text"],
        horizontal=True
    )

    resume = ""

    if resume_input_mode == "Upload Resume File":
        resume_file = st.file_uploader(
            "Upload Resume PDF/TXT",
            type=["pdf", "txt"],
            key="resume_file"
        )

        if resume_file is not None:
            if st.button("Extract Resume Text"):
                with st.spinner("Extracting resume text..."):
                    st.session_state.resume_text = extract_text_from_file(resume_file)

            if st.session_state.resume_text:
                st.success("Resume text extracted successfully.")
                resume = st.text_area(
                    "Extracted Resume Text",
                    value=st.session_state.resume_text,
                    height=180
                )
            else:
                resume = ""
    else:
        resume = st.text_area(
            "Paste Your Resume Text",
            height=180,
            placeholder="Paste your resume content here..."
        )

    st.subheader("Job Description Input")

    jd_input_mode = st.radio(
        "Choose JD input method",
        ["Upload JD File", "Paste JD Text"],
        horizontal=True
    )

    job_description = ""

    if jd_input_mode == "Upload JD File":
        jd_file = st.file_uploader(
            "Upload Job Description PDF/TXT",
            type=["pdf", "txt"],
            key="jd_file"
        )

        if jd_file is not None:
            if st.button("Extract JD Text"):
                with st.spinner("Extracting job description text..."):
                    st.session_state.jd_text = extract_text_from_file(jd_file)

            if st.session_state.jd_text:
                st.success("Job description text extracted successfully.")
                job_description = st.text_area(
                    "Extracted Job Description Text",
                    value=st.session_state.jd_text,
                    height=180
                )
            else:
                job_description = ""
    else:
        job_description = st.text_area(
            "Paste Job Description",
            height=180,
            placeholder="Paste the job description here..."
        )

    st.divider()

    if st.session_state.interview_mode == "Practice Specific Round":
            if st.button("Generate Interview Questions"):
                if not role or not resume or not job_description:
                    st.warning("Please fill role, resume, and job description.")
                else:
                    try:
                        with st.spinner("Building local RAG index from resume and JD..."):
                            total_chunks = build_rag_index(resume, job_description)
                            st.session_state.rag_ready = True
                            st.success(f"RAG index created successfully with {total_chunks} chunks.")
                    except Exception as e:
                        st.error(f"⚠️ **RAG Index Error**: {str(e)}")
                        st.stop()

                    with st.spinner("Retrieving relevant context..."):
                        rag_query = f"{role} interview questions based on resume and job description"
                        rag_context = retrieve_context(rag_query, top_k=1)

                    with st.spinner(f"{selected_agent} is generating questions using local Ollama model + RAG..."):
                        # Step 10 Replacement: Dynamic prompt generation by agent
                        prompt = get_question_prompt_by_round(
                            agent_type=selected_agent,
                            role=role,
                            resume=resume,
                            job_description=job_description,
                            rag_context=rag_context
                        )

                        response = ask_ollama(prompt)

                        st.session_state.raw_questions = response
                        st.session_state.questions_list = get_questions_from_response(response, agent_type=selected_agent, role=role)
                        st.session_state.user_answers = {}
                        st.session_state.all_evaluations = []

            if st.session_state.questions_list:
                st.subheader("📋 Generated Interview Questions")
                st.info("Answer all questions below. When you're ready, click 'Submit All Answers & Evaluate' at the bottom.")

                if "user_answers" not in st.session_state:
                    st.session_state.user_answers = {}

                for idx, q in enumerate(st.session_state.questions_list):
                    q_text = q.get("question", f"Question {idx+1}")
                    st.markdown(f"#### Question {idx+1}: {q_text}")
                    render_tts_button(q_text, button_label=f"🔊 Read Question {idx+1} Aloud", key=f"tts_spec_{idx}")

                    ans_key = f"text_ans_spec_{idx}"
                    val = st.session_state.user_answers.get(idx, "")

                    user_ans = st.text_area(
                        f"Your Answer for Question {idx+1}",
                        value=val,
                        height=140,
                        key=ans_key,
                        placeholder="Write your answer here..."
                    )
                    st.session_state.user_answers[idx] = user_ans
                    st.divider()

                if st.button("Submit All Answers & Evaluate", type="primary"):
                    answered_count = sum(1 for idx in range(len(st.session_state.questions_list)) if st.session_state.user_answers.get(idx, "").strip())
                    if answered_count == 0:
                        st.warning("Please provide an answer to at least one question before submitting.")
                    else:
                        all_evals = []
                        total_q = len(st.session_state.questions_list)
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for idx, q in enumerate(st.session_state.questions_list):
                            q_text = q.get("question", "")
                            candidate_answer = st.session_state.user_answers.get(idx, "").strip()
                            if not candidate_answer:
                                continue

                            status_text.text(f"Evaluating Question {idx+1}/{total_q}...")
                            progress_bar.progress((idx + 1) / total_q)

                            test_cases = q.get("test_cases", "")
                            execution_output = ""
                            if selected_agent == "Coding Agent":
                                passed, exec_out = run_python_code(candidate_answer, test_cases)
                                execution_output = exec_out

                            rag_context = retrieve_context(q_text, top_k=1)
                            eval_prompt = get_evaluation_prompt_by_round(
                                agent_type=selected_agent,
                                role=role,
                                question=q_text,
                                answer=candidate_answer,
                                rag_context=rag_context,
                                execution_output=execution_output
                            )

                            raw_evaluation = ask_ollama(eval_prompt)
                            evaluation_data = extract_json(raw_evaluation)

                            if evaluation_data:
                                llm_score = int(evaluation_data.get("score", 0))
                                technical_feedback = evaluation_data.get("technical_feedback", "")
                                communication_feedback = evaluation_data.get("communication_feedback", "")
                                missing_points = evaluation_data.get("missing_points", [])
                                improved_answer = evaluation_data.get("improved_answer", "")
                                ideal_answer = evaluation_data.get("ideal_answer", improved_answer)
                                expected_keywords = evaluation_data.get("expected_keywords", [])
                                final_suggestion = evaluation_data.get("final_suggestion", "")

                                hybrid_result = calculate_hybrid_score(
                                    candidate_answer=candidate_answer,
                                    ideal_answer=ideal_answer,
                                    expected_keywords=expected_keywords,
                                    llm_score=llm_score
                                )

                                final_score = hybrid_result["final_score"]

                                evaluation_data["question"] = q_text
                                evaluation_data["candidate_answer"] = candidate_answer
                                evaluation_data["llm_score"] = hybrid_result["llm_score"]
                                evaluation_data["keyword_score"] = hybrid_result["keyword_score"]
                                evaluation_data["semantic_score"] = hybrid_result["semantic_score"]
                                evaluation_data["answer_quality_score"] = hybrid_result["answer_quality_score"]
                                evaluation_data["final_score"] = hybrid_result["final_score"]
                                evaluation_data["missing_keywords"] = hybrid_result["missing_keywords"]
                                evaluation_data["execution_output"] = execution_output

                                save_interview(
                                    role=role,
                                    resume=resume,
                                    job_description=job_description,
                                    question=q_text,
                                    answer=candidate_answer,
                                    score=final_score,
                                    technical_feedback=technical_feedback,
                                    communication_feedback=communication_feedback,
                                    missing_points=json.dumps(missing_points),
                                    improved_answer=improved_answer,
                                    final_suggestion=final_suggestion,
                                    raw_evaluation=json.dumps(evaluation_data)
                                )

                                all_evals.append(evaluation_data)

                        progress_bar.empty()
                        status_text.empty()
                        st.session_state.all_evaluations = all_evals

            if st.session_state.get("all_evaluations"):
                evals = st.session_state.all_evaluations
                st.divider()
                st.subheader("📊 Full Candidate Evaluation Report")

                avg_score = sum(e.get("final_score", 0) for e in evals) / len(evals) if evals else 0
                st.metric("Overall Aggregate Score", f"{avg_score:.1f} / 100")

                tab_labels = [f"Q{i+1} Result" for i in range(len(evals))]
                eval_tabs = st.tabs(tab_labels)

                for i, data in enumerate(evals):
                    with eval_tabs[i]:
                        st.markdown(f"### Question {i+1}: {data.get('question', '')}")
                        st.info(f"**Your Answer:** {data.get('candidate_answer', '')}")

                        if data.get("execution_output"):
                            st.subheader("Code Execution Output")
                            st.text_area(f"Execution Output Q{i+1}", value=data["execution_output"], height=120, disabled=True, key=f"exec_out_display_{i}")

                        st.metric("Final Score", f"{data.get('final_score', 0)}/100")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("LLM Score", f"{data.get('llm_score', 0)}/100")
                        with col2:
                            st.metric("Keyword Score", f"{data.get('keyword_score', 0)}/100")
                        with col3:
                            st.metric("Semantic Score", f"{data.get('semantic_score', 0)}/100")
                        with col4:
                            st.metric("Answer Quality", f"{data.get('answer_quality_score', 0)}/100")

                        st.markdown("### Technical Feedback")
                        st.write(data.get("technical_feedback", ""))

                        st.markdown("### Communication Feedback")
                        st.write(data.get("communication_feedback", ""))

                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.markdown("### Missing Points")
                            mp = data.get("missing_points", [])
                            if mp:
                                for pt in mp:
                                    st.write(f"- {pt}")
                            else:
                                st.write("No major missing points.")

                        with col_m2:
                            st.markdown("### Missing Keywords")
                            mk = data.get("missing_keywords", [])
                            if mk:
                                for kw in mk:
                                    st.write(f"- {kw}")
                            else:
                                st.write("No major missing keywords.")

                        st.markdown("### Improved Answer")
                        st.write(data.get("improved_answer", ""))

                        st.markdown("### Ideal Answer")
                        st.write(data.get("ideal_answer", ""))

                        st.markdown("### Final Suggestion")
                        st.write(data.get("final_suggestion", ""))

            elif st.session_state.raw_questions and not st.session_state.questions_list:
                st.warning("Could not parse questions as JSON. Raw response:")
                st.write(st.session_state.raw_questions)



    else:
        FULL_INTERVIEW_STAGES = ["HR Agent", "Technical Agent", "Coding Agent", "System Design Agent"]
        
        if not st.session_state.interview_started:
            if st.button("Start Full Mock Interview"):
                if not role or not resume or not job_description:
                    st.warning("Please fill role, resume, and job description.")
                else:
                    st.session_state.interview_started = True
                    st.session_state.current_stage_index = 0
                    st.session_state.full_interview_results = []
                    st.session_state.current_stage_questions = []
                    st.session_state.current_question_index = 0
                    st.rerun()
        else:
            if st.session_state.current_stage_index < len(FULL_INTERVIEW_STAGES):
                current_agent = FULL_INTERVIEW_STAGES[st.session_state.current_stage_index]
                st.subheader(f"Round {st.session_state.current_stage_index + 1}/4: {current_agent}")
                
                # Check if we need to generate questions for this stage
                if "current_stage_questions" not in st.session_state or not st.session_state.current_stage_questions:
                    with st.spinner(f"{current_agent} is generating your questions..."):
                        if not st.session_state.rag_ready:
                            try:
                                build_rag_index(resume, job_description)
                                st.session_state.rag_ready = True
                            except Exception as e:
                                st.error(f"⚠️ **RAG Index Error**: {str(e)}")
                                st.stop()
                        
                        rag_query = f"{role} interview questions based on resume and job description"
                        rag_context = retrieve_context(rag_query, top_k=1)

                        prompt = get_question_prompt_by_round(
                            agent_type=current_agent,
                            role=role,
                            resume=resume,
                            job_description=job_description,
                            rag_context=rag_context
                        )
                        response = ask_ollama(prompt)
                        questions = get_questions_from_response(response, agent_type=current_agent, role=role)
                        st.session_state.current_stage_questions = questions[:3]
                        st.session_state.current_question_index = 0
                    st.rerun()
                
                # Display Question
                q_data = st.session_state.current_stage_questions[st.session_state.current_question_index]
                st.session_state.current_test_cases = q_data.get("test_cases", "")
                q_num = st.session_state.current_question_index + 1
                q_total = len(st.session_state.current_stage_questions)
                st.info(f"**Question {q_num}/{q_total}:** {q_data['question']}")
                render_tts_button(q_data['question'], button_label=f"🔊 Read Question Aloud", key=f"tts_full_{st.session_state.current_stage_index}_{q_num}")
                
                # Answer Input
                st.subheader("Answer Input")

                answer_mode = st.radio(
                    "Choose answer method",
                    ["Type Answer", "Voice Answer"],
                    horizontal=True,
                    key=f"answer_mode_{st.session_state.current_stage_index}"
                )

                candidate_answer = ""

                if answer_mode == "Type Answer":
                    candidate_answer = st.text_area(
                        "Your Answer",
                        height=180,
                        placeholder="Write your answer here...",
                        key=f"type_answer_{st.session_state.current_stage_index}"
                    )
                else:
                    st.info("Record your answer using the built-in microphone input.")

                    audio = st.audio_input("Start Recording", key=f"audio_input_{st.session_state.current_stage_index}")

                    if audio is not None:
                        audio_bytes = audio.getvalue()
                        st.audio(audio_bytes, format="audio/wav")

                        if st.button("Transcribe Voice Answer", key=f"transcribe_{st.session_state.current_stage_index}"):
                            with st.spinner("Transcribing your voice locally using faster-whisper..."):
                                audio_path = save_audio_file(audio_bytes)
                                transcript = transcribe_audio(audio_path)
                                st.session_state.voice_transcript = transcript

                    candidate_answer = st.text_area(
                        "Voice Transcript",
                        value=st.session_state.voice_transcript,
                        height=180,
                        placeholder="Your transcription will appear here...",
                        key=f"voice_answer_{st.session_state.current_stage_index}"
                    )
                
                if st.button("Submit Answer & Proceed"):
                    if not candidate_answer:
                        st.warning("Please provide an answer before proceeding.")
                    else:
                        execution_output = ""
                        test_cases = st.session_state.get("current_test_cases", "")
                        if current_agent == "Coding Agent":
                            with st.spinner("Executing your Python code..."):
                                passed, exec_out = run_python_code(candidate_answer, test_cases)
                                execution_output = exec_out

                        with st.spinner("Retrieving RAG context for evaluation..."):
                            rag_context = retrieve_context(q_data['question'], top_k=1)

                        with st.spinner(f"{current_agent} is evaluating your answer..."):
                            eval_prompt = get_evaluation_prompt_by_round(
                                agent_type=current_agent,
                                role=role,
                                question=q_data['question'],
                                answer=candidate_answer,
                                rag_context=rag_context,
                                execution_output=execution_output
                            )
                            raw_evaluation = ask_ollama(eval_prompt)
                            evaluation_data = extract_json(raw_evaluation)
                            
                            if evaluation_data:
                                llm_score = int(evaluation_data.get("score", 0))
                                improved_answer = evaluation_data.get("improved_answer", "")
                                ideal_answer = evaluation_data.get("ideal_answer", improved_answer)
                                expected_keywords = evaluation_data.get("expected_keywords", [])
                                
                                hybrid_result = calculate_hybrid_score(
                                    candidate_answer=candidate_answer,
                                    ideal_answer=ideal_answer,
                                    expected_keywords=expected_keywords,
                                    llm_score=llm_score
                                )
                                
                                final_score = hybrid_result["final_score"]
                                evaluation_data["final_score"] = final_score
                                evaluation_data["agent"] = current_agent
                                evaluation_data["question"] = q_data['question']
                                evaluation_data["candidate_answer"] = candidate_answer
                                evaluation_data["technical_feedback"] = evaluation_data.get("technical_feedback", "")
                                evaluation_data["communication_feedback"] = evaluation_data.get("communication_feedback", "")
                                evaluation_data["missing_points"] = evaluation_data.get("missing_points", [])
                                evaluation_data["improved_answer"] = improved_answer
                                evaluation_data["final_suggestion"] = evaluation_data.get("final_suggestion", "")

                                st.session_state.full_interview_results.append(evaluation_data)
                                
                                save_interview(
                                    role=role,
                                    resume=resume,
                                    job_description=job_description,
                                    question=q_data['question'],
                                    answer=candidate_answer,
                                    score=final_score,
                                    technical_feedback=evaluation_data["technical_feedback"],
                                    communication_feedback=evaluation_data["communication_feedback"],
                                    missing_points=json.dumps(evaluation_data["missing_points"]),
                                    improved_answer=improved_answer,
                                    final_suggestion=evaluation_data["final_suggestion"],
                                    raw_evaluation=json.dumps(evaluation_data)
                                )

                                # Proceed to next question or stage
                                if st.session_state.current_question_index < len(st.session_state.current_stage_questions) - 1:
                                    st.session_state.current_question_index += 1
                                    st.session_state.voice_transcript = ""
                                    st.rerun()
                                else:
                                    st.session_state.current_stage_index += 1
                                    st.session_state.current_stage_questions = []
                                    st.session_state.current_question_index = 0
                                    st.session_state.voice_transcript = ""
                                    st.rerun()
                            else:
                                st.error("Failed to evaluate answer properly. Please try again.")
            else:
                st.success("Full Mock Interview Complete!")
                
                total_score = sum(res["final_score"] for res in st.session_state.full_interview_results)
                avg_score = total_score / len(st.session_state.full_interview_results)
                
                st.metric("Final Aggregate Score", f"{avg_score:.2f} / 100")
                
                st.subheader("Performance Breakdown")
                for res in st.session_state.full_interview_results:
                    with st.expander(f"{res['agent']} - Score: {res['final_score']}"):
                        st.write(f"**Question:** {res['question']}")
                        st.write(f"**Technical Feedback:** {res['technical_feedback']}")
                        st.write(f"**Communication Feedback:** {res['communication_feedback']}")
                        st.write(f"**Final Suggestion:** {res['final_suggestion']}")
                
                import datetime
                pdf_buffer = generate_full_report_pdf_agent(
                    role=role,
                    average_score=avg_score,
                    results_list=st.session_state.full_interview_results,
                    created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
                st.download_button(
                    label="Download Full Interview Report (PDF)",
                    data=pdf_buffer,
                    file_name="full_interview_report.pdf",
                    mime="application/pdf"
                )
                
                if st.button("Finish & Start Over"):
                    st.session_state.interview_started = False
                    st.session_state.current_stage_index = 0
                    st.session_state.full_interview_results = []
                    st.session_state.current_stage_questions = []
                    st.session_state.current_question_index = 0
                    st.rerun()
with tab2:
    st.header("Interview History")

    rows = get_interviews()

    if not rows:
        st.info("No interview history found.")
    else:
        for row in rows:
            (
                interview_id,
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
            ) = row

            with st.expander(f"{role} | Score: {score}/100 | {created_at}"):
                st.subheader("Question")
                st.write(question)

                st.subheader("Your Answer")
                st.write(answer)

                st.subheader("Technical Feedback")
                st.write(technical_feedback)

                st.subheader("Communication Feedback")
                st.write(communication_feedback)

                st.subheader("Missing Points")
                try:
                    points = json.loads(missing_points)
                    for point in points:
                        st.write(f"- {point}")
                except Exception:
                    st.write(missing_points)

                st.subheader("Improved Answer")
                st.write(improved_answer)

                st.subheader("Final Suggestion")
                st.write(final_suggestion)

                # Generate and offer PDF download via Report Agent
                pdf_buffer = generate_report_pdf(
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

                st.download_button(
                    label="Ask Report Agent to Generate PDF",
                    data=pdf_buffer,
                    file_name=f"interview_report_{interview_id}.pdf",
                    mime="application/pdf"
                )


with tab3:
    st.header("Performance Dashboard")

    rows = get_interviews()

    if not rows:
        st.info("No data available yet.")
    else:
        data = []

        for row in rows:
            data.append({
                "Role": row[1],
                "Score": row[4],
                "Date": row[10]
            })

        df = pd.DataFrame(data)

        st.subheader("Score Table")
        st.dataframe(df, use_container_width=True)

        st.subheader("Average Score")
        avg_score = df["Score"].mean()
        st.metric("Average Score", f"{avg_score:.2f}/100")

        st.subheader("Score Trend")
        st.line_chart(df["Score"])

        st.subheader("Role-wise Performance")
        role_scores = df.groupby("Role")["Score"].mean().reset_index()
        st.bar_chart(role_scores.set_index("Role"))

        st.subheader("Monthly Performance")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        monthly_scores = (
            df.dropna(subset=["Date"])
            .resample("ME", on="Date")["Score"]
            .mean()
            .reset_index()
        )

        if not monthly_scores.empty:
            st.line_chart(monthly_scores.set_index("Date"))
        else:
            st.info("Not enough date data for monthly performance.")

with tab4:
    st.header("Weak Area Tracking & Learning Plan")

    rows = get_interviews()

    if not rows:
        st.info("No interview data available yet. Complete some interviews first.")
    else:
        analysis = extract_weak_areas_from_history(rows)

        average_score = analysis["average_score"]
        role_average_scores = analysis["role_average_scores"]
        top_weak_areas = analysis["top_weak_areas"]
        low_score_questions = analysis["low_score_questions"]

        st.subheader("Overall Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Average Score", f"{average_score}/100")

        with col2:
            st.metric("Performance Level", get_performance_level(average_score))

        st.divider()

        st.subheader("Role-wise Average Score")

        if role_average_scores:
            role_df = pd.DataFrame([
                {"Role": role, "Average Score": score}
                for role, score in role_average_scores.items()
            ])

            st.dataframe(role_df, use_container_width=True)
            st.bar_chart(role_df.set_index("Role"))
        else:
            st.info("No role-wise score data available.")

        st.divider()

        st.subheader("Top Weak Areas")

        if top_weak_areas:
            weak_df = pd.DataFrame([
                {"Weak Area": area, "Frequency": count}
                for area, count in top_weak_areas
            ])

            st.dataframe(weak_df, use_container_width=True)
            st.bar_chart(weak_df.set_index("Weak Area"))
        else:
            st.success("No repeated weak areas found yet.")

        st.divider()

        st.subheader("Personalized Learning Plan")

        learning_plan = generate_learning_plan(top_weak_areas)

        if learning_plan:
            for item in learning_plan:
                with st.expander(
                    f"{item['priority']} Priority: {item['weak_area']}"
                ):
                    st.write(f"**Repeated:** {item['frequency']} time(s)")
                    st.write(f"**Study Action:** {item['study_action']}")
                    st.write(f"**Practice Task:** {item['practice_task']}")
        else:
            st.info("Complete more interviews to generate a learning plan.")

        st.divider()

        st.subheader("Low Score Questions")

        if low_score_questions:
            for item in low_score_questions:
                with st.expander(
                    f"{item['role']} | Score: {item['score']}/100 | {item['created_at']}"
                ):
                    st.write("**Question:**")
                    st.write(item["question"])

                    st.write("**Technical Feedback:**")
                    st.write(item["technical_feedback"])

                    st.write("**Final Suggestion:**")
                    st.write(item["final_suggestion"])
        else:
            st.success("No low-score questions found. Good progress!")


with tab5:
    st.header("Study & Revision")
    st.caption(
        f"PDFs, saved questions, notes, answers, and review history stay in "
        f"`{PDF_DIRECTORY}` and your local SQLite/Chroma data. PDF-grounded "
        "generation and feedback use your local Ollama models."
    )

    study_sections = st.tabs([
        "PDF library & notes",
        "Practice & due review",
        "Public question research",
        "Progress & history"
    ])

    with study_sections[0]:
        uploaded_pdfs = st.file_uploader(
            "Add text-based PDFs to your local study library",
            type=["pdf"],
            accept_multiple_files=True,
            max_upload_size=50,
            key="study_pdf_upload"
        )
        if st.button("Save and index uploaded PDFs", key="upload_study_pdfs"):
            if not uploaded_pdfs:
                st.warning("Choose one or more PDF files first.")
            else:
                try:
                    with st.spinner("Saving and indexing PDFs locally..."):
                        upload_result = add_pdf_files(uploaded_pdfs)
                    st.success(
                        f"Added {len(upload_result['uploaded'])} PDF(s); "
                        f"{len(upload_result['duplicates'])} duplicate(s) already existed."
                    )
                    for message in upload_result["errors"]:
                        st.warning(message)
                except Exception as error:
                    st.error(f"Could not add PDFs: {error}")

        first_sync = "study_workflow_synced" not in st.session_state
        if first_sync or st.button("Sync PDF folder", key="sync_study_pdfs"):
            try:
                with st.spinner("Syncing the PDF study folder..."):
                    sync_result = sync_pdf_library()
                st.session_state.study_sync_result = sync_result
                st.session_state.study_workflow_synced = True
                st.session_state.pop("study_sync_error", None)
            except Exception as error:
                st.session_state.study_sync_error = str(error)
                st.session_state.study_workflow_synced = True

        if st.session_state.get("study_sync_error"):
            st.error(f"Could not sync the PDF library: {st.session_state.study_sync_error}")
            if st.button("Retry PDF sync", key="retry_study_pdfs"):
                st.session_state.pop("study_sync_error", None)
                st.session_state.pop("study_workflow_synced", None)
                st.rerun()

        sync_result = st.session_state.get("study_sync_result")
        if sync_result:
            if sync_result["indexed"] or sync_result["removed"]:
                st.success(
                    f"PDF library synced: {sync_result['indexed']} indexed, "
                    f"{sync_result['unchanged']} unchanged, "
                    f"{sync_result['removed']} removed."
                )
            for message in sync_result["errors"]:
                st.warning(message)

        documents = get_study_documents()
        if documents:
            st.subheader("Your PDF library")
            for document in documents:
                status = document["status"]
                label = (
                    f"{document['filename']} — {document['page_count']} pages "
                    f"— {status}"
                )
                with st.expander(label):
                    st.caption(document["file_path"])
                    if document["error"]:
                        st.warning(document["error"])
                    st.caption(
                        "Removing a PDF also removes questions and practice "
                        "history tied to that document."
                    )
                    confirm_key = f"confirm_remove_pdf_{document['id']}"
                    confirmed = st.checkbox(
                        "Confirm removal",
                        key=confirm_key
                    )
                    if st.button(
                        "Remove PDF and related study data",
                        key=f"remove_pdf_{document['id']}",
                        disabled=not confirmed
                    ):
                        try:
                            remove_pdf(document["id"])
                            st.success(f"Removed {document['filename']}.")
                            st.rerun()
                        except Exception as error:
                            st.error(f"Could not remove the PDF: {error}")
        else:
            st.info("No PDFs yet. Upload files here or add them to the `pdf/` folder.")

        indexed_documents = [
            item for item in documents if item["status"] == "indexed"
        ]
        if indexed_documents:
            pdf_names = {
                item["file_path"]: item["filename"]
                for item in indexed_documents
            }
            selected_paths = st.multiselect(
                "Limit study to selected PDFs (leave empty to search all)",
                options=list(pdf_names),
                format_func=lambda path: pdf_names[path],
                key="study_pdf_selection"
            )
            study_query = st.text_input(
                "Topic to revise",
                placeholder="For example: Kafka consumer groups and rebalancing",
                key="study_query"
            )
            notes_column, questions_column = st.columns(2)
            with notes_column:
                if st.button("Create and save short notes", key="generate_pdf_notes"):
                    if not study_query.strip():
                        st.warning("Enter a topic first.")
                    else:
                        try:
                            with st.spinner("Writing cited notes with local Ollama..."):
                                generated_note = generate_pdf_notes(
                                    study_query,
                                    file_paths=selected_paths or None
                                )
                            st.session_state.latest_study_note = generated_note
                        except Exception as error:
                            st.error(f"Could not create notes: {error}")
            with questions_column:
                question_count = st.number_input(
                    "Questions to generate",
                    min_value=1,
                    max_value=8,
                    value=5,
                    key="pdf_question_count"
                )
                if st.button("Generate grounded questions", key="generate_pdf_questions"):
                    if not study_query.strip():
                        st.warning("Enter a topic first.")
                    else:
                        try:
                            with st.spinner("Generating PDF-grounded questions..."):
                                created_questions = generate_pdf_questions(
                                    study_query,
                                    file_paths=selected_paths or None,
                                    count=question_count
                                )
                            st.success(
                                f"Saved {len(created_questions)} questions to your local bank."
                            )
                        except Exception as error:
                            st.error(f"Could not generate questions: {error}")

            if st.session_state.get("latest_study_note"):
                note = st.session_state.latest_study_note
                st.subheader(note["topic"])
                st.markdown(note["note_text"])
                st.markdown("**Sources**")
                for source in note["source_evidence"]:
                    st.write(f"- {source['citation']}")

            if study_query.strip() and st.button(
                "Find relevant PDF passages",
                key="find_study_passages"
            ):
                try:
                    passages = retrieve_study_context(
                        study_query,
                        top_k=5,
                        file_paths=selected_paths or None
                    )
                    if passages:
                        st.session_state.study_context = passages
                    else:
                        st.info("No matching passages found in the selected PDFs.")
                except Exception as error:
                    st.error(f"Could not search the PDF library: {error}")
            if st.session_state.get("study_context"):
                st.subheader("Relevant passages")
                st.markdown(st.session_state.study_context)

        st.subheader("Saved revision notes")
        for note in get_study_notes(limit=10):
            with st.expander(f"{note['topic']} — {note['created_at']}"):
                st.markdown(note["note_text"])
                citations = [
                    item["citation"] for item in note["source_evidence"]
                ]
                if citations:
                    st.caption("Sources: " + "; ".join(citations))

    with study_sections[1]:
        st.subheader("One question at a time")
        due_only = st.checkbox("Show due questions only", value=True)
        questions = get_question_bank(due_only=due_only)
        if not questions and due_only:
            st.success("Nothing is due. Turn off the filter to practise any saved question.")
        elif not questions:
            st.info("Generate questions from PDFs or save questions from an imported source.")
        else:
            question_labels = {
                item["id"]: (
                    f"{item['topic']} · {item['difficulty']} · "
                    f"{item['question'][:95]}"
                )
                for item in questions
            }
            selected_question_id = st.selectbox(
                "Choose a saved question",
                options=list(question_labels),
                format_func=lambda question_id: question_labels[question_id],
                key="practice_question_id"
            )
            question = next(
                item for item in questions if item["id"] == selected_question_id
            )
            st.info(question["question"])
            st.caption(
                f"Topic: {question['topic']} · Difficulty: {question['difficulty']} "
                f"· Last score: {question['last_score'] if question['last_score'] is not None else 'not attempted'} "
                f"· Attempts: {question['attempt_count']}"
            )
            if question["provenance"] == "pdf":
                st.caption(
                    f"Grounded in {question['filename'] or 'saved PDF'}, "
                    f"page {question['page_number'] or '?'}"
                )
            elif question["provenance"] == "web":
                st.caption(
                    f"Web source: {question['source_title']} "
                    f"({question['source_url']})"
                )
            else:
                st.caption("Personal question with a saved offline reference answer.")
            with st.form(f"study_answer_form_{selected_question_id}"):
                submitted_answer = st.text_area(
                    "Your answer",
                    height=160,
                    placeholder="Answer from memory before revealing the sample answer...",
                    key=f"study_answer_{selected_question_id}"
                )
                submit_answer = st.form_submit_button("Submit and get feedback")
            if submit_answer:
                try:
                    with st.spinner("Checking your answer against saved evidence..."):
                        feedback = evaluate_study_answer(question, submitted_answer)
                    st.session_state[f"study_feedback_{selected_question_id}"] = feedback
                except Exception as error:
                    st.error(f"Could not evaluate the answer: {error}")

            feedback = st.session_state.get(
                f"study_feedback_{selected_question_id}"
            )
            if feedback:
                st.metric("Evidence-based score", f"{feedback['score']}/100")
                st.write(feedback["feedback"])
                if feedback["strengths"]:
                    st.markdown("**What you covered well**")
                    for item in feedback["strengths"]:
                        st.write(f"- {item}")
                if feedback["missing_concepts"]:
                    st.markdown("**Concepts to revisit**")
                    for item in feedback["missing_concepts"]:
                        st.write(f"- {item}")
                if feedback["unsupported_claims"]:
                    st.markdown("**Claims not supported by the saved source**")
                    for item in feedback["unsupported_claims"]:
                        st.write(f"- {item}")
                if feedback["next_step"]:
                    st.write(f"**Next step:** {feedback['next_step']}")
                st.caption(
                    "Next review: " + feedback["next_review_at"]
                )
                st.markdown("**Evidence used**")
                for item in feedback["source_evidence"]:
                    st.write(f"- {item['citation']}")
                if st.checkbox(
                    "Reveal saved sample answer",
                    key=f"reveal_sample_{selected_question_id}"
                ):
                    st.markdown(question["sample_answer"] or "No sample answer was saved.")
                if question["hint"]:
                    with st.expander("Show hint"):
                        st.write(question["hint"])

    with study_sections[2]:
        st.subheader("Optional free public-source search")
        st.caption(
            "Search is online and may be unavailable. Import only sources you "
            "want to keep; saved excerpts and questions are then available offline."
        )
        with st.expander("Add a personal question"):
            with st.form("add_manual_study_question"):
                manual_topic = st.text_input("Topic", key="manual_question_topic")
                manual_question_text = st.text_area(
                    "Interview question",
                    key="manual_question_text"
                )
                manual_reference = st.text_area(
                    "Reference answer (used for grounded feedback)",
                    key="manual_question_reference"
                )
                manual_concepts = st.text_input(
                    "Expected concepts (comma-separated)",
                    key="manual_question_concepts"
                )
                manual_hint = st.text_input(
                    "Optional hint",
                    key="manual_question_hint"
                )
                save_manual_question = st.form_submit_button("Save personal question")
            if save_manual_question:
                try:
                    add_manual_question(
                        manual_question_text,
                        manual_topic,
                        manual_reference,
                        expected_concepts=[
                            concept.strip()
                            for concept in manual_concepts.split(",")
                            if concept.strip()
                        ],
                        hint=manual_hint
                    )
                    st.success("Saved to your local question bank.")
                except Exception as error:
                    st.error(f"Could not save the question: {error}")

        web_query = st.text_input(
            "Search public interview material",
            placeholder="For example: Kafka producer idempotence interview questions",
            key="web_question_query"
        )
        if st.button("Search the web", key="search_public_questions"):
            try:
                with st.spinner("Searching public sources..."):
                    st.session_state.public_question_results = search_public_questions(
                        web_query,
                        max_results=5
                    )
            except Exception as error:
                st.error(str(error))
        for index, result in enumerate(
            st.session_state.get("public_question_results", [])
        ):
            with st.container(border=True):
                st.markdown(f"**{result['title']}**")
                st.caption(result["publisher"])
                st.write(result["snippet"])
                st.link_button("Open source", result["url"])
                if st.button(
                    "Import excerpt for offline practice",
                    key=f"import_web_source_{index}"
                ):
                    try:
                        with st.spinner("Saving a readable excerpt locally..."):
                            imported = import_web_source(
                                result["url"],
                                title=result["title"],
                                publisher=result["publisher"],
                                search_snippet=result["snippet"]
                            )
                        if (
                            imported["metadata"].get("extraction_mode")
                            == "search_result_snippet"
                        ):
                            st.warning(
                                f"The source host blocked article access. Saved "
                                f"the search-result snippet for {imported['title']} "
                                f"instead; it contains less context than the full article."
                            )
                        else:
                            st.success(f"Saved source: {imported['title']}")
                    except Exception as error:
                        st.error(f"Could not import this source: {error}")

        st.subheader("Saved web sources")
        saved_sources = get_saved_web_sources()
        if saved_sources:
            source_labels = {
                source["id"]: source["title"] for source in saved_sources
            }
            selected_source_id = st.selectbox(
                "Choose an imported source",
                options=list(source_labels),
                format_func=lambda source_id: source_labels[source_id],
                key="web_source_for_questions"
            )
            web_question_count = st.number_input(
                "Questions from this source",
                min_value=1,
                max_value=8,
                value=5,
                key="web_question_count"
            )
            if st.button(
                "Generate and save source-grounded questions",
                key="generate_web_questions"
            ):
                try:
                    with st.spinner("Generating questions with local Ollama..."):
                        created_questions = generate_web_questions(
                            selected_source_id,
                            count=web_question_count
                        )
                    st.success(
                        f"Saved {len(created_questions)} questions for offline use."
                    )
                except Exception as error:
                    st.error(f"Could not generate questions: {error}")
        else:
            st.info("No web sources have been imported.")

    with study_sections[3]:
        st.subheader("Recent practice")
        all_questions = get_question_bank()
        due_questions = get_question_bank(due_only=True)
        study_docs = get_study_documents()
        metric_columns = st.columns(3)
        metric_columns[0].metric("PDFs", len(study_docs))
        metric_columns[1].metric("Saved questions", len(all_questions))
        metric_columns[2].metric("Questions due", len(due_questions))
        history = get_practice_history(limit=30)
        if not history:
            st.info("Submitted answers and review progress will appear here.")
        else:
            scores = [item["score"] for item in history if item["score"] is not None]
            if scores:
                st.metric("Average recent score", f"{sum(scores) / len(scores):.1f}/100")
            topic_scores = {}
            for item in history:
                if item["score"] is not None:
                    topic_scores.setdefault(item["topic"], []).append(item["score"])
            weakest_topics = sorted(
                topic_scores.items(),
                key=lambda entry: sum(entry[1]) / len(entry[1])
            )[:3]
            if weakest_topics:
                st.markdown("**Topics to revisit**")
                for topic, topic_history in weakest_topics:
                    average = sum(topic_history) / len(topic_history)
                    st.write(f"- {topic}: {average:.0f}/100 average")
            for attempt in history:
                with st.expander(
                    f"{attempt['created_at']} · {attempt['score']}/100 · "
                    f"{attempt['topic']}"
                ):
                    st.markdown(f"**Question:** {attempt['question']}")
                    st.markdown(f"**Your answer:** {attempt['answer']}")
                    st.write(attempt["feedback"].get("feedback", ""))
                    citations = [
                        item["citation"] for item in attempt["source_evidence"]
                    ]
                    if citations:
                        st.caption("Evidence: " + "; ".join(citations))