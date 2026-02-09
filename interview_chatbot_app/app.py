import streamlit as st
import os
import io
import tempfile
import time
import google.generativeai as genai
from langchain_openai import ChatOpenAI
from streamlit_mic_recorder import mic_recorder
from style_utils import apply_premium_style
from dotenv import load_dotenv
import PyPDF2
from deepgram import DeepgramClient
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
except ImportError:
    webrtc_streamer = None
    WebRtcMode = None
    class AudioProcessorBase: pass
from streamlit_autorefresh import st_autorefresh
import queue
import threading
import json
import asyncio

load_dotenv()

# --- CONFIGURATION ---
OPENROUTER_API_KEY = os.getenv("GRAPH_API_KEY") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
MODEL_NAME = "google/gemini-3-flash-preview"
BASE_URL = "https://openrouter.ai/api/v1"

# Configure Generative AI (Native Google Features)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def extract_text_from_pdf(file):
    """Simple helper to extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Could not extract PDF text: {e}"

from transcription_handler import transcribe_audio_deepgram, AudioProcessor

def transcribe_audio_gemini(audio_bytes, live_mode=False):
    """
    Uses Deepgram for transcription + Gemini for question detection.
    This hybrid approach ensures high speed and smart context.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("sk-or"):
        return "Error: A valid Google Gemini API Key is required for audio analysis (Question Detection). OpenRouter keys do not work here."
    
    # 1. Use Deepgram for the text part (Nova-2 is much faster than Gemini for STT)
    transcript = transcribe_audio_deepgram(audio_bytes, DEEPGRAM_API_KEY)
    if "Error" in transcript:
        return transcript

    # 2. Use Gemini for Question Detection based on the transcript
    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        if live_mode:
            prompt = f"""
            Analyze this interview transcript segment:
            "{transcript}"
            
            1. Provide a clean summary (3-4 lines) of what was said.
            2. Detect if a specific interview question was just asked to the candidate.
            
            Return the result in this EXACT JSON format:
            {{
              "transcript": "The clean summary...",
              "question": "The detected question or null"
            }}
            """
        else:
            prompt = f"""
            Extract ONLY the core interview question from this transcript:
            "{transcript}"
            
            If no clear interview question is found, return 'No question detected'.
            """
            
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Analysis failed: {e}"

def parse_resume_details(text, api_key):
    """
    Uses LLM to extract structured information from the resume text.
    """
    try:
        llm = ChatOpenAI(
            model=MODEL_NAME, 
            api_key=api_key,
            base_url=BASE_URL,
            temperature=0
        )
        
        prompt = f"""
        Analyze the following resume text and extract the information in a structured format.
        RESUME TEXT:
        {text}

        EXTRACT THE FOLLOWING:
        1. EXPERIENCE: A list of professional experiences. For each, include Company, Role, Dates, and a few bullet points of detailed achievements (tasks, projects, metrics).
        2. KEY SKILLS: A comprehensive list of technical and soft skills.
        3. PROJECTS: Notable projects mentioned, with brief descriptions.
        4. CERTIFICATIONS: Any professional certifications or licenses.

        FORMAT: Return the output as a clean, professionally formatted Markdown document with clear headers for each section.
        """
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        return f"Error parsing resume: {e}"

def generate_profile_summary(profile, api_key):
    """
    Generates a professional and highly detailed summary of the user's profile and background.
    """
    try:
        llm = ChatOpenAI(
            model=MODEL_NAME, 
            api_key=api_key,
            base_url=BASE_URL,
            temperature=0.5
        )
        
        # Use parsed_resume_details if available, otherwise fallback to raw context
        resume_info = profile.get('parsed_resume', profile.get('resume_summary', 'N/A'))
        
        prompt = f"""
        You are an expert Career Counselor and Interview Coach. 
        Your task is to analyze the candidate's profile and the target Job Description (JD) to create a MASTER CONTEXT SUMMARY.

        INPUT DATA:
        Candidate Name: {profile['name']}
        Education: {profile.get('education', 'N/A')}
        Target Company: {profile['company']}
        Skills Provided: {profile['skills']}
        Job Description: {profile['jd']}
        DETAILED RESUME ANALYSIS:
        {resume_info}

        TASKS:
        1. JD ANALYSIS: Provide a detailed description of what the JD is asking for. What is the core mission of this role?
        2. TECHNICAL SKILLS REQUIRED: List the specific technical skills and tools this JD is looking for.
        3. CANDIDATE EXPERIENCE & PROJECTS: Synthesize the candidate's experience and projects from the resume analysis. Focus on specific achievements and value added.
        4. EDUCATION & CERTIFICATIONS: Detail the education, certifications, and any other relevant background info.

        OUTPUT FORMAT:
        - Use professional Markdown (headers, bolding, bullet points).
        - Ensure the summary is comprehensive yet easy for the AI to parse as context.
        - Start with a high-level overview and then break into the sections above.

        This summary will be used as the SOLE source of truth for answering interview questions. Make it perfect.
        """
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        return f"Error generating summary: {e}"



def detect_question_and_answer(transcript, profile, api_key):
    """Checks if a question was asked and generates an answer automatically."""
    try:
        if not transcript.strip():
            return

        model = genai.GenerativeModel(model_name=MODEL_NAME)
        prompt = f"""
        Analyze this transcript segment from an interview:
        "{transcript}"
        
        Detect if a specific interview question was just asked to the candidate.
        Return 'None' if no question is found. Otherwise, return ONLY the detected question.
        """
        response = model.generate_content(prompt)
        question = response.text.strip()
        
        if "None" not in question and len(question) > 5:
            # Check if we already answered this in current session to avoid loop
            if "last_detected_question" not in st.session_state or st.session_state.last_detected_question != question:
                st.session_state.last_detected_question = question
                return question
    except Exception:
        pass
    return None

def generate_answer(question, api_key, tone, profile, simple_english=False, bullet_points=False):
    """
    Generates a personalized interview answer by synthesizing all available context.
    Returns a stream/generator for real-time UI updates.
    """
    try:
        llm = ChatOpenAI(
            model=MODEL_NAME, 
            api_key=api_key,
            base_url=BASE_URL,
            temperature=0.4,
            streaming=True
        )
        
        # Comprehensive context synthesis
        context = f"""
CANDIDATE PROFILE SUMMARY (User Edited):
{profile.get('summary', 'N/A')}

DETAILED RESUME ANALYSIS:
{profile.get('parsed_resume', profile.get('resume_summary', 'N/A'))}

TARGET JOB DESCRIPTION (JD):
{profile.get('jd', 'N/A')}

CANDIDATE BASIC INFO:
- Name: {profile['name']}
- Education: {profile.get('education', 'N/A')}
- Target Company: {profile['company']}
- Key Skills: {profile['skills']}
"""

        english_instruction = "6. LANGUAGE: Use professional business English."
        if simple_english:
            english_instruction = "6. SIMPLE ENGLISH: Use very simple, clear, and easy-to-understand English. Avoid complex jargon or long sentences."

        format_instruction = "7. FORMAT: Use a standard professional paragraph structure."
        if bullet_points:
            format_instruction = "7. BULLET POINTS: Format the ENTIRE response using bullet points (*) for every point. Do not use paragraphs. Break the detailed answer into clear, digestible bullet points."

        system_msg = f"""You are a World-Class Interview Assistant.
Your goal is to provide a HYPER-PERSONALIZED, DIRECT answer that sounds exactly like the candidate based on their background and the job they are applying for.

FULL CONTEXT (SYNTHESIZE ALL SOURCES):
{context}

INTERVIEW TONE: {tone}

INSTRUCTIONS:
1. **DIRECTNESS (CRITICAL)**: Answer the question IMMEDIATELY. Do not use any introductory phrases, preambles, or generic openers like "That's a great question" or "Based on your background...".
2. **FOCUS**: Craft the answer SOLELY based on the specific question asked. Do not volunteer extra information not requested by the user.
3. **SYNTHESIZE**: Use the candidate's specific achievements from the summary AND resume to answer the question.
4. **MATCH**: Align the answer directly with the requirements in the JD.
5. **PERSONA**: Speak in the first person ("I", "Me", "My"). Sound like a confident professional.
6. **LENGTH**: Provide a detailed answer between 5 to 6 sentences (or 5-6 bullet points if in bullet mode). 
7. **NO PLACEHOLDERS**: Do not use [Insert Skill Here].
{english_instruction}
{format_instruction}

Goal: Provide a convincing, comprehensive answer that proves the candidate's value by drawing from all provided data, starting the answer from the very first word.
"""
        user_msg = f"The interviewer just asked: \"{question}\"\n\nBased on my full profile and the JD, what is the perfect answer for me to give right now?"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        yield f"⚠️ Stream Error: {str(e)}"

def extend_answer(original_question, original_answer, api_key, tone, profile):
    """
    Generates a 10-line continuation by synthesizing all available context.
    Returns a stream/generator for real-time UI updates.
    """
    try:
        llm = ChatOpenAI(
            model=MODEL_NAME, 
            api_key=api_key,
            base_url=BASE_URL,
            temperature=0.7,
            streaming=True
        )
        
        context = f"""
CANDIDATE SUMMARY: {profile.get('summary', 'N/A')}
DETAILED RESUME: {profile.get('parsed_resume', profile.get('resume_summary', 'N/A'))}
JD: {profile.get('jd', 'N/A')}
"""

        system_msg = f"""You are a World-Class Interview Assistant.
The candidate gave a short answer and now needs to EXTEND it with 10 more lines of deep context, specific examples, and value-add by drawing from their full profile (Summary, Resume, and JD).

**DIRECTNESS**: Start the extension IMMEDIATELY. Do not use any filler phrases or introductory sentences.

CANDIDATE DATA:
{context}

Interviewer Question: "{original_question}"
Candidate's Initial Answer: "{original_answer}"

INSTRUCTIONS:
1. CONTINUATION: Do not repeat the initial answer. Pick up where it left off.
2. DEPTH: Provide approximately 10 additional lines of content.
3. CONTEXT: Use specific technical details from the raw resume or summary to provide deeper insights.
4. PERSONA: First person ("I", "Me", "My").
5. TONE: {tone}
"""
        user_msg = "Please extend my answer with 10 more lines of professional context and examples using my full background."
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        yield f"⚠️ Stream Error: {str(e)}"

def main():
    st.set_page_config(page_title="Interview Assistant", page_icon="🎙️", layout="centered")
    apply_premium_style()
    
    # --- SESSION STATE ---
    if "onboarded" not in st.session_state:
        st.session_state.onboarded = False
    if "viewing_summary" not in st.session_state:
        st.session_state.viewing_summary = False
    if "resume_analyzed" not in st.session_state:
        st.session_state.resume_analyzed = False
    if "live_mode" not in st.session_state:
        st.session_state.live_mode = False
    if "live_transcript" not in st.session_state:
        st.session_state.live_transcript = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "profile" not in st.session_state:
        st.session_state.profile = {}
    if "test_mode" not in st.session_state:
        st.session_state.test_mode = False
    if "realtime_transcript" not in st.session_state:
        st.session_state.realtime_transcript = []
    if "last_detected_question" not in st.session_state:
        st.session_state.last_detected_question = None
    if "is_listening" not in st.session_state:
        st.session_state.is_listening = False

    # --- ONBOARDING FLOW ---
    if not st.session_state.onboarded:
        if not st.session_state.viewing_summary:
            st.title("Welcome to Interview Assistant")
            
            # Test Mode Toggle
            test_mode = st.toggle("🧪 Test Mode (Auto-fill)", value=st.session_state.test_mode)
            st.session_state.test_mode = test_mode
            
            example_data = {
                "name": "Jane Smith",
                "edu": "BS in Computer Science",
                "company": "Netflix",
                "skills": "Go, Distributed Systems, Microservices, Kubernetes",
                "jd": "We are seeking a Backend Engineer to build high-scale streaming services. Must be proficient in Go and have experience with microservices and Kubernetes."
            } if test_mode else {"name": "", "edu": "", "company": "", "skills": "", "jd": ""}

            st.markdown("##### Please set up your profile to get personalized answers.")
            
            st.markdown('<div class="onboarding-card">', unsafe_allow_html=True)
            
            # Personal Info
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name*", value=st.session_state.profile.get("name", example_data["name"]), placeholder="e.g. John Doe")
            with col2:
                edu = st.text_input("Education (Optional)", value=st.session_state.profile.get("education", example_data["edu"]), placeholder="e.g. MS in Computer Science")
            
            # Job Info
            st.divider()
            st.subheader("Target Job Details")
            company = st.text_input("Company Name*", value=st.session_state.profile.get("company", example_data["company"]), placeholder="e.g. Google")
            skills = st.text_input("Key Skills*", value=st.session_state.profile.get("skills", example_data["skills"]), placeholder="e.g. Python, React, AWS")
            jd = st.text_area("Job Description (JD)*", value=st.session_state.profile.get("jd", example_data["jd"]), placeholder="Paste the key parts of the JD here...", height=150)
            
            # Resume
            st.divider()
            st.subheader("Experience")
            resume_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
            
            # Analyse Resume Button
            if resume_file:
                if st.button("🔍 Analyse Resume", use_container_width=True):
                    with st.spinner("Extracting detailed experience..."):
                        if resume_file.type == "application/pdf":
                            resume_text = extract_text_from_pdf(resume_file)
                        else:
                            resume_text = resume_file.read().decode("utf-8")
                        
                        with st.spinner("Analyzing resume structure..."):
                            parsed = parse_resume_details(resume_text, OPENROUTER_API_KEY)
                            st.session_state.profile["parsed_resume"] = parsed
                        st.session_state.profile["resume_summary"] = resume_text[:1500]
                        st.session_state.resume_analyzed = True
                        st.rerun()

            # Show Parsed Results Inline
            if st.session_state.resume_analyzed:
                st.markdown("---")
                st.markdown("### Parsed Resume Details")
                st.info("The AI has extracted the following details. You can edit them before generating the final summary.")
                edited_parsed = st.text_area("Experience, Skills, Projects, & Certs", 
                                           value=st.session_state.profile.get("parsed_resume", ""), 
                                           height=300)
                st.session_state.profile["parsed_resume"] = edited_parsed

            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Generate Profile Summary", use_container_width=True, type="primary"):
                if not name or not company or not skills or not jd:
                    st.error("Please fill in all required fields (marked with *).")
                else:
                    st.session_state.profile.update({
                        "name": name,
                        "education": edu,
                        "company": company,
                        "skills": skills,
                        "jd": jd
                    })
                    with st.spinner("Synthesizing all sources into your master profile..."):
                        summary = generate_profile_summary(st.session_state.profile, OPENROUTER_API_KEY)
                        st.session_state.profile["summary"] = summary
                        st.session_state.viewing_summary = True
                        st.rerun()
            return
        
        else:
            # Summary Review View
            st.title("Review Your Profile Summary")
            st.markdown("##### Based on your input, here's how the AI understands your background. You can refine this below.")
            
            with st.form("summary_form"):
                edited_summary = st.text_area("Professional Summary", value=st.session_state.profile.get("summary", ""), height=300)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("← Back to Form"):
                        st.session_state.viewing_summary = False
                        st.rerun()
                with col2:
                    if st.form_submit_button("Launch Assistant →"):
                        st.session_state.profile["summary"] = edited_summary
                        st.session_state.onboarded = True
                        st.session_state.viewing_summary = False
                        st.success("Assistant is ready!")
                        time.sleep(1)
                        st.rerun()
            return

    # --- MAIN ASSISTANT VIEW ---
    st.title("Interview Assistant")
    
    # --- REAL-TIME TRANSCRIPT AT TOP ---
    if st.session_state.onboarded:
        transcript_container = st.empty()
        with transcript_container.container():
            st.markdown("""
            <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;">
                <p style="margin: 0; color: #888; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">🎙️ Real-Time Transcription (Max 4 Lines)</p>
                <div id="live-transcript" style="min-height: 80px; font-family: 'Inter', sans-serif;">
            """, unsafe_allow_html=True)
            
            display_lines = st.session_state.realtime_transcript[-4:] if st.session_state.realtime_transcript else ["Waiting for dialogue..."]
            for line in display_lines:
                st.markdown(f'<p style="margin: 5px 0; font-size: 1.1rem; color: #eee;">• {line}</p>', unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)

        # --- MIC CONTROL IN MAIN AREA ---
        st.markdown("### 🎙️ Continuous Mode")
        mic_col1, mic_col2 = st.columns([1, 1])
        with mic_col1:
            if webrtc_streamer:
                ctx = webrtc_streamer(
                    key="continuous-listening",
                    mode=WebRtcMode.SENDONLY,
                    audio_processor_factory=lambda: AudioProcessor(DEEPGRAM_API_KEY),
                    media_stream_constraints={"video": False, "audio": True},
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    async_processing=True,
                )
            else:
                st.error("Live transcription unavailable (missing dependencies).")
                ctx = None

        with mic_col2:
            if ctx and ctx.state.playing:
                st.success("Listening Active")
                st_autorefresh(interval=2000, key="transcript_refresh")
            else:
                if ctx is None:
                    st.warning("Please install ffmpeg and reinstall requirements.")
                else:
                    st.info("Click 'Start' to begin listening")

        # Process fragments from the worker
        if ctx and ctx.state.playing and ctx.audio_processor:
            while not ctx.audio_processor.transcript_queue.empty():
                text = ctx.audio_processor.transcript_queue.get()
                if "Error" not in text:
                    st.session_state.realtime_transcript.append(text)
                    if len(st.session_state.realtime_transcript) > 10:
                        st.session_state.realtime_transcript.pop(0)
                    
                    # Auto Question Detection
                    if not st.session_state.get("is_generating_answer", False):
                        detected_q = detect_question_and_answer(text, st.session_state.profile, OPENROUTER_API_KEY)
                        if detected_q:
                            st.session_state.is_generating_answer = True
                            st.session_state.current_question = detected_q
                else:
                    st.error(text)

    st.markdown(f"Hi **{st.session_state.profile.get('name', '')}**, Assistant for **{st.session_state.profile.get('company', '')}** is live.")

    # --- SIDEBAR & SETTINGS ---
    st.sidebar.title("Settings")
    
    interview_tone = st.sidebar.selectbox(
        "Interview Tone",
        ["Corporate", "Bold", "Creative", "Stress-Test"],
        help="Select the personality of your AI assistant."
    )

    simple_english = st.sidebar.toggle(
        "✨ Simple English", 
        value=False,
        help="Use easier vocabulary and shorter sentences. Great for non-native speakers or clear communication."
    )

    bullet_mode = st.sidebar.toggle(
        "📋 Bullet Points", 
        value=False,
        help="Format the response as a clear list of bullet points instead of a paragraph."
    )
    
    live_mode = st.sidebar.toggle(
        "🎙️ Live Interview Mode", 
        value=st.session_state.live_mode,
        help="Continuously listen and show transcripts. Automatically answers detected questions."
    )
    st.session_state.live_mode = live_mode

    if st.sidebar.button("Edit Profile", use_container_width=True):
        st.session_state.onboarded = False
        st.rerun()

    st.sidebar.divider()
    with st.sidebar.expander("🎧 Listen to YouTube / Zoom"):
        st.markdown("""
        **To listen to other tabs (Mac):**
        1. **Install BlackHole**: Download the free [BlackHole 2ch](https://existential.audio/blackhole/) driver.
        2. **Multi-Output**: In Mac Sound Settings, create a 'Multi-Output Device' (Built-in + BlackHole).
        3. **Browser Input**: Set Chrome's Microphone privacy to 'BlackHole 2ch'.
        4. **Listen**: Play your YouTube tab, click 'Listen' here, and Gemini will hear the audio!
        """)
        st.info("This routes clear digital audio from other tabs into the Assistant.")

    st.sidebar.divider()
    st.sidebar.header("Controls")
    if st.sidebar.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    if not st.session_state.history and not st.session_state.live_mode:
        st.info("Assistant ready. Your personalized feed will appear below.")
    
    # --- Live Feed Section ---
    if st.session_state.live_mode:
        st.markdown("### 📡 Live Interview Feed")
        with st.expander("Show Live Transcript", expanded=True):
            if not st.session_state.live_transcript:
                st.caption("Listening for dialogue...")
            for text in st.session_state.live_transcript[-5:]: # Show last 5 segments
                st.write(f"• {text}")
        st.divider()
    
    # --- Display History Chronologically (Latest at bottom) ---
    for idx, item in enumerate(st.session_state.history):
        # Interviewer Message (Question)
        with st.chat_message("user", avatar="🎙️"):
            st.markdown(f"**Interviewer:** {item['question']}")
            st.caption(f"Tone Mode: {item['tone']}")
        
        # Assistant Message (Answer)
        with st.chat_message("assistant", avatar="💡"):
            st.markdown(item['answer'])
            
            # Show Extension if it exists
            if item.get("extension"):
                st.divider()
                st.markdown("**Extended Context (10+ Lines):**")
                st.markdown(item["extension"])
            
            # Button to Extend (Only show if not already extended)
            if not item.get("extension"):
                if st.button("➕ Extend Answer", key=f"extend_{idx}"):
                    stream = extend_answer(
                        item['question'], 
                        item['answer'], 
                        OPENROUTER_API_KEY, 
                        item['tone'], 
                        st.session_state.profile
                    )
                    ext = st.write_stream(stream)
                    st.session_state.history[idx]["extension"] = ext
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # --- BOTTOM INPUT SECTION ---
    # Continuous listening controls are now at the top for better visibility
    st.sidebar.divider()
    st.sidebar.subheader("Transcription Status")
    if DEEPGRAM_API_KEY:
        st.sidebar.success("Deepgram: Connected")
    else:
        st.sidebar.error("Deepgram: API Key Missing")

    # Handling Automated Answer Generation (Separated to avoid blocking listen loop)
    if st.session_state.get("is_generating_answer", False) and st.session_state.get("current_question"):
        q = st.session_state.current_question
        with st.spinner("Assistant is crafting your answer..."):
            try:
                stream = generate_answer(
                    q, 
                    OPENROUTER_API_KEY, 
                    interview_tone, 
                    st.session_state.profile,
                    simple_english=simple_english,
                    bullet_points=bullet_mode
                )
                answer = "".join(list(stream))
                st.session_state.history.append({
                    "question": q,
                    "answer": answer,
                    "tone": interview_tone
                })
            finally:
                st.session_state.is_generating_answer = False
                st.session_state.current_question = None
                st.rerun()

    st.divider()

    # Manual Keyboard Entry (Pinned to bottom by Streamlit)

    # Manual Keyboard Entry (Pinned to bottom by Streamlit)
    if manual_question := st.chat_input("Or type the question here..."):
        with st.chat_message("assistant", avatar="💡"):
            stream = generate_answer(
                manual_question, 
                OPENROUTER_API_KEY, 
                interview_tone, 
                st.session_state.profile,
                simple_english=simple_english,
                bullet_points=bullet_mode
            )
            answer = st.write_stream(stream)
            st.session_state.history.append({
                "question": manual_question,
                "answer": answer,
                "tone": interview_tone
            })
            st.rerun()

if __name__ == "__main__":
    main()
