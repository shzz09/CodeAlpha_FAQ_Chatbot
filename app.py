
import json
import re
from pathlib import Path

import streamlit as st
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Askora | AI FAQ Assistant",
    page_icon="💬",
    layout="centered"
)


# ================= COLOURFUL UI THEME =================

st.markdown("""
<style>

/* Full page background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background:
        radial-gradient(ellipse at 5% 5%, #E5DCFF 0%, transparent 38%),
        radial-gradient(ellipse at 95% 15%, #D8E9FF 0%, transparent 35%),
        radial-gradient(ellipse at 75% 90%, #F2DFFF 0%, transparent 38%),
        linear-gradient(135deg, #F4F1FF, #EDF4FF 55%, #F8F0FF);
    background-attachment: fixed;
}

/* Main content */
.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Remove default header background */
[data-testid="stHeader"] {
    background: transparent;
}

/* Main title */
.brand-title {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #44358B;
}

.brand-subtitle {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: #8174B9;
    margin-top: -5px;
}

/* Online status */
.online-status {
    display: inline-block;
    background: #E1F7ED;
    color: #16835E;
    border: 1px solid #BDEBD6;
    padding: 8px 15px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
    text-align: center;
}

/* Welcome section */
.welcome-title {
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #352B72;
    margin-top: 35px;
    margin-bottom: 8px;
}

.welcome-subtitle {
    text-align: center;
    font-size: 15px;
    color: #777394;
    margin-bottom: 25px;
}

/* Section headings */
h2, h3 {
    color: #403579 !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #FFFFFF, #F1EEFF);
    color: #493B8B;
    border: 1px solid #D8D0F5;
    border-radius: 13px;
    min-height: 48px;
    font-size: 14px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(75, 57, 140, 0.06);
    transition: all 0.25s ease;
}

.stButton button:hover {
    background: linear-gradient(135deg, #7565D8, #6554C0);
    color: white;
    border-color: #7565D8;
    transform: translateY(-2px);
    box-shadow: 0 7px 18px rgba(91, 73, 174, 0.2);
}

/* Chat message cards */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #E2DDF5;
    border-radius: 17px;
    padding: 15px;
    box-shadow: 0 5px 18px rgba(68, 53, 139, 0.06);
    margin-bottom: 12px;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #CFC6F0;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(68, 53, 139, 0.08);
}

[data-testid="stChatInput"]:focus-within {
    border-color: #7767D2;
    box-shadow: 0 0 0 2px rgba(119, 103, 210, 0.12);
}

/* Dividers */
hr {
    border-color: rgba(111, 94, 180, 0.18) !important;
}

/* Confidence text */
[data-testid="stCaptionContainer"] {
    color: #8178A6;
}

/* Footer */
.footer {
    text-align: center;
    color: #786BAE;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 15px 0 5px 0;
}

/* Small screen support */
@media (max-width: 600px) {
    .welcome-title {
        font-size: 27px;
    }

    .brand-title {
        font-size: 28px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)


# ================= LOAD FAQ DATA =================

FAQ_FILE = Path(__file__).parent / "faqs.json"

try:
    with open(FAQ_FILE, "r", encoding="utf-8") as file:
        faqs = json.load(file)

except FileNotFoundError:
    st.error("faqs.json not found. Keep it in the same folder as app.py.")
    st.stop()

except json.JSONDecodeError:
    st.error("Invalid JSON format in faqs.json.")
    st.stop()

if not isinstance(faqs, list) or not faqs:
    st.error("FAQ dataset is empty or invalid.")
    st.stop()

try:
    faq_questions = [item["question"] for item in faqs]
    faq_answers = [item["answer"] for item in faqs]

except (KeyError, TypeError):
    st.error("Each FAQ must contain question and answer fields.")
    st.stop()


# ================= TEXT PREPROCESSING =================

stemmer = PorterStemmer()

stop_words = {
    "is", "am", "are", "was", "were", "the", "a", "an",
    "and", "or", "to", "of", "in", "on", "for", "what",
    "how", "why", "when", "where", "who", "which", "with",
    "do", "does", "did", "can", "could", "would", "should",
    "i", "you", "we", "they", "it", "this", "that",
    "my", "your", "me", "please"
}


def preprocess_text(text):
    text = text.lower()
    words = re.findall(r"\b[a-z0-9]+\b", text)

    return " ".join(
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    )


# ================= TF-IDF MODEL =================

processed_questions = [
    preprocess_text(question)
    for question in faq_questions
]


@st.cache_resource
def create_model(questions):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)
    return vectorizer, vectors


try:
    vectorizer, question_vectors = create_model(
        processed_questions
    )

except ValueError:
    st.error("Unable to build the FAQ model. Check your FAQ questions.")
    st.stop()


# ================= ANSWER MATCHING =================

def get_answer(user_question):

    processed_input = preprocess_text(user_question)

    if not processed_input.strip():
        return "Please enter a meaningful question.", 0.0

    input_vector = vectorizer.transform([processed_input])

    scores = cosine_similarity(
        input_vector,
        question_vectors
    )

    best_index = scores.argmax()
    confidence = float(scores[0][best_index])

    if confidence < 0.15:
        return (
            "Sorry, I couldn't find a relevant answer. "
            "Please try asking a question related to my FAQ database.",
            confidence
        )

    return faq_answers[best_index], confidence


# ================= SESSION STATE =================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 Welcome to Askora. "
                "I'm your AI FAQ assistant. "
                "Ask me anything related to my FAQ database!"
            ),
            "confidence": None
        }
    ]

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ================= HEADER =================

header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown(
        '<div class="brand-title">✦ Askora</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="brand-subtitle">YOUR AI FAQ ASSISTANT</div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div class="online-status">● Online</div>',
        unsafe_allow_html=True
    )

st.divider()


# ================= WELCOME SECTION =================

st.markdown(
    '<div class="welcome-title">How can I help you?</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="welcome-subtitle">'
    "Ask a question and discover the answers you need."
    '</div>',
    unsafe_allow_html=True
)


# ================= NEW CONVERSATION =================

if st.button("＋  New Conversation", use_container_width=True):

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 Welcome to Askora. "
                "I'm your AI FAQ assistant. "
                "Ask me anything related to my FAQ database!"
            ),
            "confidence": None
        }
    ]

    st.session_state.pending_question = None
    st.rerun()


# ================= SUGGESTED QUESTIONS =================

st.subheader("✨ Explore questions")

suggestions = [
    "What is CodeAlpha?",
    "What is NLP?",
    "What is TF-IDF?",
    "Explain cosine similarity."
]

columns = st.columns(2)

for index, question in enumerate(suggestions):

    with columns[index % 2]:

        if st.button(
            question,
            key=f"suggestion_{index}",
            use_container_width=True
        ):
            st.session_state.pending_question = question
            st.rerun()


st.divider()


# ================= CHAT HISTORY =================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message.get("confidence") is not None:
            st.caption(
                f"Match confidence: {message['confidence']:.1%}"
            )


# ================= CHAT INPUT =================

user_input = st.chat_input("💬  Ask Askora anything...")

question_to_process = (
    user_input or st.session_state.pending_question
)

if question_to_process:

    st.session_state.pending_question = None

    st.session_state.messages.append({
        "role": "user",
        "content": question_to_process,
        "confidence": None
    })

    answer, confidence = get_answer(question_to_process)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "confidence": confidence
    })

    st.rerun()


# ================= FOOTER =================

st.divider()

st.markdown(
    '<div class="footer">✦ ASKORA · AI KNOWLEDGE ASSISTANT</div>',
    unsafe_allow_html=True
)
