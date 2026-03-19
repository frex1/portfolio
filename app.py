import os
import json
import pickle
import faiss
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ================= CONFIG =================
MODEL_ID = "moonshotai/Kimi-K2.5"
client = InferenceClient(MODEL_ID, token=os.getenv("HUGGINGFACE_TOKEN"))

embedder = SentenceTransformer("BAAI/bge-small-en")

index = faiss.read_index("kb/index.faiss")
docs = pickle.load(open("kb/docs.pkl", "rb"))

# ================= HELPERS =================

def format_history(history):
    return [{"role": m["role"], "content": m["content"]} for m in history]

def sanitize_history(history):
    if not history:
        return []
    return [m for m in history if isinstance(m, dict) and "role" in m and "content" in m]

# ================= RETRIEVAL =================

def retrieve(query, top_k=5):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, top_k)
    return "\n\n".join([docs[i]["text"] for i in I[0]])

# ================= TOOLS =================

def query_github():
    try:
        res = requests.get("https://api.github.com/users/frex1/repos")
        return [
            {
                "name": r["name"],
                "description": r["description"],
                "url": r["html_url"]
            }
            for r in res.json() if not r.get("private")
        ]
    except:
        return []

def save_lead(email, name):
    os.makedirs("leads", exist_ok=True)
    with open("leads/leads.csv", "a") as f:
        f.write(f"{datetime.now()},{name},{email}\n")
    return "✅ Saved!"

def generate_resume():
    path = "Freeman_Goja_Resume.txt"
    with open(path, "w") as f:
        f.write(
            "Freeman Goja\nAI Engineer | Data Scientist\n\n"
            "Skills:\n- AI/ML\n- Data Science\n- LLM Engineering\n\n"
            "Projects:\n- AIlysis\n- LangChain Agents\n"
        )
    return path

# ================= DIGITAL TWIN =================

class DigitalTwin:
    def __init__(self):
        self.mode = "chat"

    def chat(self, user_input, history):
        history = sanitize_history(history)
        history.append({"role": "user", "content": user_input})

        context = retrieve(user_input)

        messages = [
            {"role": "system", "content": "You are Freeman Goja, an elite AI Engineer, Data Scientist, \
                Entrepreneur and AI Mentor. You interact with users in a friendly and engaging manner answering \
                    questions in a knowledeable and professional manner clearly and concisely."},
            {
                "role": "user",
                "content": f"Answer the following question concisely without elaborating unnecessarily. \
                    If you can't answer the question, advise the user to reach out on my contact page. \
                        Context:\n{context}\n\nQuestion:\n{user_input}"
            }
        ]

        response_text = ""

        try:
            # ✅ FIX: normal generator (NOT async)
            for chunk in client.chat_completion(
                messages=messages,
                stream=True,
                max_tokens=800
            ):
                delta = chunk.choices[0].delta.get("content", "")
                if delta:
                    response_text += delta

                    temp_history = history + [
                        {"role": "assistant", "content": response_text}
                    ]

                    yield "", format_history(temp_history)

        except Exception as e:
            response_text = f"⚠️ Error: {str(e)}"

        history.append({"role": "assistant", "content": response_text})

        yield "", format_history(history)

twin = DigitalTwin()

# ================= UI =================

LANDING_HTML = """
<div class="landing-root">
    <div class="landing-aurora" aria-hidden="true"></div>
    <div class="landing-grid" aria-hidden="true"></div>
    <div class="landing-inner">
        <p class="landing-eyebrow">Nice to meet you · I am</p>
        <h1 class="landing-title">Freeman Goja</h1>
        <p class="landing-tagline">AI Engineer · Data Scientist . Mentor .Founder</p>
        <p class="landing-lede">Glad you’re here. Take a look around, see what I’ve built, and feel free to reach out anytime.</p>
        <div class="landing-chips">
            <span class="chip">Knowledge-grounded replies</span>
            <span class="chip">Project gallery</span>
            <span class="chip">Resume &amp; leads</span>
        </div>
    </div>
</div>
"""

APP_THEME = (
    gr.themes.Soft(
        primary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
    )
    .set(
        body_background_fill="#030508",
        body_background_fill_dark="#030508",
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
        body_text_color_subdued="#94a3b8",
        body_text_color_subdued_dark="#94a3b8",
        block_background_fill="#0a0f18",
        block_background_fill_dark="#0a0f18",
        block_border_color="rgba(51, 65, 85, 0.55)",
        block_border_color_dark="rgba(51, 65, 85, 0.55)",
        block_label_text_color="#cbd5e1",
        block_label_text_color_dark="#cbd5e1",
        block_title_text_color="#f8fafc",
        block_title_text_color_dark="#f8fafc",
        block_info_text_color="#94a3b8",
        block_info_text_color_dark="#94a3b8",
        input_background_fill="#060a12",
        input_background_fill_dark="#060a12",
        input_border_color="#334155",
        input_border_color_dark="#334155",
        input_placeholder_color="#64748b",
        input_placeholder_color_dark="#64748b",
        border_color_primary="#334155",
        border_color_primary_dark="#334155",
        button_secondary_background_fill="rgba(30, 41, 59, 0.6)",
        button_secondary_background_fill_dark="rgba(30, 41, 59, 0.6)",
        button_secondary_text_color="#e2e8f0",
        button_secondary_text_color_dark="#e2e8f0",
        button_secondary_background_fill_hover="rgba(51, 65, 85, 0.75)",
        button_secondary_background_fill_hover_dark="rgba(51, 65, 85, 0.75)",
        shadow_drop="0 4px 24px rgba(0,0,0,0.35)",
    )
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body {
    background:
        radial-gradient(ellipse 100% 60% at 50% -15%, rgba(6, 182, 212, 0.14), transparent 55%),
        radial-gradient(ellipse 80% 50% at 100% 50%, rgba(99, 102, 241, 0.08), transparent 45%),
        #030508 !important;
    color: #e2e8f0 !important;
}

.gradio-container {
    max-width: 960px;
    margin: 0 auto;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}

/* Readable text on dark surfaces */
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .markdown,
.gradio-container .markdown p {
    color: #cbd5e1 !important;
}
.gradio-container .prose strong,
.gradio-container .markdown strong {
    color: #f1f5f9 !important;
}
.gradio-container label,
.gradio-container span[data-testid="block-info"] {
    color: #94a3b8 !important;
}

/* Landing */
#landing-wrap {
    min-height: 78vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem 1rem 3rem;
}
.landing-root {
    position: relative;
    width: 100%;
    max-width: 640px;
    padding: 2.75rem 2.25rem;
    border-radius: 20px;
    border: 1px solid rgba(56, 189, 248, 0.2);
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.95) 0%, rgba(8, 12, 22, 0.98) 100%);
    box-shadow:
        0 0 0 1px rgba(0,0,0,0.4),
        0 24px 80px rgba(0, 0, 0, 0.55),
        0 0 120px rgba(6, 182, 212, 0.08);
    overflow: hidden;
    animation: landingIn 0.85s ease-out;
}
.landing-aurora {
    position: absolute;
    inset: -40%;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(6, 182, 212, 0.25) 0%, transparent 45%),
        radial-gradient(ellipse at 80% 60%, rgba(99, 102, 241, 0.18) 0%, transparent 40%),
        radial-gradient(ellipse at 50% 100%, rgba(14, 165, 233, 0.12) 0%, transparent 50%);
    pointer-events: none;
}
.landing-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(148, 163, 184, 0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, 0.06) 1px, transparent 1px);
    background-size: 32px 32px;
    mask-image: radial-gradient(ellipse 80% 70% at 50% 40%, black 20%, transparent 100%);
    pointer-events: none;
}
.landing-inner {
    position: relative;
    z-index: 1;
    text-align: center;
}
.landing-eyebrow {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #22d3ee;
    margin: 0 0 1rem;
}
.landing-title {
    font-size: clamp(2rem, 5vw, 2.75rem);
    font-weight: 700;
    letter-spacing: -0.03em;
    margin: 0 0 0.35rem;
    color: #f8fafc;
    line-height: 1.1;
}
.landing-tagline {
    font-size: 1.1rem;
    font-weight: 500;
    color: #94a3b8;
    margin: 0 0 1.25rem;
}
.landing-lede {
    font-size: 1rem;
    line-height: 1.65;
    color: #cbd5e1;
    margin: 0 0 1.75rem;
    max-width: 36rem;
    margin-left: auto;
    margin-right: auto;
}
.landing-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
}
.chip {
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.4rem 0.85rem;
    border-radius: 999px;
    border: 1px solid rgba(71, 85, 105, 0.6);
    background: rgba(15, 23, 42, 0.6);
    color: #e2e8f0;
}
@keyframes landingIn {
    from { opacity: 0; transform: translateY(16px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

button.landing-cta {
    margin-top: 1.75rem !important;
    min-width: 220px;
    border-radius: 999px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    box-shadow: 0 8px 32px rgba(6, 182, 212, 0.25) !important;
}

/* App shell header */
#app-shell {
    padding-top: 0.25rem;
}
#app-header-row {
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
#header-brand {
    flex: 1 1 160px;
    min-width: 0;
}
.app-brand {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.app-brand-mark {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(120deg, #22d3ee, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-brand-sub {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
}
.header-crumb p,
.header-crumb {
    margin: 0 !important;
    font-size: 0.9rem !important;
    color: #94a3b8 !important;
}
.header-crumb strong {
    color: #e2e8f0 !important;
    font-weight: 600;
}

/* Pill menu */
.nav-pill-row {
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 1.25rem;
}
.nav-pill-row button {
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border: 1px solid rgba(51, 65, 85, 0.9) !important;
    background: rgba(15, 23, 42, 0.85) !important;
    color: #e2e8f0 !important;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s !important;
}
.nav-pill-row button:hover {
    border-color: rgba(34, 211, 238, 0.55) !important;
    box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.15) !important;
    transform: translateY(-1px);
}

/* Chat section — Gradio bubbles use CSS vars: .user → --color-accent-soft; .bot → --background-fill-secondary */
#chat-section {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(51, 65, 85, 0.55);
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    /* Cascade into Chatbot subtree (flex-wrap uses --body-text-color) */
    --body-text-color: #f8fafc;
    --body-text-color-subdued: #94a3b8;
    --color-accent-soft: #0b1220;
    --background-fill-secondary: #0b1220;
    --border-color-primary: rgba(148, 163, 184, 0.45);
    --border-color-accent: rgba(34, 211, 238, 0.55);
    --border-color-accent-subdued: rgba(34, 211, 238, 0.35);
    --color-text-link: #7dd3fc;
}
#chat-section .block {
    margin-bottom: 0.5rem;
}
#portfolio-chatbot .user,
#portfolio-chatbot .bot {
    background-color: #0b1220 !important;
    color: #f8fafc !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.35) !important;
}
#portfolio-chatbot .prose.chatbot.md {
    opacity: 1 !important;
}
#portfolio-chatbot .prose,
#portfolio-chatbot .prose p,
#portfolio-chatbot .prose li,
#portfolio-chatbot .prose td,
#portfolio-chatbot .prose th {
    color: #f8fafc !important;
}
#portfolio-chatbot .prose a {
    color: #7dd3fc !important;
}
#portfolio-chatbot .prose code {
    color: #e2e8f0 !important;
    background: rgba(30, 41, 59, 0.9) !important;
}
#portfolio-chatbot .prose pre {
    background: rgba(30, 41, 59, 0.95) !important;
    color: #e2e8f0 !important;
}

/* Inputs */
input[type="text"], textarea {
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
button.primary {
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Section cards */
.section-card {
    background: rgba(15, 23, 42, 0.45);
    border: 1px solid rgba(51, 65, 85, 0.45);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
}
.section-card h2, .section-card h4 {
    font-size: 1.2rem;
    font-weight: 600;
    color: #f1f5f9 !important;
    margin: 0 0 1rem;
}

/* Gallery */
.gallery-item img, .thumbnail-item img {
    border-radius: 10px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.gallery-item:hover img, .thumbnail-item:hover img {
    transform: scale(1.02);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
}

.section-panel {
    min-height: 280px;
}
.section-hint {
    font-size: 0.9rem;
    color: #94a3b8 !important;
    margin: 0 0 1rem;
}

/* Accordions — label contrast */
.gr-accordion summary,
details summary {
    color: #e2e8f0 !important;
}
"""

NAV_KEYS = ("chat", "projects", "resume", "contact")

SECTION_LABELS = {
    "chat": "💬 Chat",
    "projects": "🚀 Projects",
    "resume": "📄 Resume",
    "contact": "📬 Contact",
}


def _nav_visibility(active: str):
    """Show one main column at a time based on section key."""
    key = (active or "chat").lower() if isinstance(active, str) else "chat"
    if key not in NAV_KEYS:
        key = "chat"
    return tuple(gr.update(visible=(k == key)) for k in NAV_KEYS)


def go_section(key: str):
    """Switch visible panel and update header crumb."""
    k = (key or "chat").lower() if isinstance(key, str) else "chat"
    if k not in NAV_KEYS:
        k = "chat"
    vis = _nav_visibility(k)
    crumb = f"**Now viewing:** {SECTION_LABELS[k]}"
    return (*vis, crumb)


def handle_resume():
    return generate_resume()


def show_app_shell():
    return gr.update(visible=False), gr.update(visible=True)


def show_landing_page():
    return gr.update(visible=True), gr.update(visible=False)


with gr.Blocks(css=CSS, theme=APP_THEME, title="AI Freeman") as demo:

    with gr.Column(visible=True, elem_id="landing-wrap") as col_landing:
        gr.HTML(LANDING_HTML)
        enter_btn = gr.Button(
            "Enter portfolio",
            variant="primary",
            size="lg",
            elem_classes=["landing-cta"],
        )

    with gr.Column(visible=False, elem_id="app-shell") as col_app:
        with gr.Row(elem_id="app-header-row"):
            gr.HTML(
                """
                <div id="header-brand" class="app-brand">
                    <span class="app-brand-mark">AI Freeman</span>
                    <span class="app-brand-sub">Portfolio</span>
                </div>
                """
            )
            section_crumb = gr.Markdown(
                "**Now viewing:** 💬 Chat",
                elem_classes=["header-crumb"],
            )
            back_welcome = gr.Button(
                "Welcome",
                variant="secondary",
                size="sm",
                min_width=96,
            )

        with gr.Row(elem_classes=["nav-pill-row"]):
            nav_chat = gr.Button("💬 Chat", variant="secondary", size="sm", scale=1)
            nav_projects = gr.Button("🚀 Projects", variant="secondary", size="sm", scale=1)
            nav_resume = gr.Button("📄 Resume", variant="secondary", size="sm", scale=1)
            nav_contact = gr.Button("📬 Contact", variant="secondary", size="sm", scale=1)

        with gr.Column(visible=True, elem_classes=["section-panel"]) as col_chat:
            gr.Markdown(
                "<p class='section-hint'>Hi, I am Freeman. How are you doing today?</p>"
            )
            with gr.Group(elem_id="chat-section"):
                chatbot = gr.Chatbot(height=480, elem_id="portfolio-chatbot")
                msg = gr.Textbox(
                    placeholder="Ask me anything about AI or mentorship...",
                    show_label=False,
                    container=False,
                )
            msg.submit(twin.chat, [msg, chatbot], [msg, chatbot])

        with gr.Column(visible=False, elem_classes=["section-panel"]) as col_projects:
            with gr.Accordion("About this gallery", open=False):
                gr.Markdown(
                    "Highlights from recent work. Images are placeholders—swap URLs for real screenshots anytime."
                )
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("#### 🚀 Projects")
                gallery = gr.Gallery(
                    value=[
                        ("https://picsum.photos/300/200", "AI EDA App"),
                        ("https://picsum.photos/300/201", "LangChain Agent"),
                        ("https://picsum.photos/300/202", "AIlysis Platform"),
                    ],
                    columns=3,
                    show_label=False,
                )

        with gr.Column(visible=False, elem_classes=["section-panel"]) as col_resume:
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("#### 📄 Resume")
                gr.Markdown(
                    "<p class='section-hint'>Generate a fresh text resume file, then download it below.</p>"
                )
                resume_file = gr.File(label="Download Resume", show_label=True)
                resume_btn = gr.Button("Generate Resume", variant="primary")
                resume_btn.click(handle_resume, [], resume_file)

        with gr.Column(visible=False, elem_classes=["section-panel"]) as col_contact:
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("#### 📬 Stay Connected")
                gr.Markdown(
                    "<p class='section-hint'>Leave your details—I’ll get back to you.</p>"
                )
                with gr.Accordion("Optional: what happens with your info?", open=False):
                    gr.Markdown(
                        "Your name and email are appended to `leads/leads.csv` on the server for follow-up only."
                    )
                name = gr.Textbox(label="Name", placeholder="Your name")
                email = gr.Textbox(label="Email", placeholder="you@example.com")
                lead_btn = gr.Button("Submit", variant="primary")
                lead_output = gr.Textbox(label="Status", interactive=False)
                lead_btn.click(save_lead, [email, name], lead_output)

        _nav_targets = [col_chat, col_projects, col_resume, col_contact, section_crumb]
        nav_chat.click(lambda: go_section("chat"), [], _nav_targets)
        nav_projects.click(lambda: go_section("projects"), [], _nav_targets)
        nav_resume.click(lambda: go_section("resume"), [], _nav_targets)
        nav_contact.click(lambda: go_section("contact"), [], _nav_targets)

        back_welcome.click(show_landing_page, [], [col_landing, col_app])

    enter_btn.click(show_app_shell, [], [col_landing, col_app])

# ================= LAUNCH =================

if __name__ == "__main__":
    demo.launch()