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
            {"role": "system", "content": "You are Freeman Goja, an elite AI Engineer."},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{user_input}"
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

CSS = """
body {
    background: #0b1120;
    color: white;
}

#hero {
    text-align:center;
    padding:50px;
    animation: fadeIn 1.5s ease-in;
}

@keyframes fadeIn {
    from {opacity:0; transform: translateY(30px);}
    to {opacity:1; transform: translateY(0);}
}

.card:hover {
    transform: scale(1.05);
    transition: 0.3s;
}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    <div id="hero">
    <h1>🤖 AI Freeman</h1>
    <p>AI Engineer | Founder | Digital Twin</p>
    </div>
    """)

    chatbot = gr.Chatbot(height=500)

    msg = gr.Textbox(placeholder="Ask me anything about AI, projects, or hiring...")

    msg.submit(twin.chat, [msg, chatbot], [msg, chatbot])

    # ===== PROJECT GALLERY =====
    gr.Markdown("## 🚀 Projects")

    gallery = gr.Gallery(
        value=[
            ("https://picsum.photos/300/200", "AI EDA App"),
            ("https://picsum.photos/300/201", "LangChain Agent"),
            ("https://picsum.photos/300/202", "AIlysis Platform"),
        ],
        columns=3
    )

    # ===== RESUME =====
    gr.Markdown("## 📄 Resume")

    resume_file = gr.File(label="Download Resume")

    def handle_resume():
        return generate_resume()

    resume_btn = gr.Button("Generate Resume")
    resume_btn.click(handle_resume, [], resume_file)

    # ===== LEAD CAPTURE =====
    gr.Markdown("## 📬 Stay Connected")

    name = gr.Textbox(label="Name")
    email = gr.Textbox(label="Email")

    lead_btn = gr.Button("Submit")
    lead_output = gr.Textbox(label="Status")

    lead_btn.click(save_lead, [email, name], lead_output)

# ================= LAUNCH =================

if __name__ == "__main__":
    demo.launch()