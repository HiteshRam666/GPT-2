import torch
import streamlit as st
import tiktoken
import time

from GPT.GPT_Model import GPTModel
from GPT.Text_Generation import text_to_tokens, token_to_text

# ---------------------------
# 🔹 Configuration
# ---------------------------
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(MODEL_CONFIGS["gpt2-small (124M)"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")


# ---------------------------
# 🔹 Load Model
# ---------------------------
def load_model(checkpoint_path: str):
    model = GPTModel(BASE_CONFIG)
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # fix naming mismatches
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("trf_blocks", "trf_block")
        k = k.replace("ff.", "ffn.")
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model.to(DEVICE).eval()


# ---------------------------
# 🔹 Streaming Generator
# ---------------------------
def generate_stream(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)[:, -1, :]

            # Top-k filtering
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, float("-inf"), logits)

            # Sampling
            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_id is not None and idx_next.item() == eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            yield idx


# ---------------------------
# 🔹 Streamlit UI
# ---------------------------
st.set_page_config(page_title="Chota ChatGPT 🚀", page_icon="🤖", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #0D0D0D;
        color: white;
        font-family: "Segoe UI", sans-serif;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 15px;
        color: #00FFB2;
    }
    .chat-container {
        max-height: 550px;
        overflow-y: auto;
        padding: 15px;
        border-radius: 12px;
        background: rgba(30, 30, 30, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    }
    .user-bubble {
        background: linear-gradient(135deg, #00A67E, #007B5E);
        padding: 12px 18px;
        border-radius: 15px 15px 0px 15px;
        margin: 8px 0;
        color: white;
        display: inline-block;
        max-width: 80%;
        float: right;
        clear: both;
        animation: fadeIn 0.3s ease-in-out;
    }
    .bot-bubble {
        background: #2C2C2C;
        padding: 12px 18px;
        border-radius: 15px 15px 15px 0px;
        margin: 8px 0;
        color: #EDEDED;
        display: inline-block;
        max-width: 80%;
        float: left;
        clear: both;
        animation: fadeIn 0.3s ease-in-out;
    }
    .typing {
        color: #AAAAAA;
        font-style: italic;
        padding-left: 10px;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Chota ChatGPT 😁</div>", unsafe_allow_html=True)

# Sidebar settings
st.sidebar.header("⚙️ Settings")
max_tokens = st.sidebar.slider("Max New Tokens", 50, 300, 120, 10)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.9, 0.1)
top_k = st.sidebar.slider("Top-K Sampling", 10, 100, 40, 5)

# Load model once
@st.cache_resource
def get_model():
    return load_model("C:\\Users\\hites\\OneDrive\\Desktop\\GPT\\notebooks\\gpt2-small-124M.pth")

gpt = get_model()

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear button with confirmation
if st.sidebar.button("🗑️ Clear Chat"):
    if st.sidebar.checkbox("Confirm Clear", False):
        st.session_state.messages = []

# Chat history display
chat_container = st.container()
with chat_container:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for role, msg in st.session_state.messages:
        if role == "user":
            st.markdown(f"<div class='user-bubble'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>{msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Input box
with st.container():
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "💬 Type your message:", 
            "", 
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        # Save user message
        st.session_state.messages.append(("user", user_input))
        chat_container.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)

        # Typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown("<div class='typing'>🤖 Bot is typing...</div>", unsafe_allow_html=True)

        # Generate bot response
        input_tokens = text_to_tokens(user_input, tokenizer).to(DEVICE)
        response_placeholder = st.empty()
        response_text = ""

        for output_tokens in generate_stream(
            model=gpt,
            idx=input_tokens,
            max_new_tokens=max_tokens,
            context_size=BASE_CONFIG["context_length"],
            top_k=top_k,
            temperature=temperature,
        ):
            response_text = token_to_text(output_tokens, tokenizer)
            response_placeholder.markdown(f"<div class='bot-bubble'>{response_text}▌</div>", unsafe_allow_html=True)
            time.sleep(0.02)

        # Finalize bot response
        typing_placeholder.empty()
        response_placeholder.markdown(f"<div class='bot-bubble'>{response_text}</div>", unsafe_allow_html=True)
        st.session_state.messages.append(("bot", response_text))
