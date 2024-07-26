import streamlit as st
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import ollama

# Function to simulate streaming data
def stream_data(text, delay: float = 0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# Initialize chat history, uploaded image, and selected model in session state if not already done
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

if 'prompt' not in st.session_state:
    st.session_state.prompt = ""

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'TinyLlama'

@st.cache_resource
def load_moondream_model():
    # Load the Moondream model and tokenizer
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"  # Update with the latest revision
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer

# Load the Moondream model and tokenizer
moondream_model, moondream_tokenizer = load_moondream_model()

# Streamlit app title
st.title("LOCAL JARVIS")
st.write("PROJECT BY REHAN ALI")
st.write("rehanalikhan4790@gmail.com")


# Add some spacing and style to the layout
st.markdown("""
    <style>
    .main-container {
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        margin-bottom: 20px.
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        background-color: #000000;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-top: 20px;
    }
    .user-message, .ai-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: #ffffff;
    }
    .user-message {
        background-color: #333333;
        text-align: left.
    }
    .ai-message {
        background-color: #444444;
        text-align: left.
    }
    .upload-btn, .model-switcher {
        display: flex;
        justify-content: center.
        margin-bottom: 20px.
    }
    </style>
    """, unsafe_allow_html=True)

# Main container for better styling
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Chat input from user
    prompt = st.text_input("ASK ANYTHING")

    # Model selection buttons
    st.markdown('<div class="model-switcher">', unsafe_allow_html=True)
    if st.button("Switch to JarVIS"):
        st.session_state.selected_model = 'TinyLlama'
    if st.button("Switch to VisION"):
        st.session_state.selected_model = 'Moondream'
    st.markdown('</div>', unsafe_allow_html=True)

    # Display current model
    if st.session_state.selected_model == 'TinyLlama':
        st.write(f"**Current Model:** JarVIS")
    else:
        st.write(f"**Current Model:** VisIon")

    # Button to generate response
    if st.button("Generate Response"):
        if prompt:
            # Add user's message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Display user's message
            st.write(f"**User:** {prompt}")

            if st.session_state.selected_model == 'Moondream' and st.session_state.uploaded_image:
                # Use Moondream for image-related tasks
                with st.spinner("Analyzing image..."):
                    image = Image.open(st.session_state.uploaded_image)
                    enc_image = moondream_model.encode_image(image)
                    response = moondream_model.answer_question(enc_image, prompt, moondream_tokenizer)
            else:
                # Use TinyLlama for text generation
                with st.spinner("Thinking..."):
                    result = ollama.chat(model="tinyllama", messages=st.session_state.chat_history)
                    response = result["message"]["content"]

            # Add response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display chat history
    st.header("Chat History")
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>User:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-message"><strong>AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # File uploader for the image
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.session_state.uploaded_image = uploaded_image

    # Display the uploaded image
    if st.session_state.uploaded_image:
        image = Image.open(st.session_state.uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
