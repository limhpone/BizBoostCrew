import streamlit as st
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyCATJWITXbrexqBwu6amQjm1pgBXFzVoW4")  # Replace with your actual API key


# Create the generative model with the configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Initialize chat session
chat_session = model.start_chat(
    history=[
        {"role": "user", "parts": ["hi / hello"]},
        {
            "role": "model",
            "parts": [
                "Hello! How can I help you today? This is Lim, your personalized BizBoost assistant! May I know your name?",
            ],
        },
    ]
)

# Streamlit UI setup
st.set_page_config(page_title="BizBoost Fitness & Wellness Chatbot", layout="wide")
st.title("BizBoost Fitness & Wellness Chatbot")

st.sidebar.image("images/logo.png", use_column_width=True) 

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# CSS for fixing alignment and ensuring the input bar stays fixed at the bottom
st.markdown(
    """
    <style>
    body {
        margin: 0;
    }
    .chat-container {
        max-height: calc(100vh - 100px); /* Adjust space for input bar */
        overflow-y: auto;
        padding: 10px;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        margin-left: auto; /* Align user messages to the right */
        background-color: #cce5ff;
        color: #004085;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 70%;
        font-size: 1em;
        text-align: right;
    }

    .bot-message {
        margin-right: auto; /* Align bot messages to the left */
        background-color: #f8f9fa;
        color: #495057;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 70%;
        font-size: 1em;
        text-align: left;
    }

    .fixed-input {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: #ffffff;
        padding: 10px;
        border-top: 1px solid #ccc;
        display: flex; /* Use flexbox for side-by-side alignment */
        align-items: center; /* Align input box and button vertically */
        gap: 10px; /* Space between input box and button */
        z-index: 100; /* Ensure input bar stays above content */
    }

    .input-box {
        flex-grow: 1; /* Input box takes up remaining space */
        color: #004085;
        font-size: 1em;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        box-sizing: border-box; /* Ensure padding is included in size */
        
    }

    .send-button {
        background-color: #f8f9fa;
        color: blue;
        border: none;
        border-radius: 5px;
        padding: 10px 20px; /* Adjust padding for consistent height with input box */
        cursor: pointer;
        text-align: center;
    }

    .send-button:hover {
        background-color: #f8f9fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Chat container for displaying messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    elif message["role"] == "model":
        st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Fixed Input Bar using Streamlit's native components
with st.container():
    col1, col2 = st.columns([4, 1], gap="small")
    with col1:
        user_input = st.text_input("Type your message here:", key="user_input", label_visibility="collapsed")
    with col2:
        if st.button("Send", key="send_button"):
            if user_input.strip():  # Check for non-empty input
                # Add user message
                st.session_state["messages"].append({"role": "user", "content": user_input})

                # Generate chatbot response
                response = chat_session.send_message(user_input)
                chatbot_message = response.text

                # Add bot message
                st.session_state["messages"].append({"role": "model", "content": chatbot_message})
