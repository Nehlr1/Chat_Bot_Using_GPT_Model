import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the model and tokenizer
model_path = './trained_gpt_model'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to ask the model a question
def ask_model(question):
    input_ids = tokenizer.encode(question, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=150, 
                            num_return_sequences=1, 
                            do_sample=True, 
                            no_repeat_ngram_size=2, 
                            top_k=50, top_p=0.95, 
                            pad_token_id=tokenizer.eos_token_id)
    full_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the original question from the generated text
    answer = full_answer[len(question):].strip()
    return answer

st.title("GPT-2 Chatbot")

# Initialize the session state if not present
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# User input
user_input = st.text_input("You:")

if user_input:
    if user_input.lower() in ['quit', 'exit']:
        st.session_state.conversation_history = []
    else:
        # Get model response
        response = ask_model(user_input)
        st.session_state.conversation_history.append(("You", user_input))
        st.session_state.conversation_history.append(("Bot", response))

# Display the conversation history
for role, line in st.session_state.conversation_history:
    st.write(f"{role}: {line}")



