import os
import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools import ImageCaptionTool, ObjectDetectionTool
import tempfile

# Specify the tools
tools = [ImageCaptionTool(), ObjectDetectionTool()]

# Specify the conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# Initialize the ChatOpenAI agent
llm = ChatOpenAI(
    openai_api_key='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',
    temperature=0,
    model_name='gpt-3.5-turbo'
)

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    max_iterations=50,
    verbose=True,
    memory=conversational_memory,
    early_stoppy_method='generate'
)

# Set up Streamlit UI
st.title("Please Ask A Question")
st.header("Please Upload A Image")

# File uploader
file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    st.image(file, use_column_width=True)
    user_question = st.text_input('Ask a Question')
    
    # Create a temporary file in the system's default temporary directory
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file.getbuffer())
        image_path = f.name

        # Check if the user question is provided
        if user_question and user_question != "":
            # Run the agent with the user question and image path
            response = agent.run('{},this is a image path:{}'.format(user_question, image_path))

            # Display the response
            with st.spinner(text='Please Wait....'):
                st.write(response)
