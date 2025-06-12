import streamlit as st
import os
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain import hub

# Load environment variable
from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(page_title="Wander", page_icon="✈️")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Page title and description
st.title("🌍 Wander: Your Travel Buddy")
st.write("Your AI-powered travel guide ready to answer all your travel queries!")

# Initialize Pinecone Vector Store
@st.cache_resource
def initialize_vector_store():
    """
    Initialize vector store and embeddings as a singleton resource.
    """
    # Use environment variables for credentials
    index_name = "langchain-pinecone-travel-chatbot"
    
    # Initialize embeddings
    embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create vector store (assuming index already exists)
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        namespace="main",
        embedding=embed
    )
    
    return vectorstore, embed

@st.cache_resource
def initialize_rag_pipeline():
    """
    Initialize RAG Pipeline by having conversation state of 5 messages.
    The chain type is stuff, which concatenates all the information into
    one string. This makes the retrieval part of the chain easier.
    """
    # Initialize vector store
    vectorstore, embed = initialize_vector_store()
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    # Conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history', 
        k=5, 
        return_messages=True
    )
    
    # Retrieval QA chain
    qa_db = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever()
    )
    
    # Custom prompt template
    template = '''
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]. Always look first in Pinecone Document Store if not then use the tool
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat 2 times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    '''
    prompt = PromptTemplate.from_template(template)
    
    # Set up tools
    tavily = TavilySearchResults(
        max_results=10, 
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    tools = [
        Tool(
            name="Pinecone Document Store",
            func=qa_db.run,
            description="This agent looks up information from the Pinecone Document Store"
        ),
        Tool(
            name="Tavily",
            func=tavily.run,
            description="If the information is not found by Pinecone Agent, this lookup information from Tavily",
        )
    ]
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        tools=tools,
        agent=agent,
        handle_parsing_errors=True,
        verbose=True,
        memory=conversational_memory
    )
    
    return agent_executor

# Sidebar for additional information
with st.sidebar:
    st.sidebar.header("🤖 About Wander")
    st.sidebar.info(
        "Wander chatbot is powered by Google's Gemini model and Pinecone vector search. "
        "It can help you with detailed travel-related queries based on a comprehensive travel guide."
    )

    st.sidebar.subheader("🤖 Wander's Architecture")
    st.markdown("""
        The locations covered by the chatbot are primarily Italy.
        The chatbot works on dual RAG (multimodal RAG) architecture, where the primary agent hits up information
        in the database, and if in case the primary fails, the secondary uses the internet to search the relevant information.
    """)

    st.sidebar.subheader("About the Developer")
    st.markdown("""
    Developed by: 2021-CS-12  
    Supervisor: Prof. Dr. Usman Ghani Khan  
    Course: Natural Language Processing
    """)

# Main chat interface
def main():
    # Initialize RAG pipeline
    try:
        agent_executor = initialize_rag_pipeline()
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        return

    # Chat input
    user_question = st.chat_input("Ask me anything about travel!")
   
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
   
    # Process new user question
    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
       
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
       
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Preparing your travel insights..."):
                try:
                    response = agent_executor.invoke({"input": user_question})
                    assistant_response = response.get('output', 'Sorry, I could not find specific information about your query.')
                   
                    # Display assistant response
                    st.markdown(assistant_response)
                   
                    # Check if Tavily search was used and display URLs
                    if 'intermediate_steps' in response:
                        for step in response['intermediate_steps']:
                            if step[0].tool == 'Tavily':
                                # Parse Tavily results
                                tavily_results = step[1]
                                if isinstance(tavily_results, list) and len(tavily_results) > 0:
                                    st.subheader("Top 5 Reference URLs")
                                    for i, result in enumerate(tavily_results[:5], 1):
                                        st.markdown(f"{i}. {result['url']}")
                   
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
               
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Run the app
if __name__ == "__main__":
    main()
