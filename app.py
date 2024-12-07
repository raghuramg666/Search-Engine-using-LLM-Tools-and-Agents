import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Custom DuckDuckGo Tool with Rate Limit Handling
class RateLimitedDuckDuckGo(DuckDuckGoSearchRun):
    def _run(self, query: str, verbose: bool = False):
        try:
            return super()._run(query, verbose=verbose)
        except Exception as e:
            if "RatelimitException" in str(e):
                time.sleep(5)  # Retry after 5 seconds
                return super()._run(query, verbose=verbose)
            else:
                raise e  # Re-raise other exceptions

# Arxiv tool setup
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# Wikipedia tool setup
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# DuckDuckGo tool setup
search = RateLimitedDuckDuckGo(name="Search")

# Streamlit Title
st.title("LangChain - Chat with Search Tools")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq AI API key", type="password")

# Check if API key is provided
if not api_key:
    st.warning("Please enter your Groq AI API key in the sidebar to proceed.")
    st.stop()

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display existing chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Save user input to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize ChatGroq and tools
    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
        tools = [search, arxiv, wiki]
        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True  # Adjusted argument
        )

        # Generate response using the agent
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

    except Exception as e:
        # Handle errors gracefully
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"}
        )
        st.chat_message("assistant").write(f"Sorry, I encountered an error: {str(e)}")
