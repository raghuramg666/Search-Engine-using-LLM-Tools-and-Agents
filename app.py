import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import requests
from bs4 import BeautifulSoup

# Custom Bing Search Tool
class BingSearchTool:
    """A tool to perform web searches using Bing."""
    name = "BingSearch"
    description = "A tool to perform web searches using Bing."

    def run(self, query: str):
        """Search Bing for the query and return the top results."""
        try:
            url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            results = []
            for result in soup.select("li.b_algo h2 a"):
                results.append(result["href"])
                if len(results) >= 5:  # Limit to top 5 results
                    break

            return "\n".join(results) if results else "No results found."
        except Exception as e:
            return f"An error occurred while searching: {str(e)}"

# Initialize Bing Search as a LangChain Tool
bing_search_tool = Tool(
    name="BingSearch",
    func=BingSearchTool().run,
    description="Use this tool to search the web using Bing."
)

# Arxiv tool setup
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = Tool(
    name="Arxiv",
    func=ArxivQueryRun(api_wrapper=api_wrapper_arxiv).run,
    description="Use this tool to search academic papers on Arxiv."
)

# Wikipedia tool setup
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_tool = Tool(
    name="Wikipedia",
    func=WikipediaQueryRun(api_wrapper=api_wrapper_wiki).run,
    description="Use this tool to search Wikipedia for summaries and details."
)

# Streamlit App
st.title("LangChain - Chat with Search Tools")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq AI API key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a chatbot who can search the web, academic papers, and Wikipedia. How can I help you?"}
    ]

# Display chat messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input("What would you like to ask?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize ChatGroq and tools
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [bing_search_tool, arxiv_tool, wiki_tool]  # Adding Bing to the tools
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=50,  # Allow up to 50 iterations
        max_execution_time=None  # No time limit
    )

    # Generate response using the agent
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            st.write("Processing your query...")
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.write(response)
        except Exception as e:
            # Handle invalid tool selection
            if "None is not a valid tool" in str(e):
                st.warning("The agent could not find the right tool. Defaulting to BingSearch.")
                fallback_response = bing_search_tool.func(prompt)
                st.session_state["messages"].append({"role": "assistant", "content": fallback_response})
                st.chat_message("assistant").write(fallback_response)
            else:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.session_state["messages"].append({"role": "assistant", "content": error_message})
                st.chat_message("assistant").write(error_message)
