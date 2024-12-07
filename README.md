# LangChain Chatbot with Search Tools

This repository hosts a chatbot application built with LangChain and Streamlit. The chatbot leverages multiple tools and agents to answer user queries intelligently, including:
- Bing Search for web-based information
- Arxiv for academic research papers
- Wikipedia for factual knowledge

The application uses LangChain's agent framework to dynamically choose the best tool for a query and integrates a CI/CD pipeline with GitHub Actions for automated testing and deployment.

---

## Features

- Web Search: Powered by Bing scraping, providing up-to-date search results.
- Academic Research: Retrieves summaries from Arxiv for research-based queries.
- Knowledge Summaries: Uses Wikipedia to fetch concise explanations.
- Tool-Based Reasoning: Dynamically selects the most appropriate tool using LangChain agents.
- Fallback Mechanism: Ensures valid responses even in case of tool selection issues.
- CI/CD Pipeline: Automated build and deployment workflows with GitHub Actions.

---

## Tools Used

The application uses the following tools, integrated via LangChain:

### Bing Search
- Purpose: Provides up-to-date information from the web.
- Implementation: A custom scraping-based tool using requests and BeautifulSoup.
- Use Case: General web queries, such as "Top programming languages in 2024" or "Latest trends in AI."

### Arxiv
- Purpose: Retrieves research papers and summaries from the Arxiv database.
- Integration: Utilizes LangChain's ArxivAPIWrapper.
- Use Case: Academic queries, such as "Recent papers on quantum computing" or "Research in neural networks."

### Wikipedia
- Purpose: Fetches factual and concise information from Wikipedia.
- Integration: Utilizes LangChain's WikipediaAPIWrapper.
- Use Case: General knowledge queries, such as "History of artificial intelligence" or "What is machine learning?"

---

## Agents

The application uses LangChain's agent framework to intelligently select tools and process user queries.

### Agent Configuration
- Type: ZERO_SHOT_REACT_DESCRIPTION
  - Enables the agent to interpret user queries dynamically and choose the most relevant tool based on the query description.
- Tools Included: Bing Search, Arxiv, Wikipedia.
- Fallback Handling: If an invalid tool is selected, the agent defaults to Bing Search for a valid response.


