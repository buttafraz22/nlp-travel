# -*- coding: utf-8 -*-
"""Travel Chatbot Model.ipynb

Automatically generated by Colab.

This is a code for preproessing the pdf dataset into the pinecone database.

# Travel Chatbot

## Setup
"""

!pip install -qU langchain python-dotenv tiktoken langchain-pinecone langchainhub pandas langchain_community pymupdf langchain-google-genai

# GLOBAL
import os
import pandas as pd
import numpy as np
import tiktoken
from uuid import uuid4
# from tqdm import tqdm
from dotenv import load_dotenv
from tqdm.autonotebook import tqdm


# LANGCHAIN
import langchain
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate

# VECTOR STORE
import pinecone
from pinecone import Pinecone, ServerlessSpec

# AGENTS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, Tool, AgentType
from langchain.agents.react.agent import create_react_agent
from langchain import hub

import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'redacted info, please get your own'
os.environ['GOOGLE_API_KEY'] = 'redacted info, please get your own'
os.environ['PINECONE_API_KEY'] = 'redacted info, please get your own'
os.environ['TAVILY_API_KEY'] = 'redacted info, please get your own'


# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
import google.generativeai as genai
GOOGLE_API_KEY= os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

"""## Loading Documents
There are several Document Loaders in the LangChain library depending on the type of file to be used. The most common ones include CSV, HTML, JSON, Markdown, File Directory or Microsoft Office formats. The complete list can be found [here](https://python.langchain.com/docs/modules/data_connection/document_loaders/office_file/).

However, there is a more extensive [list](https://python.langchain.com/docs/integrations/document_loaders/google_drive/), where you can load directly from Google Cloud, Notion, Youtube or many other services.

We will be using a Pdf file, so we will use the PyMuPdfLoader. Below you can find the code to load the file. As arguments we are using:

- **file path**

Loading the data in this way will benefit our RAG pipeline. The benefits of metadata are listed further below.
"""

from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("./Travel Guide.pdf") # Load the document
data = loader.load()
len(data)

"""## Indexing
The **Vector Store Index** is a tool that embeds your documents into vector representations. When you want to search through these embeddings, your query is also converted into a vector embedding. Then, the Vector Store Index performs a mathematical operation to rank all the document embeddings based on how semantically similar they are to your query embedding.

The key steps are:
- Embedding your documents into vectors
- Turning your search query into a vector
- Comparing the query vector to all the document vectors
- Ranking the document vectors by their similarity to the query vector
- Returning the most relevant documents based on this ranking

This allows you to search your document collection in a semantic, meaning-based way, rather than just looking for exact keyword matches.

To understand the process of vector search, we will analyze the concepts of tokenization, similarity, and embedding, which are implemented by embedding models.

According to OpenAI, as a rule of thumb 1 token corresponds to 4 characters of text for common English text. This means that 100 tokens correspond to 75 words.

## Embeddings
Embeddings are a way to represent high-dimensional sparse data like words in a more compact, lower-dimensional form while preserving the meaningful similarities between the original data points.. The key ideas are:

- **Capturing Similarities:** Similar items, like synonymous words, will have embedding vectors that are close to each other.

- **Spatial Representation:** The embedding vectors are positioned in a multi-dimensional space such that the distance between them (e.g. cosine similarity) reflects how related the original data points are

<img width="1000" alt="Untitled (1)" src="https://github.com/benitomartin/nlp-news-classification/assets/116911431/a7e044ab-2c40-47a2-bb05-e86962790ce0">


**Source**: https://openai.com/index/new-embedding-models-and-api-updates

### Cosine Similarity
The most common metric used for similarity search is **cosine similarity**. It finds application in scenarios like semantic search and document classification, because it enables the comparison of vector directions, effectively assessing the overall content of documents. By comparing the vector representations of the query and the documents, cosine similarity can identify the most similar and relevant documents to return in the search results.

<img width="1000" alt="Screenshot 2024-05-02 123447" src="https://github.com/benitomartin/nlp-news-classification/assets/116911431/f5356422-29d6-4a4c-8e11-267ad9115b51">

**Source:** https://www.pinecone.io/learn/vector-similarity/

Cosine similarity is a measure of the similarity between two non-zero vectors. It calculates the cosine of the angle between the two vectors, which results in a value between 1 (identical) and -1 (opposite).

<img width="1000" alt="Screenshot 2024-05-02 122629" src="https://github.com/benitomartin/nlp-news-classification/assets/116911431/4215ba02-1fb9-4a72-ad9c-e88740b5a71a">
"""

def cosine_similarity(query_emb, document_emb):

    # Calculate the dot product of the query and document embeddings
    dot_product = np.dot(query_emb, document_emb)

    # Calculate the L2 norms (magnitudes) of the query and document embeddings
    query_norm = np.linalg.norm(query_emb)
    document_norm = np.linalg.norm(document_emb)

    # Calculate the cosine similarity
    cosine_sim = dot_product / (query_norm * document_norm)
    return cosine_sim

"""**Simple Example of Cosine Similarity**

"""

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

question = "what is UET Lahore?"
document = "University of Engineering and Technology (UET) Lahore is a public university located in Lahore, Punjab, Pakistan specializing in STEM subjects. It is one of the oldest and most prestigious institutions of higher learning in Pakistan. UET Lahore was established in 1921 as Mughalpura Technical College and was later renamed as Maclagan Engineering College after Sir Edward Maclagan, the then Governor of the Punjab. In 1923, the college was affiliated with the University of the Punjab for awarding a bachelor's degree in engineering. In 1962, the college was granted university status and was renamed as UET Lahore. The university offers undergraduate, postgraduate, and doctoral programs in various engineering disciplines. UET Lahore is known for its strong emphasis on research and innovation and has produced several notable alumni who have made significant contributions to the field of engineering and technology."

# Using Google Generative AI EMbeddings Modael

query_emb = embeddings.embed_query(question)
document_emb = embeddings.embed_query(document)
cosine_sim = cosine_similarity(query_emb, document_emb)
print(f'Query Dimensions: {len(query_emb)}')
print(f'Document Dimensions: {len(document_emb)}')
print("Cosine Similarity:", cosine_sim)

"""## Text Splitting

Unfortunatelly, LLM models have some limitations when it comes to the point of processing text. One of those is the **context window**. The context window represents the maximum amount of text/tokens that a model can process at one time as an input to generate a response. Therefore we need to split our documents into smaller chunks that can fit into the model's context window. A complete list of OpenAI models can be found [here](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4). It spans from 4'096 tokens for the `gpt-3.5-turbo-instruct` to the `gpt-4-turbo` with 128'000 tokens.

Like the data loaders, LangChain offers several text splitters. In the table below you can see the main splitting methods and when to use which one. The `Adds Metadata` does not mean that it will add (or not) the metadata from the previous loader. For example for HTML has a `HTMLHeaderTextSplitter` and it means it will splits text at the element level and adds metadata for each chunk based on header text.

In our case we already have the metadata available and we do not need to add them using and splitter.

<img width="863" alt="Screenshot 2024-04-30 132645" src="https://github.com/benitomartin/mlops-car-prices/assets/116911431/7b4dcefb-9320-4085-821f-1b21c81f4d28">


**Source:** https://js.langchain.com/v0.1/docs/modules/data_connection/document_transformers/

The `RecursiveCharacterTextSplitter` is the recommended tool for splitting general text. It segments the text based on a defined chunk size, using a list of characters as separators.

According to LangChain, the default separators include ["\n\n", "\n", " ", ""]. This means it aims to keep paragraphs together first, followed by sentences and words, as they typically exhibit the strongest semantic connections in text.

To leverage this feature, we can utilize the `RecursiveCharacterTextSplitter` along with the tiktoken library to ensure that splits do not exceed the maximum token chunk size allowed by the language model. Each split will be recursively divided if its size exceeds the limit.

The final design of our text splitter will be as follows:

- Model: `Gemini Pro` with a context window of 16,385 tokens

- Chunk Size: number of tokens of one chunk

- Chunk Overlap: number of tokens that overlap between two consecutive chunks

- Separators: the order of separators
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ""]
)

"""## Vector Stores
A Vector Store is a specialized database that is designed to store and manage high-dimensional vector data. Vector databases store data in the form of vector embedding, which can be retrieved by the LLMs and allow them to understand the context and meaning of the data, allowing better responses.

### Indexing
Pinecone is a serverless vector store, which shows a very good performance for a fast vector search and retrieval process.

The first step to use Pinecone is to create an Index where our embeddings will be stored. There are several parameters to be considered for this:

- Index name
- Dimension: must be equal to the embedding model dimensions
- Metric: must match with the used to tain the embedding model for better results
- Serverless specifications
"""

index_name = "langchain-pinecone-travel-chatbot"
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key = PINECONE_API_KEY)

pc.create_index(
    name=index_name,
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"))

index = pc.Index(index_name)

pc.list_indexes()

# pc.delete_index(index_name)   # Deletes the index

index.describe_index_stats()

"""### Namespaces
Pinecone allows you to split the data into namespaces within an index. This allows to send queries to an specific namespace. You could for example split your data by content, language or any other index suitable for your use case.
"""

splits = text_splitter.split_documents(data)
embed = embedding= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = PineconeVectorStore.from_documents(documents=splits,
                                        embedding=embed,
                                        index_name=index_name,
                                        namespace="main"
                                        )

vectorstore = PineconeVectorStore(index_name=index_name,
                                  namespace="main",
                                  embedding=embed)

query = "Give me the best spots for diving."   # This information is on Page 52 of the book.
similarity = vectorstore.similarity_search(query, k=4)

for i in range(len(similarity)):
  print(f"-------Result Nr. {i}-------")
  print(f"Page Content: {similarity[i].page_content}")
  print(f" ")

index.describe_index_stats()

"""# RAG Pipeline"""

embed = embedding= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = PineconeVectorStore(index_name=index_name,
                                  namespace="main",
                                  embedding=embed)

"""### Retrieval"""

from langchain_google_genai import ChatGoogleGenerativeAI  # Import the Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
# Conversational memory
conversational_memory = ConversationBufferWindowMemory(memory_key='chat_history',k=5,return_messages=True)
# Retrieval qa chain
qa_db = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=vectorstore.as_retriever())

"""### Augmented
We are going to use a slightly modified prompt template. First we download the react template, which is a common template using toools and agents and then we will add the instruction of in which tool to look up first.

A collection of templates can be found in the [langchain hub](https://smith.langchain.com/hub)
"""

prompt = hub.pull("hwchase17/react")
print(prompt.template)

"""Now we will replace this line:

`Action: the action to take, should be one of [{tool_names}]`

By this line:

`Action: the action to take, should be one of [{tool_names}]. Always look first in Pinecone Document Store`


"""

template= '''
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

"""### Generation With Agent

We are going to set up 2 tools for our agent:

- Tavily Search API: Tavily search over several sources like Bing or Google and returns the most relevant content. It offers 1000 API calls per month for free.

- Vectorstore: Our vector store will be used to look for the information first.
"""

# Set up tools and agent
import os

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily = TavilySearchResults(max_results=10, tavily_api_key=TAVILY_API_KEY)

tools = [
    Tool(
        name = "Pinecone Document Store",
        func = qa_db.run,
        description = "This agent looks up information from the Pinecone Document Store"
    ),

    Tool(
        name="Tavily",
        func=tavily.run,
        description="If the information is not found by Pinecone Agent, this lookup information from Tavily",
    )

     # email sender tool

]
agent = create_react_agent(llm,tools,prompt)

agent_executor = AgentExecutor(
                        tools=tools,
                        agent=agent,
                        handle_parsing_errors=True,
                        verbose=True,
                        memory=conversational_memory)

"""Now that every thing is set up, let's run the agent."""

ques = input('Enter your query about the document: ')
response = agent_executor.invoke({"input": f"{ques}"})

"""Let's try a query that doesn't have the answer in the document."""

ques = input('Enter your query about the document: ')
response = agent_executor.invoke({"input": f"{ques}"})

"""Let's check the chat context."""

conversational_memory.load_memory_variables({})

"""Now that we are done, let's clear the memory!"""

agent_executor.memory.clear()