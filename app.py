# %% [markdown]
# ## Importing Libraries

# %%
import textwrap
import numpy as np
import pandas as pd

from typing import List

import google.generativeai as genai
import google.ai.generativelanguage as glm

from PyPDF2 import PdfReader

# Importing the CharacterTextSplitter class from the langchain library to split the text into chunks
from langchain.text_splitter import CharacterTextSplitter

from pinecone import Pinecone


from IPython.display import Markdown

import getpass
import os


# %% [markdown]
# ## Google Gemini (LLM) model Configuration

# %%
#GOOGLE_API_KEY=getpass.getpass()
os.environ["GOOGLE_API_KEY"] = "AIzaSyBkNoxMj5tWf7k0JnddvxGW83zsXqVG2ak"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# %% [markdown]
# ## Pinecone (Vector DB) Configuration

# %%
pc = Pinecone("492fe419-7850-4384-8dc4-d2019c9d1ab2")

# %%
# Extract the content of the PDF
pdf_content = ""
# Loop through the PDF files
pdf_docs = ["CV.pdf"]
for pdf in pdf_docs:
    # Read the PDF file
    pdf_reader = PdfReader(pdf)
    # Loop through the pages of the PDF file
    for page in pdf_reader.pages:
        # Extract the text from the PDF page and add it to the pdf_content variable
        pdf_content += page.extract_text()
# st.write(pdf_content)
# Get chunks of the content
# Split the text into chunks of 2000 characters with an overlap of 200 characters
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)
# Split the text into chunks of 2000 characters with an overlap of 200 characters
chunks = text_splitter.split_text(pdf_content)

# %%
#chunks[0]

# %%
df = pd.DataFrame(chunks)
df.columns = ["Content"]
#df

# %%


embeddings = pc.inference.embed(
    "multilingual-e5-large",
    inputs=df['Content'].tolist(),
    parameters={
        "input_type": "passage"
    }
)

vectors = []
for i, (d, e) in enumerate(zip(df['Content'], embeddings)):
    vectors.append({
        "id": str(i),
        "values": e['values'],
        "metadata": {'text': d}
    })

index = pc.Index('store')

index.upsert(
    vectors=vectors,
    namespace="ns1"
)

# %% [markdown]
# ## Check which Gemini models are available for use

# %%
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

# %% [markdown]
# ## We'll be using Gemini 1.5-flash

# %%
model = genai.GenerativeModel('models/gemini-1.5-flash')

# %% [markdown]
# ## Build the prompt for the LLM

# %%
def build_prompt(query: str, context: List[str]) -> str:
    """
    Builds a prompt for the LLM. #

    This function builds a prompt for the LLM. It takes the original query,
    and the returned context, and asks the model to answer the question based only
    on what's in the context, not what's in its weights.

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A prompt for the LLM (str).
    """

    base_prompt = {
        "content": "You are an expert recruiter. You are tasked with verifying the CVs given to you. Answer only based on the context provided when related to the domain that you are expert in otherwise have a normal conversation. Explain your answer briefly.",
    }
    user_prompt = {
        "content": f" The question is '{query}'. Here is all the context you have:"
        f'{(" ").join(context)}',
    }

    # combine the prompts to output a single prompt string
    system = f"{base_prompt['content']} {user_prompt['content']}"

    return system


# %% [markdown]
# ## Generating Gemini response

# %%
def get_gemini_response(query: str, context: List[str]) -> str:
    """
    Queries the Gemini API to get a response to the question.

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A response to the question.
    """

    response = model.generate_content(build_prompt(query, context))

    return response.text

# %% [markdown]
# ## Chatting with the LLM

# %%
import streamlit as st


st.title("Chat with Gemini")

# Initialize session state for query and response
if "query" not in st.session_state:
    st.session_state.query = ""
if "response" not in st.session_state:
    st.session_state.response = ""

# Input box for user query
query = st.text_input("Query:", value=st.session_state.query)

if st.button("Submit"):
    if len(query) == 0:
        st.write("Please enter a question.")
    else:
        st.session_state.query = query
        st.write("Thinking...")

        x = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={
                "input_type": "query"
            }
        )

        results = index.query(
            namespace="ns1",
            vector=x[0].values,
            top_k=3,
            include_values=False,
            include_metadata=True
        )

        context = [match['metadata']['text'] for match in results['matches']]
        response = get_gemini_response(query, context)

        st.session_state.response = response


# Display the response
if st.session_state.response:
    st.write(f"Question: {st.session_state.query}")
    st.write(f"Response: {st.session_state.response}")
    st.session_state.query = ""

# %%



