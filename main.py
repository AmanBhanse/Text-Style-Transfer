import getpass
import os

# Adityar

# 1. Access to Hugging face

# Prompt user for Hugging Face API token if not already set
def init_huggingface_api():
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your Huggingfacehub API token: ")


# 2. Packages
# !pip install -q langchain
# !pip install -q langchain-community
# !pip install -q langchain-chroma
# !pip install -q langchain-huggingface
# !pip install -q bs4
# !pip install -q rank_bm25
# !pip install -q huggingface_hub
# !pip install -q requests

# 3. Problem Statement



# 4 : Fetch and Parse

import os
import requests
import numpy as np
from bs4 import BeautifulSoup
from langchain.schema import Document

def fetch_and_parse(url: str) -> str:
    """
    Fetch the webpage content at `url` and return a cleaned string of text.

    Parameters:
    - url (str): The URL of the webpage to fetch.

    Returns:
    - str: Cleaned text content extracted from the webpage.
    """

    try:
        # Fetch the content of the URL.
        response = requests.get(url)

        # Ensure the request is successful.
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Step 2: Parse the HTML content using BeautifulSoup.
        soup = BeautifulSoup(response.content, 'html.parser')

        # Step 3: Extract the text content from the parsed HTML.
        text = soup.get_text()

        # Step 4: Return the cleaned text.
        # Aman : Not sure what cleaned text is ? Probably removing unnesscessary blank space
        cleaned_text = ' '.join(text.split())

        return cleaned_text

    except requests.RequestException as e:
        return f"Error fetching the webpage: {e}"   

def test_fetch_and_parse():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    fetch_and_parse(url)


def split_text_into_documents(text: str, chunk_size: int = 1000, overlap: int = 100):
    """
    Split a long text into overlapping chunks and return them as a list of Documents.

    Parameters:
    - text (str): The long text to split.
    - chunk_size (int): The size of each chunk (default is 1000 characters).
    - overlap (int): The number of overlapping characters between consecutive chunks (default is 100).

    Returns:
    - list: A list of Documents, each containing a chunk of text.
    """

    # Initialize an empty list to store the chunks.
    docs = []

    # Error check
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than the chunk size.")

    # Logic to split into chucks
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        chunk = text[start:end]
        
        docs.append(Document(page_content=chunk))
        
        # Move the start index forward by chunk_size - overlap.
        start += chunk_size - overlap

    return docs


# 5. Calculate Word Stats

def calculate_word_stats(texts):
    # I will work on this

    """
    Calculate and display average word and character statistics for a list of documents.

    Parameters:
    - texts (list): A list of Document objects, where each Document contains a `page_content` attribute.

    Returns:
    - None: Prints the average word and character counts per document.
    """

    # Step 1: Initialize variables to keep track of total words and total characters.
    total_words, total_characters = 0, 0

    # Step 2: Iterate through each document in the `texts` list.
    for doc in texts:
        # Hint: `doc.page_content` contains the text of the document.
        page_content = doc.page_content
        total_characters += len(page_content)
        words = page_content.split() #this also takes care of triming or striping
        total_words += len(words)

    # Step 3: Calculate the average words and characters per document.
    # - Avoid division by zero by checking if the `texts` list is not empty.
    num_documents = len(texts)
    avg_words = total_words / num_documents if num_documents > 0 else 0
    avg_characters = total_characters / num_documents if num_documents > 0 else 0

    # Step 4: Print the calculated averages in a readable format.
    # Example: "Average words per document: 123.45"
    print(f"Average words per document: {avg_words}")
    print(f"Average characters per document: {avg_characters}")


# Execute this cell to test your calculate_word_stats function.
# Create sample Document objects with text content for testing your code above.
def test_calculate_word_stats():
    sample_docs = [
        Document(page_content="This is the first test document."),
        Document(page_content="Here is another example document for testing."),
        Document(page_content="Short text."),
        Document(page_content="This document has more content. It's longer and has more words in it for testing purposes."),
    ]

    # Call the function with the sample documents to calculate word statistics.
    calculate_word_stats(sample_docs)



# 6. Set Up LLM
from langchain_huggingface import HuggingFaceEndpoint

def setup_llm(repo_id="mistralai/Mistral-7B-Instruct-v0.3" , temperature=1.0):
    """
    Set up and return a Hugging Face LLM using the specified model repository ID and generation parameters.

    Parameters:
    - repo_id (str): The repository ID of the Hugging Face model to use (default: "mistralai/Mistral-7B-Instruct-v0.3").
    - temperature (float): The generation temperature to control creativity in outputs (default: 1.0).

    Returns:
    - HuggingFaceEndpoint: A configured LLM object ready for text generation.
    """

    hugging_face_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

    # Step 1: Import the HuggingFaceEndpoint class & Configure the LLM connection
    # - This class allows you to connect to a Hugging Face model hosted on an endpoint.
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=temperature,
        huggingfacehub_api_token=hugging_face_token,
    )

    # Step 3: Return the configured LLM object.
    # - The returned LLM can be used for generating text based on input prompts.
    return llm


# 7. BM25 Retriever

from rank_bm25 import BM25Okapi
from langchain_core.runnables import RunnablePassthrough

class BM25Retriever:
    """
    A class to implement BM25-based document retrieval.

    Attributes:
    - documents (list): A list of Document objects.
    - corpus (list): A list of strings representing the document contents.
    - tokenized_corpus (list): A list of tokenized documents (lists of words).
    - bm25 (BM25Okapi): The BM25 retriever initialized with the tokenized corpus.
    """

    def __init__(self, documents):
        """
        Initialize the BM25 retriever with the given documents.

        Parameters:
        - documents (list): A list of Document objects.
        """
        # Step 1: Store the input documents.
        # Hint: Use the `page_content` attribute of each Document object to extract text.
        self.documents=documents
        texts=[]
        for doc in documents:
            content=doc.page_content
            texts.append(content)
            # word_count= len(content.split())
            # char_count=len(content)
            # total_words+=word_count
            # total_characters+=char_count
        
        num_docs=len(documents)
        # Step 2: Tokenize the corpus.
        # Hint: Use the `.split()` method to tokenize each document into words.
        tokenized_corpus=[i.lower().split(" ") for i in texts]

        # Step 3: Initialize the BM25 retriever with the tokenized corpus.
        self.bm25 = BM25Okapi(tokenized_corpus)
        # pass  # Replace this with your implementation.
        # doc_scores = bm25.get_scores()

    def retrieve(self, query, k=5):
        """
        Retrieve the top `k` most relevant documents for a given query.

        Parameters:
        - query (str): The input query as a string.
        - k (int): The number of top documents to return (default is 5).

        Returns:
        - list: A list of the top `k` relevant documents as strings.
        """
        # Step 1: Tokenize the input query.
        # Hint: Use `.split()` to tokenize the query into words.
        # res=[query.split()]
        tokenized_query = query.lower().split(" ")

        # Step 2: Use the BM25 retriever to score and rank documents.
        # Hint: Use the `bm25.get_top_n()` method to retrieve the top `k` documents.
        
        # Step 3: Return the top `k` relevant documents.
        # pass  # Replace this with your implementation.

        top_documents= self.bm25.get_top_n(tokenized_query,self.documents,k)

        return top_documents


# TESTING
from langchain.schema import Document

def test_b25():
    # Create sample Document objects.
    sample_docs = [
        Document(page_content="Machine learning is a method of data analysis that automates analytical model building."),
        Document(page_content="Deep learning is a subset of machine learning that uses neural networks with three or more layers."),
        Document(page_content="Artificial intelligence encompasses a wide range of technologies, including machine learning and deep learning."),
        Document(page_content="Natural language processing is a field of AI focused on the interaction between computers and human language."),
    ]

    # Initialize the retriever with the sample documents.
    retriever = BM25Retriever(sample_docs)

    # Test the retriever with a query.
    query = "What is machine learning?"
    top_docs = retriever.retrieve(query, k=2)

    # Print the results.
    print("Top Relevant Documents:")
    for idx, doc in enumerate(top_docs, 1):
        print(f"{idx}. {doc}")


# 8. Build Chroma
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def build_chroma(documents: list[Document]) -> Chroma:
    """
    Build a Chroma vector store using Hugging Face embeddings
    and add the documents to it.

    Parameters:
    - documents (list[Document]): A list of Document objects to add to the vector store.

    Returns:
    - Chroma: The Chroma vector store containing the embedded documents.
    """

    # Step 1: Initialize Hugging Face embeddings.
    # - Use a pre-trained embedding model (e.g., "sentence-transformers/all-mpnet-base-v2").
    # - HuggingFaceEmbeddings generates dense vector representations for text.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Step 2: Initialize the Chroma vector store.
    # - Set the collection name for the vector store (e.g., "EngGenAI").
    # - Pass the Hugging Face embeddings as the embedding function.
    vector_store = Chroma(collection_name="EngGenAI", embedding_function=embeddings)

    # Step 3: Add the input documents to the Chroma vector store.
    # - Use the `add_documents` method to embed and store the documents.
    vector_store.add_documents(documents)

    # Step 4: Return the Chroma vector store for later use.
    return vector_store


# Testing 

from langchain.schema import Document

def test_build_chroma():
    # Create sample Document objects.
    sample_docs = [
        Document(page_content="Machine learning is a method of data analysis that automates analytical model building."),
        Document(page_content="Deep learning is a subset of machine learning that uses neural networks with three or more layers."),
        Document(page_content="Artificial intelligence encompasses a wide range of technologies, including machine learning and deep learning."),
        Document(page_content="Natural language processing is a field of AI focused on the interaction between computers and human language."),
    ]

    # Call the function to build the Chroma vector store.
    vector_store = build_chroma(sample_docs)

    # Test retrieval (optional, if supported).
    print("Vector store built successfully!")
    print(vector_store)  # Print the vector store object to verify.


# 9. Ensemble Retriever
from langchain.schema import Document

class EnsembleRetriever:
    """
    Merges results from Chroma similarity search and BM25 lexical search.
    """

    def __init__(self, chroma_store, bm25_retriever):
        """
        Initialize the EnsembleRetriever with Chroma and BM25 retrievers.

        Parameters:
        - chroma_store: The Chroma vector store for semantic retrieval.
        - bm25_retriever: The BM25 retriever for lexical retrieval.
        """
        # Step 1: Store the Chroma vector store and BM25 retriever.
        # Hint: Assign the inputs `chroma_store` and `bm25_retriever` to instance variables.
        self.chroma_store = chroma_store  # Replace with your implementation.
        self.bm25_retriever = bm25_retriever  # Replace with your implementation.

    def get_relevant_documents(self, query: str, k: int = 5):
        """
        Retrieve relevant documents by combining results from Chroma and BM25.

        Parameters:
        - query (str): The input search query.
        - k (int): The number of top unique documents to return (default: 5).

        Returns:
        - list[Document]: A list of unique relevant documents.
        """

        # Step 1: Retrieve top-k documents from Chroma (semantic similarity).
        chroma_docs = self.chroma_store.similarity_search(query,k)  # Replace with your implementation.

        # Step 2: Retrieve top-k documents from BM25 (lexical matching).
        bm25_docs = self.bm25_retriever.retrieve(query,k)  # Replace with your implementation.

        # Step 3: Combine results from both retrievers into a single list.
        combined = chroma_docs + bm25_docs  # Replace with your implementation.

        # Step 4: Deduplicate the combined results.
        # Hint: Use a `set` to track seen content based on document text.
        seen = set()
        unique_docs = []
        for doc in combined:
            # Retrieve content for deduplication (check if `page_content` exists).
            # Hint: Use `doc.page_content` if it's a Document object; otherwise, use `doc` as is.
            if hasattr(doc,'page_content'):
                content = doc.page_content
            else:
                content = doc  # Replace with your implementation.

            # Use the first 60 characters of the document text as a key for deduplication.
            key = content[:60]  # Replace with your implementation.

            if key not in seen:
                # Convert plain strings to Document objects if necessary.
                # Hint: Use `Document(page_content=doc)` for plain text.
                if isinstance(doc, str):
                    doc = Document(page_content=content)  # Replace with your implementation.0
                unique_docs.append(doc)
                seen.add(key)

        # Step 5: Return the top-k unique documents.
        return unique_docs[:k]  # Replace with your implementation.


from langchain.schema import Document



# Sample documents
sample_docs = [
    Document(page_content="Machine learning automates model building using data."),
    Document(page_content="Deep learning is a type of machine learning using neural networks."),
    Document(page_content="AI includes technologies like machine learning and deep learning."),
    Document(page_content="Natural language processing focuses on human-computer language interaction."),
]

# Sample Chroma and BM25 retrievers (mock behavior)
class MockChroma:
    def similarity_search(self, query, k):
        return [Document(page_content="Machine learning automates model building using data.")]

class MockBM25:
    def retrieve(self, query, k):
        return ["Deep learning is a type of machine learning using neural networks."]

def test_mocking():
    # Initialize mock retrievers
    chroma = MockChroma()
    bm25 = MockBM25()

    # Initialize EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(chroma, bm25)

    # Test the retriever with a query
    query = "What is machine learning?"
    results = ensemble_retriever.get_relevant_documents(query, k=3)

    # Print the results
    print("Ensemble Retrieval Results:")
    for idx, doc in enumerate(results, 1):
        print(f"{idx}. {doc.page_content}")

from langchain_core.output_parsers import BaseOutputParser

class StrOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text

# 10. Format Documents
from langchain.prompts import PromptTemplate

def format_docs(docs):
    """
    Format a list of documents into a numbered, readable string.

    Parameters:
    - docs (list[Document]): A list of Document objects to format.

    Returns:
    - str: A string containing the formatted documents or a default message if no documents are provided.
    """

    # Step 1: Check if the list of documents is empty.
    # Hint: If `docs` is empty, return the string "No relevant context found."
    if not docs:
        return "No relevant context found."

    # Step 2: Initialize an empty list to store formatted snippets.
    snippet_list = []

    # Step 3: Iterate over the documents and format each one.
    # - Use `enumerate` to get the index and document.
    # - Extract and clean the `page_content` of the document.
    # - Replace newlines with spaces and remove unnecessary whitespace.
    # - Add a formatted string to the `snippet_list` (e.g., "1. Cleaned content").
    for i, doc in enumerate(docs):
        content = doc.page_content
        cleaned_content = " ".join(content.split())  # Replace newlines and extra spaces with a single space.

        # Add a formatted string to the `snippet_list`.
        snippet_list.append(f"{i + 1}. {cleaned_content}")

    # Step 4: Join the snippets with newline characters and return the result.
    return "\n".join(snippet_list)


# Define the style transfer prompt template
style_prompt = PromptTemplate(
    input_variables=["style", "context", "original_text"],
    template="Given the style '{style}' and context '{context}', transform the text: {original_text} into the desired style."
)

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEndpoint  # Or the specific LLM library you're using

# Example setup for LLM (ensure this is compatible with your LLM)
# def setup_llm():
#     return HuggingFaceEndpoint(
#         repo_id="mistralai/Mistral-7B-Instruct-v0.3",  # Replace with the appropriate model
#         temperature=0.7
#     )

# TEST 

# Sample documents
sample_docs = [
    Document(page_content="Machine learning automates data analysis."),
    Document(page_content="Deep learning uses neural networks to learn patterns."),
    Document(page_content="Artificial intelligence includes various technologies."),
]

def test_format_docs():
    # Test the format_docs function
    formatted_docs = format_docs(sample_docs)
    print("Formatted Documents:\n")
    print(formatted_docs)

    # Test the style_prompt with sample inputs
    style = "poetic"
    context = formatted_docs
    original_text = "Artificial intelligence is transforming the world."

    styled_prompt = style_prompt.format(
        style=style,
        context=context,
        original_text=original_text,
    )

    print("\nGenerated Prompt for Style Transfer:\n")
    print(styled_prompt)

    # Pass the prompt to the LLM
    llm = setup_llm()  # Initialize the LLM
    styled_output = llm(styled_prompt)  # Generate the styled text

    print("\n--- Rewritten (Styled) Text ---")
    print(styled_output)


# 11. RAG chain

from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

def build_rag_chain(llm, chroma_store, bm25_retriever):
    """
    Build a RAG chain using an ensemble retriever with Chroma and BM25,
    followed by formatting the context, applying the prompt, and parsing the output.

    Parameters:
    - llm: The language model for generating styled text.
    - chroma_store: Chroma vector store for semantic retrieval.
    - bm25_retriever: BM25 retriever for lexical retrieval.

    Returns:
    - rag_chain: A function that processes inputs through the RAG pipeline.
    """
    # Step 1: Define the Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(chroma_store, bm25_retriever)  # Replace with your implementation.

    # Step 2: Define a function to retrieve and format context
    def retrieve_and_format_context(query, k=5):
        """
        Retrieve relevant documents and format them into a readable context.

        Parameters:
        - query (str): The input query.
        - k (int): The number of documents to retrieve (default: 5).

        Returns:
        - str: The formatted context string.
        """
        # Step 2.1: Retrieve relevant documents using the ensemble retriever.
        context_docs = ensemble_retriever.get_relevant_documents(query,k)   # Replace with your implementation.

        # Step 2.2: Format the retrieved documents.
        context = format_docs(context_docs)  # Replace with your implementation.

        return context

    # Step 3: Define the RAG chain
    def rag_chain(inputs):
        """
        Process inputs through the RAG pipeline to generate styled output.

        Parameters:
        - inputs (dict): A dictionary containing:
            - "question" (str): The query for retrieving context.
            - "style" (str): The desired writing style.
            - "original_text" (str): The text to be rewritten.

        Returns:
        - str: The final styled output.
        """

        # Step 3.1: Retrieve and format the context using the helper function.
        query = inputs["question"]
        context = retrieve_and_format_context(query,k=5)

        # Step 3.2: Generate the prompt using the `style_prompt`.
        prompt = style_prompt.format(
            style=inputs["style"],
            context=context,
            original_text=inputs["original_text"],
        )

        # Step 3.3: Pass the prompt through the LLM to generate the output.
        llm_response = llm(prompt)

        # Step 3.4: Parse the LLM's output to extract the final styled text.
        parser = RunnablePassthrough()
        result = parser.invoke(llm_response)

        return result

    return rag_chain


# 12. Final response

def main():
    """
    Main script for scraping, building retrievers, setting up the RAG chain,
    and running a neural style transfer demo.
    """

    # Step 1: Scrape content and split into documents
    print("Step 1: Scraping content and splitting into documents...")
    example_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]

    # Step 1A: Initialize an empty list to store all documents
    all_docs = []

    # Step 1B: Iterate through the URLs to fetch and process content
    for url in example_urls:
        print(f"Scraping content from: {url}")

        # Step 1B.1: Fetch and parse the raw text from the URL
        raw_text = None  # Replace with your implementation

        # Step 1B.2: Split the raw text into chunks (documents)
        splits = None  # Replace with your implementation

        # Step 1B.3: Add the chunks to the list of documents
        all_docs.extend(splits)

    print(f"Total number of documents: {len(all_docs)}")

    # Step 2: Build Chroma and BM25 retrievers
    print("Step 2: Building Chroma vector store and BM25 retriever...")

    # Step 2A: Build the Chroma vector store
    chroma_store = None  # Replace with your implementation

    # Step 2B: Build the BM25 retriever
    bm25_retriever = None  # Replace with your implementation

    # Step 3: Build the RAG chain
    print("Step 3: Building RAG chain...")

    # Step 3A: Set up the LLM
    llm = None  # Replace with your implementation

    # Step 3B: Build the RAG chain
    rag_chain = None  # Replace with your implementation

    # Step 4: Neural Style Transfer Demo
    print("\nStep 4: Neural Style Transfer Demo...")

    # Step 4A: Define the user query and target style
    user_text = "Explain machine learning."
    target_style = "as if it were a recipe for cooking"
    inputs = {"question": user_text, "style": target_style, "original_text": user_text}

    print("\n============================================")
    print("        Neural Style Transfer Demo          ")
    print("============================================")
    print(f"Original Text : {user_text}")
    print(f"Desired Style : {target_style}")

    # Step 5: Run the RAG chain
    print("\nStep 5: Running the RAG chain...")

    # Hint: Pass `inputs` through the RAG chain to generate styled output.
    styled_result = None  # Replace with your implementation

    print("\n--- Styled Output ---")
    print(styled_result)



if __name__ == "__main__":
    test_mocking()
