from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent / ".env")

client = OpenAI()

# Embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

#Vector DB
vector_db = QdrantVectorStore.from_existing_collection(
    embedding= embedding_model,
    url="http://localhost:6333",
    collection_name= "learning_rag"
)

def process_query(query:str):
    print("Searching Chunks", query)
    search_results = vector_db.similarity_search(query=query)

    context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile location: {result.metadata['source']}"
    for result in search_results])

    # System Prompt
    SYSTEM_PROMPT = f"""
        You are a helpful AI Assistant who answers user query based on the availabel context retrieved from a PDF file along with page_contents and page nubmer.

        You should only answer the user based on the following context and navigate the user to open the right page number to know more.

        Context: {context}
    """
    #Response
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": query }
        ]
    )

    print(f"Response: {response.choices[0].message.content}")
    return response.choices[0].message.content