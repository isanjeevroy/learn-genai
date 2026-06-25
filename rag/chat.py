from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

# Open AI Client
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

# Take User Input
user_query = input("Ask something: ")

# Relevant chunks from the vector DB
search_results = vector_db.similarity_search(query=user_query)

# Context
context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile location: {result.metadata['source']}"
for result in search_results])

# System Prompt
SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who answers user query based on the availabel context retrieved from a PDF file along with page_contents and page nubmer.

    You should only answer the user based on the following context and navigate the user to open the right page number to know more.

    Context: {context}
"""

#Response
response = client.chat.complete.create(
    model="gpt-5",
    messages=[
        { "role": "system", "content": SYSTEM_PROMPT },
        { "role": "user", "content": user_query }
    ]
)

# Printing the response data
print(f"Response: {response.choices[0].message.content}")