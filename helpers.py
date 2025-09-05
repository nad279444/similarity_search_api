from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import UpstashVectorStore
from dotenv import load_dotenv
import traceback
import os
import requests




load_dotenv()

UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")

print(f"URL configured: {UPSTASH_VECTOR_REST_URL is not None}")
print(f"Token configured: {UPSTASH_VECTOR_REST_TOKEN is not None}")
if UPSTASH_VECTOR_REST_URL:
    print(f"URL: {UPSTASH_VECTOR_REST_URL[:50]}...")


# Test Upstash connection first
def test_upstash_connection():
    """Test if Upstash API is accessible and returns valid JSON"""
    try:
        headers = {
            'Authorization': f'Bearer {UPSTASH_VECTOR_REST_TOKEN}',
            'Content-Type': 'application/json'
        }
        
        # Test with info endpoint
        response = requests.get(f"{UPSTASH_VECTOR_REST_URL}/info", headers=headers, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Raw Response: {response.text[:500]}...")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ Upstash connection successful!")
                print(f"Index info: {data}")
                return True, data
            except json.JSONDecodeError as e:
                print(f"❌ JSONDecodeError: {e}")
                print(f"Response is not valid JSON: {response.text}")
                return False, None
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return False, None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False, None

# Test the connection
connection_ok, index_info = test_upstash_connection()



# Only proceed if connection is working
if not connection_ok:
    print("\n🚨 UPSTASH CONNECTION FAILED!")
    print("\nPossible solutions:")
    print("1. Check your UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN in .env")
    print("2. Ensure your Upstash Vector index exists and is active")
    print("3. Create a new index at https://console.upstash.com with:")
    print("   - Dimension: 768 (for nomic-embed-text)")
    print("   - Embedding model enabled")
    print("4. Check if your index region is accessible")
    raise Exception("Cannot proceed without valid Upstash connection")
else:
    print("✅ Upstash connection verified, proceeding...")



# Initialize embeddings
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Test embedding generation
    test_text = "This is a test embedding"
    test_embedding = embeddings.embed_query(test_text)
    
    print(f"✅ Embeddings working!")
    print(f"Embedding dimension: {len(test_embedding)}")
    print(f"Sample values: {test_embedding[:5]}")
    
    # Check if dimensions match index
    if index_info and 'dimension' in index_info:
        index_dim = index_info['dimension']
        if len(test_embedding) != index_dim:
            print(f"⚠️  DIMENSION MISMATCH!")
            print(f"Embedding dimension: {len(test_embedding)}")
            print(f"Index dimension: {index_dim}")
            print(f"You need to create a new index with dimension {len(test_embedding)}")
        else:
            print(f"✅ Dimensions match: {len(test_embedding)}")
    
except Exception as e:
    print(f"❌ Embedding error: {e}")
    traceback.print_exc()
    raise


# Initialize Upstash Vector Store with error handling
try:
    store = UpstashVectorStore(
        embedding=embeddings,
        index_url=UPSTASH_VECTOR_REST_URL,
        index_token=UPSTASH_VECTOR_REST_TOKEN,
    )
    print("✅ UpstashVectorStore initialized successfully")
    
except Exception as e:
    print(f"❌ Failed to initialize UpstashVectorStore: {e}")
    traceback.print_exc()
    raise



llm = ChatOllama(model="llama3", temperature=0)


# prompt
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([("human", message)])

retriever = store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 2}
)

parser = StrOutputParser()


def get_chain():
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | parser