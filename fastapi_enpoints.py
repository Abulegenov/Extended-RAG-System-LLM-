import os
import asyncio
from typing import Dict, Optional, List, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import gradio as gr
import uvicorn
import logging

import psycopg2
from psycopg2.extras import RealDictCursor

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase


# Load environment variables
load_dotenv()

# Configuration
class Config:
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    @property
    def database_url(self):
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def langchain_database_url(self):
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

config = Config()

# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query")

class QueryResponse(BaseModel):
    route: str = Field(..., description="Route decision: SQL or VECTOR")
    answer: str = Field(..., description="Answer to the query")

# Global resources
class Resources:
    def __init__(self):
        self.llm: Optional[ChatOpenAI] = None
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.sql_agent = None
        self.db = None

resources = Resources()

# Create FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing LLM and embeddings...")
    resources.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY
    )
    resources.embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=config.OPENAI_API_KEY
    )
    
    print("Initializing SQL agent...")
    resources.db = SQLDatabase.from_uri(config.langchain_database_url)
    
    resources.sql_agent = create_sql_agent(
        llm=resources.llm,
        toolkit=SQLDatabaseToolkit(db=resources.db, llm=resources.llm),
        verbose=True,
        top_k=None,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        agent_executor_kwargs = {"return_intermediate_steps": True}
    )
    
    yield
    
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="HIPAA Query API",
    description="API for querying HIPAA documentation using SQL and semantic search",
    version="1.0.0",
    lifespan=lifespan
)

def get_db_connection():
    """Get a new database connection"""
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD
    )
def execute_last_sql_query(intermediate_steps):
    #Get the last AgentAction with tool == 'sql_db_query' and execute this query in db
    last_sql = None
    for action, observation in reversed(intermediate_steps):
        if action.tool == "sql_db_query":
            last_sql = action.tool_input
            break

    if not last_sql:
        raise ValueError("No sql_db_query found in intermediate_steps.")

    #Execute the query
    # Connect to database
    conn = get_db_connection()
    rows=[]
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(last_sql) 
            rows = cursor.fetchall()
            
    finally:
        conn.close()
    if rows:
        return rows


def semantic_search(query: str) -> str:
    """Perform semantic search"""
    # Get embedding
    query_vec = resources.embeddings.embed_query(query)
    
    # Connect to database
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT chunk_text, article_number, article_title, part_number, part_name
                FROM document_chunks
                ORDER BY chunk_embedding <#> %s::vector
                LIMIT 5
            """, (query_vec,))
            rows = cursor.fetchall()
    finally:
        conn.close()
    
    if not rows:
        return "No relevant documents found in the database."
    
    # Build context
    context = "\n\n".join(
        f"{row['article_number']} {row['article_title']} from part {row['part_number']}:\n{row['chunk_text']}"
        for row in rows
    )
    
    prompt = f"""You are a legal assistant with strict and robust answers. Use following excerpts in Context from HIPAA documentation to answer the user Question
    1 Excerpt is provided in following format: "Article number Article title Part number Part Name: Article text". One context may contain several excerpts.
    When answering to user question, include the reference to the article in following format [Taken from Article Number: "Direct Citation from Article Text"]
    Context: {context}
    
    Question: {query}
    """
    
    response = resources.llm.invoke(prompt)
    return response.content

def get_route_decision(query: str) -> str:
    """Determine router that decides the nature of a query"""
    router_prompt = f"""You are an intelligent assistant that routes user queries to one of two systems:
    
    - Use **SQL** if the question refers to the document's structure, such as specific articles, parts, or uses numbering or titles.
    - Use **VECTOR** if the question is abstract, open-ended, or requires understanding of the meaning (semantic search).
    
    Respond with only one word: either SQL or VECTOR.
    
    Question: {query}
    Route:"""
    
    response = resources.llm.invoke(router_prompt)
    return "SQL" if "SQL" in response.content else "VECTOR"

# API Endpoints
@app.get("/")
async def root():
    """Redirect to Gradio UI."""
    return HTMLResponse("""
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/gradio" />
        </head>
        <body>
            <p>Redirecting to <a href="/gradio">Gradio UI</a>...</p>
        </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/api/query", response_model=QueryResponse)
async def query_hipaa(request: QueryRequest) -> QueryResponse:
    """Query HIPAA documentation using either SQL or semantic search."""
    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Get route decision
        route = await loop.run_in_executor(None, get_route_decision, request.query)
        
        if route == "SQL":
            try:
                print(f"Running SQL agent for query: {request.query}")
                answer = await loop.run_in_executor(None, resources.sql_agent.invoke, request.query)
                answer = answer['output']
                # answer = await loop.run_in_executor(None, execute_last_sql_query, answer['intermediate_steps'])
                # print(answer["intermediate_steps"])
                # answer = answer.output
            except:
                route == "VECTOR"
                print(f"Running semantic search for query: {request.query}")
                answer = await loop.run_in_executor(None, semantic_search, request.query)
        
        else:
            print(f"Running semantic search for query: {request.query}")
            answer = await loop.run_in_executor(None, semantic_search, request.query)
        
        return QueryResponse(route=route, answer=answer)
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Gradio interface functions
def query_chat(message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    """Handle chat queries for Gradio."""
    if not message.strip():
        return history, ""
    
    try:
        # Get router answer
        route = get_route_decision(message)
        # print(route)
        # Execute query
        if route == "SQL":
            try: #using try block because agent might fail to parse outputs such as "I don't know"
                print(f"Running SQL agent for query: {message}")
                answer = resources.sql_agent.invoke(message)
                if any(phrase in message.lower() for phrase in ["full text", "complete", "entire", "all content", "whole section", "text", "list", 
                "how many", "don't summarize", "without summary", "no summary"]):
                    answer = execute_last_sql_query(answer["intermediate_steps"])
                else:
                    answer = answer['output']
                # print(answer["intermediate_steps"])
                # answer = answer.output
            except: #routing to semantic search in case if sql agent is not able to the job
                route == "VECTOR"
                print(f"Running semantic search for query: {message}")
                answer = semantic_search(message)
        else:
            print(f"Running semantic search for query: {message}")
            answer = semantic_search(message)
        
        bot_response = f"**[Route: {route}]**\n\n{answer}"
        # print(bot_response)
    except Exception as e:
        bot_response = f"Error: {str(e)}"
        print(f"Chat error: {str(e)}")
    
    # history.append({"role": "user", "content": message})
    # history.append({"role": "assistant", "content": bot_response})
    history.append((message, bot_response))
    return history, ""



# Mount Gradio
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting combined FastAPI + Gradio application...")
    logging.info("Access the application at:")
    logging.info("  - Gradio UI: http://localhost/gradio")
    logging.info("  - API Docs: http://localhost/docs")
    logging.info("  - Health: http://localhost/health")
    logging.info("Or use ngrok URL if enabled.")
    
    uvicorn.run(
        app,
        host="0.0.0.0",  
        port=8000,
        reload=False  
    )