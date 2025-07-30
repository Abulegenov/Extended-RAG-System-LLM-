# HIPAA Query Assistant
A FastAPI application with Gradio UI for querying HIPAA documentation using SQL and semantic search.

HIPAA - the Health Insurance Portability and Accountability Act, is a US federal law that sets national standards for protecting sensitive patient health information from being disclosed without the patient's consent or knowledge. HIPAA aims to ensure the privacy and security of health information, facilitate the transfer and continuation of health insurance, and control healthcare costs. 

This project allows you to ask any question regarding the excerpt of this documentation available in this repository. For that the Extended RAG system was implemented. Why Extended? Because it has a routing agent that will help to distinguish between semantic and structured questions. 

## Tools:
1) LLM - `OPENAI API`
2) Embeddings - `text-embedding-3-small`
3) Vector DB - PostgreSQL + PGVector extension
4) API - Fastapi
5) UI - Gradio

### Preprocessing
Each article of HIPAA documentation was embedded and put into DB, where the metadata was included: article name, article text, part name, list of linked article, etc.  

# Start
1. Create .env file (variables are at the bottom of this doc)
2. Build and run the application using bash command:
```bash
docker compose up --build
```

This will:
- Build the Docker images
- Start PostgreSQL with pgvector extension
- Initialize the database with HIPAA documentation
- Start the FastAPI application
- Start Nginx reverse proxy

You can set up ngrok (optional, for public access)
In a new terminal, run:
```bash
ngrok http 80
```

Note: Make sure you're authenticated with ngrok. If not, run:
```bash
ngrok authtoken YOUR_AUTH_TOKEN
```

# Application Endpoints
Once running, you can access:

If ngrok is used, go to ngrok provided URL

## Gradio UI: http://localhost/gradio

Interactive chat interface for querying HIPAA documentation
Auto-routes queries to SQL or semantic search


## Health Check: http://localhost/health

Verify database connection status
Returns JSON with connection status


## API Documentation: http://localhost/docs

Interactive FastAPI Swagger documentation
Test API endpoints directly



# How It Works
## Database Structure
The HIPAA documentation is processed and stored as follows:

Text Chunking: Documentation is divided into chunks, where each chunk represents:

Full text of each article
Organized by part → subpart → article hierarchy


Vector Embeddings: 

Each chunk_text has Vector embedding (chunk_embedding) for semantic search




## Query Processing
When you submit a query, the system:

Query Routing: An LLM analyzes your query and routes it to:

SQL Agent: For structured queries (lists, specific articles, document structure)
Vector Search: For semantic queries (concepts, explanations, related content)


## SQL Agent Path:

LLM generates appropriate SQL query
Query executed against PostgreSQL database
Returns exact matches and structured data


## Vector Search Path:

Query converted to embedding
Semantic similarity search in vector space
Top 5 most relevant chunks retrieved
LLM synthesizes answer from retrieved context

! Ensure ports 80, 5432, and 8000 are available


## .ENV file
.env file should include following information:

OPENAI_API_KEY=your_key
DB_USER=ablaybulegenov
DB_PASSWORD=
DB_NAME=postgres
DB_HOST=localhost
DB_PORT=5432

## .dockerignore file
__pycache__/
env/
.DS_Store
