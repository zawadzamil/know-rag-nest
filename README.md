# RAG System with NestJS, Milvus, and Gemini AI

A Retriever-Augmented Generation (RAG) system built with NestJS that uses Milvus vector database for semantic search and Google's Gemini AI for response generation.

## Features

- **Vector Search**: Uses Milvus DB for efficient similarity search on text embeddings
- **Text Chunking**: Automatically chunks static text files for optimal retrieval
- **Embeddings**: Generates embeddings using OpenAI's text-embedding-ada-002 model
- **AI Integration**: Integrates with Google Gemini AI for intelligent response generation
- **RESTful API**: Provides clean REST endpoints for querying and management

## Architecture

```
User Query → Embedding → Milvus Search → Context Retrieval → Gemini AI → Response
```

## Prerequisites

- Node.js (v18 or higher)
- Docker and Docker Compose
- OpenAI API Key (for embeddings)
- Google Gemini API Key (for AI responses)

## Setup Instructions

### 1. Install Dependencies

```bash
npm install
```

### 2. Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# OpenAI Configuration (for embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent

# Collection Configuration
COLLECTION_NAME=context_chunks
VECTOR_DIMENSION=1536
```

### 3. Start Milvus with Docker

```bash
# Start Milvus services
docker-compose up -d

# Check if services are running
docker-compose ps
```

Wait for all services to be healthy before proceeding.

### 4. Prepare Your Text Data

Edit the `about_me.txt` file with your own content. This file will be automatically chunked and embedded when the application starts.

### 5. Start the Application

```bash
# Development mode
npm run start:dev

# Production mode
npm run start:prod
```

## API Endpoints

### Query Endpoints

#### POST /query
Send a question to the RAG system.

**Request Body:**
```json
{
  "question": "Tell me about yourself"
}
```

**Response:**
```json
{
  "answer": "Generated response from Gemini AI",
  "relevantChunks": ["chunk1", "chunk2", "chunk3"],
  "confidence": 85.5
}
```

#### GET /query?q=your_question
Alternative GET endpoint for simple queries.

**Example:**
```
GET /query?q=What is your experience with Node.js?
```

### Management Endpoints

#### POST /reprocess
Reprocess the text file and update embeddings.

**Response:**
```json
{
  "message": "Text file reprocessed and embeddings updated successfully"
}
```

#### GET /health
Check application health status.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-08-11T10:30:00.000Z"
}
```

## Services Overview

### MilvusService
- Manages Milvus database connections
- Creates and manages vector collections
- Handles vector search operations
- Inserts and retrieves embeddings

### EmbeddingsService
- Generates embeddings using OpenAI's API
- Handles batch embedding generation
- Manages API rate limiting

### TextContextService
- Loads and processes text files
- Chunks text into optimal sizes
- Coordinates embedding generation and storage
- Handles text reprocessing

### AiQueryService
- Processes user queries end-to-end
- Retrieves relevant context from Milvus
- Integrates with Gemini AI for response generation
- Calculates confidence scores

## Configuration Options

### Text Chunking
- **Chunk Size**: Default 500 characters
- **Overlap**: Default 50 characters for context continuity

### Vector Search
- **Top K**: Default 5 most relevant chunks
- **Similarity Metric**: Inner Product (IP)
- **Index Type**: IVF_FLAT for balance of speed and accuracy

### AI Generation
- **Temperature**: 0.7 for balanced creativity
- **Max Tokens**: 1024 for comprehensive responses
- **Top K/P**: 40/0.95 for quality control

## Docker Services

The `docker-compose.yml` includes:

- **Milvus**: Vector database (port 19530)
- **etcd**: Metadata storage for Milvus
- **MinIO**: Object storage for Milvus

## Troubleshooting

### Common Issues

1. **Milvus Connection Failed**
   - Ensure Docker services are running: `docker-compose ps`
   - Check ports are available: `lsof -i :19530`

2. **OpenAI API Errors**
   - Verify API key is valid and has sufficient credits
   - Check rate limits if processing large texts

3. **Gemini API Errors**
   - Ensure Gemini API key is correctly configured
   - Verify API endpoint URL is correct

4. **Empty Search Results**
   - Check if text file was processed: Look for logs during startup
   - Verify embeddings were generated and stored

### Logs

Enable detailed logging by setting the log level:
```typescript
// In main.ts
app.useLogger(['log', 'error', 'warn', 'debug', 'verbose']);
```

## Development

### Adding New Text Sources
1. Place text files in the project root
2. Update `TextContextService` to process multiple files
3. Restart the application to reprocess

### Customizing Chunking Strategy
Modify the `chunkText` method in `TextContextService` to implement different chunking strategies:
- Sentence-based chunking
- Paragraph-based chunking
- Semantic chunking

### Extending AI Integration
The system can be extended to support other AI providers by:
1. Creating new service classes
2. Implementing the same interface
3. Updating the dependency injection

## Performance Optimization

- **Batch Processing**: Process multiple queries in parallel
- **Caching**: Implement Redis for frequently accessed embeddings
- **Index Tuning**: Adjust Milvus index parameters for your use case
- **Connection Pooling**: Use connection pools for database operations

## License

This project is licensed under the UNLICENSED license.
