# RAG Course Project

This repository is a small Python CLI project used for educational purposes.
It is intentionally simple, readable, and easy to extend.

## Project rules

- Keep modules small and focused on one responsibility.
- Put CLI parsing in the CLI layer, config loading in the config layer, and command behavior in command modules.
- Prefer descriptive names over short names when the intent is not obvious.
- Keep examples and defaults safe for local development.
- Do not store secrets in the repository.
- Use `.env` for local secret/config overrides and keep `.env.example` updated when settings change.
- Keep files short when possible and split new behavior into new modules instead of growing one large file.
- Document non-obvious decisions directly in code or in this README.

## Structure

- `rag_course/cli.py` contains command-line argument parsing and dispatch.
- `rag_course/config.py` loads environment variables and validates configuration.
- `rag_course/embeddings.py` contains reusable embedding helpers and similarity math.
- `rag_course/chunker.py` contains the reusable plain-text chunker.
- `rag_course/sources.py` reads local files and HTTP sources.
- `rag_course/commands/` contains individual command implementations.
- `rag_course/__main__.py` provides the module entry point.
- `docker-compose.yml` runs local infrastructure needed for vector storage.

## Environment setup

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and adjust values as needed.
4. Run the app with `python -m rag_course --help`.
5. Start Qdrant locally with `docker compose up -d`.

## Local vector store

- `docker-compose.yml` starts Qdrant on ports `6333` and `6334`.
- Qdrant persists data in the local `./qdrant_storage` directory.
- Keep that directory out of version control; it is ignored by `.gitignore`.

## CLI commands

- `python -m rag_course hello [name]` prints a greeting.
- `python -m rag_course status` shows the loaded configuration.
- `python -m rag_course embed "some text"` prints the embedding vector.
- `python -m rag_course embed-chunks INPUT OUTPUT` reads chunk YAML and writes a new YAML file with embeddings.
- `python -m rag_course import-embeddings INPUT` imports an embedded YAML file into Qdrant.
- `python -m rag_course query [term ...]` searches Qdrant with a prompt or asks for one interactively.
- `python -m rag_course chat` starts an interactive LLM chat loop and writes audit logs to `auditlog/`.
- `python -m rag_course similarity` prompts for two lines and prints cosine similarity.
- `python -m rag_course chunk INPUT OUTPUT` reads a local file path or an `http(s)` URL and writes chunk metadata to YAML.

## Chunking rules

- Chunking is sentence-based and respects paragraph boundaries.
- Headings in the form `1. Jakobs Haus.(1)` are treated as metadata, not chunk text.
- Chunks never split a sentence in half.
- Chunks with `min_words_per_chunk` or fewer words are dropped. The default is `2`.
- Estimated token count uses `1 word = 1.3 tokens`.
- Default chunking uses `250` max tokens, `3` target sentences, and `1` sentence overlap.
- Dialogues and quoted passages stay together as long as the token budget allows.
- Intended text blocks are treated as summaries.

## Embedding configuration

- `OPENAI_BASE_URL` sets an OpenAI-compatible endpoint. Leave it empty to use the default OpenAI API URL.
- `OPENAI_API_KEY` sets the authentication token used by the client.
- `EMBEDDING_MODEL` sets the embedding model. The default is `text-embedding-3-small`.
- `CHAT_MODEL` sets the chat model used by the interactive chat loop. The default is `gpt-4.1-mini`.
- `RAG_SCORE_THRESHOLD` sets the minimum similarity score used when retrieving context for chat. The default is `0.5`.
- `RAG_TOP_K` sets how many retrieval candidates are requested before budgeting. The default is `5`.
- `RAG_CONTEXT_TOKEN_BUDGET_TOTAL` sets the total token budget for RAG context injection. The default is `800`.
- `RAG_CONTEXT_TOKEN_BUDGET_PER_ENTRY` sets the token budget for each retrieved entry. The default is `200`.
- `QDRANT_URL` sets the Qdrant endpoint. The default is `http://localhost:9333`, which matches the exposed Docker Compose port.
- `QDRANT_COLLECTION_NAME` sets the target collection name. The default is `rag_chunks`.
- `QDRANT_VECTOR_SIZE` sets the collection vector length. The default is `1536`.
- `QDRANT_API_KEY` sets an optional Qdrant API key.
- `--min-words` on the chunk command controls the minimum kept chunk length. The default is `2`.
- `--batch-size` on the `embed-chunks` command controls how many chunks are sent per embedding request. The default is `64`.

## Maintenance notes

- Update `requirements.txt` when adding or changing dependencies.
- Update `.env.example` whenever a new environment variable is introduced.
- Keep command output user-friendly and deterministic for teaching purposes.
- Add tests when behavior becomes more than a trivial example.
