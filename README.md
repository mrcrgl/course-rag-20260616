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
- `rag_course/commands/` contains individual command implementations.
- `rag_course/__main__.py` provides the module entry point.

## Environment setup

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and adjust values as needed.
4. Run the app with `python -m rag_course --help`.

## CLI commands

- `python -m rag_course hello [name]` prints a greeting.
- `python -m rag_course status` shows the loaded configuration.
- `python -m rag_course embed "some text"` prints the embedding vector.
- `python -m rag_course similarity` prompts for two lines and prints cosine similarity.

## Embedding configuration

- `OPENAI_BASE_URL` sets an OpenAI-compatible endpoint. Leave it empty to use the default OpenAI API URL.
- `OPENAI_API_KEY` sets the authentication token used by the client.
- `EMBEDDING_MODEL` sets the embedding model. The default is `text-embedding-3-small`.

## Maintenance notes

- Update `requirements.txt` when adding or changing dependencies.
- Update `.env.example` whenever a new environment variable is introduced.
- Keep command output user-friendly and deterministic for teaching purposes.
- Add tests when behavior becomes more than a trivial example.
