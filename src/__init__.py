"""Enterprise Knowledge Assistant — a small, production-minded RAG system."""

import warnings

# Silence two harmless, cosmetic warnings so the demo terminal stays clean.
# These do NOT affect correctness — answers, citations, and confidence are
# unaffected. Set here (package import) so it applies before LangChain loads.
#   1. LangChain's internal Pydantic-v1 compatibility shim on Python 3.14+.
#   2. langchain-openai's structured-output serializer noise ("parsed" field).
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

__version__ = "1.0.0"
