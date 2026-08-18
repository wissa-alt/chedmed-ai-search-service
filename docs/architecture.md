# Architecture ChedMed AI Search

The public application boundary follows the compact, modality-oriented layout
of the reference implementation. The robust domain engines remain split where
they have independent algorithms and tests.

```text
app/
├── config.py
├── models/
│   ├── audio.py
│   └── search.py
└── services/
    ├── audio_service.py
    ├── catalog_service.py
    ├── embedding_service.py
    ├── faiss_service.py
    ├── image_service.py
    └── text_service.py
```

## Request flows

```text
POST /search
  -> AssistantService
  -> SearchService
  -> query understanding / semantic query / FAISS
  -> structured filters / simple ranking / relevance gate

POST /voice-search
  -> AudioTranscriptionService
  -> AssistantService
  -> the same SearchService flow

POST /image-search
  -> ImageSearchService
  -> literal visible-product description
  -> AssistantService
  -> the same SearchService flow
```

All three routes serialize products through the same HTTP adapter. Audio and
image providers do not select catalogue products. `config.py` remains the one
settings implementation; `app.config` is its stable application import.

## Internal engines

The modules under `search/`, `embeddings/`, `vector_store/`, `database/`, and
`llm/` are implementation engines rather than alternative application APIs.
They remain separate when they have a distinct contract, failure mode, model
lifecycle, or focused unit tests. In particular, FAISS retrieval, structured
filtering, ranking, and relevance gating are not folded into modality services.
