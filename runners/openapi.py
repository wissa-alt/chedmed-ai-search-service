"""Compact OpenAPI specification and Swagger UI for the Flask API."""

from __future__ import annotations

from typing import Any

from flask import Flask, Response, jsonify, redirect


def register_openapi(application: Flask) -> None:
    """Expose the OpenAPI document and its interactive Swagger UI."""

    @application.get("/openapi.json")
    def openapi_document() -> Response:
        return jsonify(OPENAPI_SPEC)

    @application.get("/docs")
    def swagger_docs() -> Response:
        return Response(_SWAGGER_HTML, mimetype="text/html")

    @application.get("/swagger")
    def swagger_alias() -> Response:
        return redirect("/docs", code=302)


def _ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/components/schemas/{name}"}


def _response(schema: dict[str, Any], description: str = "Succès") -> dict[str, Any]:
    return {
        "description": description,
        "content": {"application/json": {"schema": schema}},
    }


def _error_responses(*codes: int) -> dict[str, Any]:
    descriptions = {
        400: "Requête invalide",
        500: "Erreur interne",
        502: "Provider externe indisponible",
        503: "Fonctionnalité indisponible",
    }
    return {str(code): _response(_ref("ErrorResponse"), descriptions[code]) for code in codes}


_TOP_K = {
    "type": "integer",
    "minimum": 1,
    "default": 5,
    "description": "Nombre maximal de résultats.",
}

_SEARCH_REQUEST = {
    "type": "object",
    "additionalProperties": False,
    "required": ["query"],
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "topK": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum retourné; absent signifie aucun plafond métier fixe.",
        },
        "includeAll": {
            "type": "boolean",
            "default": False,
            "description": "Retourne tout le catalogue indexé classé, tous statuts inclus.",
        },
    },
}

_TRANSCRIPTION_RESPONSE = {
    "type": "object",
    "required": ["text", "provider", "usedFallback", "resolutionReason", "qualityScore", "latencyMs"],
    "properties": {
        "text": {"type": "string"},
        "provider": {"type": "string", "enum": ["whisper", "gemini"]},
        "usedFallback": {"type": "boolean"},
        "resolutionReason": {"type": "string"},
        "qualityScore": {"type": "number"},
        "latencyMs": {"type": "integer", "minimum": 0},
    },
}

_VOICE_RESPONSE = {
    "allOf": [
        _ref("SearchResponse"),
        {
            "type": "object",
            "required": [
                "transcription", "provider", "whisperText", "geminiText",
                "geminiTriggered", "voiceSearchTotalLatencyMs",
            ],
            "properties": {
                "transcription": {"type": "string"},
                "whisperText": {"type": ["string", "null"]},
                "geminiText": {"type": ["string", "null"]},
                "provider": {"type": "string", "enum": ["whisper", "gemini"]},
                "usedFallback": {"type": "boolean"},
                "resolutionReason": {"type": "string"},
                "qualityScore": {"type": "number"},
                "geminiTriggered": {"type": "boolean"},
                "geminiTriggerReason": {"type": ["string", "null"]},
                "whisperLatencyMs": {"type": "integer", "minimum": 0},
                "whisperSearchLatencyMs": {"type": "integer", "minimum": 0},
                "geminiLatencyMs": {"type": "integer", "minimum": 0},
                "geminiSearchLatencyMs": {"type": "integer", "minimum": 0},
                "transcriptionLatencyMs": {"type": "integer", "minimum": 0},
                "voiceSearchTotalLatencyMs": {"type": "integer", "minimum": 0},
            },
        },
    ]
}

_IMAGE_RESPONSE = {
    "allOf": [
        _ref("SearchResponse"),
        {
            "type": "object",
            "required": ["imageDescription", "provider", "visionLatencyMs", "imageSearchTotalLatencyMs"],
            "properties": {
                "imageDescription": {"type": "string"},
                "provider": {"type": "string", "example": "gemini"},
                "visionLatencyMs": {"type": "integer", "minimum": 0},
                "imageSearchTotalLatencyMs": {"type": "integer", "minimum": 0},
            },
        },
    ]
}

_SELLER_PRODUCT_INPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "sellerPrice", "currency"],
    "properties": {
        "productId": {"type": ["string", "null"], "description": "ID à exclure des comparables."},
        "title": {"type": "string", "minLength": 1},
        "description": {"type": "string", "default": ""},
        "category": {"type": ["string", "null"]},
        "brand": {"type": ["string", "null"]},
        "color": {"type": ["string", "null"]},
        "condition": {"type": ["string", "null"]},
        "sellerPrice": {"type": "number", "exclusiveMinimum": 0},
        "currency": {"type": "string", "pattern": "^[A-Za-z]{3}$", "example": "MAD"},
    },
}

_SELLER_ASSISTANT_RESPONSE = {
    "type": "object",
    "properties": {
        "suggestedDescription": {"type": "string"},
        "descriptionGenerated": {"type": "boolean"},
        "descriptionQuality": {"type": "string", "enum": ["limited", "good"]},
        "sellerPrice": {"type": "number"},
        "currency": {"type": "string"},
        "estimatedPrice": {"type": ["number", "null"]},
        "recommendedRange": {
            "type": ["object", "null"],
            "properties": {"min": {"type": "number"}, "max": {"type": "number"}},
        },
        "priceAssessment": {
            "type": "string",
            "enum": ["too_low", "low", "reasonable", "high", "too_high", "insufficient_data"],
        },
        "message": {"type": "string"},
        "confidence": {"type": "string", "enum": ["none", "very_low", "low", "medium", "good"]},
        "comparablesCount": {"type": "integer", "minimum": 0},
        "comparables": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"}, "title": {"type": "string"},
                    "price": {"type": "number"}, "currency": {"type": "string"},
                    "condition": {"type": ["string", "null"]},
                    "matchType": {"type": "string", "enum": ["relevant", "similar", "broad_similar"]},
                    "similarityScore": {"type": "number"},
                },
            },
        },
    },
}

_COMPARABLE_PRICE = {
    "type": "object",
    "properties": {
        "name": {"type": "string"}, "price": {"type": "number"},
        "similarity_score": {"type": "number"},
        "match_type": {"type": "string", "enum": ["exact", "relevant", "similar", "broad_similar"]},
        "status": {"type": "string"}, "isSold": {"type": "boolean"},
    },
}

_PRICE_ESTIMATE_RESPONSE = {
    "type": "object",
    "properties": {
        "suggested_price": {"type": ["number", "null"]},
        "mean_price": {"type": ["number", "null"]},
        "price_range": {
            "type": ["object", "null"],
            "properties": {"min": {"type": "number"}, "max": {"type": "number"}},
        },
        "comparable_products": {"type": "array", "items": _COMPARABLE_PRICE},
        "based_on_n_products": {"type": "integer"},
        "candidate_products_count": {"type": "integer"},
        "total_catalog_products": {"type": "integer"},
    },
}

_PRICE_CHECK_RESPONSE = {
    "type": "object",
    "properties": {
        "alert": {"type": "string", "enum": ["too_low", "low", "fair", "high", "too_high", "insufficient_data"]},
        "message": {"type": "string"},
        "seller_price": {"type": "number"},
        "market_stats": {
            "type": ["object", "null"],
            "properties": {
                "mean": {"type": "number"}, "median": {"type": "number"},
                "p25": {"type": "number"}, "p75": {"type": "number"},
                "min": {"type": "number"}, "max": {"type": "number"},
            },
        },
        "based_on_n_products": {"type": "integer"},
        "candidate_products_count": {"type": "integer"},
        "total_catalog_products": {"type": "integer"},
        "comparable_products": {"type": "array", "items": _COMPARABLE_PRICE},
    },
}

_PRODUCT_PROPERTIES = {
    "id": {"type": "string", "example": "123"},
    "title": {"type": "string", "example": "Laptop Dell XPS"},
    "description": {"type": "string"},
    "category": {"type": ["string", "null"], "example": "Électronique"},
    "brand": {"type": ["string", "null"], "example": "Dell"},
    "color": {"type": ["string", "null"], "example": "Noir"},
    "condition": {"type": ["string", "null"]},
    "price": {"type": ["string", "number"], "example": "400.0"},
    "currency": {"type": ["string", "null"], "example": "MAD"},
    "city": {"type": ["string", "null"], "example": "Casablanca"},
    "imageUrls": {"type": "array", "items": {"type": "string", "format": "uri"}},
    "status": {"type": ["string", "null"], "example": "ACCEPTED"},
    "isSold": {"type": "boolean", "example": False},
    "updatedAt": {"type": ["string", "null"], "format": "date-time"},
}

OPENAPI_SPEC: dict[str, Any] = {
    "openapi": "3.1.0",
    "info": {
        "title": "ChedMed AI Search API",
        "version": "1.0.0",
        "description": "Recherche marketplace multilingue via un pipeline SearchService commun.",
    },
    "servers": [{"url": "/", "description": "Serveur Flask courant"}],
    "tags": [
        {"name": "System"}, {"name": "Search"}, {"name": "Audio"},
        {"name": "Image"}, {"name": "Seller assistant"}, {"name": "Seller"},
    ],
    "paths": {
        "/health": {"get": {
            "tags": ["System"], "summary": "Vérifier que l’API est disponible", "operationId": "health",
            "responses": {"200": _response({
                "type": "object", "required": ["status"],
                "properties": {"status": {"type": "string", "example": "ok"}},
            })},
        }},
        "/search": {"post": {
            "tags": ["Search"], "summary": "Rechercher des produits depuis du texte", "operationId": "searchProducts",
            "requestBody": {"required": True, "content": {"application/json": {
                "schema": _SEARCH_REQUEST, "example": {"query": "Dell laptop", "topK": 5},
            }}},
            "responses": {"200": _response(_ref("SearchResponse")), **_error_responses(400, 500)},
        }},
        "/transcriptions": {"post": {
            "tags": ["Audio"], "summary": "Transcrire un fichier audio", "operationId": "transcribeAudio",
            "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
                "type": "object", "required": ["audio"],
                "properties": {"audio": {"type": "string", "format": "binary"}},
            }}}},
            "responses": {"200": _response(_TRANSCRIPTION_RESPONSE), **_error_responses(400, 502, 500)},
        }},
        "/voice-search": {"post": {
            "tags": ["Audio"], "summary": "Transcrire puis rechercher avec le pipeline texte", "operationId": "voiceSearch",
            "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
                "type": "object", "required": ["audio"],
                "properties": {"audio": {"type": "string", "format": "binary"}, "topK": _TOP_K},
            }}}},
            "responses": {"200": _response(_VOICE_RESPONSE), **_error_responses(400, 502, 500)},
        }},
        "/image-search": {"post": {
            "tags": ["Image"], "summary": "Décrire une image puis rechercher avec le pipeline texte", "operationId": "imageSearch",
            "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
                "type": "object", "required": ["image"],
                "properties": {"image": {"type": "string", "format": "binary"}, "topK": _TOP_K},
            }}}},
            "responses": {"200": _response(_IMAGE_RESPONSE), **_error_responses(400, 502, 503, 500)},
        }},
        "/seller-assistant": {"post": {
            "tags": ["Seller assistant"], "summary": "Estimer un prix vendeur depuis le catalogue", "operationId": "sellerAssistant",
            "requestBody": {"required": True, "content": {"application/json": {
                "schema": _SELLER_PRODUCT_INPUT,
                "example": {"title": "Dell XPS", "description": "Laptop Dell en très bon état", "category": "Électronique", "brand": "Dell", "color": "Noir", "condition": "Très bon état", "sellerPrice": 1200, "currency": "MAD"},
            }}},
            "responses": {"200": _response(_SELLER_ASSISTANT_RESPONSE), **_error_responses(400, 500, 503)},
        }},
        "/api/seller/suggest-description": {"post": {
            "tags": ["Seller"], "summary": "Générer une description grounded, avec image facultative", "operationId": "sellerSuggestDescription",
            "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
                "type": "object", "required": ["product_name", "category"],
                "properties": {
                    "product_name": {"type": "string"}, "category": {"type": "string"},
                    "keywords": {"type": "string"}, "language": {"type": "string", "default": "fr"},
                    "image": {"type": "string", "format": "binary"},
                },
            }}}},
            "responses": {"200": _response({
                "type": "object", "required": ["fr", "image_analysis"],
                "properties": {"fr": {"type": "string"}, "image_analysis": {"type": ["string", "null"]}, "description_generated": {"type": "boolean"}},
            }), **_error_responses(400, 502, 503, 500)},
        }},
        "/api/seller/estimate-price": {"post": {
            "tags": ["Seller"], "summary": "Estimer un prix depuis les comparables catalogue", "operationId": "sellerEstimatePrice",
            "requestBody": {"required": True, "content": {"application/json": {
                "schema": {
                    "type": "object", "required": ["description"],
                    "properties": {"description": {"type": "string"}, "category": {"type": ["string", "null"]}, "currency": {"type": "string", "default": "MAD"}},
                },
                "example": {"description": "Laptop Dell XPS", "category": "Électronique"},
            }}},
            "responses": {"200": _response(_PRICE_ESTIMATE_RESPONSE), **_error_responses(400, 503, 500)},
        }},
        "/api/seller/check-price": {"post": {
            "tags": ["Seller"], "summary": "Contrôler un prix vendeur avec les quartiles catalogue", "operationId": "sellerCheckPrice",
            "requestBody": {"required": True, "content": {"application/json": {
                "schema": {
                    "type": "object", "required": ["description", "seller_price"],
                    "properties": {"description": {"type": "string"}, "category": {"type": ["string", "null"]}, "seller_price": {"type": "number", "exclusiveMinimum": 0}, "currency": {"type": "string", "default": "MAD"}},
                },
                "example": {"description": "dell laptop xps", "category": "electronique", "seller_price": 2},
            }}},
            "responses": {"200": _response(_PRICE_CHECK_RESPONSE), **_error_responses(400, 503, 500)},
        }},
    },
    "components": {"schemas": {
        "ErrorResponse": {
            "type": "object", "required": ["error"],
            "properties": {"error": {"type": "string"}},
            "example": {"error": "Requête invalide."},
        },
        "UnderstoodQuery": {
            "type": ["object", "null"],
            "properties": {
                "category": {"type": ["string", "null"]}, "productType": {"type": ["string", "null"]},
                "brand": {"type": ["string", "null"]}, "color": {"type": ["string", "null"]},
                "condition": {"type": ["string", "null"]}, "city": {"type": ["string", "null"]},
                "minPrice": {"type": ["number", "string", "null"]}, "maxPrice": {"type": ["number", "string", "null"]},
                "currency": {"type": ["string", "null"]}, "searchText": {"type": "string"},
            },
        },
        "Product": {
            "type": "object", "required": ["id", "title", "description", "price", "isSold"],
            "properties": _PRODUCT_PROPERTIES,
        },
        "SearchResultItem": {
            "type": "object", "required": ["product", "score", "matchType"],
            "properties": {
                "product": _ref("Product"), "score": {"type": "number", "format": "float"},
                "semanticScore": {"type": "number", "format": "float"},
                "similarityScore": {"type": "number", "format": "float"},
                "relevanceReason": {"type": ["string", "null"]},
                "lexicalTerms": {"type": "array", "items": {"type": "string"}},
                "matchType": {"type": "string", "enum": ["exact", "relevant", "similar", "broad_similar", "unrelated"]},
            },
        },
        "SearchResponse": {
            "type": "object",
            "required": ["query", "answer", "understoodQuery", "matchType", "primaryResultsCount", "similarResultsCount", "results"],
            "properties": {
                "query": {"type": "string"}, "answer": {"type": "string"},
                "understoodQuery": _ref("UnderstoodQuery"),
                "matchType": {"type": "string", "enum": ["exact", "relevant", "similar", "broad_similar", "unrelated", "none"]},
                "primaryResultsCount": {"type": "integer", "minimum": 0},
                "similarResultsCount": {"type": "integer", "minimum": 0},
                "broadSimilarResultsCount": {"type": "integer", "minimum": 0},
                "totalCatalogProducts": {"type": "integer", "minimum": 0},
                "candidateProductsCount": {"type": "integer", "minimum": 0},
                "totalResultsCount": {"type": "integer", "minimum": 0},
                "results": {"type": "array", "items": _ref("SearchResultItem")},
            },
        },
    }},
}


_SWAGGER_HTML = """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ChedMed AI Search API</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = () => SwaggerUIBundle({
      url: '/openapi.json', dom_id: '#swagger-ui', deepLinking: true,
      displayRequestDuration: true, tryItOutEnabled: true,
      presets: [SwaggerUIBundle.presets.apis], layout: 'BaseLayout'
    });
  </script>
</body>
</html>
"""
