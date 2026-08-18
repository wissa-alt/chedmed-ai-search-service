from decimal import Decimal
from datetime import datetime, timezone

from models.product import Product
from search.product_text_builder import ProductTextBuilder


def create_product() -> Product:
    return Product(
        id="123",
        title="Laptop Dell XPS",
        description="Ordinateur portable Dell XPS",
        category="Électronique",
        brand="Dell",
        color="Noir",
        condition="Très bon état",
        price=Decimal("400"),
        currency="MAD",
        city="Casablanca",
        image_urls=(),
        status="ACCEPTED",
        is_sold=False,
        updated_at=datetime.now(timezone.utc),
    )


def test_build_contains_semantic_fields():
    builder = ProductTextBuilder()
    product = create_product()

    document = builder.build(product)

    assert "Titre: Laptop Dell XPS" in document
    assert "Description: Ordinateur portable Dell XPS" in document
    assert "Catégorie: Électronique" in document
    assert "Marque: Dell" in document
    assert "Couleur: Noir" in document
    assert "État: Très bon état" in document
    assert "Ville: Casablanca" in document
    assert "Prix: 400 MAD" in document
    assert "Statut: ACCEPTED" in document


def test_build_does_not_add_embedding_prefix():
    builder = ProductTextBuilder()
    product = create_product()

    document = builder.build(product)

    assert not document.startswith("passage: ")


def test_build_is_deterministic():
    builder = ProductTextBuilder()
    product = create_product()

    document_1 = builder.build(product)
    document_2 = builder.build(product)

    assert document_1 == document_2


def test_optional_fields_are_omitted():
    builder = ProductTextBuilder()

    product = create_product()

    product = Product(
        id=product.id,
        title=product.title,
        description=product.description,
        category=product.category,
        brand=None,
        color=None,
        condition=None,
        price=product.price,
        currency=product.currency,
        city=None,
        image_urls=(),
        status=product.status,
        is_sold=False,
        updated_at=product.updated_at,
    )

    document = builder.build(product)

    assert "Marque:" not in document
    assert "Couleur:" not in document
    assert "État:" not in document
    assert "Ville:" not in document


def test_invalid_product_raises_type_error():
    builder = ProductTextBuilder()

    try:
        builder.build("not a product")
        assert False
    except TypeError:
        assert True
