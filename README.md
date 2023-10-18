# Pinecone Database README

## Functions

### Create Embedding

Uses OpenAI to create an embedding for a given text.

#### Parameters

-   `message`: string

#### Returns

-   `embedding`: list of floats

---

### Insert Text

Inserts a text into the vector database.

#### Parameters

-   `message`: string
-   `metadata` (optional): dictionary
-   `namespace` (optional): string

#### Returns

-   Nothing

---

### Query Text

Queries the vector database for the most similar text.

#### Parameters

-   `message`: string

#### Returns

-   `results`: dictionary containing the most similar text and its metadata
