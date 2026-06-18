import re
import pymupdf4llm
from nltk.tokenize import sent_tokenize
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid

def chunk_by_headings(md_text: str) -> list[dict]:
   """Split markdown into chunks at heading boundaries."""
   pattern = r'(?=^#{1,6} .+$)'
   sections = re.split(pattern, md_text, flags=re.MULTILINE)
   chunks = []
   for section in sections:
       section = section.strip()
       if not section:
           continue
       lines = section.split('\n')
       title = lines[0].strip('#').strip() if lines[0].startswith('#') else "Untitled"
       chunks.append({
           "title": title,
           "text": section
       })
   return chunks

def create_markdown_from_pdf(path: str):
    md_text = pymupdf4llm.to_markdown(path)
    return md_text

def split_large_chunk(chunks: dict, max_chars: int = 800) -> list[dict]:
   """If a chunk is too large, split it by sentences with overlap."""

   text = chunks["text"]
   if len(text) <= max_chars:
       return [chunks]
   sentences = sent_tokenize(text)
   sub_chunks = []
   current = ""
   for sent in sentences:
       if len(current) + len(sent) > max_chars and current:
           sub_chunks.append({
               "title": chunks["title"],
               "text": current.strip()
           })
           #overlapping
           current = sent + " "
       else:
           current += sent + " "
   if current.strip():
       sub_chunks.append({"title": chunks["title"], "text": current.strip()})
   return sub_chunks

def create_final_chunks(chunks: list[dict]):
    final_chunks = []
    for chunk in chunks:
        final_chunks.extend(split_large_chunk(chunk, max_chars=800))

    return final_chunks


def embed_and_store(final_chunks: list[dict], collection: str, embedding_model, client, metadata: dict=None):

    texts_to_embed = [f"{c['title']}\n\n{c['text']}" for c in final_chunks]
    embeddings = embedding_model.encode(
        texts_to_embed,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=len(embeddings[0]),
                distance=Distance.COSINE
            ),
        )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload={"text": texts_to_embed[i], "chunk_index": i, **metadata}
        )
        for i, _ in enumerate(final_chunks)
    ]

    client.upsert(
        collection_name=collection,
        points=points,
    )

    print(f"Text embedded and stored in `{collection}` collection!")



def build_filter(metadata: dict):
    must_conditions = []

    for key, value in metadata.items():
        must_conditions.append({
            "key": key,
            "match": {
                "value": value
            }
        })

    return {"must": must_conditions}



def run_query(query: str, embedding_model, re_ranking_model, client, collection: str, metadata: dict=None):

    filter = {}

    if metadata:
        filter = build_filter(metadata=metadata) 

    query_embedding = embedding_model.encode(
        sentences=[query],
        task="retrieval",
        prompt_name="query",
        normalize_embeddings=True,
    )

    result = client.query_points(
        collection_name=collection,
        query=query_embedding[0].tolist(),
        limit=20,
        filter=filter
    )

    top20 = [
        {
        "title": r.payload.get("text", "").split("\n")[0],
        "text": r.payload.get("text")
        }
        for r in result.points
        ]

    # print(f"TOP 20:\n {top20}")

    pairs = [(query, t["text"]) for t in top20]

    rerank_scores = re_ranking_model.predict(pairs)

    ##selecting only top 3
    reranked = sorted(
        zip(rerank_scores, top20),
        key=lambda x: x[0],
        reverse=True
    )[:4]

    full_text = ""

    for rank, (score, chunk) in enumerate(reranked, start=1):
        full_text += chunk['text']

    # print("\n")
    # print(full_text)

    return full_text
