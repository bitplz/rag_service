from methods import *
from models import get_embedding_model, get_reranking_model
from qdrant_db import get_qdrant_client
import nltk

client = None
em = None
rm = None

class RagService:
    
    @classmethod
    def init_controller():
        global client, em, rm

        client = get_qdrant_client()
        em = get_embedding_model()
        rm = get_reranking_model()

        nltk.download('punkt_tab')

    @classmethod
    def check_controllers() -> bool:
        global client, em, rm

        if not client or not em or not rm:
            print("\nInitiate Controllers First!")
            return False
        else: return True

    @classmethod
    def create_chunks(pdf_path: str):
        global em
        print(f"EM: {em}")
        if not RagService.check_controllers(): return

        print("\n\nGenerating markdown from pdf....\n")
        md_text = create_markdown_from_pdf(pdf_path)


        print("\n\nCreating chunks....\n")
        chunks = chunk_by_headings(md_text=md_text)
        final_chunks = create_final_chunks(chunks=chunks)

        return final_chunks

    @classmethod
    def embed_and_store_chunks(collection: str, chunks: list[dict]):
        global client, em

        if not RagService.check_controllers(): return

        print("\n\nEmbedding & Storing chunks....\n")
        embed_and_store(final_chunks=chunks, collection=collection, embedding_model=em, client=client)

    @classmethod
    def get_context(query: str, collection: str) -> str:
        global client, em, rm
        
        if not RagService.check_controllers(): return
        
        return run_query(query=query, embedding_model=em, re_ranking_model=rm, client=client, collection=collection)
















