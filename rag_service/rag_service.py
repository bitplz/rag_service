from rag_service.methods import *
from rag_service.models import get_embedding_model, get_reranking_model
from rag_service.qdrant_db import get_qdrant_client
import nltk

client = None
em = None
rm = None

class RagService:

    @classmethod
    def init_controller(cls, hf_token: str = None):
        global client, em, rm

        client = get_qdrant_client()
        em = get_embedding_model(hf_token=hf_token)
        rm = get_reranking_model(hf_token=hf_token)

        nltk.download('punkt_tab')

    @classmethod
    def check_controllers(cls) -> bool:
        global client, em, rm

        if not client or not em or not rm:
            print("\nInitiate Controllers First!")
            return False
        else: return True

    @classmethod
    def create_chunks_from_md_text(cls, md_text: str):
        if not RagService.check_controllers(): return

        print("\n\nCreating chunks....\n")
        chunks = chunk_by_headings(md_text=md_text)
        final_chunks = create_final_chunks(chunks=chunks)

        return final_chunks


    @classmethod
    def create_chunks(cls, pdf_path: str):
        if not RagService.check_controllers(): return

        print("\n\nGenerating markdown from pdf....\n")
        md_text = create_markdown_from_pdf(path=pdf_path)


        print("\n\nCreating chunks....\n")
        chunks = chunk_by_headings(md_text=md_text)
        final_chunks = create_final_chunks(chunks=chunks)

        return final_chunks

    @classmethod
    def embed_and_store_chunks(cls, collection: str, chunks: list[dict], metadata: dict = None):
        global client, em

        if not RagService.check_controllers(): return

        print("\n\nEmbedding & Storing chunks....\n")
        embed_and_store(final_chunks=chunks, collection=collection, embedding_model=em, client=client, metadata=metadata)

    @classmethod
    def get_context(cls, query: str, collection: str, metadata: dict=None) -> str:
        global client, em, rm

        if not RagService.check_controllers(): return

        return run_query(query=query, embedding_model=em, re_ranking_model=rm, client=client, collection=collection, metadata=metadata)
