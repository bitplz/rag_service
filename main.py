from rag_service import RagService
from langchain_openai import ChatOpenAI

HF_TOKEN = "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"


service = RagService()
service.init_controller(hf_token=HF_TOKEN)

def get_openAI():
    return ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key="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",
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.1,
        streaming=False
    )

def get_assistant_prompt(context: str, question: str):
  return f"""
  You are a policy analysis assistant.

Your task is to answer the user's question using ONLY the provided context.

Guidelines:
1. Identify relevant sections from the context.
2. Provide a clear, formal, and precise answer.
3. If the context partially answers the question, respond with the available details instead of rejecting.
4. Only state "The policy does not specify this information" if absolutely no relevant detail is present.
5. Structure your answer in paragraphs or bullet points.

Context:
{context}

User Question:
{question}

Final Answer:

  """


def create_chunks_and_embed(pdf_path: str, collection: str, metadata: dict = None):
    chunks = service.create_chunks(pdf_path=pdf_path)
    service.embed_and_store_chunks(collection=collection, chunks=chunks, metadata=metadata)

def ask(query: str, collection: str):
    context = service.get_context(query=query, collection=collection)
    print(f"CONTEXT: {context}")
    prompt = get_assistant_prompt(context=context, question=query)
    
    llm = get_openAI()

    messages = [("system", prompt)]
    result = llm.invoke(messages)

    print(f"\n\nRESULT: {result.content}")

def run():
    pdf_path="policy.pdf"
    query = "can company save my personal and where it will be used for?"
    collection = "policy"

    metadata = {
        "doc_type": "hr",
        "department": "hr",
        "source": pdf_path,
        "collection": collection
    }

    create_chunks_and_embed(pdf_path=pdf_path, collection=collection, metadata=metadata)

