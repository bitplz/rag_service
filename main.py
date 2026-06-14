from rag_service import RagService

service = RagService()
service.init_controller()

chunks = service.create_chunks(pdf_path="policy.pdf")

print(chunks)