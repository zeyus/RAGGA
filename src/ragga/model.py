

# encoder = Encoder()
# faiss_db = VectorDatabase()


# passages = faiss_db.create_passages_from_documents(docs)
# faiss_db.store_passages_db(passages, encoder.encoder)

# generator = Generator(simple_prompt, faiss_db, encoder)

# QUERY = "What happened today?"
# # QUERY = "What is happening in Israel?"
# print(QUERY)
# print("Answer:")
# result = generator.get_answer(QUERY)



# while True:
#     try:
#         query = input("Ask a question: ")
#         print("Answer:")
#         result = generator.get_answer(query)
#         # print(chunk, end='', flush=True)
#         # print("", flush=True)
#         print(result)
#     except KeyboardInterrupt:
#         print("Bye!")
#         break
#     except Exception as e:
#         print("Exception occurred. Quitting, bye!")
#         print(e)
#         break
