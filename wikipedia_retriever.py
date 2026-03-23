from langchain_community.retrievers import WikipediaRetriever

#Initialize the retriever
retriever = WikipediaRetriever(
    top_k_results=2,
    lang='en'
)

#Define the query
query = "the geopolitical history of india and pakistan from the perspective of a chinese"

#get relevant wikipedia documents
docs = retriever.invoke(query)

print(docs)

for i, doc in enumerate(docs):
    print(f"\n--- result {i+1} ---")
    print(f"Content:\n {doc.page_content} ---")     