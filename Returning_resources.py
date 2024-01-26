import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["OPENAI_API_KEY"]

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.lancedb import LanceDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import lancedb

vectordb = lancedb.connect("/temp/lancedb")
vector_table = vectordb.create_table("my_table",data=[
        {
            "vector": OpenAIEmbeddings().embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
        }
    ], mode="overwrite")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

bs_strainer = bs4.SoupStrainer(class_ = ("post-content", "post-title", "post-header"))

loader = WebBaseLoader(
    web_path="https://lilianweng.github.io/posts/2023-06-23-agent/",
    bs_kwargs=dict(parse_only = bs_strainer))

# Loading the documents
# This will fetch the necessary text and metadata from the webpage and convert
# them into docuemts
docs = loader.load()

textSplitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 250)
splits = textSplitter.split_documents(docs)
vectorstore = LanceDB.from_documents(documents=splits, embedding=OpenAIEmbeddings(), connection = vector_table)

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(temperature=0.2)



rag_chain = (
    RunnablePassthrough.assign(context = (lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer = rag_chain)

def main():
    question = ""
    response = ""
    while True:
        question = input("Question: ")
        if question == "quit":
            return
        response = rag_chain_with_source.invoke(question)
        print("Answer: ", response['answer'],"\n")
        for doc in response['context']:
            print(doc.page_content)

main()