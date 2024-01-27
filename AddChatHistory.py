import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.lancedb import LanceDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import lancedb
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage


# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

vectordb = lancedb.connect("/temp/lancedb")
vector_table = vectordb.create_table("my_table",
                                    data = [{
            "vector": OpenAIEmbeddings().embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
        }], mode="overwrite")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = LanceDB.from_documents(documents=splits, embedding=OpenAIEmbeddings(), connection = vector_table)

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_system_prompt = """You are an assistant for question-answering tasks.\
Use the following pieces of reterieved context to answer the question. \
If you don't Know the answer, just say that you don't know.\
Use three sentences maximum and keep the answer concise.\

{context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

contextualize_q_system_propmt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_propmt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

def cotextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

rag_chain = (
    RunnablePassthrough.assign(
        context = cotextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
)


def main():
    chat_history = []
    while True:
        question = input("Question: ")
        if question == "quit": return
        ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
        print("Answer: ", ai_msg.content,"\n")
        chat_history.extend([HumanMessage(content= question), ai_msg])


main()
