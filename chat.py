import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
import tempfile

st.title("Khush hoja Khushali :)")
st.write("Sorry the application isn't more helpful par PDF upload karke try kar lena :()")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    query = "Give 20 exact responses. No response shorter than 4 words should be included. Do not take into consideration the things said by the interviewer. The responses should only be by the participant. Do not summarize or generate new text. Give a summary at the end." + st.text_input("Ask a question about the PDF:")
    
    if query:
        docs = vectorstore.similarity_search(query)
        answer = qa_chain.run(input_documents=docs, question=query)
        
        st.write(f"Answer: {answer}")