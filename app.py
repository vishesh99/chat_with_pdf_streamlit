import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Retrieve environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if environment variables are loaded correctly
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set correctly")

# Configure Google Generative AI
genai.configure(api_key=google_api_key)

# Define the directory to save PDF files
PDF_DIR = "uploaded_pdfs"
if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR)


def save_pdf_locally(pdf_docs):
    for pdf in pdf_docs:
        pdf_path = os.path.join(PDF_DIR, pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())


def get_pdf_text_from_local(uploaded_pdf_filename):
    text = ""
    pdf_path = os.path.join(PDF_DIR, uploaded_pdf_filename)
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


async def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context". Don't provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


async def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response


def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF File and Click on the Submit & Process Button",
                                    accept_multiple_files=False)  # Accept only one file

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    save_pdf_locally([pdf_docs])  # Save only the uploaded file
                    raw_text = get_pdf_text_from_local(pdf_docs.name)  # Pass the filename
                    text_chunks = get_text_chunks(raw_text)
                    asyncio.run(get_vector_store(text_chunks))
                    st.success("Done")
            else:
                st.warning("Please upload a PDF file.")

    if user_question:
        response = asyncio.run(user_input(user_question))
        st.write("Reply: ", response["output_text"])


if __name__ == "__main__":
    main()
