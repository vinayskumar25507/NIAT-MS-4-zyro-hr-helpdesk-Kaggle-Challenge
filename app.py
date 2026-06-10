import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration and Theming
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢", layout="wide")
st.title("🏢 Zyro Dynamics Corporate HR Help Desk")
st.markdown("Welcome to the secure employee verification portal. Ask any question regarding internal corporate policy guidelines.")

# 2. Authentication & API Key Setup
if "GROQ_API_KEY" in os.environ:
    groq_api_key = os.environ["GROQ_API_KEY"]
elif "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    st.sidebar.warning("⚠️ GROQ_API_KEY not found in environment or secrets.")
    groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not groq_api_key:
    st.info("Please provide your Groq API Key to initiate the HR secure session.")
    st.stop()

# 3. Cached RAG Engine Initialization
@st.cache_resource(show_spinner="Indexing corporate policies and spinning up vector store...")
def initialize_rag_engine():
    # Look for documents in standard Kaggle input directory or local app directory fallback
    paths_to_check = ["/kaggle/input/zyro-dynamics-hr-corpus/*.pdf", "./*.pdf", "./policies/*.pdf"]
    pdf_paths = []
    for path in paths_to_check:
        found = glob.glob(path)
        if found:
            pdf_paths = found
            break

    documents = []
    for path in pdf_paths:
        # ADVERSARIAL FILTER: Prevent poisoned Acrux Handbook from corrupting the index
        if "01_Employee_Handbook.pdf" in os.path.basename(path):
            continue
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    if not documents:
        return None, None

    # Structural Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Compute Embeddings locally on CPU/GPU environment
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # High-Diversity Maximal Marginal Relevance (MMR) Routing
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 25, "lambda_mult": 0.4}
    )
    
    # Initialize Core Language Model Instance
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key, temperature=0.0)
    return retriever, llm

retriever, llm = initialize_rag_engine()

if retriever is None or llm is None:
    st.error("Could not locate any corporate policy PDF files. Place them in your app directory.")
    st.stop()

# 4. Compilation of Pipelines and Guardrails
OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI guardrail classifier for Zyro Dynamics corporate systems.
Determine whether a user's question is an internal corporate HR policy question or if it is out-of-scope (general knowledge, coding, world events, other companies).
Respond with exactly one word: 'IN_SCOPE' or 'OUT_SCOPE'. Do not include punctuation or markdown."""),
    ("human", "Question: {question}")
])

REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the official Zyro Dynamics HR Verification Engine.
Answer internal employee questions using ONLY the verified context fragments provided below.
1. Rely strictly on facts directly mentioned. Do not extrapolate.
2. If the answer cannot be cleanly derived, respond EXACTLY with: "I don't have that information in the documentation."
3. NO PREAMBLE: Deliver dense, factual policy answers directly. Do not include conversational introductory prefixes."""),
    ("human", "Context:
{context}

Question: {question}
Answer:")
])

def format_docs_and_sanitize(docs):
    cleaned = []
    for doc in docs:
        content = doc.page_content
        content = content.replace("Acrux Dynamics", "Zyro Dynamics")
        content = content.replace("acruxdynamics.com", "zyrodynamics.com")
        content = content.replace("AcruxHR", "ZyroHR")
        content = content.replace("AcruxDesk", "ZyroDesk")
        cleaned.append(content)
    return "\n\n".join(cleaned)

def ask_bot(question):
    classification_chain = OOS_PROMPT | llm | StrOutputParser()
    decision = classification_chain.invoke({"question": question}).strip().upper()
    
    if "OUT_SCOPE" in decision:
        return REFUSAL_MESSAGE
    
    rag_pipeline = (
        {"context": retriever | format_docs_and_sanitize, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    return rag_pipeline.invoke(question)

# 5. Streamlit Conversational State Management & Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a policy question (e.g., 'What is our Earned Leave policy?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Verifying corporate policies..."):
            response = ask_bot(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})