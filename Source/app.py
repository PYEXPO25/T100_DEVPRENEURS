import os
from flask import Flask, render_template, request, send_file, jsonify
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Make sure to set it in your .env file.")
genai.configure(api_key=API_KEY)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# PDF Processing
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(api_key=API_KEY, model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(api_key=API_KEY, model="models/embedding-001")
    try:
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except FileNotFoundError:
        return None

def get_conversational_chain():
    prompt_template = """
    Answer the question in as detailed a manner as possible using the provided context.
    If the answer is not in the context, say "answer is not available in the context."
    
    Context:\n {context}?\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    pdf_file = request.files["pdf_file"]
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)
    pdf_file.save(pdf_path)

    try:
        raw_text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return jsonify({"message": "PDF processed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/ask", methods=["POST"])
def ask_question():
    question = request.json.get("question")
    if not question:
        return jsonify({"error": "No question provided"})

    vector_store = load_vector_store()
    if not vector_store:
        return jsonify({"error": "No processed PDF found. Upload and process a PDF first."})

    docs = vector_store.similarity_search(question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

    return jsonify({"answer": response["output_text"]})

if __name__ == "__main__":
    app.run(debug=True)
