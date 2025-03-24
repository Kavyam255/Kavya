from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from accelerate import load_checkpoint_and_dispatch

app = Flask(__name__)

# Log in to Hugging Face (ensure this is done)
token = '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'  # Replace with your actual token
login(token)

# Load the Llama model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with proper offloading to handle large size
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True
)

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyPDF2."""
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    return text

def generate_answer(question, context):
    """Generate an answer based on a given question and extracted PDF text."""
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF file upload and extract text."""
    if 'pdf' not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No selected file!"}), 400

    if file and file.filename.endswith('.pdf'):
        pdf_text = extract_text_from_pdf(file)
        return jsonify({"message": "PDF uploaded successfully!", "pdf_text": pdf_text[:500]})
    else:
        return jsonify({"error": "Invalid file format!"}), 400
@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle user queries based on the extracted PDF text."""
    data = request.json
    print("Received Data:", data)  # Debugging print
    
    if not data:
        return jsonify({"error": "No JSON data received!"}), 400
    
    question = data.get("question")
    context = data.get("pdf_text")

    if not question or not context:
        return jsonify({"error": "Question or context is missing!"}), 400

    answer = generate_answer(question, context)
    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(debug=True)
