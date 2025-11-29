from flask import Flask, request, render_template
import requests
import json
import os
import io
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- ONLINE API CONFIGURATION ---
API_BASE_URL = "http://ai.collegebuzz.in"
CHAT_URL = f"{API_BASE_URL}/cerebras/chat"
DEFAULT_MODEL = "llama3.1-8b"

# --- OCR API CONFIGURATION ---
OCR_API_URL = "http://ocr.collegebuzz.in/api/ocr"

# --- LOCAL MODEL CONFIGURATION ---
HF_MODEL_REPO = "DrGPT2025/dhruva"
MODEL_FILENAME = "medical_mistral.gguf"
LOCAL_MODEL_PATH = f"models/{MODEL_FILENAME}"
local_llm = None

def download_model_from_hf():
    if os.path.exists(LOCAL_MODEL_PATH): return True
    try:
        from huggingface_hub import hf_hub_download
        print(f"📥 Downloading model from {HF_MODEL_REPO}...")
        os.makedirs("models", exist_ok=True)
        hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME, local_dir="models", local_dir_use_symlinks=False)
        print("✅ Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def load_local_model():
    global local_llm
    if local_llm is None:
        if not os.path.exists(LOCAL_MODEL_PATH): download_model_from_hf()
        if os.path.exists(LOCAL_MODEL_PATH):
            try:
                from llama_cpp import Llama
                print(f"🔄 Loading local model...")
                local_llm = Llama(model_path=LOCAL_MODEL_PATH, n_ctx=2048, n_threads=4, n_gpu_layers=0, verbose=False)
                print("✅ Local model loaded!")
            except Exception as e: print(f"❌ Failed to load local model: {e}")
    return local_llm

def extract_text_from_file(file_storage, filename):
    try:
        file_content = file_storage.read()
        file_storage.seek(0)
        files = {'file': (filename, io.BytesIO(file_content), get_mime_type(filename))}
        print(f"🌐 Sending to OCR API: {filename}")
        response = requests.post(OCR_API_URL, files=files, timeout=180)
        response.raise_for_status()
        result = response.json()
        return ' '.join(result.get('text', '').split())
    except Exception as e:
        print(f"❌ OCR Error: {e}")
        return ""

def get_mime_type(filename):
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    mime_types = {'pdf': 'application/pdf', 'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg'}
    return mime_types.get(ext, 'application/octet-stream')

def clean_ai_response(text):
    text = text.replace("```json", "").replace("```", "")
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match: return match.group(0)
    return text

def call_online_ai(prompt):
    try:
        headers = {"Content-Type": "application/json"}
        payload = {"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": prompt}]}
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if "choices" in result: return result["choices"][0].get("message", {}).get("content", "")
        return str(result)
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None

def call_local_ai(prompt):
    try:
        llm = load_local_model()
        if llm is None: return None
        response = llm(prompt, max_tokens=1024, stop=["</s>"], echo=False)
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"❌ Local Model Error: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    local_model_available = True
    
    if request.method == 'POST':
        file = request.files['report']
        model_choice = request.form.get('model_choice', 'online')
        
        if file:
            filename = file.filename
            content = ""
            
            supported_extensions = ['.pdf', '.docx', '.png', '.jpg', '.jpeg']
            file_ext = '.' + filename.lower().split('.')[-1] if '.' in filename else ''
            
            if file_ext in supported_extensions:
                content = extract_text_from_file(file, filename)
            else:
                content = file.read().decode('utf-8', errors='ignore')

            if not content.strip():
                return "Error: Could not extract text."

            # --- DYNAMIC PROMPT (No Hardcoded Categories) ---
            # We ask the AI to be the "Architect" of the data structure.
            prompt = f"""Analyze the medical report below. 
            Return a JSON object containing a detailed breakdown of the clinical data.
            
            Do NOT limit yourself to specific keys. Create keys that best describe the data found in the report.
            Examples of keys you might create: "Patient Demographics", "Chief Complaint", "Clinical History", "Lab Results", "Medications", "Imaging Findings", "Plan", "Allergies", etc.
            
            Values should be lists of strings or detailed text descriptions.

            Medical Report:
            {content[:3000]}

            Respond with ONLY the JSON object."""

            print(f"🚀 STARTING DYNAMIC ANALYSIS ({model_choice})...")
            
            if model_choice == 'local':
                raw_text = call_local_ai(prompt) or call_online_ai(prompt)
            else:
                raw_text = call_online_ai(prompt)
            
            cleaned_text = clean_ai_response(raw_text)
            
            try:
                data = json.loads(cleaned_text)
            except:
                print("⚠️ JSON Parse Failed. Using Raw Text.")
                data = {
                    "Analysis Failed": ["Could not parse JSON."],
                    "Raw Output": [raw_text]
                }

            return render_template('result.html', data=data)

    return render_template('index.html', local_model_available=local_model_available)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7860)