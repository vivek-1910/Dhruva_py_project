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

# --- SESSION STORAGE FOR CHAT HISTORY ---
chat_sessions = {}

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
                # Increased context window to 4096 for better analysis
                local_llm = Llama(model_path=LOCAL_MODEL_PATH, n_ctx=4096, n_threads=4, n_gpu_layers=0, verbose=False)
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

def smart_parse_data(text):
    """
    Robust Parser: Extracts JSON even if the AI adds extra text or formatting errors.
    Filters out unwanted keys like contact info, personal details, etc.
    """
    data = {}
    
    # 1. Clean Markdown Code Blocks
    text = text.replace("```json", "").replace("```", "").strip()
    
    # 2. Try Standard JSON Parsing first
    try:
        # regex to find the largest {} block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            # Filter unwanted keys
            filtered = filter_unwanted_keys(parsed)
            return filtered
    except:
        pass

    # 3. Fallback: Manual Regex Extraction (If JSON is broken)
    print("⚠️ Standard JSON failed. Attempting Regex extraction...")
    
    # Capture "Key": ["Value", "Value"] pattern
    pattern = r'["\'](.*?)["\']\s*:\s*(\[.*?\])'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for key, raw_list in matches:
        # Extract items inside the list string
        items = re.findall(r'["\'](.*?)["\']', raw_list)
        if items:
            data[key] = items
            
    # Capture "Key": "Value" pattern
    summary_match = re.search(r'["\'](Summary|Patient Summary|Overview)["\']\s*:\s*["\'](.*?)["\']', text, re.IGNORECASE)
    if summary_match:
        data["Patient Summary"] = summary_match.group(2)
        
    # Filter unwanted keys
    filtered = filter_unwanted_keys(data)
    
    # If empty, put raw text in a debug key
    if not filtered:
        filtered["Medical Analysis"] = [text]
        
    return filtered

def filter_unwanted_keys(data):
    """
    Remove unwanted keys like contact info, personal identifiers, etc.
    Keep only clinically relevant information.
    """
    if not isinstance(data, dict):
        return data
    
    unwanted_patterns = [
        'contact', 'phone', 'email', 'address', 'patient_id', 'hospital_id',
        'clinic', 'name', 'patient_name', 'doctor_name', 'provider', 
        'id_number', 'insurance', 'account', 'mrn', 'medical_record',
        'zip', 'postal', 'street', 'city', 'state', 'country',
        'fax', 'mobile', 'phone_number', 'contact_person',
        'dob', 'date_of_birth', 'age_in_years', 'sex',
        'occupation', 'employer', 'reference', 'next_of_kin'
    ]
    
    filtered = {}
    for key, value in data.items():
        key_lower = key.lower().replace(' ', '_')
        
        # Skip if key matches unwanted patterns
        if any(pattern in key_lower for pattern in unwanted_patterns):
            print(f"⏭️ Skipping unwanted key: {key}")
            continue
        
        # Skip if value is empty
        if not value or (isinstance(value, list) and len(value) == 0):
            continue
        
        # Skip if it's just personal details
        if isinstance(value, (str, list)):
            value_str = str(value).lower()
            if any(pattern in value_str for pattern in ['@', 'tel:', 'phone:', '+91', '+1']):
                continue
        
        filtered[key] = value
    
    return filtered

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

def is_medical_text(text: str) -> bool:
    """DEPRECATED: kept for fallback compatibility."""
    if not text or not isinstance(text, str):
        return False
    return False


def classify_medical(text: str, use_online=True) -> bool:
    """Ask the online AI whether `text` is medical.

    Returns True when the AI labels the text as MEDICAL. If the online AI
    cannot be reached or returns an unclear answer, fall back to a minimal
    keyword heuristic.
    """
    if not text or not isinstance(text, str):
        return False

    # Use only the first 2000 chars for classification to keep payload small
    sample = text.strip()[:2000]

    prompt = (
        "You are a classifier. Decide whether the following user-provided text is a MEDICAL document or query.\n"
        "Respond with a single word: MEDICAL or NON-MEDICAL. Do not add any other text.\n\n"
        f"Text:\n{sample}\n"
    )

    ai_response = None
    if use_online:
        try:
            ai_response = call_online_ai(prompt)
        except Exception:
            ai_response = None

    decision = None
    if ai_response and isinstance(ai_response, str):
        r = ai_response.strip().upper()
        if 'MEDICAL' in r and 'NON-MEDICAL' not in r:
            decision = True
        elif 'NON-MEDICAL' in r and 'MEDICAL' not in r:
            decision = False
        else:
            # sometimes model returns explanation; look for keywords
            if 'MEDICAL' in r:
                decision = True
            elif 'NON-MEDICAL' in r:
                decision = False
    # If AI is unreachable or ambiguous, default to MEDICAL (do not hard-reject)
    if decision is None:
        return True

    return bool(decision)


def is_medical_query(text: str) -> bool:
    """Prefer AI classifier; fallback to deprecated function."""
    return classify_medical(text)

def call_local_ai(prompt):
    try:
        llm = load_local_model()
        if llm is None: return None
        
        # --- FIXED: Mistral Prompt Formatting ---
        # We wrap the prompt in [INST] tags so the model follows instructions
        formatted_prompt = f"""[INST] {prompt} 
        
        Ensure valid JSON output. [/INST]"""
        
        print("🏠 Running Local Inference...")
        response = llm(formatted_prompt, max_tokens=1500, stop=["</s>"], echo=False)
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"❌ Local Model Error: {e}")
        return None

@app.route('/chat', methods=['POST'])
def chat():
    """Handle multiturn medical chatbot conversations"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        history = data.get('history', [])
        
        if not user_message:
            return json.dumps({'error': 'Empty message'}), 400
        
        # Build conversation context from history
        system_prompt = """You are a medical AI assistant. Prioritize medical topics and provide accurate, concise medical information, analysis, and guidance.
        If a user asks a question that is not medical in nature, do NOT rigidly repeat a single refusal phrase. Instead, briefly state that your primary focus is medical topics and either:
        - Offer to reframe the question toward health/medical aspects if appropriate, or
        - Provide a short, neutral, high-level response when the question is harmless and does not require specialist non-medical expertise.

        When you do answer medical questions, remind users that your responses are informational and not a replacement for professional medical advice. Keep tone helpful and concise."""
        
        # Prepare messages for API
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-10:]:  # Keep last 10 messages for context
            messages.append({"role": msg['role'], "content": msg['content']})
        messages.append({"role": "user", "content": user_message})
        
        # Call AI API
        try:
            headers = {"Content-Type": "application/json"}
            payload = {"model": DEFAULT_MODEL, "messages": messages}
            response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result:
                ai_reply = result["choices"][0].get("message", {}).get("content", "")
                return json.dumps({'reply': ai_reply})
            else:
                return json.dumps({'error': 'Invalid API response'}), 500
        except Exception as e:
            print(f"❌ Chat API Error: {e}")
            return json.dumps({'error': f'API Error: {str(e)}'}), 500
            
    except Exception as e:
        print(f"❌ Chat Error: {e}")
        return json.dumps({'error': str(e)}), 500

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

            # Validate that the uploaded report is medical using AI classifier
            try:
                is_med = classify_medical(content)
            except Exception:
                is_med = True

            if not is_med:
                return render_template('index.html', local_model_available=local_model_available, upload_error='Please upload only medical-related reports (PDFs or text).')

            # --- DYNAMIC PROMPT ---
            prompt = f"""You are an expert medical AI. Analyze the medical report below.
            
            Return a JSON object containing a detailed breakdown of the clinical data.
            Do NOT use hardcoded keys. Dynamically create keys that best describe the data found (e.g., "Patient_Info", "Chief_Complaint", "Lab_Values", "Medications", "Imaging_Results").
            
            Values must be LISTS of strings.
            
            Medical Report:
            {content[:3000]}

            Respond ONLY with the JSON object."""

            print(f"🚀 STARTING DYNAMIC ANALYSIS ({model_choice})...")
            
            if model_choice == 'local':
                raw_text = call_local_ai(prompt)
                # Fallback to online if local fails/returns nothing
                if not raw_text:
                    print("⚠️ Local returned empty. Switching to Online.")
                    raw_text = call_online_ai(prompt)
            else:
                raw_text = call_online_ai(prompt)
            
            # Use the new Smart Parser instead of basic cleaning
            data = smart_parse_data(raw_text)

            return render_template('result.html', data=data)

    return render_template('index.html', local_model_available=local_model_available)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7860)