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
AUTO_SWITCH_URL = f"{API_BASE_URL}/api/auto-switch"
CHAT_URL = f"{API_BASE_URL}/cerebras/chat"
DEFAULT_MODEL = "llama3.1-8b"

# --- OCR API CONFIGURATION ---
OCR_API_URL = "http://ocr.collegebuzz.in/api/ocr"

print("🌐 Using Online AI Model API...")
print("📄 Using Online OCR API for document processing...")

def extract_text_from_file(file_storage, filename):
    """Extracts text from any supported file using the OCR API."""
    try:
        # Read file content
        file_content = file_storage.read()
        file_storage.seek(0)  # Reset file pointer
        
        # Prepare the file for upload
        files = {
            'file': (filename, io.BytesIO(file_content), get_mime_type(filename))
        }
        
        print(f"🌐 Sending file to OCR API: {filename}")
        response = requests.post(OCR_API_URL, files=files, timeout=180)
        response.raise_for_status()
        
        result = response.json()
        text = result.get('text', '')
        file_type = result.get('fileType', 'unknown')
        processing_time = result.get('processingTime', 0)
        
        print(f"✅ OCR completed: {file_type} processed in {processing_time}ms")
        
        # Clean the text - remove excessive newlines
        text = ' '.join(text.split())
        return text
        
    except requests.exceptions.RequestException as e:
        print(f"❌ OCR API Error: {e}")
        return ""
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return ""

def get_mime_type(filename):
    """Returns the MIME type based on file extension."""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    mime_types = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'ppt': 'application/vnd.ms-powerpoint',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'xls': 'application/vnd.ms-excel',
        'txt': 'text/plain',
        'csv': 'text/csv',
        'rtf': 'application/rtf',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'tiff': 'image/tiff',
        'tif': 'image/tiff',
        'bmp': 'image/bmp',
        'gif': 'image/gif',
    }
    return mime_types.get(ext, 'application/octet-stream')

def clean_ai_response(text):
    """Cleans the AI response to find the JSON block."""
    text = text.replace("```json", "").replace("```", "")
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def call_online_ai(prompt):
    """Calls the online AI API to get a response."""
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        print(f"🌐 Sending request to {CHAT_URL}...")
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        # Extract the response content from the API response
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("message", {}).get("content", "")
        elif "response" in result:
            return result["response"]
        elif "content" in result:
            return result["content"]
        else:
            return str(result)
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['report']
        if file:
            filename = file.filename
            content = ""

            # --- 1. EXTRACT TEXT USING OCR API ---
            # Supported: PDF, DOCX, PPTX, XLSX, DOC, PPT, XLS, TXT, CSV, RTF, PNG, JPEG, TIFF
            supported_extensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', 
                                   '.txt', '.csv', '.rtf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif']
            
            file_ext = '.' + filename.lower().split('.')[-1] if '.' in filename else ''
            
            if file_ext in supported_extensions:
                print(f"📄 Processing {file_ext.upper()} file via OCR API...")
                content = extract_text_from_file(file, filename)
            else:
                print("📄 Processing as plain text...")
                content = file.read().decode('utf-8', errors='ignore')

            if not content.strip():
                return "Error: Could not extract text from file. File may be empty or unsupported."

            # --- 2. THE PROMPT ---
            prompt = f"""Analyze the medical report below. Extract the following details into a JSON object.
Use simple lists of strings. Do not use nested objects.

Keys required:
- "summary": A brief string summary.
- "conditions": A list of strings.
- "medications": A list of strings.
- "vitals": A list of strings.
- "treatments": A list of strings.

Medical Report:
{content[:2500]}

Respond with ONLY the JSON object, no other text."""

            print("🚀 STARTING AI ANALYSIS...")
            
            # --- 3. CALL ONLINE AI API ---
            raw_text = call_online_ai(prompt)
            
            if raw_text is None:
                return "Error: Failed to connect to AI service. Please try again."
            
            # --- 4. CLEAN & PARSE ---
            cleaned_text = clean_ai_response(raw_text)
            
            try:
                # Try standard JSON parsing
                start = cleaned_text.find('{')
                end = cleaned_text.rfind('}') + 1
                if start != -1 and end != -1:
                    data = json.loads(cleaned_text[start:end])
                else:
                    raise ValueError("No JSON found")
            except:
                print("⚠️ JSON Failed. Showing Raw Text.")
                data = {
                    "summary": raw_text, 
                    "conditions": ["See Summary"], 
                    "medications": ["See Summary"], 
                    "vitals": ["See Summary"], 
                    "treatments": ["See Summary"]
                }

            return render_template('result.html', data=data)

    return render_template('index.html')

if __name__ == '__main__':
    # Cloud Config: Host 0.0.0.0 and Port 7860
    app.run(debug=True, host='0.0.0.0', port=7860)