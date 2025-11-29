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

# Startup messages
print("🌐 Online AI API available")
print("📄 Online OCR API available")

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
        
        print(f"🌐 Sending request to online API...")
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
            
            supported_extensions = ['.pdf', '.docx', '.png', '.jpg', '.jpeg']
            file_ext = '.' + filename.lower().split('.')[-1] if '.' in filename else ''
            
            if file_ext in supported_extensions:
                content = extract_text_from_file(file, filename)
            else:
                content = file.read().decode('utf-8', errors='ignore')

            if not content.strip():
                return "Error: Could not extract text."

            # --- DYNAMIC PROMPT (No Hardcoded Categories) ---
            prompt = f"""Analyze the medical report below. 
            Return a JSON object containing a detailed breakdown of the clinical data.
            
            Do NOT limit yourself to specific keys. Create keys that best describe the data found in the report.
            Examples of keys you might create: "Patient Demographics", "Chief Complaint", "Clinical History", "Lab Results", "Medications", "Imaging Findings", "Plan", "Allergies", etc.
            
            Values should be lists of strings or detailed text descriptions.

            Medical Report:
            {content[:3000]}

            Respond with ONLY the JSON object."""

            print(f"🚀 STARTING DYNAMIC ANALYSIS (Online)...")
            
            # Directly call the online AI
            raw_text = call_online_ai(prompt)
            
            if raw_text is None:
                return "Error: Failed to connect to AI service. Please try again."

            # Clean & Parse
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

    return render_template('index.html')

if __name__ == '__main__':
    # Cloud Config: Host 0.0.0.0 and Port 7860
    app.run(debug=True, host='0.0.0.0', port=7860)