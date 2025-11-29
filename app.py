from flask import Flask, request, render_template
import requests
import json
import os
import io
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- 🌐 ONLINE API CONFIGURATION ---
# The Brain that analyzes the text
API_BASE_URL = "http://ai.collegebuzz.in"
CHAT_URL = f"{API_BASE_URL}/cerebras/chat"
DEFAULT_MODEL = "llama3.1-8b"

# --- 📄 OCR API CONFIGURATION ---
# The Eyes that read the file (PDF/Image)
OCR_API_URL = "http://ocr.collegebuzz.in/api/ocr"

print("✅ System Ready: Using Online AI & OCR APIs")

def extract_text_with_ocr(file_storage, filename):
    """
    Sends the file to the OCR API to extract text.
    Works for PDF, Images, Word Docs, etc.
    """
    try:
        # Reset file pointer to beginning
        file_storage.seek(0)
        file_content = file_storage.read()
        file_storage.seek(0)
        
        # Determine MIME type
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        mime_map = {
            'pdf': 'application/pdf', 'png': 'image/png', 'jpg': 'image/jpeg', 
            'jpeg': 'image/jpeg', 'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        mime_type = mime_map.get(ext, 'application/octet-stream')

        # Send to OCR API
        print(f"📤 Sending {filename} to OCR API...")
        files = {'file': (filename, io.BytesIO(file_content), mime_type)}
        response = requests.post(OCR_API_URL, files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            raw_text = result.get('text', '')
            # Clean up extra whitespace
            return ' '.join(raw_text.split())
        else:
            print(f"❌ OCR Failed: {response.status_code}")
            return ""
    except Exception as e:
        print(f"❌ OCR Error: {e}")
        return ""

def call_ai_api(prompt):
    """
    Sends the prompt to the AI API to get the analysis.
    """
    try:
        print("🧠 Calling AI API...")
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        
        # Handle different API response formats
        if "choices" in result and len(result) > 0:
            return result["choices"][0].get("message", {}).get("content", "")
        elif "content" in result:
            return result["content"]
        else:
            return str(result)
            
    except Exception as e:
        print(f"❌ AI API Error: {e}")
        return None

def smart_parse_data(text):
    """
    Cleans the AI response and ensures it's valid JSON for the dashboard.
    """
    data = {
        "summary": "",
        "conditions": [], "medications": [], "vitals": [], "treatments": []
    }

    # 1. Cleanup Markdown and Response Tags
    if "### Response:" in text: text = text.split("### Response:")[-1]
    text = text.replace("```json", "").replace("```", "")
    
    # 2. Extract Summary
    sum_match = re.search(r'summary["\']?\s*:\s*["\'](.*?)["\']', text, re.IGNORECASE | re.DOTALL)
    if sum_match: 
        data["summary"] = sum_match.group(1)
    else:
        # Fallback: Grab the first 500 chars if no summary key found
        clean_text = re.sub(r'[{}"]', '', text).strip()
        data["summary"] = clean_text[:500] + "..." if len(clean_text) > 500 else clean_text

    # 3. Extract Lists (Regex to find arrays like key: ["val1", "val2"])
    def extract_items(key, source_text):
        match = re.search(rf'{key}["\']?\s*:\s*\[(.*?)\]', source_text, re.IGNORECASE | re.DOTALL)
        items = []
        if match:
            raw_list = match.group(1)
            # Find strings inside quotes
            items = re.findall(r'["\'](.*?)["\']', raw_list)
            # If no quotes, try comma separation
            if not items:
                items = [x.strip() for x in raw_list.split(',') if x.strip()]
        return items

    data["conditions"] = extract_items("conditions", text)
    data["medications"] = extract_items("medications", text)
    data["vitals"] = extract_items("vitals", text)
    if not data["vitals"]: data["vitals"] = extract_items("vital signs", text)
    data["treatments"] = extract_items("treatments", text)

    return data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('report')
        
        if file and file.filename:
            print(f"📂 Received File: {file.filename}")
            
            # --- 1. GET TEXT (OCR) ---
            # We use the Online OCR for everything to be safe
            content = extract_text_from_file(file, file.filename)
            
            # Fallback for plain text files if OCR fails or returns empty
            if not content.strip() and file.filename.endswith('.txt'):
                file.seek(0)
                content = file.read().decode('utf-8', errors='ignore')

            if not content.strip():
                return render_template('result.html', data={"summary": "Error: Could not read text from file."})

            # --- 2. SEND TO AI ---
            prompt = f"""Analyze the medical report below. 
            Return a JSON object with keys: "summary", "conditions", "medications", "vitals", "treatments".
            
            Rules:
            1. Values must be simple lists of strings.
            2. Do not use nested objects.
            3. "summary" should be a single string paragraph.

            REPORT:
            {content[:3000]}

            Respond with ONLY the JSON object."""

            raw_response = call_ai_api(prompt)
            
            if not raw_response:
                return render_template('result.html', data={"summary": "Error: AI Service Unavailable"})

            print(f"🤖 AI Response: {raw_response[:100]}...")

            # --- 3. PARSE & DISPLAY ---
            final_data = smart_parse_data(raw_response)
            
            return render_template('result.html', data=final_data)

    return render_template('index.html')

if __name__ == '__main__':
    # Standard Flask Port
    app.run(debug=True, port=8080)