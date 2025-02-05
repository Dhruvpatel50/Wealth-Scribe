from flask import Flask, request, jsonify
import os
import json
import pdfplumber
from flask_cors import CORS
import re
import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import pytesseract

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Google API
API_KEY = os.getenv('GOOGLE_API_KEY')  # Get API key from environment variable
genai.configure(api_key=API_KEY)

# Financial terms dictionaries
RT = {
    "revenue from operations": 1, "Total Revenue": 2, "Turnover": 3, "Net Sales": 4,
    "Gross Revenue": 5, "Operating Revenue": 6, "Revenues": 7, "Receipts": 8,
    "Income from Operations": 9, "Business Income": 10, "Gross Sales": 11
}

OPT = {
    "Operating Profit": 1, "EBIT": 2, "Earnings Before Interest and Tax": 3, "Profit Before Tax": 4,
    "PBIT": 5, "Operating Income": 6, "Operating Earnings": 7, "Core Earnings": 8,
    "NOP": 9, "NOPAT": 10, "Operating Margin": 11, "Pre-Tax Operating Profit": 12
}

NPT = {
    "Net Profit": 1, "Net Income": 2, "Profit After Tax": 3, "PAT": 4,
    "Earnings After Tax": 5, "Final Profit": 6, "Net Earnings": 7,
    "Total Comprehensive Income": 8, "Post-Tax Profit": 9
}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_dates_from_text(text):
    """Extract all dates from text and determine the latest quarter."""
    date_pattern = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
    dates = re.findall(date_pattern, text)
    formatted_dates = []
    
    for date_str in dates:
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y"):
            try:
                formatted_dates.append(datetime.strptime(date_str, fmt))
                break
            except ValueError:
                continue
    
    if not formatted_dates:
        return None, None
    
    sorted_dates = sorted(formatted_dates, reverse=True)
    
    latest_date = sorted_dates[0]
    latest_quarter = (latest_date.month - 1) // 3 + 1
    latest_year = latest_date.year
    
    previous_date = sorted_dates[1] if len(sorted_dates) > 1 else None
    previous_quarter = (previous_date.month - 1) // 3 + 1 if previous_date else None
    previous_year = previous_date.year if previous_date else None
    
    return f"Q{latest_quarter} {latest_year}", f"Q{previous_quarter} {previous_year}" if previous_date else None

def extract_table_or_text(pdf_path):
    """Extracts table data from PDF using pdfplumber. If no table is found, uses OCR."""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if any("Particulars" in str(cell) for cell in table[0] if cell):
                        logger.info(f"Table found on page {page.page_number}")
                        return table
            
            logger.info(f"No table found on page {page.page_number}. Using OCR...")
            text = extract_text_from_image(pdf_path, page.page_number)
            return text if text.strip() else None 
    
    logger.warning("No tables or text found in the PDF.")
    return None

def extract_text_from_image(pdf_path, page_number):
    """Extracts text from an image-based PDF page using OCR."""
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    extracted_text = ""
    
    for img in images:
        text = pytesseract.image_to_string(img.convert("L"), config='--psm 6')
        extracted_text += text + "\n"
    return extracted_text.strip()

def extract_financial_values(table):
    """Extract financial values for current quarter and annual data."""
    extracted_data = {
        "Current Quarter": {"Revenue": None, "Operating Profit": None, "Net Profit": None},
        "Annual Data": {"Revenue": None, "Operating Profit": None, "Net Profit": None}
    }
    
    if not table:
        return extracted_data 
    
    header = table[0]
    current_quarter_col_index = None
    previous_quarter_col_index = None
    annual_col_index = None  
    
    for i, cell in enumerate(header):
        if cell and "Particular" in str(cell):
            current_quarter_col_index = i + 1  
        if cell and "ended" in str(cell).lower():
            previous_quarter_col_index = i - 1  
        if cell and "year ended" in str(cell).lower():
            annual_col_index = i  
    
    if current_quarter_col_index is None or annual_col_index is None:
        return extracted_data 
    
    def select_highest_priority(term_dict, row_text):
        if row_text is None: 
            return None
        matches = [(term, priority) for term, priority in term_dict.items() if term.lower() in row_text.lower()]
        return min(matches, key=lambda x: x[1])[0] if matches else None
    
    for row in table:
        if not row or row[0] is None:
            continue
        
        revenue_match = select_highest_priority(RT, row[0])
        op_profit_match = select_highest_priority(OPT, row[0])
        net_profit_match = select_highest_priority(NPT, row[0])
    
        if revenue_match:
            extracted_data["Current Quarter"]["Revenue"] = row[current_quarter_col_index] if current_quarter_col_index < len(row) else None
            extracted_data["Annual Data"]["Revenue"] = row[annual_col_index] if annual_col_index < len(row) else None
        if op_profit_match:
            extracted_data["Current Quarter"]["Operating Profit"] = row[current_quarter_col_index] if current_quarter_col_index < len(row) else None
            extracted_data["Annual Data"]["Operating Profit"] = row[annual_col_index] if annual_col_index < len(row) else None
        if net_profit_match:
            extracted_data["Current Quarter"]["Net Profit"] = row[current_quarter_col_index] if current_quarter_col_index < len(row) else None
            extracted_data["Annual Data"]["Net Profit"] = row[annual_col_index] if annual_col_index < len(row) else None
    
    return extracted_data

def use_gemini_extraction(text):
    """Use Gemini AI to extract financial data."""
    prompt = f"""
    Identify the latest quarters' financial data and annual data, and extract values for:
    1. Revenue
    2. Operating Profit
    3. Net Profit
    4. Financial Unit (Crores, Lakhs, Millions, Billions)
    Search for heading "Statement of" and find the latest quarter and annual financial data(column marked with 'year ended').
    Financial unit will be mentioned above the table.
    Provide output in JSON:
    {{
      "Current Quarter": {{
        "Quarter": "Qx YYYY",
        "Revenue": X,
        "Operating Profit": Y,
        "Net Profit": Z,
        "Unit": "Detected financial unit"
      }},
      "Annual Data": {{
        "Year": "YYYY",
        "Revenue": D,
        "Operating Profit": E,
        "Net Profit": F,
        "Unit": "Detected financial unit"
      }}
    }}
    Text to analyze:
    {text}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    
    try:
        json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        return None
    
    return None

def detect_fin_unit(text):
    """Detect financial unit from the extracted text."""
    units = ["Crores", "Lakhs", "Millions", "Billions"]
    for unit in units:
        if unit.lower() in text.lower():
            return unit
    return "Unknown"

def extract_fin_data(pdf_path):
    """Main function to extract financial data."""
    try:
        extracted_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text() or "" 
        
        if not extracted_text.strip():
            return {"error-status": 404, "message": "No financial data found in the document."}
        
        current_quarter, previous_quarter = extract_dates_from_text(extracted_text)
        fin_unit = detect_fin_unit(extracted_text)
        
        table = extract_table_or_text(pdf_path)
        fin_data = extract_financial_values(table) 
        
        if not any(fin_data["Current Quarter"].values()):
            ai_data = use_gemini_extraction(extracted_text) or {}
            fin_data["Current Quarter"].update(ai_data.get("Current Quarter", {}))
            fin_data["Annual Data"].update(ai_data.get("Annual Data", {}))
        
        if not any(fin_data["Current Quarter"].values()) and not any(fin_data["Annual Data"].values()):
            return {"error-status": 404, "message": "No financial data found in the document."}
        
        fin_data["Current Quarter"]["Quarter"] = current_quarter
        fin_data["Annual Data"]["Year"] = re.search(r"\b\d{4}\b", extracted_text).group() if re.search(r"\b\d{4}\b", extracted_text) else "Unknown Year"
        fin_data["Current Quarter"]["Unit"] = fin_unit
        fin_data["Annual Data"]["Unit"] = fin_unit
        
        return fin_data
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return {"error-status": 500, "message": f"Error processing PDF: {str(e)}"}

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/v1/extract-financial-data', methods=['POST'])
def extract_financial_data():
    """Extract financial data from uploaded PDF file"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'status': 400
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'status': 400
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Only PDF files are allowed',
                'status': 400
            }), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        logger.info(f"File saved: {filepath}")

        try:
            extracted_data = extract_fin_data(filepath)
            os.remove(filepath)
            logger.info(f"File processed and removed: {filepath}")

            if 'error-status' in extracted_data:
                return jsonify({
                    'error': extracted_data['message'],
                    'status': extracted_data['error-status']
                }), extracted_data['error-status']

            return jsonify({
                'status': 200,
                'data': extracted_data
            }), 200

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({
                'error': 'Error processing PDF file',
                'details': str(e),
                'status': 500
            }), 500

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'status': 500
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeding limit"""
    return jsonify({
        'error': 'File too large. Maximum size is 16MB',
        'status': 413
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    return jsonify({
        'error': 'Internal server error',
        'status': 500
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)