import os

def extract_pdf(pdf_path):
    text = ""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except ImportError:
        pass
    
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except ImportError:
        pass

    return "Could not extract PDF (PyPDF2 or PyMuPDF/fitz not installed)."

def extract_docx(docx_path):
    try:
        import docx
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except ImportError:
        return "Could not extract DOCX (python-docx not installed)."

def main():
    base_dir = r"c:\Users\Manish Kumar\Downloads\Projects of Data Scientist\MatRisk-Data-Set-Sample"
    pdf_path = os.path.join(base_dir, "483555A_Data_Scientist_MatRisk_AI.docx.pdf")
    docx_path = os.path.join(base_dir, "ReadME for Dataset.docx")
    
    pdf_text = extract_pdf(pdf_path)
    docx_text = extract_docx(docx_path)
    
    out_path = os.path.join(base_dir, "extracted_tasks.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== PDF TASKS ===\n")
        f.write(pdf_text)
        f.write("\n\n=== DOCX README ===\n")
        f.write(docx_text)
        
    print("Extraction complete. Text saved to extracted_tasks.txt")

if __name__ == "__main__":
    main()
