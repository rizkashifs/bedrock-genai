import PyPDF2
from app.models.bedrock_client import invoke_claude

def file_read(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            content = ""
            for page in reader.pages:
                content += page.extract_text()
            return content
    except Exception as e:
        return f"An error occurred: {e}"
    

# A function to read a pdf just using file_read()
def read_pdf_content(pdf_path):
    return file_read(pdf_path)

# Example usage
if __name__ == "__main__":
    pdf_path = r"C:\Users\admin\Downloads\bedrock-testing\app\testdocs\\test.pdf"  # Replace with the path to your PDF file
    #content = read_pdf_content(pdf_path) # doesn't work well
    content = file_read(pdf_path)
    print(content[:50])
    print (invoke_claude(f"Summarize the following content in one sentence: {content[:100]}"))

    