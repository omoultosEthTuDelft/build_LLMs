from PyPDF2 import PdfReader, PdfWriter

def split_first_pages(input_pdf, output_pdf):
    try:
        # Open the PDF file
        reader = PdfReader(input_pdf)
        writer = PdfWriter()
        
        # Add the first 10 pages to the new PDF
        for page_num in range(min(22, len(reader.pages))):
            writer.add_page(reader.pages[page_num])
        
        # Save the new PDF to the specified output file
        with open(output_pdf, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"Successfully created {output_pdf} with the first 10 pages of {input_pdf}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_pdf_path = "/Users/omoultos/coding/build_LLMs/S.Raschka_book.pdf"  # Replace with your input PDF file
output_pdf_path = "/Users/omoultos/coding/build_LLMs/22.pdf"  # Desired output file name
split_first_pages(input_pdf_path, output_pdf_path)
