import os
import glob
import shutil
import fitz
import io
from PIL import Image
import pytesseract

from typing import List, Tuple

# Find the tesseract binary portably instead of hardcoding a Homebrew path
# (which only exists on Apple-Silicon Macs). Falls back to the common install
# locations, and if none are found OCR is simply skipped (see extract_pdf_text).
_TESSERACT_CANDIDATES = (
    '/opt/homebrew/bin/tesseract',   # Apple Silicon Homebrew
    '/usr/local/bin/tesseract',      # Intel Homebrew
    '/usr/bin/tesseract',            # Linux
)
_TESSERACT_BIN = shutil.which('tesseract') or next(
    (p for p in _TESSERACT_CANDIDATES if os.path.exists(p)), None
)
if _TESSERACT_BIN:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_BIN
    _bin_dir = os.path.dirname(_TESSERACT_BIN)
    if _bin_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = _bin_dir + os.pathsep + os.environ.get('PATH', '')

HAS_TESSERACT = _TESSERACT_BIN is not None


class PDFProcessor:
    """
    Process PDF files, extracting text and using OCR to analyze images within the PDFs.
    """

    @staticmethod
    def get_pdf_paths(folder_path: str):
        """
        Fetches all PDF file paths from the folder.

        Parameters:
        - path: Path to the folder that contains the PDF files.

        Returns:
        - pdf_paths: List of PDF file paths.
        """
        pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
        print(f"Found {len(pdf_paths)} PDF files")
        return pdf_paths

    @staticmethod
    def extract_pdf_text(path: str) -> Tuple[List[str], List[int]]:
        """
        Extract text from the input PDF file, with images analyzed using OCR.

        Parameters:
        - path: Path to the PDF file.

        Returns:
        - texts: List of texts extracted from each page
        - page_numbers: List of page numbers
        """
        texts = []
        page_numbers = []
        with fitz.open(path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = ""  # initialize for this page

                # Extract text
                text = page.get_text()
                if text.strip():
                    page_text = text

                # Extract images and perform OCR (skip entirely if tesseract
                # is unavailable, and never let a single bad image abort the run)
                if HAS_TESSERACT:
                    for img_tuple in page.get_images(full=True):
                        try:
                            xref = img_tuple[0]
                            image_bytes = doc.extract_image(xref)["image"]
                            image = Image.open(io.BytesIO(image_bytes))
                            ocr_text = pytesseract.image_to_string(image)
                            if ocr_text.strip():
                                page_text += "\n" + ocr_text
                        except Exception:
                            # OCR is best-effort enrichment; keep going.
                            continue

                page_numbers.append(page_num + 1)
                texts.append(page_text)

        return texts, page_numbers

    @staticmethod
    def extract_text_from_pdfs_in_folder(folder_path) -> Tuple[List[str], List[str]]:
        """
        Processes all PDFs in the folder and extracts text and OCR content.

        Parameters:
        - path: Path to the folder that contains the PDF files.

        Returns:
        - texts: List of texts extracted from each page
        - info: List of info, i.e., PDF filenames and page numbers 
        where the text has been extracted.
        """
        pdf_paths = PDFProcessor.get_pdf_paths(folder_path)

        texts = []
        info = []
        for path in pdf_paths:
            print(f"Processing {path}")
            texts_cur, page_numbers_cur = PDFProcessor.extract_pdf_text(path)

            texts += texts_cur
            fname_cur = os.path.basename(path)
            info += [f'{fname_cur} {page_num}' 
                    for page_num in page_numbers_cur]

        return texts, info
