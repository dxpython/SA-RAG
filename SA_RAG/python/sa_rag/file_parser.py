import io
import logging
from typing import Optional
import fitz  # PyMuPDF
from docx import Document

logger = logging.getLogger(__name__)

class FileParser:
    """Parses various file formats to extract text"""
    
    @staticmethod
    def parse_pdf(file_content: bytes) -> str:
        """Parse PDF content"""
        text = ""
        try:
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text() + "\n\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise

    @staticmethod
    def parse_docx(file_content: bytes) -> str:
        """Parse DOCX content"""
        text = ""
        try:
            doc = Document(io.BytesIO(file_content))
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error parsing DOCX: {e}")
            raise

    @staticmethod
    def parse_txt(file_content: bytes) -> str:
        """Parse Text content"""
        try:
            return file_content.decode("utf-8").strip()
        except Exception as e:
            logger.error(f"Error parsing TXT: {e}")
            raise

    @classmethod
    def parse(cls, filename: str, content: bytes) -> str:
        """Parse file based on extension"""
        filename = filename.lower()
        if filename.endswith(".pdf"):
            return cls.parse_pdf(content)
        elif filename.endswith(".docx"):
            return cls.parse_docx(content)
        elif filename.endswith(".txt") or filename.endswith(".md"):
            return cls.parse_txt(content)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
