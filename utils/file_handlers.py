"""File handling utilities for different document types."""

import logging
import mimetypes
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib

from langchain.schema import Document

logger = logging.getLogger(__name__)

class FileValidator:
    """Validates uploaded files before processing."""

    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    @classmethod
    def validate_file(cls, file_path: Path) -> Dict[str, Any]:
        """Validate a file and return validation results."""
        result = {
            'valid': False,
            'error': None,
            'file_info': {}
        }

        try:
            # Check if file exists
            if not file_path.exists():
                result['error'] = "File does not exist"
                return result

            # Check file extension
            file_extension = file_path.suffix.lower()
            if file_extension not in cls.ALLOWED_EXTENSIONS:
                result['error'] = f"Unsupported file type: {file_extension}"
                return result

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > cls.MAX_FILE_SIZE:
                result['error'] = f"File too large: {file_size / 1024 / 1024:.1f}MB (max: {cls.MAX_FILE_SIZE / 1024 / 1024:.1f}MB)"
                return result

            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
            except Exception as e:
                result['error'] = f"File is not readable: {str(e)}"
                return result

            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))

            result.update({
                'valid': True,
                'file_info': {
                    'name': file_path.name,
                    'size': file_size,
                    'extension': file_extension,
                    'mime_type': mime_type,
                    'path': str(file_path)
                }
            })

        except Exception as e:
            result['error'] = f"Validation error: {str(e)}"
            logger.error(f"Error validating file {file_path}: {e}")

        return result

    @classmethod
    def validate_multiple_files(cls, file_paths: List[Path]) -> Dict[str, Any]:
        """Validate multiple files."""
        results = {
            'valid_files': [],
            'invalid_files': [],
            'total_size': 0,
            'summary': {}
        }

        for file_path in file_paths:
            validation = cls.validate_file(file_path)

            if validation['valid']:
                results['valid_files'].append(validation['file_info'])
                results['total_size'] += validation['file_info']['size']
            else:
                results['invalid_files'].append({
                    'path': str(file_path),
                    'error': validation['error']
                })

        results['summary'] = {
            'total_files': len(file_paths),
            'valid_count': len(results['valid_files']),
            'invalid_count': len(results['invalid_files']),
            'total_size_mb': results['total_size'] / 1024 / 1024
        }

        return results

class DocumentMetadata:
    """Handles document metadata extraction and management."""

    @staticmethod
    def extract_metadata(file_path: Path, content: str = None) -> Dict[str, Any]:
        """Extract metadata from a document file."""
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size,
            'created_time': file_path.stat().st_ctime,
            'modified_time': file_path.stat().st_mtime,
        }

        # Add content-based metadata if content is provided
        if content:
            metadata.update({
                'content_length': len(content),
                'word_count': len(content.split()),
                'char_count': len(content),
                'line_count': content.count('\n') + 1,
            })

        # Add file hash for deduplication
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                metadata['file_hash'] = file_hash
        except Exception as e:
            logger.warning(f"Could not generate hash for {file_path}: {e}")

        return metadata

    @staticmethod
    def enhance_document_metadata(document: Document, additional_metadata: Dict[str, Any] = None) -> Document:
        """Enhance a LangChain document with additional metadata."""
        if additional_metadata:
            document.metadata.update(additional_metadata)

        # Add processing timestamp
        import time
        document.metadata['processed_at'] = time.time()

        return document

class ContentExtractor:
    """Extracts and cleans content from different file types."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove or replace problematic characters
        text = text.replace('\x00', '')  # Remove null bytes
        text = text.replace('\r\n', '\n')  # Normalize line breaks
        text = text.replace('\r', '\n')

        # Remove excessive newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')

        return text.strip()

    @staticmethod
    def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text (simple implementation)."""
        if not text:
            return []

        # Simple keyword extraction based on word frequency
        words = text.lower().split()

        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }

        # Count word frequencies
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        key_phrases = [word for word, freq in sorted_words[:max_phrases]]

        return key_phrases

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []

        # Simple sentence splitting (can be improved with NLTK or spaCy)
        import re

        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short sentences
                cleaned_sentences.append(sentence)

        return cleaned_sentences

class FileUtils:
    """General file utilities."""

    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """Ensure directory exists, create if it doesn't."""
        directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_filename(filename: str) -> str:
        """Create a safe filename by removing/replacing problematic characters."""
        import re

        # Replace problematic characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        safe_name = re.sub(r'[^\w\s.-]', '', safe_name)
        safe_name = safe_name.strip()

        # Ensure it's not empty and not too long
        if not safe_name:
            safe_name = "unnamed_file"
        if len(safe_name) > 255:
            safe_name = safe_name[:255]

        return safe_name

    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """Get comprehensive file information."""
        try:
            stat = file_path.stat()
            return {
                'name': file_path.name,
                'path': str(file_path),
                'size': stat.st_size,
                'size_mb': stat.st_size / 1024 / 1024,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'extension': file_path.suffix.lower(),
                'exists': file_path.exists(),
                'is_file': file_path.is_file(),
                'is_readable': file_path.exists() and file_path.is_file()
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {'error': str(e)}

    @staticmethod
    def cleanup_temp_files(directory: Path, max_age_hours: int = 24) -> int:
        """Clean up temporary files older than specified hours."""
        if not directory.exists():
            return 0

        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0

        try:
            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.info(f"Cleaned up old file: {file_path}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        return cleaned_count
