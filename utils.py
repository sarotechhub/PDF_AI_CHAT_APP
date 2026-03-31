import os
import re
import tempfile


def validate_pdf(file) -> bool:
    """
    Validate an uploaded file:
      - Must be a valid PDF (header check)
      - Must be under 200 MB

    Raises ValueError with a user-friendly message on failure.
    """
    # Check file size (200 MB limit for large documents)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    max_size = 200 * 1024 * 1024  # 200 MB
    if file_size > max_size:
        raise ValueError(
            f"File size ({format_file_size(file_size)}) exceeds the 200 MB limit. "
            "Please upload a smaller file."
        )

    # Validate PDF magic bytes
    header = file.read(5)
    file.seek(0)
    if header != b"%PDF-":
        raise ValueError(
            "Invalid file format. The uploaded file does not appear to be a valid PDF."
        )

    return True


def sanitize_input(text: str) -> str:
    """
    Sanitize user input:
      - Strip HTML tags
      - Remove dangerous characters
      - Limit to 500 characters
    """
    if not isinstance(text, str):
        return ""

    text = text[:500]
    text = re.sub(r"<[^>]*>", "", text)                 # strip HTML
    text = re.sub(r"[^\w\s.,!?'\"\-:;()/]", "", text)   # keep safe chars
    return text.strip()


def format_chat_history(messages: list) -> list:
    """Return the last 5 messages for context-window use."""
    return messages[-5:] if len(messages) > 5 else messages


def save_uploaded_file(uploaded_file) -> str:
    """
    Persist an uploaded file to ./temp/ using tempfile for security.
    Returns the path to the saved file.
    """
    temp_dir = "./temp/"
    os.makedirs(temp_dir, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return temp_path


def cleanup_temp_files(temp_dir: str = "./temp/"):
    """Remove all files in the temp directory."""
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except OSError:
                pass


def format_file_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string (e.g. 4.2 MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
