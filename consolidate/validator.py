import os

from consolidate.logger import logger

def validate_file(file_path, extensions, file_type):
    """
    Validate the file path for existence and correct extension.

    Args:
        file_path (str): The path to the file.
        extensions (list): Valid file extensions (e.g., ['.mp3', '.wav']).
        file_type (str): Description of the file type (e.g., 'audio file').

    Returns:
        str: Validated file path.

    Raises:
        ValueError: If the file does not exist or the extension is invalid.
    """
    if not os.path.exists(file_path):
        error_msg = f"Error: The {file_type} at '{file_path}' does not exist."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if not any(file_path.lower().endswith(ext) for ext in extensions):
        error_msg = f"Error: The {file_type} must have one of the following extensions: {', '.join(extensions)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return file_path