import logging
import os
import re
import sys
from pathlib import Path
from time import time, sleep

import nltk
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from tqdm import tqdm

# Set to False to enable actual GPT API calls
DRY_RUN = False
# Delimiter used to separate text in GPT messages
DELIMITER = '\n"""\n'
# Set the max number of chunks to summarize (recommended: -1 for unlimited)
MAX_CHUNKS = -1
# Size of each text chunk for summarization (recommended: 1,000 to 5,000)
CHUNK_SIZE = sys.maxsize
# Timeout for GPT API requests in seconds (recommended: >= 60s)
GPT_REQUEST_TIMEOUT = 60
# Maximum number of retries for GPT API requests (recommended: 2 to 5)
GPT_REQUEST_MAX_RETRY = 3

# Initialize logging
logging.basicConfig(
    filename="summarizer.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)

# Download the necessary NLTK models for splitting by sentence
nltk.download("punkt")

load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))


def open_file(filepath: str | Path) -> str:
    """
    Opens and reads the content of a file.

    Args:
        filepath (str | Path): Path to the file to be read.
    Returns:
        str: The content of the file.
    Raises:
        IOError: If the file cannot be opened or read.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as infile:
            return infile.read()
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found {filepath}: {fnf_error}")
        raise IOError(f"File not found: {filepath}") from fnf_error
    except IOError as io_error:
        logging.error(f"IO error while opening file {filepath}: {io_error}")
        raise IOError(f"IO error: {filepath}") from io_error
    except Exception as exc:
        logging.error(f"Unexpected error while opening file {filepath}: {exc}")
        raise IOError(f"Unexpected error: {filepath}") from exc


def save_file(content: str, filepath: str | Path) -> None:
    """
    Saves content to a file.

    Args:
        content (str): Content to be saved.
        filepath (str | Path): Path where the content will be saved.
    Raises:
        IOError: If the file cannot be written.
    """
    if not filepath:
        logging.error("File path is empty")
        raise ValueError("File path is empty")

    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as outfile:
            outfile.write(content)
    except IOError as io_error:
        logging.error(f"IO error while saving file {filepath}: {io_error}")
        raise IOError(f"IO error: {filepath}") from io_error
    except Exception as exc:
        logging.error(f"Unexpected error while saving file {filepath}: {exc}")
        raise IOError(f"Unexpected error: {filepath}") from exc


def remove_extra_whitespace(text: str) -> str:
    """
    Removes extra whitespaces from the text.

    Args:
        text (str): The original text.
    Returns:
        str: The text with extra whitespaces removed.
    """
    return re.sub(r"\s+", " ", text.strip()).strip()


def chunk_text_by_sentences(text, max_chunk_size):
    """
    Splits the text into chunks, each containing whole sentences and having a length close to max_chunk_size.

    Args:
        text (str): The text to be chunked.
        max_chunk_size (int): The approximate maximum size of each chunk.

    Returns:
        list: A list of text chunks.
    """
    # Return the original text if the chunk size is set to infinity
    if max_chunk_size is sys.maxsize:
        return [text]

    # Split the text into sentences
    sentences = sent_tokenize(text)

    _chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Check if adding the next sentence would exceed the max_chunk_size
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            # If so, add the current_chunk to the chunks list and start a new chunk
            _chunks.append(current_chunk)
            current_chunk = sentence
        else:
            # Otherwise, add the sentence to the current chunk
            current_chunk += " " + sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        _chunks.append(current_chunk)

    return _chunks


def summarize_with_gpt(_input_txt: str, _model: str = "gpt-3") -> str:
    """
    Summarizes text using the GPT model.

    Implements an exponential backoff strategy for retries in case of request failures.

    Args:
        _input_txt (str): Text to be summarized.
        _model (str, optional): The GPT model to be used. Defaults to 'gpt-3'.
    Returns:
        str: Summarized text or error message in case of failure.
    """
    global err
    if DRY_RUN:
        # Return the original text to simulate a summary in dry-run mode
        return remove_extra_whitespace(_input_txt)

    if not _model or _model == "gpt-3":
        _model = "gpt-3.5-turbo"
    if _model == "gpt-4":
        _model = "gpt-4-1106-preview"

    _messages = [
        {
            "role": "system",
            "content": "You are a writing assistant, skilled in revising and summarizing complex technical writing "
            "with accuracy and precision.",
        },
        {
            "role": "user",
            "content": "Provide an executive summary of the following text (delimited by triple quotes). "
            "Present the key ideas and findings directly and to the point, without bullet points, "
            "as if for a busy professional who needs to record the essential points quickly for"
            "their own reference later. Since you are the one writing the notes for your own use,"
            "you should never mention that the text you write is a summary of a larger text. "
            "Do not limit your summary to complete sentences or grammatical correctness. "
            "Abbreviate long and repetitive words to reduce character count. "
            "This is very important for my career, so please try your best. "
            f"{DELIMITER}{_input_txt}{DELIMITER}",
        },
    ]

    err = ""
    retry = 0
    extra_wait = 2  # Additional wait time in seconds

    while retry < GPT_REQUEST_MAX_RETRY:
        try:
            response = client.chat.completions.create(
                model=_model, messages=_messages, timeout=GPT_REQUEST_TIMEOUT
            )
            cleaned_input = remove_extra_whitespace(_input_txt)
            cleaned_output = remove_extra_whitespace(
                response.choices[0].message.content
            )
            log_text = (
                "PROMPT:\n\n"
                + cleaned_input
                + "\n\n==========\n\nRESPONSE:\n\n"
                + cleaned_output
            )
            save_file(log_text, f'gpt_logs/{"%.2f" % time()}_gpt.txt')
            return cleaned_output
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                match = re.search(r"Please try again in (\d+)m(\d+)s", str(e))
                if match:
                    minutes, seconds = match.groups()
                    minutes, seconds = int(minutes) + 1, int(seconds) + extra_wait
                    print(f"Waiting for {minutes} minutes, {seconds} seconds...")
                    wait_time = minutes * 60 + seconds
                    sleep(wait_time)
                else:
                    print(f"Waiting 1 minute before trying again...")
                    # If wait time is not found, use a default wait time
                    sleep(60 + extra_wait)
            else:
                logging.error(
                    f"General GPT request error on try {retry + 1}/{GPT_REQUEST_MAX_RETRY}: {e}"
                )
        finally:
            retry += 1

        if retry >= GPT_REQUEST_MAX_RETRY:
            logging.error(f"GPT request failed after {GPT_REQUEST_MAX_RETRY} retries")
            return f"GPT error: {err}" if err else "GPT error: Unknown error"

    return "Max retry attempts exceeded"


if __name__ == "__main__":
    try:
        lectures_dir = Path("lectures_individual")
        summarize_dir = Path("summaries_individual")
        summarize_dir.mkdir(exist_ok=True)

        for module_dir in lectures_dir.iterdir():
            if module_dir.is_dir():
                module_summarize_dir = summarize_dir / module_dir.name
                module_summarize_dir.mkdir(exist_ok=True)

                for file_path in module_dir.glob("*.txt"):
                    output_path = module_summarize_dir / file_path.name

                    # Skip this file if a matching filename exists in the output directory
                    if output_path.exists():
                        logging.info(
                            f"Skipping previously summarized lecture: {file_path.name}"
                        )
                        continue

                    try:
                        original_text = open_file(file_path)
                        chunks = chunk_text_by_sentences(original_text, CHUNK_SIZE)
                        chunks = chunks[:MAX_CHUNKS] if MAX_CHUNKS > 0 else chunks

                        result = []
                        for chunk in tqdm(
                            chunks,
                            desc=f"Summarizing lecture {file_path.name}",
                            leave=True,
                        ):
                            summary = summarize_with_gpt(chunk, _model="gpt-4")
                            result.append(summary)

                        final_combined_text = "\n".join(result)
                        save_file(final_combined_text, output_path)

                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {e}")

    except FileNotFoundError as fnf_error:
        logging.critical(f"Critical file not found error: {fnf_error}")
    except IOError as io_error:
        logging.critical(f"Critical input/output error: {io_error}")
    except Exception as exc:
        logging.critical(f"Unexpected critical error occurred: {exc}")
