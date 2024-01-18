import os
import re
from unidecode import unidecode

# Define the root directory
root_dir = "summaries_individual"


# Function to escape LaTeX special characters
def latex_escape(text):
    special_chars = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "[": r"\[",
        "]": r"\]",
        '"': r"''",
        "\\": r"\textbackslash{}",
        "~": r"\textasciitilde{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
        "^": r"\textasciicircum{}",
        "`": r"\textasciigrave{}",
    }
    regex = re.compile("|".join(re.escape(key) for key in special_chars.keys()))
    return regex.sub(lambda x: special_chars[x.group()], text)


# Function to process and clean text
def process_text(txt):
    # Replace non-ASCII characters with nearest equivalents
    txt = unidecode(txt)

    # Escape LaTeX special characters
    txt = latex_escape(txt)

    # Replace ellipses '...' with '\ldots'
    txt = txt.replace("...", r"\ldots")

    # Replace straight double-quotes with straight single-quotes.
    txt = txt.replace("''", "'")

    # Replace straight single-quotes with single backticks.
    txt = txt.replace("'", "`")

    # Ensure each sentence starts on a new line (optional, for cleaner LaTeX)
    txt = re.sub(r"(?<!\n)\. (?=[A-Z])", ".\n", txt)

    # Handle end-of-sentence spacing for sentences ending with a capital letter
    txt = re.sub(r"([A-Z])\.", r"\1\\@.", txt)

    # Escape periods in abbreviations or in the middle of sentences
    # Ensure it doesn't affect ellipses or periods at the end of sentences
    txt = re.sub(r"(?<!\.\.)\.(?=\s+[a-z])", r"\.", txt)

    return txt


# Traverse the root directory
for module_dir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, module_dir)):
        # Extract module number and name
        module_number, module_name = module_dir.split("__", maxsplit=1)
        module_name = module_name.replace("_", " ")

        # Prepare the LaTeX document
        latex_content = (
            f"\\section{{{process_text(module_number + ': ' + module_name)}}}\n\n"
        )

        # List to hold (lecture number, lecture name, file content)
        lectures = []

        # Read all text files in the subdirectory
        for file in os.listdir(os.path.join(root_dir, module_dir)):
            if file.endswith(".txt"):
                # Extract lecture number and name
                lecture_number, lecture_name = file.split(" - ", maxsplit=1)
                lecture_number = int(lecture_number)
                lecture_name = lecture_name.replace(".txt", "").replace("_", " ")

                # Read the content of the text file
                with open(os.path.join(root_dir, module_dir, file), "r") as f:
                    content = f.readline().strip()

                lectures.append((lecture_number, lecture_name, content))

        # Sort lectures by lecture number
        lectures.sort(key=lambda x: x[0])

        # Add lectures to the LaTeX content
        for lecture_number, lecture_name, content in lectures:
            latex_content += f"\\subsection{{{process_text(lecture_name)}}}\n{process_text(content)}\n\n"

        # Write the LaTeX content to a file
        with open(f"{module_number}__{module_name.replace(' ', '_')}.tex", "w") as f:
            f.write(latex_content)
