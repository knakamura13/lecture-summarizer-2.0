"""
Converts `.srt` files to `.txt` files.
"""
import os
import re


def convert_srt_to_txt(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    text_lines = []
    for line in lines:
        # Remove lines with timestamps and line numbers
        if re.match(r"\d+", line) or "-->" in line:
            continue
        line = line.strip()
        if line:
            text_lines.append(line)

    return " ".join(text_lines)


def find_and_convert_srt(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".srt"):
                full_path = os.path.join(root, file)
                txt_content = convert_srt_to_txt(full_path)
                new_file_path = full_path.replace(".srt", ".txt")
                with open(new_file_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(txt_content)
                print(f"Converted {file} to txt")


if __name__ == "__main__":
    find_and_convert_srt(input("Enter the root directory containing .srt files:"))
