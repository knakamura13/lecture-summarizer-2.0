import os
import re


def clean_name(name):
    # Remove '_subtitles', replace underscores with spaces
    name = name.replace("_subtitles", "").replace("_", " ")
    # Replace two or more spaces with a single space
    return re.sub(r"\s{2,}", " ", name)


def combine_lectures_in_module(root_directory):
    for module in os.listdir(root_directory):
        module_path = os.path.join(root_directory, module)
        if os.path.isdir(module_path):
            cleaned_module_name = clean_name(module)
            combined_lectures = ""
            for lecture_file in sorted(os.listdir(module_path)):
                if lecture_file.endswith(".txt"):
                    lecture_number, lecture_name = lecture_file.split(" - ", 1)
                    cleaned_lecture_name = clean_name(lecture_name[:-4])
                    lecture_path = os.path.join(module_path, lecture_file)
                    with open(lecture_path, "r", encoding="utf-8") as file:
                        lecture_content = file.read().strip()
                    combined_lectures += f"{cleaned_module_name} - {lecture_number} - {cleaned_lecture_name}\n{lecture_content}\n\n"

            combined_file_path = os.path.join(
                root_directory, f"{cleaned_module_name}.txt"
            )
            with open(combined_file_path, "w", encoding="utf-8") as combined_file:
                combined_file.write(combined_lectures)
            print(
                f"Combined lectures for module {cleaned_module_name} into {combined_file_path}"
            )


if __name__ == "__main__":
    combine_lectures_in_module(
        input("Enter the root directory of the lecture modules:")
    )
