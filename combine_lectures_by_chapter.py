import os


def combine_by_chapter(root_directory):
    chapter_contents = {}
    files_sorted = sorted(os.listdir(root_directory))

    for file_name in files_sorted:
        if file_name.endswith(".txt"):
            chapter = file_name[:2]  # Extract the first two characters for chapter
            file_path = os.path.join(root_directory, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip() + "\n\n"
                chapter_contents.setdefault(chapter, "")
                chapter_contents[chapter] += content

    for chapter, content in chapter_contents.items():
        combined_file_path = os.path.join(
            root_directory, "lectures_combined_by_chapter", f"{chapter}_combined.txt"
        )
        with open(combined_file_path, "w", encoding="utf-8") as combined_file:
            combined_file.write(content)
        print(f"Combined content for chapter {chapter} into {combined_file_path}")


if __name__ == "__main__":
    combine_by_chapter("lectures_combined_by_module")
