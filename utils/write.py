"""
This file includes the file writing function.
"""


def write_to_file(file_path, content):
    """
    Write content to a file.

    Args:
        - file_path (str): The path of the file to write to.
        - content (str): The content to be written to the file.
    """
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(content)
    except IOError:
        print("An error occurred while writing to the file")

