from symforce import python_util
from symforce import types as T


def format_cpp(file_contents: str, filename: str) -> str:
    """
    Autoformat a given C++ file using clang-format

    Args:
        file_contents (str): The unformatted contents of the file
        filename (str): A name that this file might have on disk; this does not have to be a real
            path, it's only used for clang-format to find the correct style file (by traversing
            upwards from this location) and to decide if an include is the corresponding .h file
            for a .cc file that's being formatted (this affects the include order)

    Returns:
        formatted_file_contents (str): The contents of the file after formatting
    """
    formatted_file_contents = T.cast(
        str,
        python_util.execute_subprocess(
            ["clang-format-8", f"-assume-filename={filename}"],
            stdin_data=file_contents,
            log_stdout=False,
        ),
    )

    return formatted_file_contents
