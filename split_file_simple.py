"""
A script to split a large text file into smaller parts.
Simplified version without delimiter support.

Usage:
  - By number of lines per file:
    python split_file_simple.py [input_file] --lines [number_of_lines] -o [output_directory]
    Example: python split_file_simple.py my_large_file.txt --lines 10000 -o split_files

  - By approximate size in bytes per file:
    python split_file_simple.py [input_file] --bytes [number_of_bytes] -o [output_directory]
    Example: python split_file_simple.py my_large_file.txt --bytes 5000000 -o split_files

  - By a specific number of equal parts:
    python split_file_simple.py [input_file] --parts [number_of_parts] -o [output_directory]
    Example: python split_file_simple.py my_large_file.txt --parts 4 -o split_files

  - By number of words per file:
    python split_file_simple.py [input_file] --words [number_of_words] -o [output_directory]
    Example: python split_file_simple.py my_large_file.txt --words 50000 -o split_files
"""
import os
import argparse

# --- CONFIGURATION / НАСТРОЙКИ ---
# Путь к файлу или папке, которую нужно разделить.
# Оставьте пустым (""), чтобы обязательно использовать аргументы командной строки.
# Пример: "d:/my_files/large_text.txt" или "d:/my_files/input_folder"
INPUT_PATH = r""

# Папка, куда будут сохранены разделенные файлы.
OUTPUT_DIR = r"./split_output/"

# Метод разделения: "lines" (строки), "bytes" (байты), "parts" (части), "words" (слова)
SPLIT_METHOD = "words"

# Значение для метода (например, 50000 для слов, 1000 для строк)
SPLIT_VALUE = 50000
# ---------------------------------

def split_file_by_lines(filepath, lines_per_file, output_dir):
    """Splits a file into multiple files of a specified number of lines."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(filepath, 'r', encoding='utf-8') as f:
        file_count = 1
        lines = []
        for line in f:
            lines.append(line)
            if len(lines) >= lines_per_file:
                write_part(filepath, file_count, lines, output_dir)
                lines = []
                file_count += 1
        if lines:
            write_part(filepath, file_count, lines, output_dir)

def split_file_by_bytes(filepath, bytes_per_file, output_dir):
    """Splits a file into multiple files of a specified byte size."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(filepath, 'r', encoding='utf-8') as f:
        file_count = 1
        lines = []
        current_bytes = 0
        for line in f:
            line_bytes = len(line.encode('utf-8'))
            if current_bytes + line_bytes > bytes_per_file and lines:
                write_part(filepath, file_count, lines, output_dir)
                lines = []
                current_bytes = 0
                file_count += 1
            lines.append(line)
            current_bytes += line_bytes
        if lines:
            write_part(filepath, file_count, lines, output_dir)

def split_file_by_words(filepath, words_per_file, output_dir):
    """Splits a file into multiple files of a specified number of words."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        file_count = 1
        current_chunk = []
        current_words = 0
        
        for line in f:
            line_words = len(line.split())
            
            if current_words + line_words > words_per_file and current_chunk:
                write_part(filepath, file_count, current_chunk, output_dir)
                current_chunk = []
                current_words = 0
                file_count += 1
            
            current_chunk.append(line)
            current_words += line_words
        
        # Write any remaining content
        if current_chunk:
            write_part(filepath, file_count, current_chunk, output_dir)

def split_file_into_parts(filepath, num_parts, output_dir):
    """Splits a file into a specified number of equal parts."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    lines_per_part = (total_lines + num_parts - 1) // num_parts  # Ceiling division

    for i in range(num_parts):
        start = i * lines_per_part
        end = start + lines_per_part
        part_lines = lines[start:end]
        if part_lines:
            write_part(filepath, i + 1, part_lines, output_dir)

def write_part(original_filepath, part_number, lines, output_dir):
    """Writes a part of the file to a new file."""
    base, ext = os.path.splitext(os.path.basename(original_filepath))
    output_filename = os.path.join(output_dir, f"{base}_part_{part_number}{ext}")
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        out_file.writelines(lines)
    print(f"Created {output_filename}")

def process_path(input_path, output_dir, method_func, method_arg):
    """
    Processes a single file or a directory of files using the specified split method.
    """
    if not input_path:
        print("Error: Input path is not specified.")
        return

    if os.path.isfile(input_path):
        method_func(input_path, method_arg, output_dir)
    elif os.path.isdir(input_path):
        # Iterate over files in the directory
        files_found = False
        for filename in os.listdir(input_path):
            filepath = os.path.join(input_path, filename)
            if os.path.isfile(filepath):
                files_found = True
                print(f"Processing {filepath}...")
                method_func(filepath, method_arg, output_dir)
        if not files_found:
            print(f"No files found in directory: {input_path}")
    else:
        print(f"Error: {input_path} is not a valid file or directory.")

def main():
    parser = argparse.ArgumentParser(description="Split large text files or all files in a folder.")
    # Arguments are now optional
    parser.add_argument("input_path", nargs='?', help="The file or directory to split.")
    parser.add_argument("-o", "--output-dir", help="The directory to save the split files.")
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-l", "--lines", type=int, help="Number of lines per file.")
    group.add_argument("-b", "--bytes", type=int, help="Approximate number of bytes per file.")
    group.add_argument("-n", "--parts", type=int, help="Number of equal parts to split the file into.")
    group.add_argument("-w", "--words", type=int, help="Number of words per file.")

    args = parser.parse_args()

    # Determine Input Path
    input_path = args.input_path or INPUT_PATH
    
    # Determine Output Directory
    output_dir = args.output_dir or OUTPUT_DIR

    if not input_path:
        print("Error: Please provide an input file/folder as an argument or set INPUT_PATH in the script.")
        return

    # Determine Split Method and Value
    method = None
    value = 0

    if args.lines:
        method = split_file_by_lines
        value = args.lines
    elif args.bytes:
        method = split_file_by_bytes
        value = args.bytes
    elif args.parts:
        method = split_file_into_parts
        value = args.parts
    elif args.words:
        method = split_file_by_words
        value = args.words
    else:
        # Fallback to config
        if SPLIT_METHOD == "lines":
            method = split_file_by_lines
        elif SPLIT_METHOD == "bytes":
            method = split_file_by_bytes
        elif SPLIT_METHOD == "parts":
            method = split_file_into_parts
        elif SPLIT_METHOD == "words":
            method = split_file_by_words
        
        value = SPLIT_VALUE

    if method and value:
        process_path(input_path, output_dir, method, value)
    else:
        print("Error: No split method specified. Use arguments or configure SPLIT_METHOD/SPLIT_VALUE in the script.")

if __name__ == "__main__":
    main()
