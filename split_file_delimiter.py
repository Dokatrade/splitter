"""
A script to split a large text file into smaller parts.

Usage:
  - By number of lines per file:
    python split_file.py [input_file] --lines [number_of_lines] -o [output_directory]
    Example: python split_file.py my_large_file.txt --lines 10000 -o split_files

  - By approximate size in bytes per file:
    python split_file.py [input_file] --bytes [number_of_bytes] -o [output_directory]
    Example: python split_file.py my_large_file.txt --bytes 5000000 -o split_files

  - By a specific number of equal parts:
    python split_file.py [input_file] --parts [number_of_parts] -o [output_directory]
    Example: python split_file.py my_large_file.txt --parts 4 -o split_files

  - By a delimiter string (each file starts with the delimiter line):
    python split_file.py [input_file] --delimiter "-- Chat:" -o [output_directory]
    Example: python split_file.py my_large_file.txt --delimiter "-- Chat:" -o split_files

  - By multiple delimiters:
    python split_file.py [input_file] --delimiter "-- Chat:" "---" -o [output_directory]
    Example: python split_file.py my_large_file.txt -d "-- Chat:" "-- New:" -o split_files
"""
import os
import argparse

# --- CONFIGURATION / НАСТРОЙКИ ---
# Путь к файлу или папке, которую нужно разделить.
# Оставьте пустым (""), чтобы обязательно использовать аргументы командной строки.
# Пример: "d:/my_files/large_text.txt" или "d:/my_files/input_folder"
INPUT_PATH = r"d:/results/chatgpt/all-projects.txt"

# Папка, куда будут сохранены разделенные файлы.
OUTPUT_DIR = r"d:/results/chatgpt/split_output/"

# Метод разделения: "lines" (строки), "bytes" (байты), "parts" (части), "words" (слова), или "delimiter" (по разделителю)
SPLIT_METHOD = "words"

# Значение для метода (например, 50000 для слов, 1000 для строк)
SPLIT_VALUE = 50000

# Строка-разделитель (или список строк) для метода "delimiter"
# Каждый файл начинается со строки, содержащей любой из разделителей
# Можно указать одну строку: "-- Chat:"
# Или несколько: ["-- Chat:", "-- New conversation:", "---"]
DELIMITER = ["-- Chat:", "=== Project:", "### Account:"]
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
    """
    Splits a file into multiple files of a specified number of words.
    If DELIMITER is set, respects delimiter boundaries - splits happen at delimiters.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if we should respect delimiter boundaries
    use_delimiters = DELIMITER and len(DELIMITER) > 0
    
    def line_contains_delimiter(line):
        if not use_delimiters:
            return False
        return any(d in line for d in DELIMITER)
    
    def find_prev_non_empty_line_is_delimiter(all_lines, idx):
        """Check if the previous non-empty line is a delimiter."""
        prev_idx = idx - 1
        while prev_idx >= 0 and all_lines[prev_idx].strip() == '':
            prev_idx -= 1
        if prev_idx < 0:
            return False
        return line_contains_delimiter(all_lines[prev_idx])
    
    # Read all lines first (needed for delimiter-aware splitting)
    with open(filepath, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    if not all_lines:
        return
    
    file_count = 1
    current_chunk = []
    current_words = 0
    i = 0
    
    while i < len(all_lines):
        line = all_lines[i]
        line_words = len(line.split())
        is_delim = line_contains_delimiter(line)
        
        # Check if this delimiter starts a new logical section
        # (i.e., previous non-empty line is content, not another delimiter)
        is_split_point = False
        if use_delimiters and is_delim and i > 0:
            is_split_point = not find_prev_non_empty_line_is_delimiter(all_lines, i)
        
        # If we've exceeded word limit and hit a split point, save current chunk
        if current_words >= words_per_file and is_split_point and current_chunk:
            write_part(filepath, file_count, current_chunk, output_dir)
            current_chunk = []
            current_words = 0
            file_count += 1
            # Don't increment i - we want to include this delimiter in next chunk
            continue
        
        # If no delimiters configured, use simple word-based splitting
        if not use_delimiters:
            if current_words + line_words > words_per_file and current_chunk:
                write_part(filepath, file_count, current_chunk, output_dir)
                current_chunk = []
                current_words = 0
                file_count += 1
        
        current_chunk.append(line)
        current_words += line_words
        i += 1
    
    # Write any remaining content
    if current_chunk:
        write_part(filepath, file_count, current_chunk, output_dir)

def split_file_by_delimiter(filepath, delimiters, output_dir):
    """
    Splits a file by delimiter string(s).
    Each output file starts with delimiter line(s) and ends before the next delimiter.
    Consecutive delimiters are grouped at the beginning of their section.
    
    Args:
        filepath: Path to the input file
        delimiters: A single delimiter string or a list of delimiter strings
        output_dir: Directory to save output files
    """
    # Normalize delimiters to a list
    if isinstance(delimiters, str):
        delimiters = [delimiters]
    
    def line_contains_delimiter(line):
        """Check if line contains any of the delimiters."""
        return any(d in line for d in delimiters)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read all lines first
    with open(filepath, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    if not all_lines:
        print(f"Warning: {filepath} is empty.")
        return
    
    # Step 1: Find all delimiter line indices
    delimiter_indices = []
    for i, line in enumerate(all_lines):
        if line_contains_delimiter(line):
            delimiter_indices.append(i)
    
    if not delimiter_indices:
        # No delimiters found - write entire file as single part
        write_part(filepath, 1, all_lines, output_dir)
        print(f"Warning: No delimiters {delimiters} found in {filepath}. File copied as single part.")
        return
    
    # Step 2: Find real split points (where delimiter comes after content, not after another delimiter)
    # A split point is a delimiter index where the previous NON-EMPTY line is NOT a delimiter
    split_points = []
    for idx in delimiter_indices:
        if idx == 0:
            # First line is delimiter - this is a split point (start of first section)
            split_points.append(idx)
        else:
            # Find the previous non-empty line (skip blank lines)
            prev_idx = idx - 1
            while prev_idx >= 0 and all_lines[prev_idx].strip() == '':
                prev_idx -= 1
            
            if prev_idx < 0:
                # All previous lines are empty - treat as first content
                split_points.append(idx)
            else:
                # Check if the previous non-empty line is a delimiter
                prev_is_delim = line_contains_delimiter(all_lines[prev_idx])
                if not prev_is_delim:
                    # Previous non-empty line is content - this is a split point
                    split_points.append(idx)
                # If previous non-empty line IS a delimiter, this is consecutive - not a split point
    
    # Step 3: Write sections
    file_count = 0
    
    # Write header (content before first split point) if exists
    # If the first line is a split point, first_split_idx will be 0, so header will be empty (correct)
    if split_points:
        first_split_idx = split_points[0]
        if first_split_idx > 0:
            header_lines = all_lines[:first_split_idx]
            write_part(filepath, file_count, header_lines, output_dir)
            file_count += 1
    elif not split_points and all_lines:
         # No split points found but file has content - treat as single part header
         write_part(filepath, 1, all_lines, output_dir)
         return
    
    # Write each section (from one split point to the next)
    for i, start_idx in enumerate(split_points):
        if i + 1 < len(split_points):
            end_idx = split_points[i + 1]
        else:
            end_idx = len(all_lines)
        
        section_lines = all_lines[start_idx:end_idx]
        if section_lines:
            file_count += 1
            write_part(filepath, file_count, section_lines, output_dir)

def write_part(original_filepath, part_number, lines, output_dir):
    """Writes a part of the file to a new file."""
    base, ext = os.path.splitext(os.path.basename(original_filepath))
    output_filename = os.path.join(output_dir, f"{base}_part_{part_number}{ext}")
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        out_file.writelines(lines)
    print(f"Created {output_filename}")

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
    group.add_argument("-d", "--delimiter", nargs='+', type=str, 
                        help="Delimiter string(s) to split by. Can specify multiple (e.g., -d '-- Chat:' '---').")

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
    elif args.delimiter:
        method = split_file_by_delimiter
        value = args.delimiter
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
        elif SPLIT_METHOD == "delimiter":
            method = split_file_by_delimiter
            value = DELIMITER
        
        value = SPLIT_VALUE

    if method and value:
        process_path(input_path, output_dir, method, value)
    else:
        print("Error: No split method specified. Use arguments or configure SPLIT_METHOD/SPLIT_VALUE in the script.")

if __name__ == "__main__":
    main()