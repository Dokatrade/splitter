import os

filepath = r"d:/LetsProgram/python-vscode/TEST_PROJECTS/splitter/account-huntj.txt"
delimiters = ["-- Chat:", "=== Project:", "### Account:"]

def line_contains_delimiter(line):
    return any(d in line for d in delimiters)

with open(filepath, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()

print(f"Total lines: {len(all_lines)}")

# Step 1: Find all delimiter line indices
delimiter_indices = []
for i, line in enumerate(all_lines):
    if line_contains_delimiter(line):
        delimiter_indices.append(i)

print(f"\nTotal delimiters found: {len(delimiter_indices)}")
print(f"First 20 delimiter indices: {delimiter_indices[:20]}")

# Step 2: Find real split points
split_points = []
for idx in delimiter_indices:
    if idx == 0:
        split_points.append(idx)
    else:
        prev_idx = idx - 1
        while prev_idx >= 0 and all_lines[prev_idx].strip() == '':
            prev_idx -= 1
        
        if prev_idx < 0:
            split_points.append(idx)
        else:
            prev_is_delim = line_contains_delimiter(all_lines[prev_idx])
            if not prev_is_delim:
                split_points.append(idx)

print(f"\nTotal split points: {len(split_points)}")
print(f"All split points: {split_points}")

print("\n--- Section ranges ---")
for i, start_idx in enumerate(split_points):
    if i + 1 < len(split_points):
        end_idx = split_points[i + 1]
    else:
        end_idx = len(all_lines)
    
    print(f"Section {i+1}: lines {start_idx} to {end_idx-1} (indices), length={end_idx-start_idx} lines")
    print(f"  Starts with: {repr(all_lines[start_idx][:60])}...")
    print(f"  Ends with:   {repr(all_lines[end_idx-1][:60])}...")
