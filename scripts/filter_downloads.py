def should_keep(line: str) -> bool:
    line_lower = line.lower()
    if "emg" in line_lower or "mov" in line_lower:
        return False
    if "eeg" in line_lower and "events" not in line_lower:
        return False
    return True

with open("ds005873-1.1.0.sh", "r", encoding="utf-8") as infile, open("filtered_download_script.sh", "w", encoding="utf-8") as outfile:
    for line in infile:
        if should_keep(line):
            outfile.write(line)
