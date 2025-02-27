import os
import string

# Define the punctuation marks to be removed
punctuation = string.punctuation  # Contains all common punctuation marks

# Or explicitly specify the punctuation marks to remove
# punctuation = '.,?!"\';:-()[]{}'

# Open the metadata.csv file and read lines
with open('LJSpeech-1.1/metadata.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Prepare lists to hold the data
ids = []
transcriptions = []
normalized_transcriptions = []

# Process each line
for line_num, line in enumerate(lines, start=1):
    # Remove trailing newline and any leading/trailing whitespace
    line = line.strip()
    # Split on '|' character, maximum of 2 splits
    parts = line.split('|', 2)
    if len(parts) != 3:
        print(f"Skipping line {line_num} due to incorrect format: {line}")
        continue
    id, transcription, normalized_transcription = parts
    ids.append(id)
    transcriptions.append(normalized_transcription)

# Function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punctuation))

# Now, create .lab files
os.makedirs('lab_files', exist_ok=True)
for id, text in zip(ids, transcriptions):
    # Remove punctuation if needed
    clean_text = remove_punctuation(text)
    # Remove extra whitespace
    clean_text = ' '.join(clean_text.split())
    # Write to .lab file
    with open(f'LJSpeech-1.1/wavs_16k/{id}.lab', 'w', encoding='utf-8') as f_out:
        f_out.write(clean_text)