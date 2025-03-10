import os
import glob
import re
import torch
import torchaudio
import textgrid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE

# ==================== Configuration ====================

# Path configurations
root_folder = "phoneme_codec_visualization"
data_folder = root_folder
model_name = "SpeechTokenizer"
model_type = "hubert_avg_16k"
save_dir = os.path.join(root_folder, "results", model_name, model_type)
audio_dir = os.path.join(data_folder, "LJSpeech-1.1/wavs_16k")
textgrid_dir = os.path.join(root_folder, "textgrids")

# Audio processing parameters
target_sample_rate = 16000  # 16kHz
hop_length = 320  # Set according to the actual model
time_per_step = hop_length / target_sample_rate

# ==================== Load Model ====================

def load_model():
    try:
        from speechtokenizer import SpeechTokenizer
        import nlp2
    except ImportError:
        raise Exception("Please install SpeechTokenizer first. pip install -U speechtokenizer")
    
    # Download configuration and checkpoint
    nlp2.download_file(
        'https://huggingface.co/fnlp/SpeechTokenizer/raw/main/speechtokenizer_hubert_avg/config.json',
        'speechtokenizer_hubert_avg')
    config_path = "speechtokenizer_hubert_avg/config.json"
    nlp2.download_file(
        'https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt',
        "speechtokenizer_hubert_avg")
    ckpt_path = "speechtokenizer_hubert_avg/SpeechTokenizer.pt"
    
    # Load the model
    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    model = model.cuda()
    model.eval()
    print("Model loaded successfully.")
    return model

# ==================== Data Preparation ====================

def get_file_lists(audio_dir, textgrid_dir):
    audio_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
    textgrid_files = sorted(glob.glob(os.path.join(textgrid_dir, '*.TextGrid')))
    print(f"Number of audio files: {len(audio_files)}")
    print(f"Number of TextGrid files: {len(textgrid_files)}")
    return audio_files, textgrid_files

def extract_phoneme_set(textgrid_files):
    phoneme_set = set()
    for textgrid_path in textgrid_files:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        phoneme_tier = tg.getFirst('phones')
        for interval in phoneme_tier:
            phoneme_label = interval.mark.strip()
            if phoneme_label != '':
                phoneme_set.add(phoneme_label)
    print(f"Number of unique phonemes: {len(phoneme_set)}")
    return phoneme_set

# ==================== Audio Processing ====================

def process_audio_files(model, audio_files):
    all_codes = []
    for file_path in tqdm(audio_files, desc='Processing audio files'):
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        input_audio = waveform.unsqueeze(0).cuda()

        # Get model codes
        with torch.no_grad():
            codes = model.encode(input_audio, n_q=1).squeeze(0).squeeze(0).cpu().numpy()
            all_codes.append(codes)
        del input_audio
    return all_codes

# ==================== Build Co-occurrence Matrix ====================

def build_cooccurrence_matrix(all_codes, audio_files, textgrid_dir, phoneme_to_idx):
    max_code_index = int(max([codes.max() for codes in all_codes]))
    print(f"Maximum code index: {max_code_index}")

    co_occurrence_matrix = np.zeros((len(phoneme_to_idx), max_code_index + 1), dtype=np.int32)

    for idx, file_path in tqdm(enumerate(audio_files), desc='Building co-occurrence matrix', total=len(audio_files)):
        codes = all_codes[idx]
        num_codes = len(codes)
        base_name = os.path.basename(file_path).replace('.wav', '')
        textgrid_path = os.path.join(textgrid_dir, f'{base_name}.TextGrid')
        if not os.path.exists(textgrid_path):
            print(f"TextGrid file not found: {textgrid_path}")
            continue

        tg = textgrid.TextGrid.fromFile(textgrid_path)
        phoneme_tier = tg.getFirst('phones')
        for interval in phoneme_tier:
            phoneme_label = interval.mark.strip()
            if phoneme_label == '':
                continue
            start_time = interval.minTime
            end_time = interval.maxTime
            start_idx = int(np.ceil(start_time / time_per_step))
            end_idx = int(np.floor(end_time / time_per_step))
            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, num_codes - 1)
            phoneme_codes = codes[start_idx:end_idx + 1]
            phoneme_index = phoneme_to_idx.get(phoneme_label)
            if phoneme_index is None:
                continue
            for code in phoneme_codes:
                co_occurrence_matrix[phoneme_index, int(code)] += 1
    return co_occurrence_matrix

# ==================== Merge Phonemes ====================

def merge_phonemes(phoneme_list, phoneme_to_idx, co_occurrence_matrix):
    phoneme_suffix_pattern = re.compile(r'^(.*?)([012])$')
    merged_phoneme_mapping = {}
    merged_phoneme_set = set()

    for phoneme in phoneme_list:
        match = phoneme_suffix_pattern.match(phoneme)
        if match:
            prefix, suffix = match.groups()
            if prefix.isalpha():
                merged_phoneme = prefix
            else:
                merged_phoneme = phoneme
        else:
            merged_phoneme = phoneme
        merged_phoneme_mapping[phoneme] = merged_phoneme
        merged_phoneme_set.add(merged_phoneme)

    merged_phoneme_list = sorted(merged_phoneme_set)
    merged_phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(merged_phoneme_list)}

    # Create a mapping from old indices to new indices
    old_idx_to_new_idx = {}
    for phoneme in phoneme_list:
        old_idx = phoneme_to_idx[phoneme]
        merged_phoneme = merged_phoneme_mapping[phoneme]
        new_idx = merged_phoneme_to_idx[merged_phoneme]
        old_idx_to_new_idx[old_idx] = new_idx

    # Merge rows of the co-occurrence matrix
    num_codes = co_occurrence_matrix.shape[1]
    co_occurrence_matrix_new = np.zeros((len(merged_phoneme_list), num_codes), dtype=co_occurrence_matrix.dtype)

    for old_idx in range(co_occurrence_matrix.shape[0]):
        new_idx = old_idx_to_new_idx[old_idx]
        co_occurrence_matrix_new[new_idx, :] += co_occurrence_matrix[old_idx, :]

    print(f"Number of phonemes after merging: {len(merged_phoneme_list)}")
    return merged_phoneme_list, co_occurrence_matrix_new

# ==================== Heatmap Visualization ====================

def visualize_heatmap(co_occurrence_matrix, phoneme_list):
    # 1. Calculate the maximum co-occurrence value and corresponding phoneme index for each code index
    max_values = np.max(co_occurrence_matrix, axis=0)
    code_to_phoneme_idx = np.argmax(co_occurrence_matrix, axis=0)
    code_to_phoneme_idx[max_values == 0] = -1  # Unassigned code indices

    # 2. Group code indices according to their assigned phonemes
    code_indices_grouped = {}
    unassigned_code_indices = []

    for code_idx in range(co_occurrence_matrix.shape[1]):
        phoneme_idx = code_to_phoneme_idx[code_idx]
        if phoneme_idx == -1:
            unassigned_code_indices.append((code_idx, 0))
            continue
        co_occurrence_count = co_occurrence_matrix[phoneme_idx, code_idx]
        if phoneme_idx not in code_indices_grouped:
            code_indices_grouped[phoneme_idx] = []
        code_indices_grouped[phoneme_idx].append((code_idx, co_occurrence_count))

    # 3. Sort code indices for each phoneme
    ordered_code_indices = []
    for phoneme_idx in range(len(phoneme_list)):
        if phoneme_idx in code_indices_grouped:
            codes_with_counts = code_indices_grouped[phoneme_idx]
            sorted_codes = sorted(codes_with_counts, key=lambda x: -x[1])
            sorted_code_indices = [code_idx for code_idx, _ in sorted_codes]
            ordered_code_indices.extend(sorted_code_indices)
        else:
            print(f"Phoneme '{phoneme_list[phoneme_idx]}' has no assigned code indices.")

    # Add unassigned code indices to the end
    if unassigned_code_indices:
        print(f"There are {len(unassigned_code_indices)} unassigned code indices, adding them to the end of the list.")
        unassigned_code_indices_sorted = [code_idx for code_idx, _ in unassigned_code_indices]
        ordered_code_indices.extend(unassigned_code_indices_sorted)
    else:
        print("All code indices have been assigned.")

    # 4. Reorder the columns of the co-occurrence matrix
    co_occurrence_matrix_reordered = co_occurrence_matrix[:, ordered_code_indices]

    # 5. Normalize the columns
    column_sums = co_occurrence_matrix_reordered.sum(axis=0, keepdims=True)
    column_sums[column_sums == 0] = 1  # Avoid division by zero
    co_occurrence_normalized = co_occurrence_matrix_reordered / column_sums

    # 6. Check if the value on the "staircase" line in each column is the maximum value
    print("Checking if the value on the staircase line in each column is the maximum:")
    for idx, code_idx in enumerate(ordered_code_indices):
        phoneme_idx = code_to_phoneme_idx[code_idx]
        column_values = co_occurrence_normalized[:, idx]
        max_value = np.max(column_values)
        if phoneme_idx == -1:
            print(f"Column {idx} (Code index {code_idx}): Unassigned code index.")
        else:
            staircase_value = co_occurrence_normalized[phoneme_idx, idx]
            if staircase_value == max_value:
                print(f"Column {idx} (Code index {code_idx}): Staircase value is the max in this column ({max_value:.4f}).")
            else:
                print(f"Column {idx} (Code index {code_idx}): Staircase value {staircase_value:.4f} is not the max ({max_value:.4f}).")

    # 7. Plot the heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(co_occurrence_normalized, cmap='Blues',
                xticklabels=False, yticklabels=phoneme_list, cbar=False)
    plt.xlabel('Code index')
    plt.ylabel('Phoneme')
    plt.title('Co-occurrence matrix heatmap of code indices and phonemes')
    plt.tight_layout()
    plt.savefig('co_occurrence_heatmap.png', dpi=300)
    plt.close()

    return code_to_phoneme_idx

# ==================== T-SNE Visualization ====================

def visualize_tsne(co_occurrence_matrix, code_to_phoneme_idx, phoneme_list):
    # Prepare code feature vectors
    code_features = co_occurrence_matrix.T
    code_features_sum = code_features.sum(axis=1, keepdims=True)
    code_features_sum[code_features_sum == 0] = 1  # Avoid division by zero
    code_features_normalized = code_features / code_features_sum
    code_features_normalized = np.nan_to_num(code_features_normalized)

    # Perform T-SNE analysis
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    code_embeddings_2d = tsne.fit_transform(code_features_normalized)
    np.save('tsne_embeddings.npy', code_embeddings_2d)

    # Assign colors to each code index based on their assigned phoneme
    num_phonemes = len(phoneme_list)
    palette = sns.color_palette("hls", num_phonemes + 1)  # Extra color for unassigned code indices
    colors = []
    for i in range(len(code_to_phoneme_idx)):
        phoneme_idx = code_to_phoneme_idx[i]
        if phoneme_idx == -1:
            colors.append(palette[-1])  # Gray for unassigned code indices
        else:
            colors.append(palette[phoneme_idx])

    # Plot the T-SNE scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(code_embeddings_2d[:, 0], code_embeddings_2d[:, 1], c=colors, s=50)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.title('Visualization of code indices using T-SNE')
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300)
    plt.close()

# ==================== Main Function ====================

def main():
    os.makedirs(save_dir, exist_ok=True)  # Recursively create directories if they do not exist
    os.chdir(save_dir)  # Change the current working directory
    # Load the model
    model = load_model()

    # Get file lists
    audio_files, textgrid_files = get_file_lists(audio_dir, textgrid_dir)

    # Extract phoneme set
    phoneme_set = extract_phoneme_set(textgrid_files)
    phoneme_list = sorted(phoneme_set)
    phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(phoneme_list)}

    # Process audio files and get codes
    all_codes = process_audio_files(model, audio_files)

    # Build co-occurrence matrix
    co_occurrence_matrix = build_cooccurrence_matrix(all_codes, audio_files, textgrid_dir, phoneme_to_idx)

    np.save('co_occurrence_matrix.npy', co_occurrence_matrix)
    # co_occurrence_matrix = np.load('co_occurrence_matrix.npy')
    
    # Merge phonemes
    phoneme_list, co_occurrence_matrix = merge_phonemes(phoneme_list, phoneme_to_idx, co_occurrence_matrix)
    phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(phoneme_list)}

    # Heatmap visualization
    code_to_phoneme_idx = visualize_heatmap(co_occurrence_matrix, phoneme_list)

    # T-SNE visualization
    visualize_tsne(co_occurrence_matrix, code_to_phoneme_idx, phoneme_list)

if __name__ == '__main__':
    main()