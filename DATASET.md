# Dataset Manifest Structure

This document describes the required structure of the input evaluation dataset manifest for the `uwebasr-calibrate` tool. For installation and usage instructions, please refer to [README.md](README.md).

The manifest file can be provided in **JSONL**, **CSV**, or a **JSON array** format.

---

## Required Fields

For each utterance (row), the following fields are required:

- **`audio_path`**: The file path to the source audio file. Relative paths are resolved relative to the directory containing the manifest file.
- **`reference`**: The reference transcript (ground-truth transcription) of the audio segment.

*Note on `utt_id`: While `utt_id` (a stable unique utterance identifier) is highly recommended, if it is missing, the tool will automatically derive it from the stem of the audio file name. The tool will exit with a fatal error if the resulting utterance IDs are not unique.*

---

## Optional Fields

To support speaker-disjoint splitting and realistic long-recording evaluations (`test_real`), you can provide these optional fields:

- **`speaker_id`**: A stable identifier for the speaker. Used as the group key to partition train/test splits, ensuring no speaker's voice leaks between training and test sets. If not provided, the script will attempt to extract it from the `utt_id` using the regular expression `(?<!\d)(\d{4,5})(?!\d)` (e.g. `12345_A_001_000123` -> speaker `12345`).
- **`video_id`**: A group identifier representing a single long recording (e.g., a video or program). Used during `test_real` aggregation. If not provided, it falls back to the `video_id` patterns or the `utt_id` itself.
- **`split`**: An explicit split assignment. Supported values are `"train"` or `"test"`. If provided, the tool will skip its random speaker-disjoint partitioning and assign utterances directly to the train or test set as specified.

---

## Supported Formats

### 1. JSONL (JSON Lines) Format

Each line of the file must be a valid JSON object.

```json
{"utt_id": "00026_M_003_0065", "audio_path": "audio/00026_M_003_0065.wav", "reference": "yeah exactly that I wanna talk about it", "speaker_id": "00026", "video_id": "00026_M_003"}
{"utt_id": "00026_M_003_0068", "audio_path": "audio/00026_M_003_0068.wav", "reference": "so finally when I moved to", "speaker_id": "00026", "video_id": "00026_M_003"}
```

### 2. CSV Format

Must contain columns matching the field names.

```csv
utt_id,audio_path,reference,speaker_id,video_id
00026_M_003_0065,audio/00026_M_003_0065.wav,yeah exactly that I wanna talk about it,00026,00026_M_003
00026_M_003_0068,audio/00026_M_003_0068.wav,so finally when I moved to,00026,00026_M_003
```

### 3. MALACH-style JSON Array Format

For compatibility with standard MALACH datasets, a JSON array containing `filename` and `text` fields is also accepted:

```json
[
  {
    "filename": "/path/to/dataset/audio/00026_M_003_0065.wav",
    "text": "yeah exactly that I wanna talk about it"
  },
  {
    "filename": "/path/to/dataset/audio/00026_M_003_0068.wav",
    "text": "so finally when I moved to"
  }
]
```

When this format is detected, the fields are mapped internally as:
- `utt_id` = stem of the `filename` (e.g. `00026_M_003_0065`)
- `audio_path` = `<manifest_dir>/audio/<basename of filename>`
- `reference` = `text`
