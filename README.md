# AutoLyrics: Whisper Fine-Tuning for Lyrics Transcription 🎵🎙️

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green.svg)

> **AutoLyrics** is an automated pipeline for fine-tuning OpenAI's `whisper-small` model specifically for Singing/Lyrics Transcription (SLT). Leveraging Low-Rank Adaptation (LoRA) via Hugging Face's PEFT library, the pipeline achieves highly efficient parameter-efficient fine-tuning (PEFT), significantly improving Word Error Rate (WER) over the zero-shot baseline.

## 📑 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Model & Training Architecture](#-model--training-architecture)
- [Results & Performance](#-results--performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)

---

## 🔍 Overview

Singing transcription is notoriously difficult for standard ASR models due to extreme pitch variations, background instrumentation, and stylistic pronunciations. 

This repository provides a complete, end-to-end Jupyter Notebook pipeline to:
1. Load and thoroughly preprocess raw audio & lyric pairs.
2. Inject LoRA adapters into both the Encoder and Decoder of Whisper.
3. Train the model using gradient accumulation and mixed precision.
4. Systematically evaluate zero-shot vs. fine-tuned performance.
5. Auto-generate comprehensive qualitative and quantitative performance reports.

## 💾 Dataset

The project utilizes the [gmenon/slt-lyrics-audio](https://huggingface.co/datasets/gmenon/slt-lyrics-audio) dataset from Hugging Face.

* **Total Samples:** ~9,500
* **Filtered Samples:** ~8,985 (after duration & audio validity checks)
* **Data Split:** Handled dynamically via `singer` grouping to prevent data leakage across splits.
  * **Train:** 70%
  * **Validation:** 15%
  * **Test:** 15%

## ⚙️ Preprocessing Pipeline

To handle the highly dynamic nature of musical audio, a robust multi-stage preprocessing chain is applied prior to feature extraction:

### Audio Preprocessing (`librosa` & `pyloudnorm`)
1. **DC Offset Removal:** Centers the waveform.
2. **Silence Trimming:** Removes leading/trailing digital silence (Threshold: 30dB).
3. **Loudness Normalization:** Standardizes integrated loudness to `-23.0 LUFS` (EBU R128 standard) with a `1.0 dB` headroom, falling back to peak normalization for extreme edge-cases.

### Text Normalization
1. **Structural Stripping:** Removes bracketed annotations (e.g., `[Chorus]`, `(Verse 1)`).
2. **Lowercasing & Punctuation:** Converts to lowercase and strips all punctuation except essential apostrophes and hyphens.
3. **Whitespace Normalization:** Collapses multi-spaces.

## 🧠 Model & Training Architecture

* **Base Model:** `openai/whisper-small`
* **PEFT Method:** LoRA (Low-Rank Adaptation)
* **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`
* **LoRA Config:** `r=8`, `alpha=16`, `dropout=0.1`
* **Trainable Parameters:** ~3.24M (approx. 1.32% of total parameters)
* **Optimization:** `AdamW` optimizer with a Linear Learning Rate Scheduler + Warmup.
* **Decoding Strategy:** Greedy decoding (`num_beams=1`)

## 📊 Results & Performance

The model was evaluated against a purely zero-shot baseline using **Word Error Rate (WER)** and **Character Error Rate (CER)**. 

| Metric | Zero-Shot Baseline | Best Validation | Test Set |
| :--- | :---: | :---: | :---: |
| **WER (%)** | 74.32% | 58.45% | **60.52%** |
| **CER (%)** | 49.39% | 39.61% | **41.29%** |
| **Loss** | 6.5030 | 1.0716 | 1.1426 |

* **Relative WER Improvement:** **18.6%** over zero-shot.
* *Target >15% WER reduction successfully achieved.*

## 📁 Project Structure

Running the pipeline will automatically generate the following directory structure:

```text
.
├── autolyrics_v2_slt_preprocessed.ipynb  # Main pipeline notebook
├── checkpoints/                          # Model artifacts
│   ├── lora_best/                        # Best saved LoRA weights & config
│   ├── results.json                      # Raw validation tracking data
│   └── test_results.json                 # Final test set metrics
└── report/                               # Auto-generated performance artifacts
    ├── performance_report.png            # Visual graphs (Loss, WER, CER)
    ├── qualitative_predictions.txt       # Side-by-side REF vs HYP lyric samples
    └── report_summary.json               # Machine-readable comprehensive summary
