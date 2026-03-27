# RenAIssance Printed OCR – GSoC 2026 Test I



This repository contains my solution for \*\*Specific Test I: Optical Character Recognition of printed sources\*\* for the RenAIssance GSoC 2026 project.  

The goal is to build an OCR pipeline for early modern printed sources using a convolutional‑recurrent model and a late‑stage LLM clean‑up step.
The repo contains data preparation utilities, training and evaluation scripts, and notebooks demonstrating the full workflow.

## Features

- Line‑level OCR for early‑modern Spanish printed sources using a CRNN model.
- Configurable training pipeline (batch size, image size, augmentations, optimizer, scheduler, etc.).
- Evaluation scripts with character‑ and word‑level metrics.
- Optional LLM‑based post‑processing for spelling and normalization experiments.
- Reproducible experiments aligned with the GSoC 2026 test description.


## 1. Project overview



 **Task:** Recognize the main printed text in scanned early modern sources, ignoring marginalia and decorations.

 **Model:** Convolutional‑Recurrent Neural Network (CRNN) with CTC loss for line‑level OCR.

 **Baselines:** Tesseract OCR on preprocessed images.

 **LLM step:** A large language model is used after OCR to correct obvious recognition errors while preserving historical spelling.

 **Metrics:** Character Error Rate (CER) and Word Error Rate (WER).



## 2. Repository structure



```text

.

├── notebooks/

│   ├── 00\_data\_preparation.ipynb   # PDF → pages → lines + CSV creation

│   ├── 01\_tesseract\_baseline.ipynb # baseline OCR + CER/WER

│   ├── 02\_crnn\_training.ipynb      # CRNN training and evaluation

│   └── 03\_llm\_postprocessing.ipynb # LLM clean‑up and final metrics

├── src/

│   ├── data.py                     # datasets, preprocessing, line segmentation

│   ├── models.py                   # CRNN model definition

│   ├── train.py                    # training loop utilities

│   └── eval.py                     # evaluation utilities (CER, WER)

│   └── llm_clean.py                # clean up OCR output using  LLM 

├── configs/

│   └── default.yaml                # paths, hyperparameters, LLM settings

├── data/                           # (ignored) PDFs, images, CSVs

├── checkpoints/                    # (ignored) trained model weights

├── requirements.txt

├── README.md

└── .gitignore

```

## Results

| System              | CER    | WER   |
|---------------------|------- |-------|
| Tesseract baseline  | 359.68 | 255.29 |
| CRNN                | 0.832  | 1.021 |
| CRNN + LLM decoding | 0.833  | 1.020 |

CER = Character Error Rate, WER = Word Error Rate on the test set.

## Installation

```bash
git clone https://github.com/RavinduFernando/RenAIssance-ocr-test2.git
cd RenAIssance-ocr-test2

# Create and activate a virtual environment (example with venv)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### I) Prepare data

Adjust the paths in the data‑preparation notebook:

```bash
jupyter notebook notebooks/00_data_preparation.ipynb
```

This step assumes you have access to the underlying page images and PDFs locally; they are not stored in this repository.

### II) Train the OCR model

From the project root:

```bash
python src/train.py --config configs/default.yaml
```

This will:

- Load the train/validation splits from `data/`.
- Instantiate the CRNN model.
- Save model checkpoints into `checkpoints/`.

### III) Evaluate

```bash
python src/eval.py --config configs/default.yaml --checkpoint checkpoints/crnn_best.pt
```

This reports character‑ and word‑level accuracy on the test split.

### IV) LLM‑based post‑processing (optional)

Open:

```bash
jupyter notebook notebooks/03_llm_postprocessing.ipynb
```

This notebook shows how to apply an LLM on top of the raw OCR output for correction and normalization experiments.

## Data availability and limitations

- The repository includes **only lightweight metadata**: line‑level CSV files and transcription documents derived from several early‑modern Spanish printed sources.
- The original high‑resolution PDFs and page images are **not** committed to GitHub because they exceed size limits and may be subject to separate licensing terms.
- To fully reproduce the experiments, you need local access to equivalent page images or to the same printed sources from your institution / project.
- Checkpoints are ignored by default; users are expected to train their own models or download weights from an external link if provided.

## License

This project is released under the **Apache‑2.0** license. See `LICENSE` for details.




