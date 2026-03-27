\# RenAIssance Printed OCR – GSoC 2026 Test I



This repository contains my solution for \*\*Specific Test I: Optical Character Recognition of printed sources\*\* for the RenAIssance GSoC 2026 project.  

The goal is to build an OCR pipeline for early modern printed sources using a convolutional‑recurrent model and a late‑stage LLM clean‑up step.



\## 1. Project overview



\- \*\*Task:\*\* Recognize the main printed text in scanned early modern sources, ignoring marginalia and decorations.

\- \*\*Model:\*\* Convolutional‑Recurrent Neural Network (CRNN) with CTC loss for line‑level OCR.

\- \*\*Baselines:\*\* Tesseract OCR on preprocessed images.

\- \*\*LLM step:\*\* A large language model is used after OCR to correct obvious recognition errors while preserving historical spelling.

\- \*\*Metrics:\*\* Character Error Rate (CER) and Word Error Rate (WER).



\## 2. Repository structure



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



