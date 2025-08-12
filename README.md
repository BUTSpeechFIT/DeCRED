# DeCRED: Decoder-Centric Regularization for Encoder-Decoder Based Speech Recognition

[![Hugging Face Models](https://img.shields.io/badge/🤗-Models-blue.svg)](https://huggingface.co/collections/BUT-FIT/decred-671669beae78266f694ec918)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/XXXXX)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

**DeCRED** (**De**coder-**C**entric **R**egularization for **E**ncoder-**D**ecoder ASR) is a lightweight regularization
method for the **internal language model** (ILM) inside encoder-decoder speech recognition models.
It improves both **in-domain** and **out-of-domain** robustness without adding computational overhead.

**Key aspects:**

* **Method** – Adds *auxiliary classifiers* to intermediate decoder layers, enabling next-token prediction from
  intermediate logits.
* **Effect on ILM** – Reduces mean internal LM BPE perplexity by **36.6%** across 11 test sets.
* **In-domain WER** – Improves over the baseline in **5/7** test sets, reducing macro WER from **6.4% → 6.3%**.
* **Out-of-domain WER** – Improves in **3/4** test sets, reducing macro WER from **18.2% → 16.2%** (≈2.0 absolute
  points).
* **Single-domain gains** – On TEDLIUM3, achieves **7.0% WER**, surpassing the baseline and encoder-centric InterCTC by
  0.6% and 0.5%, respectively.
* **Competitiveness** – Matches or beats much larger models like **OWSM v3.1** and **Whisper-medium**, despite using
  less training data and having fewer parameters.
* **Domain adaptation** – A simple adaptation scheme further improves out-of-domain WER by **0.3 points**.

**Limitations:**

* Trained on **English only** (direct multilingual comparison not possible).
* Experiments scaled only to **6k hours** of training data and **172M parameters**.
* Gains are smaller when using large-beam beam-search decoding (with added inference cost).

---

## Results

### In-Domain WER

| Model              | CV-13 | SB eval2000 | LS clean | LS other | TEDLIUM3 | VoxPopuli | WSJ | Macro Avg. |
|--------------------|-------|-------------|----------|----------|----------|-----------|-----|------------|
| ED (baseline)      | 11.9  | 9.2         | 2.5      | 5.7      | 6.6      | 7.5       | 1.8 | 6.4        |
| DeCRED (baseline)  | 12.0  | 9.4         | 2.4      | 5.5      | 6.3      | 7.3       | 1.5 | 6.3        |
| DeCRED (per-token) | 12.2  | 9.1         | 2.3      | 5.5      | 5.7      | 7.3       | 1.5 | 6.2        |
| Whisper medium     | 12.4  | 14.7        | 3.0      | 5.9      | 4.2      | 8.0       | 3.2 | 7.3        |
| OWSM v3.1          | 12.9  | 11.2        | 2.4      | 5.0      | 5.0      | 8.5       | 3.5 | 6.9        |

### Out-of-Domain WER

| Model              | FLEURS | AMI ihm | Gigaspeech | Earnings-22 | Macro Avg. |
|--------------------|--------|---------|------------|-------------|------------|
| ED (baseline)      | 6.4    | 24.8    | 20.1       | 21.4        | 18.2       |
| DeCRED (baseline)  | 6.7    | 22.1    | 16.9       | 19.0        | 16.2       |
| DeCRED (per-token) | 6.7    | 21.9    | 16.7       | 18.3        | 15.9       |
| OWSM v3.1          | 7.2    | 23.3    | 19.2       | 14.0        | 15.9       |
| Whisper medium     | 4.5    | 16.6    | 13.8       | 11.7        | 11.7       |

### ILM Perplexity

| Model             | CV-13 | LS clean | LS other | SB eval2000 | TEDLIUM3 | VoxPopuli | WSJ   | FLEURS | AMI-ihm | Gigaspeech | Earnings-22 |
|-------------------|-------|----------|----------|-------------|----------|-----------|-------|--------|---------|------------|-------------|
| ED (baseline)     | 455.8 | 459.8    | 473.3    | 474.0       | 297.6    | 286.2     | 676.8 | 306.7  | 537.8   | 297.7      | 592.1       |
| DeCRED (baseline) | 215.7 | 209.0    | 197.5    | 271.6       | 140.4    | 141.0     | 723.2 | 161.1  | 310.4   | 134.1      | 266.7       |

---

## Models on Hugging Face

* [ED Base](https://huggingface.co/BUT-FIT/ED-base)
* [ED Small](https://huggingface.co/BUT-FIT/ED-small)
* [DeCRED Base](https://huggingface.co/BUT-FIT/DeCRED-base)
* [DeCRED Small](https://huggingface.co/BUT-FIT/DeCRED-small)

---

## 🔍 Inference

You can try DeCRED in two ways:

* **Cloud demo** → [Hugging Face Space](https://huggingface.co/spaces/BUT-FIT/DeCRED-ASR) (runs on Hugging Face’s
  free-tier hardware, slower for long audio).
* **Local demo** → [`demo.ipynb`](demo.ipynb) (runs the [🤗 Transformers
  `pipeline` for ASR](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)
  on your own machine using the downloaded model).

> 💡 The local notebook requires Python, PyTorch, `transformers`, and `torchaudio` installed.
> Running locally avoids the hardware limits of the free cloud Space and gives full control over inference speed and
> resources.

Example snippet from the notebook:

```python
from transformers import pipeline

model_id = "BUT-FIT/DeCRED-base"
pipe = pipeline("automatic-speech-recognition", model=model_id, feature_extractor=model_id, trust_remote_code=True)
# In newer versions of transformers (>4.31.0), there is a bug in the pipeline inference type.
# The warning can be ignored.
pipe.type = "seq2seq"
```

---

## 🏋️ Training

**Full training requires following the complete recipe** provided in the [`recipes/`](recipes) folder.
The process includes environment setup, dataset preparation, tokenizer training, model initialization, and full training
scripts.

**Steps:**

1. **Clone and set up environment**

   ```bash
   git clone https://github.com/BUTSpeechFIT/DeCRED.git
   cd DeCRED

   python -m venv decred_venv
   source decred_venv/bin/activate

   git submodule init
   git submodule update
   cd huggingface_asr
   pip install -r requirements.txt
   cd ..
   ```

2. **Configure environment variables** in `env.sh`

   ```bash
   source decred_venv/bin/activate
   export PROJECT="DeCRED"
   export WORK_DIR="/path/to/DeCRED"
   export HF_HOME="${WORK_DIR}/huggingface_cache"
   export OMP_NUM_THREADS=64
   export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/huggingface_asr"
   ```

3. **Prepare datasets**

* Update the paths to **WSJ** and **Fisher** datasets in Kaldi format inside [`recipes/datasets.json`](recipes/datasets.json).
* If you do not have local copies of these datasets:
  * Remove their entries from `datasets.json`, **or**
  * Use the already pruned [`recipes/datasets_hf.json`](recipes/datasets_hf.json), which contains only datasets available on the [Hugging Face Hub](https://huggingface.co/datasets) and requires no local copies.
* Run the data preparation script. *(Optionally, store the prepared dataset using the `--dump_prepared_dataset_to` argument to speed up future loading.)*

   ```bash
   sbatch recipes/data_prep.sh
   ```

4. **Train tokenizer** (optional if using existing tokenizer)

   ```bash
   sbatch recipes/tokenizer.sh
   ```

5. **Initialize model configs** (optional if using existing models)

   ```bash
   sbatch recipes/initialize_models.sh
   ```

6. **Run training**

   ```bash
   sbatch recipes/decred_base.sh
   ```

> 📄 See [`recipes/`](recipes) for alternative configurations (small/base models, domain adaptation, etc.).

---

## Citation

```bibtex
@inproceedings{polok2025decred,
  title        = {{DeCRED}: Decoder-Centric Regularization for Encoder-Decoder Based Speech Recognition},
  author       = {Polok, Alexander and Kesiraju, Santosh and Bene{\v s}, Karel and Yusuf, Bolaji and Burget, Luk{\'a}{\v s} and {\v C}ernock{\'y}, Jan},
  booktitle    = {2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},  
  year         = {2025},
}
```

---

## Contact

Questions? → [ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz)

Contributions welcome! Please open an issue or PR.
