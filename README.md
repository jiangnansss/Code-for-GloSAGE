# GloSAGE: core research code

This preliminary release collects a subset of the GloSAGE code: binary lesion-recognition fine-tuning, GS_class2 inference, and visual-adapter training with an open causal LLM. More experiments, models, training utilities, and checkpoints will be organized and released progressively. No images, text examples, patient data, or weights are included here.

## Important reproduction status

`train_recognition.py` implements the GloPath encoder + first visual projection Linear + GELU + classifier used by GloSAGE recognition, with all model parameters unfrozen. It uses explicit user-provided splits and simplified horizontal/vertical flipping; the sampling, augmentations, and validation-based checkpoint selection are not identical to the original `clsV4/train.py` pipeline. Results should not be claimed as exact reproduction without matching the original splits and protocol.

For VQA, the original experiment's `glopath2` branch passed the 768-D GloPath encoder output directly to the adapter; it did **not** apply the GloSAGE projection. `train_vqa.py` retains this as `--visual-representation raw` and offers `projected` as a separate configuration, which applies the first Linear + GELU of the GloSAGE visual projection. The latter is not the original reported VQA configuration. This release saves adapter-only checkpoints and validates using teacher-forced loss; unlike the original training script it does not generate answers or compute test metrics at each epoch.

## Setup and files

Use Python 3.10 or later and an appropriate PyTorch/CUDA build, then install dependencies:

```bash
pip install -r requirements.txt
```

Paths below are relative to the repository root. Create the indicated directories and obtain the assets separately:

```text
data/
  GS_class2/normal/           # non-GS images
  GS_class2/sclerosed/        # GS images
  vqa.jsonl                   # your own image/question-answer records
  recognition.csv             # your own binary recognition split
weights/
  glopath_pretrained.pth
  glosage_projection.pt
  glosage_pas_gs.pth          # fine-tuned PAS [19] classifier
  glopath_pas_gs.pth
  uni_pas_gs.pth
  plip_pas_gs.pth
  conch_pas_gs.pth
  keep_pas_gs.pth
  KEEP/config.json            # downloaded KEEP configuration
outputs/
```

GloPath and GloSAGE weights: **Google Drive `xxxx`** (placeholder; links and distribution terms to be added). The public GS classification dataset AIDPATH DATASET_B can be requested/downloaded at [Mendeley Data](https://data.mendeley.com/datasets/k7nvtgn2x6/3); see its licence and usage conditions. Keep the `normal` and `sclerosed` images in their respective directories.

Other foundation model releases are [UNI](https://huggingface.co/MahmoodLab/UNI), [PLIP](https://huggingface.co/vinid/plip), [CONCH](https://huggingface.co/MahmoodLab/CONCH), and [KEEP](https://huggingface.co/MAGIC-AI4Med/KEEP). Some require access approval. Download the KEEP release (including `config.json`) into `weights/KEEP` using Hugging Face tools. The default VQA LLM is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct); obtain its model weights and tokenizer from its open release, subject to its licence. It can be downloaded automatically through Transformers or supplied as a local directory.

**Pretrained foundation weights are not the PAS-GS binary fine-tuned checkpoints.** The inference comparison below requires a separately trained PAS `[19]` recognition checkpoint for each family; merely downloading a foundation model from Hugging Face will not produce a valid result. This preliminary release does not distribute the baseline fine-tuned checkpoints.

## 1. Binary lesion recognition

Prepare `data/recognition.csv` with columns `path,label,split`, labels `0`/`1`, and `split` values `train`, `val`, `test`. Each split must contain both classes. Construct patient-disjoint splits externally; this script does not create or validate patient identifiers. The checkpoint is selected by validation ROC-AUC; the test split is reserved for downstream evaluation.

```bash
python train_recognition.py \
  --manifest data/recognition.csv \
  --encoder-weight weights/glopath_pretrained.pth \
  --projection-weight weights/glosage_projection.pt \
  --output outputs/glosage_pas_gs.pth --device cuda:0
```

The visual projection uses only its first Linear and GELU for classification; the final projection Linear is not used in this forward path. The checkpoint contains `modelT` and can be evaluated on GS_class2 with the next command.

## 2. GS_class2 inference

Place a separately fine-tuned PAS-GS classifier at each `weights/*_pas_gs.pth` path. The inference script uses strict checkpoint loading, historical 224-pixel BGR preprocessing for GloSAGE/GloPath/UNI/PLIP/CONCH and RGB for KEEP. It reports ROC-AUC, accuracy, and macro-F1. Do not use the external test set to select the best fold; select checkpoints on an independent validation set first.

```bash
python evaluate_gs.py --data data/GS_class2 --keep-config weights/KEEP/config.json \
  --checkpoint glosage=weights/glosage_pas_gs.pth \
  --checkpoint glopath=weights/glopath_pas_gs.pth \
  --checkpoint uni=weights/uni_pas_gs.pth \
  --checkpoint plip=weights/plip_pas_gs.pth \
  --checkpoint conch=weights/conch_pas_gs.pth \
  --checkpoint keep=weights/keep_pas_gs.pth \
  --output outputs/gs_class2_metrics.csv --device cuda:0
```

Alternatively open `notebooks/gs_class2_inference.ipynb`, adjust paths and execute its code cell. To evaluate a subset, supply only those `--checkpoint` arguments. Other checkpoint formats, models, and timm versions may require adaptation; strict loading intentionally fails instead of silently evaluating random weights.

## 3. Visual adapter and LLM training

Create `data/vqa.jsonl` with one JSON object per line. Each record must contain `patch_path` (an absolute path or a path relative to the JSONL file), `group_text` (the reference morphology description), and `split` (`train`, `val`, or `test`). The optional `staining` field supports `--staining pas_20x`, `masson`, or `pasm`. Prepare patient-disjoint splits externally: the script checks duplicate image paths across splits but cannot identify different images from the same patient. Do not publish private images, descriptions, or manifests.

The fixed question is `Describe the renal glomerular morphology and identified lesions in this image. Answer:`. Images use RGB, bicubic resize to 224 x 224, and ImageNet normalization, following the original VQA data loader. The image encoder and LLM are frozen; an adapter maps the image representation to eight visual prefix tokens by default. Only the adapter is optimized.

To train the original `glopath2` VQA representation (raw 768-D encoder features):

```bash
python train_vqa.py --data data/vqa.jsonl \
  --encoder-weight weights/glopath_pretrained.pth \
  --llm Qwen/Qwen2.5-0.5B-Instruct \
  --visual-representation raw --output-dir outputs/vqa_raw --device cuda:0
```

To use GloSAGE's first projection Linear + GELU instead, use a distinct output directory:

```bash
python train_vqa.py --data data/vqa.jsonl \
  --encoder-weight weights/glopath_pretrained.pth \
  --projection-weight weights/glosage_projection.pt \
  --visual-representation projected --llm Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/vqa_projected --device cuda:0
```

For offline operation, download the LLM and tokenizer in advance, pass their local directory to `--llm`, and add `--local-files-only`. The output directory contains `best.pt` (lowest validation loss), `last.pt`, and `history.json`. The checkpoints contain only the adapter weights and configuration, not the frozen encoder or LLM. The test split is kept separate and is not evaluated by this training script. Use a fresh output directory for each run to avoid overwriting previous results.

Only load checkpoints from trusted sources: PyTorch pickle-based checkpoints can execute arbitrary code.
