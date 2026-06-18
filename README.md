# NLP Project 2 — Korean Review Reply Assistant

> Automated response generation for small business owners via persona-aware prompting and LoRA fine-tuning

---

## Overview

This project implements an AI-powered review reply assistant for small business owners in Korea.
The system automatically generates professional Korean replies to customer reviews using a fully local NLP pipeline — no external API or cloud service required.

**Key upgrades in Project 2:**
- **Plan A**: Training data expanded from 30 → 56 review-response pairs
- **Plan B**: Persona-aware dynamic prompting (inspired by PARAN) — infers a response persona from sentiment × review category and injects it into the system prompt
- **LoRA fine-tuning**: Qwen2.5-3B-Instruct fine-tuned on domain-specific Korean data
- **Quantitative evaluation**: BLEU-4, ROUGE-L, BERTScore comparison (base vs. fine-tuned)

---

## Project Structure

```text
nlp-project1/
├── review_reply_assistant.ipynb  # Main pipeline (Colab)
├── training_data.json            # 56 Korean review-response pairs
├── finetune.py                   # LoRA fine-tuning script
├── requirements.txt              # Dependencies
├── eval_comparison.png           # BLEU-4 / ROUGE-L / BERTScore bar chart
├── eval_by_category.png          # Per-category BERTScore chart
├── 20211781.pdf                  # Project report (IEEE format)
├── 20211781.tex                  # LaTeX source
└── README.md
```

---

## Pipeline

| Stage | Task | Model |
|-------|------|-------|
| 1 | Sentiment Classification | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` |
| 2 | Review Type Classification | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| 3 | Persona Inference & Prompt Construction | Rule-based (sentiment × category) |
| 4 | Reply Generation | `Qwen/Qwen2.5-7B-Instruct` (4-bit quantized) |

---

## Supported Review Categories

| Category | Description |
|----------|-------------|
| `delivery_issue` | Late or missing delivery |
| `food_quality` | Taste, temperature, or portion complaints |
| `service_complaint` | Rude or slow service |
| `hygiene_complaint` | Cleanliness or food safety concerns |
| `unverified_claim` | Unverifiable or unfair accusations |
| `compliment` | Positive reviews |
| `multi_complaint` | Multiple complaint types combined |

---

## How to Run (Google Colab Recommended)

### 1. Clone Repository

```bash
git clone https://github.com/binisnil/nlp-project1.git
cd nlp-project1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Main Pipeline

Open `review_reply_assistant.ipynb` in Google Colab and run cells sequentially.

### 4. (Optional) LoRA Fine-tuning

Upload `finetune.py` and `training_data.json` to Colab, then run:

```bash
python finetune.py
```

---

## Evaluation Results

| Metric | Base Model | Fine-tuned (LoRA) | Change |
|--------|-----------|-------------------|--------|
| BLEU-4 | 0.3700 | 0.3414 | -0.029 |
| ROUGE-L | 0.1667 | 0.1667 | 0.000 |
| BERTScore (F1) | 0.7734 | 0.7841 | **+0.011** |

---

## Example

**Input review:**
```
음식이 한 시간이나 늦게 왔고 완전히 식어있었어요.
```

**Generated reply:**
```
고객님, 배달이 지연되고 음식이 식은 상태로 전달된 점 진심으로 사과드립니다.
해당 문제를 내부적으로 확인하여 재발하지 않도록 개선하겠습니다.
더 나은 서비스로 보답드릴 수 있도록 노력하겠습니다. 다음에도 방문해 주시면 감사하겠습니다.
```

---

## Notes

- All models run locally without external API calls
- Google Colab with T4 GPU is recommended
- Fine-tuning takes approximately 13 minutes on a T4 GPU
- Designed for NLP Term Project #2 (academic purposes)
