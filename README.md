# NLP Project 1 — Korean Review Reply Assistant

## Overview

This project implements an AI-powered review reply assistant for small business owners in Korea.

The system helps restaurant, café, and salon owners automatically generate professional responses to customer reviews using locally executable NLP models without any external API dependency.

---

## Features

- Korean customer review sentiment classification  
- Review type detection using sentence embeddings  
- AI-generated professional Korean replies  
- Multi-complaint review detection  
- Optional LoRA fine-tuning support  
- Gradio web interface  

---

## Project Structure

```text
nlp_project1/
├── finetune.py
├── README.md
├── requirements.txt
├── review_reply_assistant.ipynb
└── training_data.json
```

## Models Used

### Stage 1 — Sentiment Classification
- `lxyuan/distilbert-base-multilingual-cased-sentiments-student`

### Stage 2 — Review Type Classification
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

### Stage 3 — Reply Generation
- `Qwen/Qwen2.5-7B-Instruct`

---

## Supported Categories

- `delivery_issue`
- `food_quality`
- `service_complaint`
- `hygiene_complaint`
- `unverified_claim`
- `compliment`
- `multi_complaint`

---

## How to Run (Google Colab Recommended)

### 1. Clone Repository

```python
!git clone https://github.com/YOUR_ID/nlp_project1.git
%cd nlp_project1
```

### Optional Fine-tuning

```python
!python finetune.py
```

## Example Input
음식이 한 시간이나 늦게 왔고 완전히 식어있었어요.

##Example Output
고객님, 배달이 지연되고 음식이 식은 상태로 전달된 점 진심으로 사과드립니다.
해당 문제를 내부적으로 확인하여 재발하지 않도록 개선하겠습니다.
더 나은 서비스로 보답드릴 수 있도록 노력하겠습니다.

##Notes
- All models run locally without external API calls.
- Designed for academic demonstration purposes.
- Google Colab GPU runtime is recommended.
