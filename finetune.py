# ============================================================
# LoRA Fine-tuning — Qwen2.5-3B-Instruct
# Colab 셀별로 나눠서 실행하세요.
# ============================================================

# ── 셀 1. 패키지 설치 ────────────────────────────────────────
# !pip install transformers peft trl accelerate bitsandbytes datasets

# ── 셀 2. 데이터 로드 ────────────────────────────────────────
import json
from datasets import Dataset

with open("training_data.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

dataset = Dataset.from_list(raw)
print(f"훈련 데이터: {len(dataset)}개")
print(dataset[0])

# ── 셀 3. 모델 & 토크나이저 로드 (4-bit 양자화) ──────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
print("모델 로드 완료")

# ── 셀 4. LoRA 설정 ──────────────────────────────────────────
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 셀 5. 데이터 포맷 변환 ───────────────────────────────────
def format_example(example):
    """messages 리스트를 Qwen chat template 문자열로 변환"""
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

formatted = dataset.map(format_example)
print("포맷 예시:")
print(formatted[0]["text"])

# ── 셀 6. 파인튜닝 실행 ──────────────────────────────────────
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./finetuned-review-reply",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="none",
    max_seq_length=512,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted,
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()
print("파인튜닝 완료!")

# ── 셀 7. LoRA 어댑터 저장 ───────────────────────────────────
model.save_pretrained("./finetuned-review-reply/adapter")
tokenizer.save_pretrained("./finetuned-review-reply/adapter")
print("어댑터 저장 완료: ./finetuned-review-reply/adapter")

# ── 셀 8. 파인튜닝된 모델로 테스트 ──────────────────────────
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype="auto", device_map="auto"
)
ft_model = PeftModel.from_pretrained(base_model, "./finetuned-review-reply/adapter")
ft_tokenizer = AutoTokenizer.from_pretrained("./finetuned-review-reply/adapter")

ft_pipe = pipeline(
    "text-generation",
    model=ft_model,
    tokenizer=ft_tokenizer,
    device_map="auto",
)

test_messages = [
    {"role": "system", "content": "당신은 소상공인 가게의 전문 고객 응대 담당자입니다. 고객 리뷰에 대해 따뜻하고 진심 어린 톤으로, 2~3문장 이내의 한국어 답변을 작성합니다."},
    {"role": "user",   "content": "리뷰: \"음식은 늦게 나오고 다 식어서 왔네요. 맛도 특별히 좋은지 모르겠고 양도 가격에 비해 너무 적습니다. 직원한테 문의했는데 대답도 퉁명스럽고 귀찮아하는 느낌이라 기분만 상했어요.\""},
]

output = ft_pipe(test_messages, do_sample=True, temperature=0.3, max_new_tokens=200)
generated = output[0]["generated_text"]
reply = generated[-1]["content"] if isinstance(generated, list) else generated
print("\n파인튜닝 모델 답변:")
print(reply)
