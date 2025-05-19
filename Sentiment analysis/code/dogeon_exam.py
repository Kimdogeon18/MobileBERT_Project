import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from transformers import get_linear_schedule_with_warmup, logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import chardet  # chardet 라이브러리로 파일 인코딩 자동 감지

# 0. GPU 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 1. 경고 메시지 제거
logging.set_verbosity_error()

# 2. 데이터 로드 (실제 컬럼명 사용)
path = "cleaned_WELFake_2000.csv"

# 인코딩 자동 감지하여 CSV 파일 읽기
with open(path, 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

# 파일을 감지된 인코딩으로 읽기
df = pd.read_csv(path, encoding=encoding)

# 💡 'Headline' 컬럼이 없으므로 실제 컬럼명으로 수정
data_X = df['title'].astype(str).values  # 'title' 컬럼 사용 (이름 다르면 수정해야 함)
labels = df['label'].values  # 'label' 컬럼 사용 (Fake=1, Real=0)

# 3. NaN 값 처리
labels = labels.astype(float)  # labels를 float로 변환하여 NaN 처리 가능하도록 함
labels = np.nan_to_num(labels, nan=0)  # NaN 값을 0으로 대체

# NaN 값이 없으면 확인 메시지 출력
if np.any(np.isnan(labels)):
    print("NaN 값이 여전히 존재합니다!")

print(f"데이터 샘플 수: {len(data_X)}")
print("헤드라인 예시:", data_X[:3])
print("레이블 예시:", labels[:3])

# 4. 토큰화
tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased', do_lower_case=True)
inputs = tokenizer(list(data_X), truncation=True, max_length=64, padding="max_length", return_tensors="pt")

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 5. 데이터 분할
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    input_ids, labels, test_size=0.2, random_state=2025
)
train_masks, val_masks = train_test_split(attention_mask, test_size=0.2, random_state=2025)

# 6. 텐서 변환 (GPU/CPU 이동)
# train_labels에 대해 NaN 값이 없다고 가정하고, 타입을 int로 변환
train_inputs, train_labels, train_masks = train_inputs.to(device), torch.tensor(train_labels, dtype=torch.long).to(device), train_masks.to(device)
val_inputs, val_labels, val_masks = val_inputs.to(device), torch.tensor(val_labels, dtype=torch.long).to(device), val_masks.to(device)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
val_data = TensorDataset(val_inputs, val_masks, val_labels)

train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=8)
val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=8)

# 7. 모델 설정 (가짜 뉴스 이진 분류)
model = MobileBertForSequenceClassification.from_pretrained('mobilebert-uncased', num_labels=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 4)

# 8. 학습 루프
epochs = 5
epoch_results = []

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    total_train_correct = 0  # 학습 정확도 계산을 위한 변수
    total_train_count = 0  # 전체 학습 데이터 수

    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")

    for batch in progress_bar:
        batch_ids, batch_mask, batch_labels = batch
        optimizer.zero_grad()

        output = model(batch_ids, attention_mask=batch_mask, labels=batch_labels)
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

        # 학습 정확도 계산
        preds = torch.argmax(output.logits, dim=1)
        total_train_correct += (preds == batch_labels).sum().item()
        total_train_count += batch_labels.size(0)

        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_accuracy = total_train_correct / total_train_count  # 정확도 계산

    # 9. 검증 평가
    model.eval()
    val_correct, val_total = 0, 0

    for batch in val_dataloader:
        batch_ids, batch_mask, batch_labels = batch
        with torch.no_grad():
            logits = model(batch_ids, attention_mask=batch_mask).logits
        preds = torch.argmax(logits, dim=1)
        val_correct += (preds == batch_labels).sum().item()
        val_total += batch_labels.size(0)

    val_accuracy = val_correct / val_total
    epoch_results.append((avg_train_loss, train_accuracy, val_accuracy))

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, Validation Accuracy = {val_accuracy:.4f}")

# 10. 모델 저장
model.save_pretrained("mobilebert_fake_news_headline_detector1")
print("모델 저장 완료!")
