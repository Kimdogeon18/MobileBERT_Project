import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# 0. GPU 있는지 확인
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device: ", device)

# 1. 경고 제거
logging.set_verbosity_error()

# 2. 데이터 로딩
path = "reduced_WELFake_Dataset2000.csv"
df = pd.read_csv(path, encoding="utf-8")
df = df[df['text'].apply(lambda x: isinstance(x, str) and pd.notna(x))]
data_X = df['text'].values
labels = df['label'].values

print('### 데이터 샘플###')
print("기사 본문 : ", data_X[:3])
print("진짜/가짜 : ", labels[:3])

# 3. 토큰화
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased', do_lower_case=True)
inputs = tokenizer(list(data_X), truncation=True, max_length=256, add_special_tokens=True, padding="max_length", return_tensors="pt")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 4. 훈련/검증 분리
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=2025)
train_masks, val_masks, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2025)

# 5. TensorDataset, DataLoader 구성
batch_size = 8

train_data = TensorDataset(train_inputs, train_masks, torch.tensor(train_labels))
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, torch.tensor(val_labels))
val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

# 6. 모델 준비
model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=2)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epochs)

# 7. 학습 루프
epoch_results = []

for e in range(epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {e+1}")

    for batch in progress_bar:
        batch_ids, batch_mask, batch_labels = [x.to(device) for x in batch]
        model.zero_grad()
        output = model(batch_ids, attention_mask=batch_mask, labels=batch_labels)
        loss = output.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)

    # 정확도 계산
    model.eval()
    def compute_accuracy(dataloader):
        all_preds, all_labels = [], []
        for batch in dataloader:
            batch_ids, batch_mask, batch_labels = [x.to(device) for x in batch]
            with torch.no_grad():
                logits = model(batch_ids, attention_mask=batch_mask).logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
        return np.mean(np.array(all_preds) == np.array(all_labels))

    train_acc = compute_accuracy(train_dataloader)
    val_acc = compute_accuracy(val_dataloader)

    epoch_results.append((avg_train_loss, train_acc, val_acc))

# 8. 결과 출력
for i, (loss, tr_acc, val_acc) in enumerate(epoch_results, 1):
    print(f"Epoch {i}: Loss {loss:.4f}, Train Acc: {tr_acc:.4f}, Val Acc: {val_acc:.4f}")

# 9. 모델 저장
save_path = "mobilebert_news_new4"
model.save_pretrained(save_path)
print("모델 저장 완료:", save_path)