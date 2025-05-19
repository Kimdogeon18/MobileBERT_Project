    import torch
    import pandas as pd
    import numpy as np
    from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
    from tqdm import tqdm

    # 디바이스 설정
    GPU = torch.cuda.is_available()
    device = torch.device("cuda" if GPU else "cpu")
    print("Using device: ", device)

    # 데이터 불러오기
    data_path = "WELFake_Dataset.csv"
    df = pd.read_csv(data_path, encoding="utf-8")

    # 컬럼명을 소문자로 통일
    df.columns = df.columns.str.lower()

    # 텍스트와 라벨 추출
    data_x = data_x = df['text'].fillna("").astype(str).tolist()
    labels = df['label'].values
    print(f"총 샘플 수: {len(data_x)}")

    # 토크나이저 준비 및 토큰화
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased", do_lower_case=True)
    inputs = tokenizer(data_x, truncation=True, max_length=512, add_special_tokens=True, padding="max_length")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    print("토큰화 완료")

    # 텐서로 변환 및 DataLoader 구성
    batch_size = 8
    test_inputs = torch.tensor(input_ids)
    test_labels = torch.tensor(labels)
    test_masks = torch.tensor(attention_mask)

    test_data = torch.utils.data.TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = torch.utils.data.RandomSampler(test_data)
    test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    print("데이터셋 구축 완료")

    # 사전 학습된 모델 불러오기
    model = MobileBertForSequenceClassification.from_pretrained("./mobilebert_news_new4")
    model.to(device)
    model.eval()

    # 예측 및 평가
    test_pred = []
    test_true = []

    for batch in tqdm(test_dataloader, desc="Inferencing Full Dataset"):
        batch_ids, batch_mask, batch_labels = batch

        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            output = model(batch_ids, attention_mask=batch_mask)
        logits = output.logits
        pred = torch.argmax(logits, dim=1)

        test_pred.extend(pred.cpu().numpy())
        test_true.extend(batch_labels.cpu().numpy())

    # 정확도 계산
    test_accuracy = np.sum(np.array(test_pred) == np.array(test_true)) / len(test_pred)
    print(f"전체 데이터 {len(test_pred)}건에 대한 감성 분류 정확도: {test_accuracy:.4f}")
