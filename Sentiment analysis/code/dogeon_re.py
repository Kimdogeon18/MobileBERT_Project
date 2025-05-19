import pandas as pd

file_path = "WELFake_Dataset_Sampled.csv"  # 원본 파일 경로
utf8_file_path = "WELFake_Dataset_Sampled_utf8.csv"  # 변환된 파일 저장 경로

# Windows-1252로 읽고 UTF-8로 저장
df = pd.read_csv(file_path, encoding="ISO-8859-1")  # 혹은 'utf-8'을 시도
df.to_csv(utf8_file_path, encoding="utf-8", index=False)

print("파일이 UTF-8로 변환되어 저장되었습니다:", utf8_file_path)
