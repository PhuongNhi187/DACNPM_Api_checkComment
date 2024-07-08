import numpy as np
from fastapi import FastAPI, Query
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import pandas as pd
import re
from pyvi import ViTokenizer
from fastapi.middleware.cors import CORSMiddleware

# https://github.com/PhuongNhi187/DACNPM_Api_checkComment
app = FastAPI()

origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    # Add any other origins you need
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định nghĩa model input
class TextInput(BaseModel):
    text: str

# Load model đã được train
model_path = "DACNPM/DACNPM_Api_checkComment/fastApiProject/dacnpmh5"
model = load_model(model_path)
# Đọc các từ điển và chuẩn bị hàm tiền xử lý văn bản
correct_char_path = pd.read_table('DACNPM/DACNPM_Api_checkComment/fastApiProject/dictionary/correct_char.txt', delimiter='\t')
correct_char_dict = dict(zip(correct_char_path['char'], correct_char_path['correct_char']))

en2vi_path = pd.read_table('DACNPM/DACNPM_Api_checkComment/fastApiProject/dictionary/top_500_adjective_eng.txt', delimiter='\t')
en2vi_dict = dict(zip(en2vi_path['Adj'], en2vi_path['Mean']))

teencode_path = pd.read_table('DACNPM/DACNPM_Api_checkComment/fastApiProject/dictionary/teencode.txt', delimiter='\t')
teencode_dict = dict(zip(teencode_path['incorrect'], teencode_path['correct']))

def preprocess_text(text):
    text = text.lower()

    # Loại bỏ các từ đặc biệt
    def remove_special_words(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'\#\w+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        return text

    text = remove_special_words(text)

    # Loại bỏ các ký tự đặc biệt
    def remove_special_characters(text):
        text_pre = re.compile("[^a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđíìỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s]")
        return re.sub(text_pre, ' ', text)

    def correct_char(text):
        words = text.split()
        for word in words:
            if word in correct_char_dict:
                text = text.replace(f' {word} ', f' {correct_char_dict[word]} ')
        return text

    text = remove_special_characters(text)
    text = correct_char(text)

    # Loại bỏ ký tự lặp lại
    def remove_repeated_characters(text):
        return re.sub(r'(\w)\1+', r'\1', text)

    # Loại bỏ từ lặp lại
    def remove_repeated_words(text):
        words = text.split()
        seen = set()
        new_words = []
        for word in words:
            if word not in seen:
                seen.add(word)
                new_words.append(word)
        return " ".join(new_words)

    text = remove_repeated_characters(text)
    text = remove_repeated_words(text)

    # Loại bỏ emoji
    def remove_emoji(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   u"\U00002500-\U00002BEF"
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r' ', text)

    text = remove_emoji(text)

    # Chuyển đổi từ tiếng Anh sang tiếng Việt
    def change_en_to_vn(text):
        words = text.split()
        for word in words:
            if word in en2vi_dict:
                text = text.replace(f' {word} ', f' {en2vi_dict[word]} ')
        return text

    text = change_en_to_vn(text)

    # Chuyển đổi teencode
    def teencode(text):
        words = text.split()
        for word in words:
            if word in teencode_dict:
                text = text.replace(f' {word} ', f' {teencode_dict[word]} ')
        return text

    text = teencode(text)

    # Loại bỏ khoảng trắng thừa
    def remove_extra_spaces(text):
        return ' '.join(text.split())

    text = remove_extra_spaces(text)

    # Loại bỏ các từ quá dài
    def remove_text_over_len(text):
        return ' '.join([w for w in text.split() if len(w) <= 7])

    text = remove_text_over_len(text)

    # Tokenize văn bản tiếng Việt
    text = ViTokenizer.tokenize(text)
    return text

# Định nghĩa API endpoint để dự đoán cảm xúc
@app.get("/predict/{input_data}")
def predict_sentiment_api(input_data: str):
    text = preprocess_text(input_data)  # Tiền xử lý văn bản
    token_input = [text]  # Chuẩn bị dữ liệu đầu vào cho mô hình
    token_input = np.array(token_input)
    prediction = model.predict(token_input)  # Dự đoán
    if prediction[0][0] <= prediction[0][1]:
        result = 1
    else:
        result = 0
    return {"prediction": result}

# Định nghĩa root endpoint
@app.get("/")
async def root():
    return {"message": "Xin chào thế giới"}

# Định nghĩa endpoint chào hỏi
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Xin chào {name}"}
