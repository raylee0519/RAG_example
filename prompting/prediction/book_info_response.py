# 책 정보 출력

from response import get_response_gpt
import pandas as pd

system_message = "사용자 쿼리에 책의 베스트셀러 점수 계산 및 분석에 대한 내용이 존재할 경우 부차적인 말 없이 도서명을 반환하세요."
user_message = f"[사용자 쿼리]\n빛의 자리라는 책의 베스트셀러 점수를 산정하여 베스트셀러 가능성을 예측해주세요. 그리고 베스트셀러 가능성이 낮은 경우(4점 이하) 판매량 증진 전략을 설명해주세요."

temperature = 0.5
max_tokens = 4096
top_p = 0.5

book_title = get_response_gpt(system_message, user_message, model_name='gpt-4o', temperature=temperature, max_tokens=max_tokens, top_p=top_p, use_json_format=False, retry_attempts=3, delay=5)
print(book_title)

def get_book_info(book_title):
    df = pd.read_csv('book_info.csv')
    book_info = df[df["제목"] == book_title].iloc[0].to_dict()
    return book_info

book_info = get_book_info(book_title) 
book_info