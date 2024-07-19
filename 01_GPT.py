# GPT의 API을 이용하여 구성
from openai import OpenAI

# OpenAI API 키 설정
client = OpenAI(api_key='Your API KEY')


# GPT-4 모델에 요청 보내기
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."}, # system의 역할 부여
        {"role": "user", "content": "Hello, how can I use GPT API with Python? in korean"}
    ],
    max_tokens=350,  # 응답에서 생성할 최대 토큰 수
    temperature=0.1  # 텍스트 생성의 창의성 수준 (값이 커지면 좀 더 창의적임)
)

# 응답 출력
print(response.choices[0].message.content.strip())