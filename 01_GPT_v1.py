# 프롬프트가 달라졌는데 GPT API만 이용했을 때의 문제점
# 가독성이 매우 떨어지고 코드 효율성이 좋지 않다.
import openai

# OpenAI API 키 설정
openai.api_key = 'API_KEY'

def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Yddd."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

def generate_response2(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You2222."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

def generate_response3(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You555t."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# 첫 번째 질문
question1 = "What are the key features of Python?"
answer1 = generate_response(question1)
print("Answer 1:", answer1)

# 두 번째 질문, 첫 번째 응답을 기반으로
question2 = f"Can you explain more about {answer1.split()[0]} in Python?"
answer2 = generate_response2(question2)
print("Answer 2:", answer2)

# 최종 응답, 두 번째 응답을 기반으로
final_prompt = f"Summarize the information about {answer1.split()[0]} and {answer2.split()[0]} in Python."
final_answer = generate_response3(final_prompt)
print("Final Answer:", final_answer)