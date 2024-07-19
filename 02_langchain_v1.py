import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# OpenAI API 키 설정
openai.api_key = 'API_KEY'

# LangChain의 OpenAI LLM 설정
llm = OpenAI(model_name="gpt-4", openai_api_key=openai.api_key)

# 첫 번째 프롬프트 템플릿
prompt_template1 = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer the following question: {question}"
)
llm_chain1 = LLMChain(llm=llm, prompt=prompt_template1)

# 두 번째 프롬프트 템플릿
prompt_template2 = PromptTemplate(
    input_variables=["feature"],
    template="You are a helpful assistant. Explain more about {feature} in Python."
)
llm_chain2 = LLMChain(llm=llm, prompt=prompt_template2)

# 최종 프롬프트 템플릿
prompt_template3 = PromptTemplate(
    input_variables=["feature1", "feature2"],
    template="You are a helpful assistant. Summarize the information about {feature1} and {feature2} in Python."
)
llm_chain3 = LLMChain(llm=llm, prompt=prompt_template3)




### 질의


# 첫 번째 질문
question1 = "What are the key features of Python?"
answer1 = llm_chain1.run(question=question1)
print("Answer 1:", answer1)

# 두 번째 질문, 첫 번째 응답을 기반으로
feature1 = answer1.split()[0]
answer2 = llm_chain2.run(feature=feature1)
print("Answer 2:", answer2)

# 최종 응답, 두 번째 응답을 기반으로
feature2 = answer2.split()[0]
final_answer = llm_chain3.run(feature1=feature1, feature2=feature2)
print("Final Answer:", final_answer)








# TODO 요약
#  LangChain을 사용하면 복잡한 작업을 더 구조화하고 간결하게 작성할 수 있으며,
#  코드의 가독성, 재사용성, 유지 보수성, 확장성을 크게 향상시킬 수 있습니다.
#  특히 여러 단계의 질문과 응답을 처리해야 하는 시나리오에서는 LangChain의 체인 구성 방식이 큰 이점을 제공
#  LangChain을 사용하지 않는 경우, 이러한 작업을 수동으로 관리해야 하며, 이는 코드 관리의 어려움을 증가시킬 수 있음