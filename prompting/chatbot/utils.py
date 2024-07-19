import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


def load_model(model_name): # 모델 불러오는 곳
    load_dotenv()
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=model_name) # langchain에서 제공해주는 chat openai모델을 이용
    return llm 


def load_prompt(character_name): # prompt 불러오기
    with open(f"prompts/{character_name}.prompt", "r", encoding="utf-8") as file:
        prompt = file.read().strip()
    return prompt


def set_memory(): # 이전 대화를 기억을 하기 위함. 
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True) # 랭체인에서 제공해주는 함수


def initialize_chain(llm, character_name, memory):
    system_prompt = load_prompt(character_name) # 캐릭터별 시스템 프로폼트 가져오기
    custom_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=custom_prompt, verbose=True, memory=memory)
    return chain


def generate_message(chain, user_input):
    result = chain({"input": user_input})
    response_content = result["text"]
    return response_content
