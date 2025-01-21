# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : browser-use-webui
# @FileName: test_llm_api.py
import os
import pdb

from dotenv import load_dotenv

load_dotenv()

import sys

sys.path.append(".")


def test_openai_model():
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="openai",
        model_name="gpt-4o",
        temperature=0.8,
        base_url=os.getenv("OPENAI_ENDPOINT", ""),
        api_key=os.getenv("OPENAI_API_KEY", "")
    )
    image_path = "assets/examples/test.png"
    image_data = utils.encode_image(image_path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )
    ai_msg = llm.invoke([message])
    print(ai_msg.content)


def test_gemini_model():
    # you need to enable your api key first: https://ai.google.dev/palm_docs/oauth_quickstart
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="gemini",
        model_name="gemini-2.0-flash-exp",
        temperature=0.8,
        api_key=os.getenv("GOOGLE_API_KEY", "")
    )

    image_path = "assets/examples/test.png"
    image_data = utils.encode_image(image_path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )
    ai_msg = llm.invoke([message])
    print(ai_msg.content)


def test_azure_openai_model():
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="azure_openai",
        model_name="gpt-4o",
        temperature=0.8,
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "")
    )
    image_path = "assets/examples/test.png"
    image_data = utils.encode_image(image_path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )
    ai_msg = llm.invoke([message])
    print(ai_msg.content)


def test_deepseek_model():
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="deepseek",
        model_name="deepseek-chat",
        temperature=0.8,
        base_url=os.getenv("DEEPSEEK_ENDPOINT", ""),
        api_key=os.getenv("DEEPSEEK_API_KEY", "")
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": "who are you?"}
        ]
    )
    ai_msg = llm.invoke([message])
    print(ai_msg.content)


def test_deepseek_r1_model():
    from openai import OpenAI
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = OpenAI(
        base_url=os.getenv("DEEPSEEK_ENDPOINT", ""),
        api_key=os.getenv("DEEPSEEK_API_KEY", "")
    )
    # Round 1
    messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
    response = llm.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages
    )

    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    pdb.set_trace()
    # Round 2
    messages.append({'role': 'assistant', 'content': content})
    messages.append({'role': 'user', 'content': "How many Rs are there in the word 'strawberry'?"})
    response = llm.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages
    )
    pdb.set_trace()


def test_ollama_model():
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="qwen2.5:7b")
    ai_msg = llm.invoke("Sing a ballad of LangChain.")
    print(ai_msg.content)


if __name__ == '__main__':
    # test_openai_model()
    # test_gemini_model()
    # test_azure_openai_model()
    # test_deepseek_model()
    # test_ollama_model()
    test_deepseek_r1_model()
