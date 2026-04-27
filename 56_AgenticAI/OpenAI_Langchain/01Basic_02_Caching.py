# C:\Users\Admin\AppData\Local\pypoetry\Cache\virtualenvs\langchain-kr-sSe9WGAd-py3.11\Scripts\python.exe
import sys
import os
if os.path.isdir("D:/DataScience/★GitHub_kimds929"):
    library_path = "D:/DataScience/★GitHub_kimds929/DS_Library"
    folder_path = "D:/DataScience/★GitHub_kimds929/CodeNote/56_AgenticAI"
    env_path = 'D:/DataScience/DataBase/Keys/.env'
    llm_case = "ChatOpenAI"
else:
    if os.path.isdir("D:/DataScience/PythonforWork"):
        base_path = "D:/DataScience/PythonforWork"
    elif os.path.isdir("C:/Users/kimds929/DataScience"):
        base_path = "C:/Users/kimds929/DataScience"
    
    library_path = f"{base_path}/DS_Library"
    folder_path = f"{base_path}/AgenticAI"
    env_path = f"{base_path}/AgenticAI/.env"
    llm_case = "PgptLLM"
sys.path.append(library_path)


import requests
import base64
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from DS_AgenticAI import langsmith, PgptLLM, StreamResponse, read_messages, PgptEmbeddings
except:
    remote_library_url = 'https://raw.githubusercontent.com/kimds929/'
    try:
        import httpimport
        with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
            from DS_AgenticAI import logging, PgptLLM, StreamResponse, read_messages, PgptEmbeddings
    except:
        import requests
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_AgenticAI.py", verify=False)
        exec(response.text)

result = load_dotenv(env_path)
print("로드 결과:", result)

##########################################################################################

# LLM 객체 생성
if llm_case == "PgptLLM":
    llm = PgptLLM(
        api_key=os.getenv("API_KEY"),
        emp_no=os.getenv("EMP_NO"),
        # model_name="gpt-4.1-nano",
        model_name="gpt-5-nano",
        # temperature=2.0,  # 정상 코드에 있던 설정값 적용
        # top_p=0.9,
        # stream_usage=True
    )
    embeddings = PgptEmbeddings(
        api_key=os.getenv("API_KEY"),
        emp_no=os.getenv("EMP_NO"),
        model_name='text-embedding-ada-002'
    )
elif llm_case == "ChatOpenAI":
    llm = ChatOpenAI(
        # temperature=0.1,  # 창의성 (0.0 ~ 2.0)
        # model_name="gpt-4o-mini",  # 모델명
        # model_name="gpt-4.1-nano",  # 모델명
        model_name="gpt-5-nano",  # 모델명
    )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    langsmith("Default project")      # LangSmith 추적을 시작합니다.
    # langsmith("Default project", set_enable=False)  # LangSmith 추적을 하지 않습니다.
print(llm)
print(embeddings)

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################








##############################################################################################################
## 프롬프트 캐싱 : LLM 비용/속도 최적화의 핵심 메커니즘
# - 참고 링크: https://platform.openai.com/docs/guides/prompt-caching
#   프롬프트 캐싱 기능을 활용하면 반복하여 동일하게 입력으로 들어가는 토큰에 대한 비용을 아낄 수 있습니다.
#   다만, 캐싱에 활용할 토큰은 고정된 PREFIX 를 주는 것이 권장됩니다.
#   아래의 예시에서는 `<PROMPT_CACHING>` 부분에 고정된 토큰을 주어 캐싱을 활용하는 방법을 설명합니다.


from langchain_teddynote.messages import stream_response

very_long_prompt = """
당신은 매우 친절한 AI 어시스턴트 입니다. 
당신의 임무는 주어진 질문에 대해 친절하게 답변하는 것입니다.
아래는 사용자의 질문에 답변할 때 참고할 수 있는 정보입니다.
주어진 정보를 참고하여 답변해 주세요.

<WANT_TO_CACHE_HERE>
#참고:
**Prompt Caching**
Model prompts often contain repetitive content, like system prompts and common instructions. OpenAI routes API requests to servers that recently processed the same prompt, making it cheaper and faster than processing a prompt from scratch. This can reduce latency by up to 80% and cost by 50% for long prompts. Prompt Caching works automatically on all your API requests (no code changes required) and has no additional fees associated with it.

Prompt Caching is enabled for the following models:

gpt-4.1 (excludes gpt-4.1-2024-05-13 and chatgpt-4.1-latest)
gpt-4.1-mini
o1-preview
o1-mini
This guide describes how prompt caching works in detail, so that you can optimize your prompts for lower latency and cost.

Structuring prompts
Cache hits are only possible for exact prefix matches within a prompt. To realize caching benefits, place static content like instructions and examples at the beginning of your prompt, and put variable content, such as user-specific information, at the end. This also applies to images and tools, which must be identical between requests.

How it works
Caching is enabled automatically for prompts that are 1024 tokens or longer. When you make an API request, the following steps occur:

Cache Lookup: The system checks if the initial portion (prefix) of your prompt is stored in the cache.
Cache Hit: If a matching prefix is found, the system uses the cached result. This significantly decreases latency and reduces costs.
Cache Miss: If no matching prefix is found, the system processes your full prompt. After processing, the prefix of your prompt is cached for future requests.
Cached prefixes generally remain active for 5 to 10 minutes of inactivity. However, during off-peak periods, caches may persist for up to one hour.

Requirements
Caching is available for prompts containing 1024 tokens or more, with cache hits occurring in increments of 128 tokens. Therefore, the number of cached tokens in a request will always fall within the following sequence: 1024, 1152, 1280, 1408, and so on, depending on the prompt's length.

All requests, including those with fewer than 1024 tokens, will display a cached_tokens field of the usage.prompt_tokens_details chat completions object indicating how many of the prompt tokens were a cache hit. For requests under 1024 tokens, cached_tokens will be zero.

What can be cached
Messages: The complete messages array, encompassing system, user, and assistant interactions.
Images: Images included in user messages, either as links or as base64-encoded data, as well as multiple images can be sent. Ensure the detail parameter is set identically, as it impacts image tokenization.
Tool use: Both the messages array and the list of available tools can be cached, contributing to the minimum 1024 token requirement.
Structured outputs: The structured output schema serves as a prefix to the system message and can be cached.
Best practices
Structure prompts with static or repeated content at the beginning and dynamic content at the end.
Monitor metrics such as cache hit rates, latency, and the percentage of tokens cached to optimize your prompt and caching strategy.
To increase cache hits, use longer prompts and make API requests during off-peak hours, as cache evictions are more frequent during peak times.
Prompts that haven't been used recently are automatically removed from the cache. To minimize evictions, maintain a consistent stream of requests with the same prompt prefix.
Frequently asked questions
How is data privacy maintained for caches?

Prompt caches are not shared between organizations. Only members of the same organization can access caches of identical prompts.
Does Prompt Caching affect output token generation or the final response of the API?
Prompt Caching does not influence the generation of output tokens or the final response provided by the API. Regardless of whether caching is used, the output generated will be identical. This is because only the prompt itself is cached, while the actual response is computed anew each time based on the cached prompt. 
Is there a way to manually clear the cache?
Manual cache clearing is not currently available. Prompts that have not been encountered recently are automatically cleared from the cache. Typical cache evictions occur after 5-10 minutes of inactivity, though sometimes lasting up to a maximum of one hour during off-peak periods.
Will I be expected to pay extra for writing to Prompt Caching?
No. Caching happens automatically, with no explicit action needed or extra cost paid to use the caching feature.
Do cached prompts contribute to TPM rate limits?
Yes, as caching does not affect rate limits.
Is discounting for Prompt Caching available on Scale Tier and the Batch API?
Discounting for Prompt Caching is not available on the Batch API but is available on Scale Tier. With Scale Tier, any tokens that are spilled over to the shared API will also be eligible for caching.
Does Prompt Caching work on Zero Data Retention requests?
Yes, Prompt Caching is compliant with existing Zero Data Retention policies.
</WANT_TO_CACHE_HERE>

#Question:
{}

"""
# 위의 LLM 요청
#   ㄴ [긴 system prompt + 설명 + 규칙 + context + user 질문]
# 
# 문제:
#   매번 동일한 instruction 반복됨
#   매번 같은 토큰 다시 계산
#   비용 ↑
#   latency ↑
#
#👉 해결:
#   “앞부분은 이미 계산했으니까 재사용하자”
#
# 어떻게 동작하냐 (핵심 원리)
# 1. 요청 들어옴
#    PREFIX (고정) + SUFFIX (변하는 질문)
# 2. 시스템이 prefix 검사
#   기존 요청과 prefix 동일한지 확인
# 3. 동일하면 캐시된 결과 재사용
#   ✅ Cache Hit → prefix 재사용
#   ❌ Cache Miss → 전체 다시 계산
# ※ 📌 중요한 조건 : "완전히 동일한 prefix"여야 함
#
# 핵심규칙
# Rule 1. prefix는 “앞부분”이어야 한다
# [GOOD]
# 고정 프롬프트
# 고정 설명
# 고정 context
#   → 사용자 질문
#
# [BAD]
# 사용자 질문
#   → 고정 프롬프트
#
# Rule 2. EXACT MATCH
#  단 하나라도 다르면 cache miss
#
# Rule 3. 1024 토큰 이상부터 작동
#   1024 tokens 이상 → 캐싱 시작
#   128 token 단위로 증가



from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    # 답변 요청
    answer = llm.invoke(
        very_long_prompt.format("프롬프트 캐싱 기능에 대해 2문장으로 설명하세요")
    )
    print(cb)
    # 캐싱된 토큰 출력
    cached_tokens = answer.response_metadata["token_usage"]["prompt_tokens_details"][
        "cached_tokens"
    ]
    print(f"캐싱된 토큰: {cached_tokens}")


with get_openai_callback() as cb:
    # 답변 요청
    answer = llm.invoke(
        very_long_prompt.format("프롬프트 캐싱 기능에 대해 2문장으로 설명하세요")
    )
    print(cb)
    # 캐싱된 토큰 출력
    cached_tokens = answer.response_metadata["token_usage"]["prompt_tokens_details"][
        "cached_tokens"
    ]
    print(f"캐싱된 토큰: {cached_tokens}")