# import json
# import requests
# response = requests.get(f"https://api.smith.langchain.com/commits/teddynote/chain-of-density-prompt/latest", verify=False)
# request_json = json.loads(response.content)
# COD_SYSTEM_PROMPT = request_json['manifest']['kwargs']['messages'][0]['kwargs']['prompt']['kwargs']['template']
# 
# with open(f"{start_script_folder}/data/templates/cod_prompt_template.txt", 'w', encoding='utf-8-sig') as f:
#     f.write(cod_prompt_template)



from DS_Markdown import MarkdownParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# COD_SYSTEM_PROMPT = Path(f"{start_script_folder}/data/templates/cod_prompt_template.txt").read_text(encoding="utf-8")  
COD_SYSTEM_PROMPT = """
As an expert copy-writer, you will write increasingly concise, entity-dense summaries of the user provided {content_category}. The initial summary should be under {max_words} words and contain {entity_range} informative Descriptive Entities from the {content_category}.

A Descriptive Entity is:
- Relevant: to the main story.
- Specific: descriptive yet concise (5 words or fewer).
- Faithful: present in the {content_category}.
- Anywhere: located anywhere in the {content_category}.

# Your Summarization Process
- Read through the {content_category} and the all the below sections to get an understanding of the task.
- Pick {entity_range} informative Descriptive Entities from the {content_category} (";" delimited, do not add spaces).
- In your output JSON list of dictionaries, write an initial summary of max {max_words} words containing the Entities.
- You now have `[{{"missing_entities": "...", "denser_summary": "..."}}]`

Then, repeat the below 2 steps {iterations} times:
- Step 1. In a new dict in the same list, identify {entity_range} new informative Descriptive Entities from the {content_category} which are missing from the previously generated summary.
- Step 2. Write a new, denser summary of identical length which covers every Entity and detail from the previous summary plus the new Missing Entities.

A Missing Entity is:
- An informative Descriptive Entity from the {content_category} as defined above.
- Novel: not in the previous summary.

# Guidelines
- The first summary should be long (max {max_words} words) yet highly non-specific, containing little information beyond the Entities marked as missing. Use overly verbose language and fillers (e.g., "this {content_category} discusses") to reach ~{max_words} words.
- Make every word count: re-write the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the {content_category} discusses".
- The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the {content_category}.
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
- You're finished when your JSON list has 1+{iterations} dictionaries of increasing density.

# IMPORTANT
- Remember, to keep each summary to max {max_words} words.
- Never remove Entities or details. Only add more from the {content_category}.
- Do not discuss the {content_category} itself, focus on the content: informative Descriptive Entities, and details.
- Remember, if you're overusing filler phrases in later summaries, or discussing the {content_category} itself, not its contents, choose more informative Descriptive Entities and include more details from the {content_category}.
- Answer with a minified JSON list of dictionaries with keys "missing_entities" and "denser_summary".
- "denser_summary" should be written in the same language as the "content".

## Example output
[{{"missing_entities": "ent1;ent2", "denser_summary": "<vague initial summary with entities 'ent1','ent2'>"}}, {{"missing_entities": "ent3", "denser_summary": "denser summary with 'ent1','ent2','ent3'"}}, ...]
"""

COD_CONTENTS_PROMPT ="""
{content_category}:
{content}
"""


cod_prompt = PromptTemplate.from_template(COD_SYSTEM_PROMPT + "\n" + COD_CONTENTS_PROMPT)

import textwrap
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import SimpleJsonOutputParser

# {content}를 제외한 모든 입력에 대한 기본값 지정
cod_chain_inputs = {
    "content": lambda d: d.get("content"),                                  # content : 요약의 원문 텍스트(혹은 요약 대상 콘텐츠)
    "content_category": lambda d: d.get("content_category", "Article"),     # content_category : 콘텐츠의 “종류/범주”를 프롬프트에 주입하는 라벨 (예: Article, Blog post, Meeting notes, Policy, Paper 등)
    "entity_range": lambda d: d.get("entity_range", "1-3"),                 # entity_range : 각 요약 단계에서 추출할 “Descriptive Entities”의 개수 범위(예. '1-3')를 문자열로 지정
    "iterations": lambda d: int(d.get("iterations", 5)),                    # iterations : Chain-of-Density 반복횟수
    "max_words": lambda d: int(d.get("max_words", 80)),                     # max_words : 각 단계의 요약이 넘지 말아야 하는 최대 단어 수
}

# Chain of Density 체인 생성
cod_chain = (
    cod_chain_inputs
    | cod_prompt
    | llm
    | SimpleJsonOutputParser()
)



# 두 번째 체인 생성, 최종 요약만 추출 (스트리밍 불가능, 최종 결과가 필요함)
cod_final_summary_chain = cod_chain | (
    lambda output: output[-1].get(
        "denser_summary", '오류: 마지막 딕셔너리에 "denser_summary" 키가 없습니다'
    )
)



##################################################################################################################################
contents = Path(f"{start_script_folder}/data/example_sources/contents001.txt").read_text(encoding="utf-8")  
md = MarkdownParser(contents)
content = md.sections("###")[0].content


##################################################################################################################################


# cod_chain을 스트리밍 모드로 실행하고 부분적인 JSON 결과를 처리
results: list[dict[str, str]] = []
for partial_json in cod_chain.stream({"content": content, "content_category": "Article"}):
    results = partial_json      # 각 반복마다 results를 업데이트
    print(results, end="\r", flush=True)        # 현재 결과를 같은 줄에 출력 (캐리지 리턴을 사용하여 이전 출력을 덮어씀)



# 총 요약 수 계산
total_summaries = len(results)
print("\n")

# 각 요약을 순회하며 처리
i = 1
for cod in results:
    # 누락된 엔티티들을 추출하고 포맷팅
    added_entities = ", ".join(
        [
            ent.strip()
            for ent in cod.get(
                "missing_entities", 'ERR: "missing_entiies" key not found'
            ).split(";")
        ]
    )
    # 더 밀도 있는 요약 추출
    summary = cod.get("denser_summary", 'ERR: missing key "denser_summary"')

    # 요약 정보 출력 (번호, 총 개수, 추가된 엔티티)
    print(
        f"### CoD Summary {i}/{total_summaries}, 추가된 엔티티(entity): {added_entities}"
        + "\n"
    )
    # 요약 내용을 80자 너비로 줄바꿈하여 출력
    print(textwrap.fill(summary, width=80) + "\n")
    i += 1

print("\n============== [최종 요약] =================\n")
print(summary)


# -------------------------------------------------------------------------------------------
results = cod_chain.invoke({"content": content, "content_category": "Article"})
summary = results[-1]['denser_summary']

# -------------------------------------------------------------------------------------------
























######################################################################################################
######################################################################################################
######################################################################################################

COD_PROMPT_SYSTEM_ROLE = """
# Role
You Are an expert copy-writer.
"""

COD_PROMPT_SYSTEM_CONTENTS = """
# Goal
you will write increasingly concise, entity-dense summaries of the user provided {content_category}. 
The initial summary should be under {max_words} words and contain {entity_range} informative Descriptive Entities from the {content_category}.

A Descriptive Entity is:
- Relevant: to the main story.
- Specific: descriptive yet concise (5 words or fewer).
- Faithful: present in the {content_category}.
- Anywhere: located anywhere in the {content_category}.

# Your Summarization Process
- Read through the {content_category} and the all the below sections to get an understanding of the task.
- Pick {entity_range} informative Descriptive Entities from the {content_category} (";" delimited, do not add spaces).
- In your output JSON list of dictionaries, write an initial summary of max {max_words} words containing the Entities.
- You now have `[{{"missing_entities": "...", "denser_summary": "..."}}]`

Then, repeat the below 2 steps {iterations} times:
- Step 1. In a new dict in the same list, identify {entity_range} new informative Descriptive Entities from the {content_category} which are missing from the previously generated summary.
- Step 2. Write a new, denser summary of identical length which covers every Entity and detail from the previous summary plus the new Missing Entities.

A Missing Entity is:
- An informative Descriptive Entity from the {content_category} as defined above.
- Novel: not in the previous summary.

# Guidelines
- The first summary should be long (max {max_words} words) yet highly non-specific, containing little information beyond the Entities marked as missing. Use overly verbose language and fillers (e.g., "this {content_category} discusses") to reach ~{max_words} words.
- Make every word count: re-write the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the {content_category} discusses".
- The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the {content_category}.
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
- You're finished when your JSON list has 1+{iterations} dictionaries of increasing density.

# IMPORTANT
- Remember, to keep each summary to max {max_words} words.
- Never remove Entities or details. Only add more from the {content_category}.
- Do not discuss the {content_category} itself, focus on the content: informative Descriptive Entities, and details.
- Remember, if you're overusing filler phrases in later summaries, or discussing the {content_category} itself, not its contents, choose more informative Descriptive Entities and include more details from the {content_category}.
- Answer with a minified JSON list of dictionaries with keys "missing_entities" and "denser_summary".
- "denser_summary" should be written in the same language as the "content".

## Example output
[{{"missing_entities": "ent1;ent2", "denser_summary": "<vague initial summary with entities 'ent1','ent2'>"}}, {{"missing_entities": "ent3", "denser_summary": "denser summary with 'ent1','ent2','ent3'"}}, ...]
"""

COD_PROMPT_HUMAN = """
{content_category}:
{content}
"""

from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

class COD_Chain():
    def __init__(self, llm, output_type='final'):
        self.llm = llm
        self.prompts = [COD_PROMPT_SYSTEM_ROLE, COD_PROMPT_SYSTEM_CONTENTS, COD_PROMPT_HUMAN]
        self.prompts_template = None

        self.json_chain = None
        self.final_chain = None
        self.output_type = output_type
        
        self.partial()
        self.summary()

    def make_template(self):
        cod_prompt = "\n".join(self.prompts)
        return PromptTemplate.from_template(cod_prompt)

    def partial(self, inputs: dict={}):
        self.prompts_template = self.make_template()
        payload = {
            "content_category": inputs.get("content_category", "Article"),
            "entity_range": inputs.get("entity_range", "1-3"),
            # "iterations": int(inputs.get("iterations", 5)),
            "iterations": int(inputs.get("iterations", 1)),
            "max_words": int(inputs.get("max_words", 80)),
        }
        print(payload)
        self.prompts_template = self.prompts_template.partial(**payload)
    
    def summary_json(self):
        self.json_chain = self.prompts_template | self.llm | SimpleJsonOutputParser()
        return self.json_chain

    def summary(self):
        self.json_chain = self.summary_json()
        self.final_chain = (self.json_chain 
                | RunnableLambda(lambda x: x[-1]["denser_summary"]) 
                )
        return self.final_chain

    def invoke(self, input, config=None, **kwargs):
        if self.output_type == 'json':
            return self.summary_json().invoke(input, config=config, **kwargs)
        else:
            return self.summary().invoke(input, config=config, **kwargs)
    
    def stream(self, input, config=None, **kwargs):
        if self.output_type == 'json':
            return self.summary_json().stream(input, config=config, **kwargs)
        else:
            return self.summary().stream(input, config=config, **kwargs)
    
    @property
    def runnable(self):
        return self.summary_json() if self.output_type == 'json' else self.summary()
    
    def __or__(self, other):
        return self.summary_json().__or__(other) if self.output_type == 'json' else self.summary().__or__(other) 

    def __ror__(self, other):
        return self.summary_json().__ror__(other) if self.output_type == 'json' else self.summary().__ror__(other)

    def __repr__(self):
        return self.json_chain.__repr__() if self.output_type == 'json' else self.final_chain.__repr__()


from DS_Markdown import MarkdownParser
contents = Path(f"{start_script_folder}/data/example_sources/contents001.txt").read_text(encoding="utf-8")  
md = MarkdownParser(contents)
content = md.sections("###")[0].content


cod_prompt = COD_Chain(llm, output_type='final')
res = cod_prompt.invoke(content)
res
res = StreamResponse(cod_prompt.stream(content))
res.content
res.object



cod_prompt = COD_Chain(llm, output_type='final')

translate_chain = PromptTemplate.from_template("{text}를 영어(english)로 번역해.")
chain = cod_prompt | RunnableLambda(lambda x: {'text':x})| translate_chain | llm
res = chain.invoke(content)
res.content
res = StreamResponse(chain.stream(content))
res.content
res.object







