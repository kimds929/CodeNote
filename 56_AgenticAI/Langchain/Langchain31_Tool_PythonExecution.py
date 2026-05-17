import os
current_file_path = os.path.abspath(__file__).replace('\\','/')

if os.path.isdir("D:/DataScience/★GitHub_kimds929"):
    start_script_folder ="D:/DataScience/★GitHub_kimds929/CodeNote/56_AgenticAI"
elif os.path.isdir("D:/DataScience/PythonforWork"):
    start_script_folder ="D:/DataScience/PythonForwork/AgenticAI"
elif os.path.isdir("C:/Users/kimds929/DataScience"):
    start_script_folder = "C:/Users/kimds929/DataScience/AgenticAI"
        
start_script_path = f'{start_script_folder}/StartingScript_AgenticAI.txt'

with open(start_script_path, 'r', encoding='utf-8') as f:
    script = f.read()
script_formatted = script.replace('{base_folder_name}', 'DataScience') \
                         .replace('{current_path}', current_file_path)
# print(script_formatted)
exec(script_formatted)

################################################################################################






from langchain_experimental.tools import PythonREPLTool

# 파이썬 코드를 실행하는 도구를 생성합니다.
python_tool = PythonREPLTool()

# 파이썬 코드를 실행하고 결과를 반환합니다.
python_tool.invoke("print(100 + 200)")

# res = StreamResponse(python_tool.stream("print(100 + 200)"))
# res.content
# res.object

##########################################################################################


from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_experimental.tools import PythonREPLTool

# 파이썬 코드를 실행하는 도구를 생성합니다.
python_tool = PythonREPLTool()

# llm.model_name = 'gpt-5-nano'
python_tool.invoke("print('hello')")

# 파이썬 코드를 실행하고 중간 과정을 출력하고 도구 실행 결과를 반환하는 함수
def print_and_execute(code, debug=True):
    if debug:
        print("CODE:")
        print(code)
    code_result = python_tool.invoke(code)
    return eval(code_result[:-1])

CODE_EXECUTION_SYSTEM_PROMPT = """
You are Raymond Hetting, an expert python programmer, well versed in meta-programming and elegant, concise and short but well documented code. 
You follow the PEP8 style guide.

Return only the code, no intro, no explanation, no chatty, no markdown, no code block, no nothing. 
Just the code.
"""

# 실행결과를 Return하고 싶을 때
CODE_EXECUTION_RETURN_PROMPT ="""
# IMPORTANT
 - Ensure that the code is written so the execution results are always shown via the `print` function.
 - The final output contains only required result without description
 - The code will be executed in a REPL-like environment where __name__ is not "__main__".
    Do not use if __name__ == "__main__": guards.
    Store the final output in a variable named RESULT in addition to printing it.
"""


# 파이썬 코드를 작성하도록 요청하는 프롬프트
prompt = ChatPromptTemplate.from_messages(
    [ ("system", CODE_EXECUTION_SYSTEM_PROMPT + "\n" +CODE_EXECUTION_RETURN_PROMPT),
        ("human", "{input}") ])



# 프롬프트와 LLM 모델을 사용하여 체인 생성
chain = prompt | llm | StrOutputParser() | RunnableLambda(print_and_execute)

res = chain.invoke("로또 번호 생성기를 출력하는 코드를 작성하세요.")
res


res = StreamResponse(chain.stream("로또 번호 생성기를 출력하는 코드를 작성하세요."))
res.content
res.object



