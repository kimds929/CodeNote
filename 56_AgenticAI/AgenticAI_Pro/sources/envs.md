# Account(.env)
 - `.env` path : `C:/Users/kimds929/DataScience/AgenticAI/.env`

# LLM model Guide
 - `Langchain`을 위한 LLM Model은 아래 Guide에 기반하여 사용
 - **Library path** : `C:/Users/kimds929/DataScience/DS_Library/DS_AgenticAI.py`
 - **Class** : `from DS_AgenticAI import PgptLLM`
 - **사용방식** : `ChatOpenAI` class와 대부분 동일 (사용 중 문제 발생시 해당 Library Path에 접근하여 class구조 확인 하여 코드 수정 필요)
 - **사용가능 model_name** : `gpt-5.2`, `gpt-5`, `gpt-5.1-codex`, `gpt-5-nano`, `gpt-5.2-chat`, `gpt-5.1-chat`, `gpt-5-chat`
 - Example_of_Use_Guidance
    ```
    from dotenv import load_dotenv
    from DS_AgenticAI import PgptLLM
    load_dotenv("{env_path}")
    llm = Pgpt(api_key=os.getenv("API_KEY"),
        emp_no=os.getenv("EMP_NO"),
        model_name="gpt-5.2"
        )
    ```

# Embeddings Guide
 - `Embedding`을 위한 Model은 아래 Guide에 기반하여 사용
 - **Library path** : `C:/Users/kimds929/DataScience/DS_Library/DS_AgenticAI.py`
 - **Class** : `from DS_AgenticAI import PgptEmbeddings`
 - **사용방식** : `OpenAIEmbeddings` class와 대부분 동일 (사용 중 문제 발생시 해당 Library Path에 접근하여 class구조 확인 하여 코드 수정 필요)
 - **사용가능 model_name** : 'text-embedding-3-small', 'text-embedding-3-large', 'test-embedding-ada-002'
  - **Example_of_Use_Guidance**
    ```
    from dotenv import load_dotenv
    from DS_AgenticAI import PgptEmbeddings
    load_dotenv("{env_path}")
    llm = PgptEmbeddings(api_key=os.getenv("API_KEY"),
        emp_no=os.getenv("EMP_NO"),
        model_name="text-embedding-3-small"
        )
    ```