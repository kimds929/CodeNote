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








import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import HDBSCAN


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------------------------------------
# 1) Load
df = pd.read_clipboard(sep='\t')
# df.columns

text_col = "추진배경 및 문제점"
df = df.dropna(subset=[text_col]).drop_duplicates(subset=[text_col]).reset_index(drop=True)
texts = df[text_col].astype(str).tolist()

# ---------------------------------------------------------------------------------------------------------
# 2) Embedding
import time
from tqdm.auto import tqdm

def embed_in_batches(embeddings_model, texts, batch_size=20, latency_time=1):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1} / {len(texts) // batch_size + 1}...")
        try:
            # 배치 단위로 임베딩 수행
            batch_embeddings = embeddings_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            # 서버 부하 방지를 위한 짧은 대기
            time.sleep(latency_time) 
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            break
    return all_embeddings

X_mat = pd.read_csv(f'{folder_path}/database/QSS_embeddings.csv', encoding='utf-8-sig').to_numpy()

# X = embeddings.embed_documents(texts, batch_size=100)  # (n, dim)
# X = embed_in_batches(embeddings, texts, batch_size=10)
# X_mat = np.array(X)
# pd.DataFrame(X_mat).to_csv(f'{folder_path}/database/QSS_embeddings.csv', encoding='utf-8-sig',index=False)


# ---------------------------------------------------------------------------------------------------------
# 3) Clustering (최적 K 검색)

# # K-Means -----------------------------------------------------------------------------
# # 여러개의 k에 대해 K-means 수행
# k_range = list(range(2, 11))

# best_k = 2
# best_score = -1
# inertias = []
# silhouette_scores = []
# for k in tqdm(k_range):
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
#     labels = kmeans.fit_predict(X_mat)

    
#     # Elbow method를 위한 inertia(SSE : 각 데이터 지점에서 자신이 속한 클러스터 중심까지 거리 제곱합) 계산
#     inertias.append(kmeans.inertia_)

#     # 실루엣 점수 계산
#     score = silhouette_score(X_mat, labels)
#     silhouette_scores.append(score)

# # -----------------------------------------------------------
# optimal_k_silhouette_idx = np.argmax(silhouette_scores)
# optimal_k_silhouette = k_range[optimal_k_silhouette_idx]       # silhouette_score가 가장큰 값을 최적의 k로 선정
# print(f" . optimal_k_silhouette : {optimal_k_silhouette}")

# # -----------------------------------------------------------
# # Elbow Method 최적 K계산
# # 첫 점(k=2)과 마지막 점(k=10)의 좌표
# p1 = np.array([k_range[0], inertias[0]])
# p2 = np.array([k_range[-1], inertias[-1]])

# inertia_distances = []
# for i in range(len(k_range)):
#     p0 = np.array([k_range[i], inertias[i]])
#     # 직선과 점 사이의 거리 계산 (외적 활용)
#     inertia_distance = np.abs(np.cross(p2 - p1, p1 - p0)) / np.linalg.norm(p2 - p1)
#     inertia_distances.append(inertia_distance)

# optimal_k_elbow_idx = np.argmax(inertia_distances)
# optimal_k_elbow = k_range[optimal_k_elbow_idx]       # 거리가 가장 먼 지점의 인덱스를 찾아 최적의 k 도출
# print(f" . optimal_k_elbow : {optimal_k_elbow}")

# # -----------------------------------------------------------
# # 그래프 시각화
# plt.figure(figsize=(16, 5))
# plt.subplot(1,2,1)
# plt.title('Elbow Method')
# plt.plot(k_range, inertias, 'o-', color='steelblue', alpha=0.7, label='Inertia (WCSS)')
# plt.xlabel('Number of Clusters (k)')
# plt.xticks(k_range)

# plt.subplot(1,2,2)
# line1 = plt.plot(k_range, inertia_distances, 'o-', color='steelblue', alpha=0.7, label='Inertia Distance')
# plt.scatter(optimal_k_elbow, inertia_distances[optimal_k_elbow_idx], color='blue', marker='*', s=200)
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia Distance', color='steelblue')
# plt.xticks(k_range)

# plt.twinx()
# line2 = plt.plot(k_range, silhouette_scores, 'o-', color='orange', alpha=0.7, label='Silhouette Score')
# plt.scatter(optimal_k_silhouette, silhouette_scores[optimal_k_silhouette_idx], color='red', marker='*', s=200)
# plt.ylabel('Silhouette Score', color='orange')
# labels = [l.get_label() for l in (line1 + line2)]
# plt.legend(line1 + line2, labels,loc='upper right', bbox_to_anchor=(1.5,1))
# plt.xlabel('Number of Clusters (k)')
# plt.xticks(k_range)
# plt.show()
# # -----------------------------------------------------------

# Optimal K
optimal_k = 4
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
df["cluster_id"] = final_kmeans.fit_predict(X_mat)

#  --------------------------------------------------------------------------------------------------------



# HDBSCAN -----------------------------------------------------------------------------

# # 탐색할 min_cluster_size의 범위 설정 (데이터 크기에 따라 5~50 등으로 조절)
# mcs_range = list(range(5, 51, 5)) 

# n_clusters_list = []
# noise_ratios = []
# silhouette_scores = []
# for mcs in tqdm(mcs_range):
#     # HDBSCAN 모델 학습
#     hdbscan = HDBSCAN(min_cluster_size=mcs, min_samples=None)
#     labels = hdbscan.fit_predict(X_mat)
    
#     # 1. 생성된 군집의 수 계산 (노이즈인 -1은 제외)
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     n_clusters_list.append(n_clusters)
    
#     # 2. 노이즈 비율 계산
#     noise_ratio = list(labels).count(-1) / len(labels)
#     noise_ratios.append(noise_ratio)
    
#     # 3. 실루엣 점수 계산 (군집이 2개 이상이고, 모든 데이터가 노이즈가 아닐 때만 계산)
#     if n_clusters > 1:
#         # 노이즈(-1)를 제외한 데이터만 마스킹
#         mask = labels != -1
#         score = silhouette_score(X_mat[mask], labels[mask])
#     else:
#         score = 0  # 군집이 1개이거나 0개면 실루엣 점수 계산 불가
#     silhouette_scores.append(score)

# # -----------------------------------------------------------
# # 최적의 min_cluster_size 선정 (실루엣 점수 기준)
# optimal_mcs_idx = np.argmax(silhouette_scores)
# optimal_mcs = mcs_range[optimal_mcs_idx]
# optimal_k_found = n_clusters_list[optimal_mcs_idx]

# print(f" . 최적의 min_cluster_size : {optimal_mcs}")
# print(f" . 이때 자동으로 찾아낸 군집 수(k) : {optimal_k_found} 개")
# print(f" . 이때의 노이즈 비율 : {noise_ratios[optimal_mcs_idx]*100:.1f}%")

# # -----------------------------------------------------------
# # 그래프 시각화
# plt.figure(figsize=(16, 5))

# # [왼쪽 그래프] min_cluster_size에 따른 '생성된 군집의 수'
# plt.subplot(1, 2, 1)
# plt.title('Number of Clusters found by HDBSCAN')
# plt.plot(mcs_range, n_clusters_list, 'o-', color='purple', alpha=0.7, label='Number of Clusters')
# plt.axvline(x=optimal_mcs, color='red', linestyle='--', alpha=0.5) # 최적 지점 표시
# plt.xlabel('min_cluster_size')
# plt.ylabel('Number of Clusters (k)')
# plt.xticks(mcs_range)
# plt.grid(True, alpha=0.3)
# plt.legend()

# # [오른쪽 그래프] 실루엣 점수와 노이즈 비율 (이중 Y축)
# plt.subplot(1, 2, 2)
# plt.title('Silhouette Score & Noise Ratio')

# # 첫 번째 선 (노이즈 비율)
# line1 = plt.plot(mcs_range, noise_ratios, 'o-', color='gray', alpha=0.7, label='Noise Ratio')
# plt.xlabel('min_cluster_size')
# plt.ylabel('Noise Ratio', color='gray')
# plt.xticks(mcs_range)

# # 이중 Y축 생성
# plt.twinx()

# # 두 번째 선 (실루엣 점수)
# line2 = plt.plot(mcs_range, silhouette_scores, 'o-', color='orange', alpha=0.7, label='Silhouette Score')
# # 최적의 실루엣 점수에 별표 마커 표시
# plt.scatter(optimal_mcs, silhouette_scores[optimal_mcs_idx], color='red', marker='*', s=200, zorder=5)
# plt.ylabel('Silhouette Score (excluding noise)', color='orange')

# # 범례 통합
# labels = [l.get_label() for l in (line1 + line2)]
# plt.legend(line1 + line2, labels, loc='upper right', bbox_to_anchor=(1.4, 1))

# plt.tight_layout()
# plt.show()
# # -----------------------------------------------------------


# 데이터 학습 및 군집 라벨 예측
optimal_mcs = 20
final_hdbscan = HDBSCAN(min_cluster_size=optimal_mcs, min_samples=None)     # optimal_mcs = 20
df["cluster_id"] = final_hdbscan.fit_predict(X_mat)
optimal_k = df["cluster_id"].value_counts().index.max()+1
#  --------------------------------------------------------------------------------------------------------




# ---------------------------------------------------------------------------------------------------------
# 4) 각 클러스터 대표 문장 뽑기 (centroid에 가장 가까운 top_n)

# Centroid 중심의 Sampling
def get_representatives(cluster_id, top_n=8):
    idx = df.index[df["cluster_id"] == cluster_id].to_list()
    Xc = X_mat[idx]
    # centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
    centroid = Xc.mean(axis=0).reshape(1, -1)

    dists = cosine_distances(Xc, centroid).reshape(-1)
    top_idx_local = np.argsort(dists)[:top_n]
    rep_texts = [texts[idx[i]] for i in top_idx_local]
    return rep_texts, idx


# MMR (Maximal Marginal Relevance) 샘플링
def get_representatives_mmr(cluster_id, top_n=10, lambda_param=0.5):
    """
    lambda_param: 1에 가까울수록 중심점(Centroid)에 가까운 문서 위주로,
                  0에 가까울수록 기존에 뽑힌 문서와 다른(다양한) 문서 위주로 추출
    """
    idx = df.index[df["cluster_id"] == cluster_id].to_list()
    Xc = X_mat[idx]
    # centroid = final_kmeans.cluster_centers_[cluster_id].reshape(1, -1)
    centroid = Xc.mean(axis=0).reshape(1, -1)
    
    # 1. 중심점과의 유사도 계산
    from sklearn.metrics.pairwise import cosine_similarity
    sim_to_centroid = cosine_similarity(Xc, centroid).reshape(-1)
    
    selected_idx_local = []
    unselected_idx_local = list(range(len(idx)))
    
    # 첫 번째 문서는 중심점과 가장 가까운 문서로 선택
    first_choice = np.argmax(sim_to_centroid)
    selected_idx_local.append(first_choice)
    unselected_idx_local.remove(first_choice)
    
    # 나머지 top_n - 1 개 문서 선택
    for _ in range(top_n - 1):
        if not unselected_idx_local:
            break
            
        # 선택되지 않은 문서들과 이미 선택된 문서들 간의 유사도 계산
        sim_to_selected = cosine_similarity(Xc[unselected_idx_local], Xc[selected_idx_local])
        max_sim_to_selected = np.max(sim_to_selected, axis=1)
        
        # MMR 점수 계산
        mmr_scores = (lambda_param * sim_to_centroid[unselected_idx_local]) - ((1 - lambda_param) * max_sim_to_selected)
        
        # MMR 점수가 가장 높은 문서 선택
        best_choice_idx = np.argmax(mmr_scores)
        best_choice = unselected_idx_local[best_choice_idx]
        
        selected_idx_local.append(best_choice)
        unselected_idx_local.remove(best_choice)
        
    rep_texts = [texts[idx[i]] for i in selected_idx_local]
    return rep_texts, [idx[i] for i in selected_idx_local]



# # Map Reduce 방식
# from langchain_core.prompts import PromptTemplate

# # 1. Map 프롬프트 (개별 청크 요약)
# map_prompt = ChatPromptTemplate.from_template(
#     "다음은 현장 직원들의 의견 일부입니다. 주요 내용을 3~4가지 포인트로 요약해주세요:\n{text}"
# )

# # 2. Reduce 프롬프트 (최종 종합 요약)
# reduce_prompt = ChatPromptTemplate.from_template(
#     "다음은 여러 그룹에서 추출된 현장 의견 요약본들입니다. "
#     "이를 종합하여 이 클러스터의 최종 대표 주제와 핵심 요약을 작성해주세요.\n\n"
#     "[요약본들]\n{text}\n\n"
#     "출력 형식:\n1) cluster_title: (15자 내외)\n2) summary_bullets: - 형태로 3~5개\n3) keywords: 콤마로 5~10개"
# )

# cluster_summaries = {}

# for cid in tqdm(range(optimal_k)):
#     # 해당 클러스터의 '모든' 텍스트 가져오기
#     cluster_texts = df[df["cluster_id"] == cid][text_col].tolist()
    
#     # 배치 사이즈(예: 20개씩)로 나누기
#     batch_size = 20
#     map_results = []
    
#     for i in range(0, len(cluster_texts), batch_size):
#         batch_texts = cluster_texts[i:i+batch_size]
#         batch_str = "\n".join([f"- {t}" for t in batch_texts])
        
#         # Map 단계: 부분 요약
#         map_res = llm.invoke(map_prompt.format_messages(text=batch_str)).content
#         map_results.append(map_res)
        
#     # Reduce 단계: 부분 요약들을 모아서 최종 요약
#     combined_map_results = "\n\n".join(map_results)
#     final_summary = llm.invoke(reduce_prompt.format_messages(text=combined_map_results)).content
    
#     cluster_summaries[cid] = final_summary





# ---------------------------------------------------------------------------------------------------------
# 5) LLM으로 클러스터 라벨/요약 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "당신은 다음의 텍스트 문장들을(한국어)들을 군집별로 요약하는 분석가입니다. "
     "주어진 문장들의 묶음에서 공통 주제(클러스터 주제명)와 핵심 요약을 간결하게 작성하세요."),
    ("user",
     "다음은 같은 그룹으로 묶인 문장 텍스트들(대표 샘플)입니다:\n\n{samples}\n\n"
     "출력 형식:\n"
     "1) cluster_title: (15자 내외)\n"
     "2) summary_bullets: - 형태로 3~5개\n"
     "3) keywords: 콤마로 5~10개\n")
])

cluster_summaries = {}
for cid in tqdm(range(optimal_k)):
    reps, idxs = get_representatives(cid, top_n=10)
    samples = "\n".join([f"- {t}" for t in reps])
    msg = prompt.format_messages(samples=samples)
    out = llm.invoke(msg).content
    cluster_summaries[cid] = out


# ---------------------------------------------------------------------------------------------------------
# 6) 비율/집계
summary = (
    df.groupby("cluster_id")
      .size()
      .rename("count")
      .reset_index()
)
summary["ratio"] = summary["count"] / len(df)


# 7) cluster_title 컬럼 만들기(LLM 출력에서 title만 파싱하거나, 일단 전체를 넣을 수도)
# 간단히: summary에 LLM 결과 텍스트를 붙임
summary["llm_summary"] = summary["cluster_id"].map(cluster_summaries)



# 8) Save to csv
summary.to_csv(f"{folder_path}/database/opinions_clustered.csv", index=False, encoding='utf-8-sig')



# ```

# 이 결과로:
# - `rows_with_cluster` 시트: 각 의견이 어떤 cluster에 속했는지
# - `cluster_summary` 시트: cluster별 count/ratio + LLM 요약

# ---

# ## 3) “K를 자동으로” (HDBSCAN) 추천 흐름
# 현장 의견은 보통 노이즈/단문/잡다한 주제가 섞여서 **KMeans보다 HDBSCAN이 실무에서 더 자연스러운 경우가 많습니다.**
# - 장점: 군집 수 자동, 애매한 의견은 `-1`(노이즈)로 분리 가능  
# - 단점: 파라미터 튜닝 필요(min_cluster_size 등)

# 구성:
# 1) 임베딩
# 2) (옵션) UMAP으로 차원 축소
# 3) HDBSCAN으로 라벨 생성
# 4) 라벨별 요약/비율

# ---

# ## 4) LangChain “Agent”는 어디에 쓰면 좋은가?
# 군집 자체는 **에이전트보다 고정 파이프라인이 안정적**입니다. 다만 아래는 Agent가 유용합니다.

# - **결과 품질 점검/재요약**: “이 군집 요약이 중복되는지”, “군집명이 너무 추상적인지” 자동 검토 후 재작성
# - **클러스터 수 조정 의사결정 지원**: Silhouette score/토픽 일관성 지표를 계산하고, LLM이 “5개 vs 7개 중 무엇이 더 설명력이 좋은지”를 리포팅
# - **한국어 도메인 룰 반영**: 특정 키워드(안전/품질/인력/설비/납기 등) 기준으로 군집을 재분류할지 제안

# 즉:
# - **클러스터링 = 임베딩+통계/ML**
# - **해석/라벨링/보고서화 = LLM(=LangChain)**

# ---

# ## 5) 실무에서 꼭 추가하면 좋은 것들
# - **대표 문장 추출**: 군집 요약의 근거(샘플)로 함께 저장
# - **군집명 중복 제거**: 서로 비슷한 군집 요약이면 병합 제안
# - **품질 평가**: 각 cluster 내 평균 유사도(응집도), cluster 간 거리(분리도)
# - **민감정보 마스킹**: 이름/전화/주소 등 정규식 또는 별도 NER로 제거

# ---

# ## 6) 질문 (정확한 코드/설계를 위해)
# 1) 의견 컬럼이 **한국어만**인가요? (혼합 언어인지)  
# 2) 대략 row 수가 100 정도인지, 수천/수만까지 커지는지?  
# 3) “항상 5개로” 같은 고정 요구인지, 자동 군집 수가 필요한지?  
# 4) 사내 보안상 **OpenAI API 사용 가능**한지(불가하면 로컬 임베딩/로컬 LLM로도 가능)

# 원하시면 위 4개 답 기준으로
# - **(A) KMeans 고정 K 버전** / **(B) HDBSCAN 자동 군집 버전**
# - 결과 엑셀 포맷(요약시트에 제목/키워드/대표문장/비율까지)
# 까지 바로 실행 가능한 형태로 더 다듬어 드릴게요.