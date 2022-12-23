df2_group = df2.groupby(['출강목표'])

# 출강목표별 TS 상하 1% Data 제거(Outlier)
df_result = pd.DataFrame()
for i, v in df2_group:
    df6 = fun_outlier_remove(data=v, x=['초_TS'], method='quantile', criteria=0.01)   # Outlier 제거
    df_result = pd.concat([df_result, v], axis=0)

df_result



from sklearn.cluster import KMeans


df_060_099 = pd.read_clipboard()
df_kmean = df_060_099.set_index('출강목표')

plt.scatter(x=df_kmean['EL'], y=df_kmean['TS'] )
plt.show()

# 계층갯수별 Distance (The Elbow Method)
SumOfSquare = []
TotalK = range(2,10)
for k in TotalK:
    km = KMeans(n_clusters=k)
    km.fit(df_kmean)
    SumOfSquare.append(km.inertia_)    # inertia_ : Total within-cluster sum of squared Value

plt.plot(TotalK, SumOfSquare, 'bx-' )
plt.xlabel('k')
plt.ylabel('Sum of Sqared Distance')
plt.title('The Elbow Method showing the potimal "k"')
plt.show()


kmean = KMeans(n_clusters = 4, init='random')    # K-Mean Model 생성,  n_cluster : Cluster 갯수
kmean.fit(df_kmean)
predict = pd.DataFrame( kmean.predict(df_kmean) )   # K-Mean model에 의한 예측값
predict.columns = ['kmean']
km_result = pd.concat([df_kmean.reset_index(),predict], axis=1)

kimg = plt.figure()
plt.scatter(x=km_result['EL'], y=km_result['TS'], c=list(km_result['kmean']))
plt.title('K-Mean Cluster: N=4')
plt.show()

fun_Img_Copy(kimg)
km_result.to_clipboard(index=False)


    # K-Mean Cluster Result
for i in range(2,10):
    kmean = KMeans(n_clusters = i, init='random', random_state=2)    # K-Mean Model 생성,  n_cluster : Cluster 갯수
    kmean.fit(df_kmean)
    predict = pd.DataFrame( kmean.predict(df_kmean) )   # K-Mean model에 의한 예측값
    predict.columns = ['kmean']
    km_result = pd.concat([df_kmean.reset_index(),predict], axis=1)

    plt.figure()
    plt.scatter(x=km_result['EL'], y=km_result['TS'], c=list(km_result['kmean']))
    plt.title('k ' + str(i))
    plt.show()



df_kgroup = pd.read_clipboard()
df10 = pd.merge(left=df_final, right=df_kgroup, on='출강목표', how='inner')

df10.groupby('강종Group')['초_TS'].mean().to_frame().to_clipboard()


df_result.to_clipboard(index=False)
df_result = pd.DataFrame()
for i, v in df10.groupby(['소둔_공정실적','강도Group','두께그룹']):
    if len(v) >=30:
        df_result = pd.concat([df_result, v], axis=0)



