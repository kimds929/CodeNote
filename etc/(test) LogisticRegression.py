

# (Decision-Boundary)
Xp = np.linspace(X.min(), X.max(), 100)
df_proba = pd.DataFrame(np.concatenate([Xp, LRC.predict_proba(Xp)], axis=1), columns=['Xp', 'proba_0', 'proba_1'])

df_proba[df_proba['proba_1'] < 0.5]
df_proba[df_proba['proba_1'] < 0.5].tail(1)
df_proba[df_proba['proba_1'] < 0.5].tail(1)['Xp']
db_1ord = df_proba[df_proba['proba_1'] < 0.5].tail(1)['Xp'].values[0]



# (Decision-Boundary)
# log( p/(1-p) ) =  β0 + β1·x1 + β2·x2 + ... + βn·xn
#    (1st order)  log( p/(1-p) ) = logit =  β0 + β1·x1
#                  → x1 = (logit - β0) / β1
threshold = 0.5
logit = np.log(threshold / (1-threshold))
decision_boundary = ( (logit - LRC.intercept_) / LRC.coef_ )[0][0]
decision_boundary

# LRC_pred_proba[:,1] > threshold
LRC_proba_X = (LRC_pred_proba[:,1] > threshold).astype(int)
LRC_proba_X



# (Graph with Decision-Boundary)
Xp = np.linspace(X.min(), X.max(), 100)
LRC_pred_proba_Xp = LRC.predict_proba(Xp)[:,1]

dot_colors = ['red','steelblue']
pred_TF = (LRC_proba_X == y).apply(lambda x: dot_colors[x])

plt.figure()
plt.scatter(X, y, c=pred_TF)
plt.plot(Xp, LRC_pred_proba_Xp, color='red')
plt.axhline(threshold, color='orange', ls='--', alpha=0.5)
plt.axvline(decision_boundary, color='orange',ls='--', alpha=0.5)
plt.show()






def draw_logistic_graph(X, y, estimator, threshold):
    decision_boundary = ( ( np.log(threshold / (1-threshold)) - estimator.intercept_) / estimator.coef_ )[0][0]

    Xp = np.linspace(X.min(), X.max(), 100)

    dot_colors = ['red','steelblue']
    y_pred_proba_X = estimator.predict_proba(X)[:,1]
    y_pred_proba_Xp = estimator.predict_proba(Xp)[:,1]
    
    y_pred_X = (y_pred_proba_X > threshold).astype(int)
    pred_TF = (y_pred_X == y).apply(lambda x: dot_colors[x])

    f= plt.figure()
    plt.scatter(X, y, c= pred_TF)
    plt.plot(Xp, y_pred_proba_Xp, color='red')
    plt.axhline(threshold, color='orange', ls='--', alpha=0.5)
    plt.axvline(decision_boundary, color='orange',ls='--', alpha=0.5)
    plt.close()

    return f

draw_logistic_graph(X,y, LRC, 0.2)
draw_logistic_graph(X,y, LRC, 0.5)
draw_logistic_graph(X,y, LRC, 0.8)

