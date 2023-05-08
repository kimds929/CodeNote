# confusion matrix 를 활용해 confusion plot draw
def draw_confusion(confusion_array) :
    conf_matrix=pd.DataFrame(confusion_array, columns=['Predicted:0(Female)','Predicted:1(Male)'],index=['Actual:0(Female)','Actual:1(Male)'])
    plt.figure(figsize = (8,6))
    sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# Deicision 영역 그리는 함수
def plot_decision(W1, W2, b1, b2):
    x_plot, y_plot = np.meshgrid(np.linspace(-0.5, 1.5), np.linspace(-0.5, 1.5))

    X_plot = np.hstack([x_plot.reshape(-1, 1), y_plot.reshape(-1, 1)])
    z1 = tf.sigmoid(X_plot @ W1 + b1)
    z2 = tf.sigmoid(z1 @ W2 + b2)

    plt.contour(x_plot.reshape(50, 50), y_plot.reshape(50, 50), z2.numpy().reshape(50, 50))
    plt.scatter(X[:, 0], X[:, 1], s=200, c=y, marker='*')
    plt.show()