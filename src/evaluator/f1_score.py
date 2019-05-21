def f1_weighted(labels, preds):
    from sklearn.metrics import f1_score
    print(preds.shape)
    preds = np.argmax(preds.reshape(12, -1), axis=0)
    score = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_weighted', score, True
