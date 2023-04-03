import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment
nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def cal_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def cluster(features, labels, n_clusters, count=10):
    features = features.to('cpu').detach().numpy()
    labels = labels if isinstance(labels, np.ndarray) else labels.to('cpu').detach().numpy()
    acc = np.zeros(count)
    nmi = np.zeros(count)
    ar = np.zeros(count)
    pred_all = []
    for i in range(count):
        km = KMeans(n_clusters=n_clusters, n_init=10)
        pred = km.fit_predict(features)
        pred_all.append(pred)
    gt = labels.copy()
    gt = np.reshape(gt, np.shape(pred))
    if np.min(gt) == 1:
        gt -= 1
    for i in range(count):
        acc[i] = cal_acc(gt, pred_all[i])
        nmi[i] = normalized_mutual_info_score(gt, pred_all[i])
        ar[i] = ari(gt, pred_all[i])
    return {"clu_acc_avg":  '%05.2f' % (acc.mean()  *100),    'clu_acc_std':  '%05.2f' % (acc.std() *100),
            "clu_nmi_avg":  '%05.2f' % (nmi.mean()  *100),    'clu_nmi_std':  '%05.2f' % (nmi.std() *100),
            "clu_ar_avg":   '%05.2f' % (ar.mean()   *100),    'clu_ar_std':   '%05.2f' % (ar.std()  *100)}