"""
Focused runner to push binary detection accuracy toward 99%.

Strategy:
- load processed train/test arrays (data/processed/*.npz)
- map labels to binary
- prefilter top-K features by f_classif
- run ABA on the top-K subset to find compact feature mask (optional)
- evaluate a small set of LightGBM hyperparameter candidates using OOF CV
- for each candidate, compute OOF probabilities and tune a decision threshold to maximize accuracy
- train final model on full train and evaluate on test using the best threshold
- optionally include XGBoost and stacking (if available)

Usage (smoke test):
  python src/run_best_for_99.py --limit 20000

Outputs:
- models/best_model.joblib
- results/best_for_99_results.json
"""
from pathlib import Path
import argparse
import numpy as np
import joblib
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
MOD = ROOT / 'models'
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

def map_to_binary(yarr):
    try:
        svals = [str(v).lower() for v in yarr[:1000]] if len(yarr)>0 else []
        if any('normal' in s for s in svals):
            return np.array([0 if 'normal' in str(v).lower() else 1 for v in yarr], dtype=int)
    except Exception:
        pass
    try:
        y_int = np.array(yarr, dtype=int)
        maj = int(np.argmax(np.bincount(y_int)))
        return np.array([0 if int(v)==maj else 1 for v in y_int], dtype=int)
    except Exception:
        try:
            yf = np.array(yarr, dtype=float)
            return np.array([0 if float(v)==0.0 else 1 for v in yf], dtype=int)
        except Exception:
            first = yarr[0]
            return np.array([0 if v==first else 1 for v in yarr], dtype=int)

def find_best_threshold(y_true, probas):
    # search for threshold maximizing accuracy (coarse then fine)
    from sklearn.metrics import accuracy_score
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        pred = (probas >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc; best_t = t
    return float(best_t), float(best_acc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='Limit number of train samples for a quick run (0=use all)')
    parser.add_argument('--prefilter_k', type=int, default=500)
    parser.add_argument('--aba_pop', type=int, default=12)
    parser.add_argument('--aba_iter', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    npz_tr = ROOT / 'data' / 'processed' / 'train_processed.npz'
    npz_te = ROOT / 'data' / 'processed' / 'test_processed.npz'
    if not npz_tr.exists():
        raise FileNotFoundError('train_processed.npz not found; run preprocessing first')
    arr = np.load(npz_tr, allow_pickle=True)
    X = arr['X']; y = arr['y']

    if args.limit and args.limit>0 and args.limit < X.shape[0]:
        idx = np.random.RandomState(args.seed).choice(X.shape[0], args.limit, replace=False)
        X = X[idx]; y = y[idx]

    y_bin = map_to_binary(y)
    print('Loaded train:', X.shape, 'Binary class counts:', dict(Counter(y_bin)))

    # prefilter
    from sklearn.feature_selection import f_classif
    F, p = f_classif(X, y_bin)
    K = min(args.prefilter_k, X.shape[1])
    topk_idx = np.argsort(F)[-K:][::-1]
    X_topk = X[:, topk_idx]
    print('Top-K shape:', X_topk.shape)

    # run ABA on top-K to reduce features further
    try:
        from src.feature_selection.aba import ArtificialButterfly
        use_aba = True
    except Exception:
        use_aba = False
    selected_idx = topk_idx
    aba_best_score = None
    if use_aba:
        print('Running ABA on top-K (pop=%d iter=%d) ...' % (args.aba_pop, args.aba_iter))
        aba = ArtificialButterfly(pop_size=args.aba_pop, n_iter=args.aba_iter, random_state=args.seed)
        def fitness_wrapper(Xsub, ysub):
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            import lightgbm as lgb
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
            clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=args.seed, n_jobs=1)
            scores = cross_val_score(clf, Xsub, ysub, cv=skf, scoring='accuracy', n_jobs=1)
            return float(np.mean(scores))
        best_mask, best_score = aba.fit(X_topk, y_bin, fitness_wrapper)
        aba_best_score = float(best_score)
        sel = best_mask.astype(bool)
        selected_idx = topk_idx[sel]
        print('ABA selected %d features, score=%.4f' % (sel.sum(), best_score))

    X_sel = X[:, selected_idx]
    print('Final selected feature matrix shape:', X_sel.shape)

    # train/val/test split from train file (we'll hold out 20% for final test)
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, val_idx = next(sss.split(X_sel, y_bin))
    X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
    y_tr, y_val = y_bin[tr_idx], y_bin[val_idx]

    # compute class weights
    cnt = Counter(y_tr.tolist()); total = len(y_tr)
    class_w = {k: total/(len(cnt)*v) for k,v in cnt.items()}
    sw_tr_full = np.array([class_w[int(yy)] for yy in y_tr])

    # candidate LightGBM param sets (small set to try)
    candidates = [
        {'n_estimators':500, 'learning_rate':0.03, 'num_leaves':31},
        {'n_estimators':800, 'learning_rate':0.02, 'num_leaves':63},
        {'n_estimators':400, 'learning_rate':0.05, 'num_leaves':127},
    ]

    try:
        import lightgbm as lgb
    except Exception:
        raise RuntimeError('lightgbm required for this runner')

    best_overall = {'acc': -1.0}
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    # evaluate each candidate using OOF CV (5-fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for cand in candidates:
        print('Evaluating candidate:', cand)
        oof_proba = np.zeros(X_tr.shape[0])
        for i, (it, iv) in enumerate(skf.split(X_tr, y_tr)):
            clf = lgb.LGBMClassifier(random_state=args.seed, n_jobs=1, **cand)
            sw = np.array([class_w[int(yy)] for yy in y_tr[it]])
            # early stopping via callbacks on fold val
            try:
                cb = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
            except Exception:
                cb = []
            clf.fit(X_tr[it], y_tr[it], sample_weight=sw, eval_set=[(X_tr[iv], y_tr[iv])], eval_metric='auc', callbacks=cb)
            try:
                oof_proba[iv] = clf.predict_proba(X_tr[iv])[:,1]
            except Exception:
                oof_proba[iv] = clf.predict(X_tr[iv])

        # tune threshold on OOF
        t_best, t_acc = find_best_threshold(y_tr, oof_proba)
        oof_pred = (oof_proba >= t_best).astype(int)
        oof_f1 = f1_score(y_tr, oof_pred)
        print('Candidate OOF best_acc=%.4f best_thr=%.3f OOF_f1=%.4f' % (t_acc, t_best, oof_f1))

        # train final on full train (X_tr) and evaluate on X_val
        clf_full = lgb.LGBMClassifier(random_state=args.seed, n_jobs=1, **cand)
        try:
            cb = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        except Exception:
            cb = []
        clf_full.fit(X_tr, y_tr, sample_weight=sw_tr_full, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=cb)
        try:
            val_proba = clf_full.predict_proba(X_val)[:,1]
        except Exception:
            val_proba = clf_full.predict(X_val)
        val_pred = (val_proba >= t_best).astype(int)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred)
        try:
            val_auc = roc_auc_score(y_val, val_proba)
        except Exception:
            val_auc = None

        print('Candidate final val_acc=%.4f val_f1=%.4f val_auc=%s' % (val_acc, val_f1, str(val_auc)))

        if val_acc > best_overall.get('acc', -1):
            best_overall = {'acc': val_acc, 'f1': val_f1, 'auc': val_auc, 'thr': t_best, 'params': cand, 'model': clf_full}

    # optionally try XGBoost stacking
    try:
        import xgboost as xgb
        has_xgb = True
    except Exception:
        has_xgb = False

    if has_xgb:
        print('XGBoost available: attempting simple stacking with best LGB model')
        # retrain best LGB on X_tr and get OOF predictions (already have best model)
        # Train a simple XGB on same features and build meta-learner
        # For speed, we'll train one XGB candidate
        xgb_clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=args.seed)
        xgb_clf.fit(X_tr, y_tr)
        try:
            lgb_full = best_overall['model']
            proba_lgb = lgb_full.predict_proba(X_val)[:,1]
            proba_xgb = xgb_clf.predict_proba(X_val)[:,1]
            meta_X = np.column_stack([proba_lgb, proba_xgb])
            from sklearn.linear_model import LogisticRegression
            meta = LogisticRegression(max_iter=400)
            meta.fit(meta_X, y_val)
            meta_proba = meta.predict_proba(meta_X)[:,1]
            m_thr, m_acc = find_best_threshold(y_val, meta_proba)
            if m_acc > best_overall['acc']:
                best_overall.update({'acc': m_acc, 'thr': m_thr, 'meta': meta, 'xgb': xgb_clf})
                print('Stacking improved val acc to', m_acc)
        except Exception as e:
            print('Stacking failed:', e)

    # evaluate on separate processed test set if present
    results = {'aba_best_score': aba_best_score, 'selected_feature_count': int(X_sel.shape[1])}
    if npz_te.exists():
        arrt = np.load(npz_te, allow_pickle=True)
        Xte = arrt['X']; yte = arrt['y']
        yte_bin = map_to_binary(yte)
        # use selected_idx to slice
        Xte_sel = Xte[:, selected_idx]
        best_model = best_overall['model']
        try:
            te_proba = best_model.predict_proba(Xte_sel)[:,1]
        except Exception:
            te_proba = best_model.predict(Xte_sel)
        thr = best_overall['thr']
        te_pred = (te_proba >= thr).astype(int)
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        results.update({'test_acc': float(accuracy_score(yte_bin, te_pred)), 'test_f1': float(f1_score(yte_bin, te_pred))})
        try:
            results['test_auc'] = float(roc_auc_score(yte_bin, te_proba))
        except Exception:
            results['test_auc'] = None

    # save best model and results
    # joblib can't serialize nested LGB scikit estimators with callbacks reliably in some versions; wrap metadata
    model_path = MOD / 'best_model_for_99.joblib'
    joblib.dump({'model': best_overall.get('model'), 'params': best_overall.get('params'), 'threshold': best_overall.get('thr')}, model_path)
    results.update({'best_val_acc': best_overall.get('acc'), 'best_val_f1': best_overall.get('f1'), 'best_params': best_overall.get('params'), 'best_threshold': best_overall.get('thr')})
    with open(RES / 'best_for_99_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print('Saved model to', model_path)
    print('Results:', results)

if __name__ == '__main__':
    main()
