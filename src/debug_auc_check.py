import joblib, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
ROOT=Path('c:/ECHO/Projects/Personal_Projects/Fruty')
model=joblib.load(ROOT/'models'/'final_lgb_model.joblib')
arr=np.load(ROOT/'data'/'processed'/'test_processed.npz', allow_pickle=True)
X=arr['X']; y=arr['y']
print('model.classes_:', getattr(model,'classes_',None))
proba_all = model.predict_proba(X)
print('proba_all shape:', proba_all.shape)
print('proba_all stats col0 mean,std:', proba_all[:,0].mean(), proba_all[:,0].std())
print('proba_all stats col1 mean,std:', proba_all[:,1].mean(), proba_all[:,1].std())
# map y to binary
if np.array(y).dtype.kind in ('U','S','O'):
    y_bin = np.array([0 if 'normal' in str(v).lower() else 1 for v in y])
else:
    y_bin = (np.array(y) != 0).astype(int)
print('y_bin distribution:', np.unique(y_bin, return_counts=True))
for i in [0,1]:
    try:
        auc=roc_auc_score(y_bin, proba_all[:,i])
    except Exception as e:
        auc=str(e)
    print(f'auc for proba col {i}:', auc)
if getattr(model,'classes_',None) is not None:
    cls = model.classes_
    for idx,cl in enumerate(cls):
        try:
            auc=roc_auc_score(y_bin, proba_all[:,idx])
        except Exception as e:
            auc=str(e)
        print('class',cl,'col idx',idx,'auc',auc)
print('done')
