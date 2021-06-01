from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier as EEC
from imblearn.ensemble import BalancedRandomForestClassifier as BRF
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

SPLIT_SEED = 810
CLASSIFIER_SEED = 114
IMB_SEED = 514

descriptors = ["AAC", "DPC", "CKSAAGP", "PAAC", "PHYC"]
stage = "Entire"

# models = {"GBDT": {"model": GBC(random_state=CLASSIFIER_SEED), "param_grid":{"n_estimators": [150, 256, 300, 512, 1024, 1500], "learning_rate":[0.1, 0.2, 0.05]}},
#           "RF": {"model": RFC(random_state=CLASSIFIER_SEED), "param_grid":{"n_estimators": [150, 256, 300, 512, 1024, 1500]}}}

# imb_strategies = {"SMOTE": SMOTE(random_state=IMB_SEED, k_neighbors=5), "NearMiss":NearMiss(version=3, n_neighbors=5)}

models = {
    "BalancedRF": {
        "model": BRF(random_state=CLASSIFIER_SEED),
        "param_grid": {'n_estimators':[100, 200, 400, 800, 1200]}
    },
    "RandomForest":{
        "model": RFC(random_state=CLASSIFIER_SEED),
        "param_grid": {'n_estimators':[100, 200, 400, 800, 1200]}
    }
}

imb_strategies = {
    "default": None
}
