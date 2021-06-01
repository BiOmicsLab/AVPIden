import numpy as np
import os, time, re
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ranksums
import json
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc, fbeta_score
from imblearn.metrics import geometric_mean_score
from Args import stage, descriptors, models, imb_strategies, SPLIT_SEED

_Families = ["Bunyaviridae", "Coronaviridae", "Flaviviridae", "Herpesviridae", 
             "Orthomyxoviridae", "Others", "Paramyxoviridae", "Retroviridae"]
_Viruses = ["ANDV", "FIV", "HCV", "HIV", "HPIV3", "HSV1", "INFVA", "Other", "RSV", "SARSCoV", "SNV"]

def evaluate(X, y, estm):
    # Performance metrics
    y_pred = estm.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    # ROC curve
    try:
        if "decision_function" not in dir(estm):
            y_prob = estm.predict_proba(X)[:, 1]
        else:
            y_prob = estm.decision_function(X)
        pre, rec, _ = precision_recall_curve(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)
        aucroc = auc(fpr, tpr)
        aucpr = auc(rec, pre)
    except AttributeError:
        print("Classifier don't have predict_proba or decision_function, ignoring roc_curve.")
        pre, rec = None, None
        fpr, tpr = None, None
        aucroc = None
        aucpr = None
    eval_dictionary = {
        "CM": confusion_matrix(y, y_pred),  # Confusion matrix
        "ACC": (tp + tn) / (tp + fp + fn + tn),  # accuracy
        "F1": fbeta_score(y, y_pred, beta=1),
        "F2": fbeta_score(y, y_pred, beta=2),
        "GMean": geometric_mean_score(y, y_pred, average='binary'),
        "SEN": tp / (tp + fn),
        "PREC": tp / (tp + fp),
        "SPEC": tn / (tn + fp),
        "MCC": matthews_corrcoef(y, y_pred),
        "PRCURVE": {"precision": pre, "recall": rec, "aucpr": aucpr},
        "ROCCURVE": {"fpr": fpr, "tpr": tpr, "aucroc": aucroc}
    }
    return eval_dictionary


def train_gridCV_imb(X, y, estm, sampler, param_grid, scoring="recall"):
    if sampler is not None:
        X_res, y_res = sampler.fit_resample(X, y)
    else:
        X_res, y_res = X, y
    grid = GridSearchCV(estm, param_grid,
                        scoring=scoring, cv=5, n_jobs=-1)  # use recall for finding results
    grid.fit(X_res, y_res)
    return grid.best_estimator_, grid.best_params_, grid.best_score_, X_res, y_res


def imb_classification(X_tr, y_tr, X_te, y_te, imb_samplers, estimator,
                       grid_params=None, X_u=None, iteration=16, scoring="recall"):
    # Imbalanced learning with different imbalance strategies
    perf_df = []
    pr_curves_all, roc_curves_all = {}, {}
    best_estimators_dict, best_params_dict = {}, {}
    for sampler_name, sampler in imb_samplers.items():
        # Train with sampled data
        acc_all, sen_all, prec_all, spec_all = [], [], [], []
        f1_all, f2_all, gmean_all = [], [], []
        mcc_all, aucpr_all, aucroc_all = [], [], []
        pr_cur, roc_cur = None, None  # Not support for mean PR-curve yet
        best_estimator, best_params = None, None
        # determine the iteration of sampler using randomized strategy
        if 'random_state' not in dir(sampler):
            iter_use = 1
        else:
            if sampler.random_state is not None:
                print("Perform single_iter sampling since random_state is set.")
                iter_use = 1
            else:
                iter_use = iteration
        print("Module of sampling strategy: {}, Evaluating with {:2d} iterations".format(sampler_name, iter_use))
        for ii in range(iter_use):
            if 'random_state' in dir(sampler):
                if sampler.random_state is None:
                    sampler.random_state = 0 + ii
            estm, params, _, _, _ = train_gridCV_imb(X_tr, y_tr, estimator, sampler, grid_params, scoring=scoring)
            eval_d = evaluate(X_te, y_te, estm)  # evaluate with test data
            # judge the best estimator
            if len(acc_all) == 0:
                best_estimator = estm
                best_params = params
            else:
                best_estimator = estm if eval_d['SEN'] > sen_all[-1] else best_estimator
                best_params = params if eval_d['SEN'] > sen_all[-1] else best_params
            acc_all.append(eval_d['ACC'])
            f1_all.append(eval_d['F1'])
            f2_all.append(eval_d['F2'])
            gmean_all.append(eval_d['GMean'])
            sen_all.append(eval_d['SEN'])
            prec_all.append(eval_d['PREC'])
            spec_all.append(eval_d['SPEC'])
            mcc_all.append(eval_d['MCC'])
            pr_cur = eval_d['PRCURVE']
            roc_cur = eval_d['ROCCURVE']
            aucpr_all.append(pr_cur['aucpr'])
            aucroc_all.append(roc_cur['aucroc'])

        perf_df.append({
            "Sampler": sampler_name,
            "ACC(%)": "{:.2f}".format(np.mean(acc_all) * 100),
            "GMean(%)": "{:.2f}".format(np.mean(gmean_all) * 100),
            "SEN(%)": "{:.2f}".format(np.mean(sen_all) * 100),
            "SPEC(%)": "{:.2f}".format(np.mean(spec_all) * 100),
            "AUCROC(%)": ("{:.2f}".format(np.mean(aucroc_all) * 100) if X_u is None else "Unavailable")
        })

        pr_curves_all[sampler_name] = pr_cur
        roc_curves_all[sampler_name] = roc_cur
        best_estimators_dict[sampler_name] = best_estimator
        best_params_dict[sampler_name] = best_params
    perf_df = pd.DataFrame(perf_df)
    print("Table: Performance of selected samplers")
    print(perf_df)
    return pr_curves_all, roc_curves_all, perf_df, best_estimators_dict, best_params_dict


"""
Function for plot pr/roc curves
"""


def plot_pr_roc(pr_curves_all, roc_curves_all, dpi=600):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)
    for sampler_name, curve_data in pr_curves_all.items():
        prec = curve_data['precision']
        recall = curve_data['recall']
        aucpr = curve_data['aucpr']
        ax[0].plot(recall, prec, lw=1.2, label="{:s} (AUC = {:.3f})".format(sampler_name, aucpr))
    ax[0].legend()
    ax[0].set_title("Precision-Recall Curve")
    ax[0].set_xlabel("recall")
    ax[0].set_ylabel("precision")
    ax[0].set_xlim([-0.05, 1.05])
    ax[0].set_ylim([-0.05, 1.05])
    for sampler_name, curve_data in roc_curves_all.items():
        fpr = curve_data['fpr']
        tpr = curve_data['tpr']
        aucroc = curve_data['aucroc']
        ax[1].plot(fpr, tpr, lw=1.2, label="{:s} (AUC = {:.3f})".format(sampler_name, aucroc))
    ax[1].legend()
    ax[1].set_title("Receiver Operating Curve")
    ax[1].set_xlabel("fpr")
    ax[1].set_ylabel("tpr")
    ax[1].set_xlim([-0.05, 1.05])
    ax[1].set_ylim([-0.05, 1.05])
    return fig


"""
Function of indexing the descriptors
"""
def match_descriptors_ind(df,descriptors):
    # here, 2
    inds = [ii for ii in range(2, len(df.columns)) for dd in descriptors if re.match(r'(.*)\|(.*)', df.columns[ii], re.M|re.I).group(1) == dd]
    return df[df.columns[inds]]


if __name__ == "__main__":
    """
    Establish result directory
    """
    if not os.path.exists("results"):
        os.mkdir("results")
    time_now = int(round(time.time() * 1000))
    time_now = time.strftime("%Y-%m-%d_%H-%M", time.localtime(time_now / 1000))
    clsdir = os.path.join("results", "rslt_{}".format(time_now))
    os.makedirs(clsdir)
    with open(os.path.join(clsdir, "arguments.txt"), 'w') as file:
        file.write("Used features:\n")
        file.write("{:s}\n".format(str(descriptors)))
        file.write("Used models:\n")
        file.write("{:s}\n".format(str(models)))
        file.write("imbalance strategies:\n")
        file.write("{:s}\n".format(str(imb_strategies)))
    """
    Construct Data
    """
    if stage == "Entire":
        clsdata = {
            "Entire(non-AMP_only)": {
                "pos": pd.read_csv("./data/Entire/set/Anti-Virus.csv"), 
                "neg": pd.read_csv("./data/Entire/set/non-AMP.csv")
                },
            "Entire(non-AVP_incl)": {
                "pos": pd.read_csv("./data/Entire/set/Anti-Virus.csv"),
                "neg": pd.concat([
                    pd.read_csv("./data/Entire/set/non-AMP.csv"), 
                    pd.read_csv("./data/Entire/set/non-AVP.csv")], axis=0)
                }
        }
    elif stage == "ByFamily":
        clsdata = dict()
        for fm in _Families:
            clsdata[fm] = {
                "pos": pd.read_csv("./data/ByFamily/set/{:s}.csv".format(fm)),
                "neg": pd.read_csv("./data/ByFamily/set/non-{:s}.csv".format(fm))
            }
    elif stage == "ByVirus":
        clsdata = dict()
        for vm in _Viruses:
            clsdata[vm] = {
                "pos": pd.read_csv("./data/ByVirus/set/{:s}.csv".format(vm)),
                "neg": pd.read_csv("./data/ByVirus/set/non-{:s}.csv".format(vm))
            }
    else:
        raise ValueError("Invalid stage code.")
    """
    Perform clssification based on k-fold split.
    """
    for tabind, tabdat in clsdata.items():
        # Make directory for the classification
        print("Perform classification for {:s}...".format(tabind))
        tabdir = os.path.join(clsdir, tabind)
        # Create data
        X = pd.concat([tabdat['pos'], tabdat['neg']], axis=0)
        X = X.reset_index(drop=True)
#         X = match_descriptors_ind(X, descriptors)
        print(X)
        y = np.concatenate([np.ones(len(tabdat['pos'])), np.zeros(len(tabdat['neg']))])
        assert len(X) == len(y)
        # Perform stratified k-fold. (k=4, randomized)
        skfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=SPLIT_SEED)
        for k, (train_split, test_split) in enumerate(skfold.split(X, y)):
            print("Resolving fold: {:d}".format(k))
            # Create Fold Directory
            folddir = os.path.join(tabdir, "fold{:d}".format(k))
            os.makedirs(folddir)
            # Create Training/Test data
            X_train, y_train = X.iloc[train_split], y[train_split]
            X_test, y_test = X.iloc[test_split], y[test_split]
            # Dump the dataset
            joblib.dump({'X': X_train, 'y': y_train}, os.path.join(folddir, "train_set.bin"))
            joblib.dump({'X': X_test, 'y': y_test}, os.path.join(folddir, "test_set.bin"))
            X_train = match_descriptors_ind(X_train, descriptors)
            X_test = match_descriptors_ind(X_test, descriptors)
            # Fit classifiers
            for nmdl, mmdl in models.items():
                mdldir = os.path.join(folddir, nmdl)
                os.makedirs(mdldir)
                clf = mmdl['model']
                clf_params = mmdl['param_grid']
                if 'imblearn.ensemble' not in clf.__module__:
                    prcs, rocs, perfs, estms, params_estm = imb_classification(X_train, y_train, X_test, y_test,
                                                                imb_samplers=imb_strategies,
                                                                estimator=clf,
                                                                grid_params=clf_params, iteration=1)
                else:
                    prcs, rocs, perfs, estms, params_estm = imb_classification(X_train, y_train, X_test, y_test,
                                                                            imb_samplers={"Default": None},
                                                                            estimator=clf,
                                                                            grid_params=clf_params, iteration=1)
                # Save the estimators, parameters, roc/pr results
                perfs.to_csv(os.path.join(mdldir, "performances.csv"), index=False)
                for spl_name in prcs.keys():
                    spldir = os.path.join(mdldir, spl_name)
                    os.makedirs(spldir)
                    np.save(os.path.join(spldir, "pr.npy"), prcs[spl_name])
                    np.save(os.path.join(spldir, "roc.npy"), rocs[spl_name])
                    joblib.dump(estms[spl_name], os.path.join(spldir, "estimator.joblib"))
                    json.dump(params_estm[spl_name], open(os.path.join(spldir, "params.json"), 'w'))
