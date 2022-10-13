from lib import *

MIL_MODELS = ['misvm', 'MissSVM', 'sbMIL', 'sMIL', 'MICA', 'NSK', 'SIL']
LASTCOL = 45

def initial_processing(df):
    df = df.drop("Dates", 1)
    df = df.drop("Day", 1)
    df = df.drop("Shift", 1)
    df = df.interpolate(method ='linear', limit_direction ='forward')
    df = df.fillna(0)
    # df = df.drop("ID", 1)
    # df = df.dropna()
    df["Labels"] = df["Labels"].astype("int64")
    return df

def create_correlation_matrix(df, filename=None):
    plt.figure(figsize=(36, 40))
    cor = df.corr()
    cor.fillna(0)
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()

def get_pos_neg_sample_count(X_train, X_test, y_train, y_test):
    num_pos_train = list(Counter(y_train).values())[1]
    num_neg_train = list(Counter(y_train).values())[0]
    num_pos_test = list(Counter(y_test).values())[1]
    num_neg_test = list(Counter(y_test).values())[0]

    baseline = (num_pos_train + num_pos_test) /  (num_neg_train + num_pos_train + num_pos_test + num_neg_test)
    return num_pos_train, num_neg_train, num_pos_test, num_neg_test, baseline

def get_model(model_type, y_train=None, **params):
    max_iters = 30
    if model_type == 'xgb':
        "RETURNING XGB"
        if y_train is not None:
            scale_pos_weight = list(Counter(y_train).values())[0] / list(Counter(y_train).values())[1]
            params['scale_pos_weight'] = scale_pos_weight
        clf = xgb.XGBClassifier(**params)
    elif model_type == 'svm':
        clf = svm.SVC(**params, probability=True)
    elif model_type == 'rfc':
        clf = rfc(max_depth=5, random_state=0)
    elif model_type == 'lr':
        clf = lr(random_state=0)
    elif model_type == 'misvm':
        clf = misvm.MISVM(**params)
    elif model_type == 'MissSVM':
        clf = misvm.MissSVM(**params)
    elif model_type == 'sMIL':
        clf = misvm.sMIL(kernel='linear', C=1.0)
    elif model_type == 'sbMIL':
        clf = misvm.sbMIL(**params)
    elif model_type == 'MICA':
        clf = misvm.MICA(kernel='linear', C=1.0, max_iters=max_iters)
    elif model_type == 'NSK':
        clf = misvm.NSK(**params)
    elif model_type == 'SIL':
        clf = misvm.SIL(**params)
    return clf, params

''' AUC, ROC, Precision, Recall, F1 Score'''
def metrics_for_binary_classification(model_type, writer, id, y_test, y_pred, labels=[0, 1], train_test='TEST'):
    writer.writerow(['ROC AUC METRICS %s SET' % train_test])
    auc_score = roc_auc_score(y_test, y_pred)
    fpr1, tpr1, thresh1 = roc_curve(y_test, y_pred, pos_label=1)
    writer.writerow(['AUC SCORE ', str(auc_score)])
    writer.writerow(['TRUE POSITIVE RATE', str(tpr1)])
    writer.writerow(['FALSE POSITIVE RATE', str(fpr1)])
    writer.writerow(['TRESHOLD', str(thresh1)])
    class_report = classification_report(y_test, y_pred)
    writer.writerow(['PRECISION SCORE', precision_score(y_test, y_pred)])
    writer.writerow(['RECALL SCORE', recall_score(y_test, y_pred)])
    writer.writerow(['F1 SCORE', f1_score(y_test, y_pred)])

    plt.clf()
    plt.rcParams.update({'font.size': 22})
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label=model_type)
    plt.title('ROC CURVE FOR %s' % model_type)
    plt.xlabel('FALSE POSITIVE RATE')
    plt.ylabel('TRUE POSITIVE RATE')
    plt.savefig(FILEPATH % id + 'graphs/' + 'ROC_%s' % model_type, dpi=300)

''' Used to write generic metrics like accuracy, confusion matrix, rmse '''
def write_prediction_stats(writer, y_train, y_test, y_train_pred, y_pred, c_m1, c_m2):
    acc = "TRAIN ACCURACY: " + \
        str(metrics.accuracy_score(y_train, y_train_pred) * 100)
    writer.writerow(['TRAIN ACCURACY', str(
        metrics.accuracy_score(y_train, y_train_pred))])
    acc = "TEST ACCURACY: " + str(metrics.accuracy_score(y_test, y_pred) * 100)
    writer.writerow(['TEST ACCURACY', str(
        metrics.accuracy_score(y_test, y_pred))])
    writer.writerow(['NUM TRAIN', str(list(Counter(y_train).values()))])
    writer.writerow(['NUM TEST', str(list(Counter(y_test).values()))])
    writer.writerow(['CONFUSION MATRIX TRAIN', str(c_m1)])
    writer.writerow(['CONFUSION MATRIX TEST', str(c_m2)])

''' For Multi-cv_type Learning 

Divides all samples into bags of size 12 +/- 2. If at least one positive cv_type
is found in the bag, then the entire bag will be labeled positive.
'''
def create_bags(df):
    bag_labels = []
    bags = [g.iloc[:, 3:LASTCOL].to_numpy(dtype=np.float32) for _, g in df.groupby(['ID'])]
    bag_labels = [g.Labels.to_numpy()[0] for _, g in df.groupby(['ID'])]

    return np.array(bags, dtype=object), np.array(bag_labels)

def get_model_weights(y_train):
    classes_weights = list(class_weight.compute_class_weight('balanced',
                           np.unique(y_train), y_train))
    print("CLASSES WEIGHTS ", classes_weights)
    weights = np.ones(len(y_train), dtype='float')
    for i, val in enumerate(y_train):
        weights[i] = 1 / classes_weights[val-1]
    return weights

def run_model(model_type, df, id, writer, cross_val=False, multi_cv_type=False, binary=False, model_weights=False, **kwargs):
    if multi_cv_type:
        clf = get_model(model_type)
        labels = np.where(df.Labels > 2, 1, -1)
        size = len(df['Labels'])
        bags, bag_labels, labels = create_bags(
            df.iloc[:, 3:LASTCOL], labels, 12, 2, size, 1)
        train_bags, test_bags, y_train, y_test = train_test_split(
            bags, bag_labels, test_size=0.3, random_state=6)
        y_train_pred, y_pred, c_m1, c_m2 = classify(
            clf, train_bags, y_train, test_bags, y_test, labels=[-1, 1], model_weights=model_weights)
        write_prediction_stats(writer, y_train, y_test, np.sign(
            y_train_pred), np.sign(y_pred), c_m1, c_m2)
        metrics_for_binary_classification(model_type, writer, id, y_train, np.sign(
            y_train_pred), labels=[-1, 1], train_test='TRAIN')
        metrics_for_binary_classification(
            model_type, writer, id, y_test, np.sign(y_pred), labels=[-1, 1])
    else:  # single cv_type learning
        labels = df.Labels
        if binary: labels = [1 if val > 2 else 0 for val in df.Labels]
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 3:LASTCOL], labels, test_size=0.2,
                                                            shuffle=True, random_state=0)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)
        if model_type == 'xgb': scale_pos_weight = X_train[0] / X_train[1]
        clf, _ = get_model(model_type, **kwargs)
        max_label = max(labels)
        unique_labels = list(range(0, max_label+1))
        y_train_pred, y_pred, c_m1, c_m2 = classify(
            clf, X_train, y_train, X_test, y_test, labels=unique_labels, cv_type='single', model_weights=model_weights)
        write_prediction_stats(writer, y_train, y_test,
                               y_train_pred, y_pred, c_m1, c_m2)

        if binary:
            metrics_for_binary_classification(
                model_type, writer, id, y_train, y_train_pred, train_test='TRAIN')
            metrics_for_binary_classification(
                model_type, writer, id, y_test, y_pred)

def run_models_and_collect_results(id, df, models, filename, cross_val=False, multi_cv_type=False,
                                   binary=False, model_weights=False):
    with open(filename % id, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for model in models:
            print("Current Model is: %s \n" % str(model))
            # writer.write("\nCURRENT MODEL IS: %s \n" % str(model))
            writer.writerow(["CURRENT MODEL", str(model)])
            run_model(model, df, id, writer, cross_val=cross_val, multi_cv_type=multi_cv_type, binary=binary,
                      model_weights=model_weights)
            writer.writerow([" "])


def run_generalized_model(model, df, filename, cross_val=False, multi_cv_type=False,
                          binary=False, model_weights=False):
    frames = []
    Path(FILEPATH % "generalized").mkdir(parents=True, exist_ok=True)
    with open(filename % 'generalized', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        Path(FILEPATH % "generalized").mkdir(parents=True, exist_ok=True)
        create_correlation_matrix(df, filename=FILEPATH % 'generalized')

        run_model(model, df, "generalized", writer, multi_cv_type=multi_cv_type,
                  binary=binary, model_weights=model_weights)

def noramlize_data(X_train, X_test, model):
    if model not in MIL_MODELS:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)
    else:
        min_max_scaler = preprocessing.MinMaxScaler()
        _ = [min_max_scaler.partial_fit(x) for x in X_train]
        X_train = [min_max_scaler.transform(x) for x in X_train]
        X_test = [min_max_scaler.transform(x) for x in X_test]
    
    return X_train, X_test

def feature_selector(X_train, y_train, X_test, clf):
    model = SelectFromModel(estimator=clf)
    model.fit(X_train, y_train)
    mask = model.get_support()

    return X_train[:, mask], X_test[:, mask], mask

def inner_loop(id, model, X, y, cv_inner, outer_fold, params, features=False, cv_type='multi_cv'):
    result = 0
    mask = None
    best_auc = 0

    idx = 0
    keys = list(params)
    # History of parameter combinations to reduce run time complexity for the repeated
    # combinations when using MISVM or SVM
    combinations = itertools.product(*map(params.get, keys))
    total = len(list(combinations))*3
    current_params = best_params = {}
    index = 1

    old_params = []
    for values in list(itertools.product(*map(params.get, keys))):
        current_params = dict(zip(keys, values))
        prev_aucs = []
        if cv_type == 'multi_cv':
            curr_C = current_params['C']
            curr_kernel = current_params['kernel']
            curr_dict = {'C': curr_C, 'kernel': curr_kernel}
            if curr_dict in old_params:
                index += 1
                print("ID %s - o%s - iFold %s of 3 (%s/%s) - Old kernel-C combo " % (id, outer_fold+1, fold+1, index, total))
                continue
            else:
                if current_params['kernel'] != 'rbf':
                    old_params.append(curr_dict)
        for fold, (train_ix, test_ix) in enumerate(cv_inner.split(X, y)):
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            clf, current_params = get_model(model, y_train=y_train, **current_params)
            X_train, X_test = noramlize_data(X_train, X_test, model)
            if model not in MIL_MODELS or model != 'svm' and features:
                X_train, X_test, mask = feature_selector(X_train, y_train, X_test, clf)
            y_train_pred, y_pred, y_train_score, y_score, c_m1, c_m2 = classify(clf, X_train, y_train, X_test, 
                y_test, cv_type=cv_type)
            auc_, pr_rc_auc, precision, recall = get_cross_val_results(y_test, y_score)
            prev_aucs.append(auc(recall, precision))
            print("ID %s - o%s - iFold %s of 3 (%s/%s) - roc_auc=%f, pr_rc_auc=%f" %
                  (id, outer_fold+1, fold+1, index, total, auc_, pr_rc_auc))
            index += 1
        if np.mean(prev_aucs) > best_auc:
            best_auc = np.mean(prev_aucs)
            best_params = current_params
    return best_params, mask

def get_cross_val_results(y_test, y_pred, plot=False):
    fpr, tpr, thresholds = metrics.roc_curve(
        y_test, y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)
    pr, rc, thresholds = precision_recall_curve(y_test, y_pred)
    pr_rc_auc = auc(rc, pr)
    if plot:
        return auc_, pr_rc_auc, pr, rc, fpr, tpr
    return auc_, pr_rc_auc, pr, rc

def classify(clf, X_train, y_train, X_test, y_test, labels=None, cv_type='multi_cv', model_weights=False):
    weights = None  # Only for single cv_type learning right now
    if model_weights:
        print("Calculating with Weights...")
        weights = get_model_weights(y_train)
    clf.fit(X_train, y_train)
    if cv_type != 'multi_cv':
        y_train_score = clf.predict_proba(X_train)[:, 1]
        y_score = clf.predict_proba(X_test)[:, 1]
        y_train_pred = clf.predict(X_train)
        y_pred = clf.predict(X_test)
    if labels == None:
        labels = [0, 1]
    if cv_type == 'multi_cv':
        y_train_score = clf.predict(X_train)
        y_score = clf.predict(X_test)
        y_train_pred = np.sign(y_train_score)
        y_pred = np.sign(y_score)
        labels = [-1, 1]
    c_m1 = confusion_matrix(y_train, y_train_pred, labels=labels)
    c_m2 = confusion_matrix(y_test, y_pred, labels=labels)
    return y_train_pred, y_pred, y_train_score, y_score, c_m1, c_m2

def export_cv_metrics(model, fold, col, feature_dict, writer, y_train, y_test,
        y_train_pred, y_pred, c_m1, c_m2, mask):
    writer.writerow(["CURRENT MODEL", str(model)])
    writer.writerow(["CURRENT FOLD", str(fold)])
    write_prediction_stats(writer, y_train, y_test,
        y_train_pred, y_pred, c_m1, c_m2)
    writer.writerow(["SELECTED FEATURES: "])
    final_features = ([col[i] for i in list(np.where(mask)[0])])
    writer.writerow(final_features)
    num_features = len([col[i] for i in list(np.where(mask)[0])])
    for feature in final_features:
        feature_dict[feature] += 1
    writer.writerow([" "])
    return num_features

def plot_cummulative_metrics(segment, baseline, model, test_labels, scores):
    auc_, pr_rc_auc, pr, rc, fpr, tpr = get_cross_val_results(test_labels, scores, plot=True)
    auc_score = roc_auc_score(test_labels, scores)
    f1, ax1 = plt.subplots(1)
    f2, ax2 = plt.subplots(1)
    lw = 2
    # AUC-ROC
    ax1.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax1.plot(fpr, tpr, lw=lw, label="Area = %0.2f" % (auc_score))
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc="lower right")
    f1.savefig('/cluster/home/t62928uhn/datafiles/participant_files_expanded/%s_%s_ROC_AUC.png' %  (segment, model), dpi=300)

    # AUC-PR
    ax2.plot([0, 1], [baseline, baseline], color="navy", lw=lw, linestyle="--")
    ax2.plot(rc, pr, lw=lw, label="Area = %0.2f" % (pr_rc_auc))
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(loc="lower right")
    f2.savefig('/cluster/home/t62928uhn/datafiles/participant_files_expanded/%s_%s_PR_AUC.png' % (segment, model), dpi=300)

def plot_ALE(classifiers, feature_dict, X_train):
    for name in feature_dict:
        f, ax = plt.subplots(9, 4, sharey='all', figsize=(20, 75))
        test = []
        for i, clf in enumerate(classifiers):
            xgb_ale = ALE(clf.predict_proba, feature_names=feature_dict, target_names=[2])
            xgb_exp = xgb_ale.explain(np.array(X_train))
            test.append(xgb_exp.ale_values)
            xgb_exp_avg = np.sum(test, axis=0) / 5
            xgb_exp.ale_values = xgb_exp_avg
            plot_ale(xgb_exp, ax=ax, line_kw={'label': 'Fold %s' % (i+1)})
            f.rcParams['figure.constrained_layout.use'] = True
            f.savefig('feature_plots/features_%s.png' % segment)

def cross_val_loop(df, id, model, features, filename, params=None, colnames=None, write=True, 
                   cv_type='multi_cv', export_file=None):
    num_outer_folds = 5
    num_inner_folds = 3
    results = matrices = []
    scores = test_labels = []
    col = list(df.columns[3:LASTCOL])
    feature_dict = dict.fromkeys(col, 0)
    segment = 'shift' if 'shift' in filename else 'path'
    # Adjust labels based on SIL or MIL method
    if model in MIL_MODELS:
        df.Labels = [1 if val > 2 else -1 for val in df.Labels]
        X, y = create_bags(df)
    else:
        X = df.iloc[:, 3:LASTCOL].to_numpy()
        y = np.array([1 if val > 2 else 0 for val in df.Labels])
    
    classifiers = []
    cv_outer = StratifiedKFold(n_splits=num_outer_folds, random_state=1, shuffle=True)
    if write == True:
        csvfile = open(export_file, 'w', newline='')
        writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL) 
    for fold, (train_ix, test_ix) in enumerate(cv_outer.split(X, y)):
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        num_pos_train, num_neg_train, num_pos_test, num_neg_test, baseline = get_pos_neg_sample_count(
            X_train, X_test, y_train, y_test)
        cv_inner = StratifiedKFold(n_splits=num_inner_folds,
            random_state=1, shuffle=True)
        parameters, mask = inner_loop( id, model, X_train, y_train, cv_inner, fold, 
            params, features=features, cv_type=cv_type)
        # Retrain model using the best parameters
        clf, parameters = get_model(model, y_train=y_train, **parameters)
        X_train, X_test = noramlize_data(X_train, X_test, model)
        if mask is not None:
            X_train = X_train[:, mask]
            X_test = X_test[:, mask]
        clf.fit(X_train, y_train)
        classifiers.append(clf)
        y_train_pred, y_pred, y_train_score, y_score, c_m1, c_m2 = classify(clf, X_train, y_train, X_test, 
            y_test, cv_type=cv_type)
        num_features = "ALL"
        if write and features: num_features = export_cv_metrics(model, fold, col, feature_dict, writer, y_train, y_test,
            y_train_pred, y_pred, c_m1, c_m2, mask)   
        auc_, pr_rc_auc, pr, rc, fpr, tpr = get_cross_val_results(y_test, y_score, plot=True)

        result = [fold, id, num_pos_train, num_neg_train, num_pos_test, 
                  num_neg_test, auc_, pr_rc_auc, np.mean(pr), np.std(pr), 
                  np.mean(rc), np.std(rc), num_features] + list(parameters.values())
        results.append(result)

        # Append all labels and scores to get cummulative ROC_AUC and ROC_PR
        test_labels = np.append(test_labels, y_test)
        scores = np.append(scores, y_score)

    plot_cummulative_metrics(segment, baseline, model, test_labels, scores)
    # if id == 'g' and features: plot_ALE(classifiers, feature_dict, X_train)
    if id != 'g' and features:
        export_features = pd.DataFrame(feature_dict, index=[0])
        feature_filename = '/'.join((filename.split('/'))[:-1]) + ('/%s_%s_feature_export.csv' % (model, segment))
        print(feature_filename)
        export_features.to_csv(feature_filename)
    
    results = pd.DataFrame(results)
    print(results)
    print(colnames)
    results.columns = colnames
    results.to_csv(filename)