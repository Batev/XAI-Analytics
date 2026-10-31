def get_pipeline(ct: ColumnTransformer, algorithm: Algorithm, data_size: int) -> GridSearchCV:
    """
    Returns a new pipeline depending on the chosen algorithm.
    :param ct: A column transformer used in the pipeline
    :param algorithm: Algorithm that is then used for training the model
    :param data_size: The size of the data used for training.
    :return: A new pipeline
    """
    grid_values = {}
    weights = np.linspace(0.0, 0.99, 200)

    if algorithm is Algorithm.LOGISTIC_REGRESSION:
        estimator = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_NUMBER)
        grid_values = [
            {
                'model__solver': ['liblinear'],
                # == [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
                'model__C': np.logspace(-4, 4, 9),
                'model__penalty': ['l1', 'l2'],
                'model__dual': [False]
            },
            {
                'model__solver': ['saga'],
                'model__C': np.logspace(-4, 4, 9),
                'model__penalty': ['l1', 'l2', 'elasticnet', 'none']
            },
            {
                'model__solver': ['newton-cg', 'lbfgs', 'sag'],
                'model__C': np.logspace(-4, 4, 9),
                'model__penalty': ['l2', 'none']
            }
        ]
    elif algorithm is Algorithm.DECISION_TREE:
        estimator = DecisionTreeClassifier(random_state=RANDOM_NUMBER)
        grid_values = [
            {
                # 'model__criterion': ['gini', 'entropy'],
                'model__criterion': ['gini'],
                # 'model__splitter': ['best', 'random'],
                'model__splitter': ['best'],
                # 'model__max_depth': np.linspace(2, 64, num=10, dtype=int),
                'model__max_depth': [8],
                # 'model__max_features': ['auto', 'sqrt', 'log2', 0.25, 0.5, 0.75, 1.0, None]
                'model__max_features': [1.0]
            }]
    elif algorithm is Algorithm.RANDOM_FOREST:
        estimator = RandomForestClassifier(
            random_state=RANDOM_NUMBER,
            n_jobs=-1)
        grid_values = [
            {
                # 'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__n_estimators': np.linspace(100, 10000, num=10, dtype=int),
                'model__max_depth': np.linspace(2, 64, num=10, dtype=int),
                'model__max_features': ['sqrt', 'log2', 0.25, 0.5, 0.75, 1.0, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__bootstrap': [True, False]
            }
        ]
    elif algorithm is Algorithm.XGB:
        estimator = XGBClassifier(
            random_state=RANDOM_NUMBER,
            n_jobs=-1)
    elif algorithm is Algorithm.SVC:
        grid_values = {'model__C': np.logspace(-4, 4, 9),
                       # == [0.000010, 0.000100, 0.001000, 0.010000, 0.100000, 1.000000]
                       'model__gamma': np.logspace(-5, 0, 6)
                       }
        estimator = SVC(kernel="rbf",
                        random_state=RANDOM_NUMBER,
                        probability=True,
                        verbose=False)
    elif algorithm is Algorithm.LINEAR_REGRESSION:
        estimator = LinearRegression(n_jobs=-1)
    elif algorithm is Algorithm.SVM:
        raise NotImplementedError
    else:
        raise NotImplementedError

    pipeline = Pipeline([("preprocessor", ct), ("model", estimator)])
    # scoring = ['accuracy', 'f1']
    scoring = 'accuracy'
    if algorithm is Algorithm.LOGISTIC_REGRESSION or algorithm is Algorithm.DECISION_TREE:
        search = GridSearchCV(estimator=pipeline, param_grid=grid_values, scoring=scoring,
                              cv=3, return_train_score=True, verbose=2, n_jobs=-1)
    else:
        search = RandomizedSearchCV(estimator=pipeline, param_distributions=grid_values, scoring=scoring,
                                    cv=3, random_state=RANDOM_NUMBER,
                                    refit="accuracy", verbose=2,
                                    return_train_score=True, n_iter=15,
                                    n_jobs=-1)
    return search


def train_model(model_type: ModelType, split: Split, df_x: pd.DataFrame, df_y: pd.Series) -> \
        (Pipeline, pd.DataFrame, pd.Series):
    """
    Train a model on a dataset.
    :param model_type: The type of model that has to be trained (training algorithm, ...)
    :param split: Whether a balanced or imbalanced split shall be used. If balanced split is used
    a feature for the split shall be selected
    :param df_x: The features of the dataset
    :param df_y: The target of the dataset
    :return: A model pipeline, X_train, y_train, X_test, y_test
    """

    num_features, cat_features = _divide_features(df_x)

    log.debug("Numerical features: {}".format(num_features))
    log.debug("Categorical features: {}".format(cat_features))

    # Transform the categorical features to numerical
    preprocessor = get_column_transformer(num_features, cat_features)

    X_train, X_test, y_train, y_test = get_split(preprocessor, split, cat_features, df_x, df_y)
    model = get_pipeline(preprocessor, model_type.algorithm, int(df_y.size))

    start = time.time()
    # Now we can fit the model on the whole training set and calculate accuracy on the test set.
    model.fit(X_train, y_train)
    end = time.time()
    _log_elapsed_time(start, end, "training a {} classifier is".format(model_type.algorithm.name))

    print("tuned hpyerparameters :(best parameters) ", model.best_params_)
    print("accuracy :", model.best_score_)

    model = model.best_estimator_

    # Generate predictions
    y_pred = model.predict(X_test)

    # classification
    if model_type.problem_type == ProblemType.CLASSIFICATION:
        log.info("Model accuracy: {}".format(accuracy_score(y_test, y_pred)))
        log.info("Classification report: \n{}".format(classification_report(y_test, y_pred)))
        # plot confusion matrix and roc curve
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
        ax1.set_title("Confusion Matrix")
        ax2.set_title("ROC Curve")
        plot_confusion_matrix(model, X_test, y_test, ax=ax1)
        plot_roc_curve(model, X_test, y_test, ax=ax2)
        fig.tight_layout()
        plt.show()
    # regression
    elif model_type.problem_type == ProblemType.REGRESSION:
        log.info("R2 score : %.2f" % r2_score(y_test, y_pred))
        log.info("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
        log.info("RMSE number:  %.2f" % pd.np.sqrt(mean_squared_error(y_test, y_pred)))

    return model, X_train, y_train, X_test, y_test