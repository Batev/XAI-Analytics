import pandas as pd
import numpy as np
import logging as log
import enum
import time
import math

from functools import partial
from random import choices, randrange
from matplotlib import figure, axes, font_manager, pyplot as plt
from IPython import display
from eli5 import show_weights, explain_weights
from shap import summary_plot, dependence_plot, force_plot
from shap.plots._force import AdditiveForceVisualizer
from lime.explanation import Explanation
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, r2_score,\
    mean_squared_error, plot_confusion_matrix, plot_roc_curve
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier
from pandas.api.types import is_numeric_dtype, is_string_dtype
from multipledispatch import dispatch
from rbo import RankingSimilarity
from pdpbox import pdp
from util.dataset import Datasets, Dataset
from util.model import Algorithm, Model, ModelType, ProblemType
from util.split import Split, SplitTypes


RANDOM_NUMBER = None
TEST_SPLIT_SIZE = 0.3


log.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
log.getLogger().setLevel(log.INFO)

# debug logger generates too much output
log.getLogger('matplotlib.font_manager').disabled = True

# remove DataFrame display limitation
pd.set_option('display.max_columns', 50)


class FeatureImportanceType(enum.Enum):
    ELI5 = 1
    SKATER = 2
    SHAP = 3


class PDPType(enum.Enum):
    PDPBox = 1
    SKATER = 2
    SHAP = 3


class LocalInterpreterType(enum.Enum):
    LIME = 1
    OPTIMIZED_LIME = 2
    SHAP = 3


class ExampleType(enum.Enum):
    RANDOM = 1
    FALSELY_CLASSIFIED = 2
    TRULY_CLASSIFIED = 3


def _divide_features(df: pd.DataFrame) -> (list, list):
    """
    Separate the numerical from the non-numerical columns of a pandas.DataFrame.
    :param df: The pandas.DataFrame to be separated.
    :return: Two lists. One containing only the numerical column names and another one only
    the non-numerical column names.
    """
    num = []
    cat = []

    for n in df.columns:
        if is_numeric_dtype(df[n]):
            num.append(n)
        elif is_string_dtype(df[n]):
            cat.append(n)

    return num, cat


def get_column_transformer(numerical: list, categorical: list) -> ColumnTransformer:
    """
    Create a column transformer for the numerical and categorical columns later
    used in the model pipeline.
    :param numerical: Numerical columns in the dataset
    :param categorical: Categorical columns in the dataset
    :return: A new column transformer
    """

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
        ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))])

    return ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical),
                    ('cat', categorical_transformer, categorical)])


def get_pipeline(ct: ColumnTransformer, algorithm: Algorithm, data_size: int) -> Pipeline:
    """
    Returns a new pipeline depending on the chosen algorithm.
    :param ct: A column transformer used in the pipeline
    :param algorithm: Algorithm that is then used for training the model
    :param data_size: The size of the data used for training.
    :return: A new pipeline
    """

    if algorithm is Algorithm.LOGISTIC_REGRESSION:
        estimator = LogisticRegression(
            max_iter=1000,
            solver='liblinear',
            C=1,
            penalty='l1',
            dual=False,
            random_state=RANDOM_NUMBER)
    elif algorithm is Algorithm.DECISION_TREE:
        estimator = DecisionTreeClassifier(
            criterion='gini',
            splitter='best',
            max_depth=8,
            max_features=1.0,
            random_state=RANDOM_NUMBER)
    elif algorithm is Algorithm.RANDOM_FOREST:
        estimator = RandomForestClassifier(
            criterion='gini',
            n_estimators=1200,
            max_depth=22,
            max_features=0.25,
            min_samples_split=10,
            min_samples_leaf=2,
            bootstrap=True,
            random_state=RANDOM_NUMBER,
            n_jobs=-1)
    elif algorithm is Algorithm.XGB:
        estimator = XGBClassifier(
            random_state=RANDOM_NUMBER,
            n_jobs=-1)
    elif algorithm is Algorithm.SVC:
        estimator = SVC(
            kernel="rbf",
            C=100,
            gamma=0.01,
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

    return pipeline


def get_split(preprocessor: ColumnTransformer, split: Split, cat_features: list, df_x: pd.DataFrame, df_y: pd.Series)\
        -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    """
    Splits the data in train and test data taking the split type (balanced/imbalanced) in
    consideration. Also this function makes sure that the number of features after encoding them
    using sklearn.preprocessing.OneHotEncoder is same for the train and the test splits.
    If for iter_limit iterations the no split is found that has the same number of features
    for the train and test split, then an error is risen.
    :param preprocessor: A preprocessor containing a sklearn.preprocessing.OneHotEncoder that
    will be used for training the model.
    :param split: Whether a balanced or imbalanced split shall be used. If balanced split is used
    a feature for the split shall be selected
    :param cat_features: The categorical features of the dataset
    :param df_x: Dataset features
    :param df_y: Dataset target
    :return: X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()

    if cat_features:
        iter_limit = 100
        are_sets_equal = False
        count = 0
        random_number = None

        while not are_sets_equal:
            X_train, X_test, y_train, y_test, random_number = _get_train_test_split(split, cat_features, df_x, df_y)

            _ = preprocessor.fit_transform(X_train)
            X_train_ohe_features = set(
                preprocessor.named_transformers_["cat"].named_steps['onehot'].get_feature_names(cat_features).tolist())
            _ = preprocessor.fit_transform(X_test)
            X_test_ohe_features = set(
                preprocessor.named_transformers_["cat"].named_steps['onehot'].get_feature_names(cat_features).tolist())
            are_sets_equal = X_train_ohe_features == X_test_ohe_features and\
                             np.array_equal(sorted(y_train.unique()), sorted(y_test.unique()))
            if not are_sets_equal:
                log.debug("Iteration counter: {}\nFeatures difference between train and test split: {}.\n"
                          "Unique target values for train split: {} and test split: {}".
                          format(count,
                                 X_train_ohe_features - X_test_ohe_features
                                 if len(X_train_ohe_features) > len(X_test_ohe_features)
                                 else X_test_ohe_features - X_train_ohe_features,
                                 y_train.unique(),
                                 y_test.unique()))

            count = count + 1
            if count == iter_limit:
                raise TimeoutError("No train/test split was found with the same number of encoded features for "
                                   "both the train and the test split after {} iterations.\n"
                                   "Features that differ: {}\n".
                                   format(count, X_train_ohe_features - X_test_ohe_features
                                 if len(X_train_ohe_features) > len(X_test_ohe_features)
                                 else X_test_ohe_features - X_train_ohe_features,
                                 y_train.unique(),
                                 y_test.unique()))
    else:
        X_train, X_test, y_train, y_test, random_number = _get_train_test_split(split, cat_features, df_x, df_y)

    if random_number != RANDOM_NUMBER:
        _set_random_number(random_number=random_number)

    return X_train, X_test, y_train, y_test


def _get_train_test_split(split: Split, cat_features: list, df_X: pd.DataFrame, df_y: pd.Series)\
        -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int):
    """
    Splits the data into train and test.
    If we use the imbalanced split option, then the data will not be balanced (down-/upsampled) on
    any feature and will be split as it is.
    If a balanced split option is selected and a feature or a list of features is available,
    the data will be sampled in such a way that there is an equal number of samples for each
    category of the features.
    If a balanced split option is selected and not additional features, an upsampling technique
    will be applied trying to balance the target (y). Either SMOTENC or ADASYN algorithm will
    be used depending on the features type.
    :param split: Whether the data should be down-/upsampled before the split
    :param cat_features: The categorical features of this dataset
    :param df_X: Dataset features (X)
    :param df_y: Dataset target (y)
    :return: (X_train, X_test, y_train, y_test, random_number (seed used for the train/test split))
    """
    random_number = _get_random_number()

    if split.type is SplitTypes.BALANCED:
        if not split.value:
            X_train, X_test, y_train, y_test = train_test_split(df_X,
                                                                df_y,
                                                                stratify=df_y,
                                                                test_size=TEST_SPLIT_SIZE,
                                                                random_state=random_number)
            if cat_features:
                from imblearn.over_sampling import SMOTENC
                X_resampled_train, y_resampled_train = \
                    SMOTENC(categorical_features=sorted([list(df_X.columns).index(f) for f in cat_features]),
                            random_state=random_number).fit_resample(X_train, y_train)
            else:
                from imblearn.over_sampling import ADASYN
                X_resampled_train, y_resampled_train = \
                    ADASYN(random_state=random_number).fit_resample(X_train, y_train)
            return X_resampled_train, X_test, y_resampled_train, y_test, random_number
        else:
            from functools import reduce
            from xai import balanced_train_test_split
            max_per_group = int(df_y.size/
                                (reduce(lambda a, b: a + b,
                                        list(map(lambda x: len(df_X[x].unique()), cat_features)))*len(df_y.unique())))-1
            X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, train_idx, test_idx =\
                balanced_train_test_split(
                    df_X, df_y, *split.value,
                    min_per_group=int(max_per_group/10),
                    max_per_group=max_per_group,
                    categorical_cols=cat_features)

            return X_train_balanced,\
                   X_test_balanced,\
                   pd.Series(data=y_train_balanced, name=df_y.name),\
                   pd.Series(data=y_test_balanced, name=df_y.name),\
                   random_number
    elif split.type is SplitTypes.NORMAL:
        X_train, X_test, y_train, y_test = train_test_split(df_X,
                                                            df_y,
                                                            stratify=df_y,
                                                            test_size=TEST_SPLIT_SIZE,
                                                            random_state=random_number)
        return X_train, X_test, y_train, y_test, random_number
    else:
        raise NotImplementedError


def _get_random_number() -> int:
    """
    Get the value of the random seed/state used for the train/test splits and passed to the external modules.
    If the value of the global variable is not set a new random number with a positive value
     lower than 100 will be returned.
    return: A random number [0;100)
    """

    if RANDOM_NUMBER:
        random_number = RANDOM_NUMBER
    else:
        random_number = randrange(100)
        log.debug("Random number value was set to: {}.".format(random_number))

    return random_number


def _set_random_number(random_number: int):
    """
    Set the value of the global RANDOM_NUMBER variable. This variable makes sure that all of the models use the same
    train/test split when trained.
    :param random_number: The new value for the global variable.
    :return: void
    """
    global RANDOM_NUMBER
    if not RANDOM_NUMBER:
        RANDOM_NUMBER = random_number
        log.info("Value of RANDOM_NUMBER is set to {0}".format(random_number))
    else:
        log.warning("Trying to change the value of a constant variable RANDOM_NUMBER from {0} to {1}."
                    .format(RANDOM_NUMBER, random_number))


def _get_categorical_ohe_features(model: Pipeline, cat_features: list) -> list:
    """
    Gets all encoded (with OneHotEncoder) features for a model.
    :param model: Pipeline for the model.
    :param cat_features: The initial categorical columns for the dataset.
    :return: All encoded features for the model.
    """
    preprocessor = model.named_steps["preprocessor"]

    # Get all categorical columns (including the newly encoded with the OHE)
    try:
        new_ohe_features = preprocessor.named_transformers_["cat"].named_steps['onehot']\
            .get_feature_names(cat_features)\
            .tolist()
    except NotFittedError:
        log.warning("No categorical features found in this dataset.")
        new_ohe_features = []

    return new_ohe_features


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


def plot_feature_importance_with_eli5(model: Model) -> (display.HTML, dict):
    """
    Global explanation for a model of type feature importance.
    :param model: The model to be interpreted.
    :return: (IPython.display.HTML element with the feature importance, feature importance in a dictionary format)
    """

    weights = []
    explanation = explain_weights(model.model.named_steps["model"],
                                  feature_names=model.features_ohe,
                                  top=len(model.features_ohe))

    try:
        weights = explanation.targets[0].feature_weights.pos
    except TypeError as e:
        log.debug("An expected error occurred. Program execution may continue: {}".format(e))
        weights = explanation.feature_importances.importances

    feature_weight = {}
    for weight in weights:
        if weight.weight > 0:
            feature_weight[weight.feature] = weight.weight

    return show_weights(model.model.named_steps["model"], feature_names=model.features_ohe), feature_weight


def plot_feature_importance_with_skater(model: Model) -> (figure.Figure, axes.Axes, dict):
    """
    Global explanation for a model of type (permutation) feature importance.
    :param model: The model to be interpreted.
    :return: (f, ax, feature_weight): (figure instance, matplotlib.axes._subplots.AxesSubplot,
    feature importance in a dictionary format)
    """

    from matplotlib import pyplot
    from itertools import cycle

    ascending = True

    weights = model.skater_interpreter.feature_importance.feature_importance(model.skater_model,
                                                                             n_samples=1000,
                                                                             ascending=ascending,
                                                                             n_jobs=2)
    feature_weight = {}

    for i in range(1, len(weights) + 1):
        if weights[i * (-1)] > 0:
            feature_weight[list(weights.keys())[i * (-1)]] = weights[i * (-1)]

    f, ax = pyplot.subplots(1)
    colors = cycle(['#328BD5', '#404B5A', '#3EB642', '#E04341', '#8665D0'])
    color = next(colors)
    weights.sort_values(ascending=ascending).plot(kind='barh', ax=ax, color=color)

    return f, ax, feature_weight


def plot_feature_importance_with_shap(model: Model, plot_type="bar") -> dict:
    """
    Plot feature importance for a given model with shap.
    :param model: Model for which the feature importance should be plotted.
    :param plot_type: The type of the plot
    :return: feature importance in a dictionary format
    """

    if not model.shap_values:
        model.init_shap()

    feature_mean = {}
    feature_order = np.argsort(np.sum(-np.mean(np.abs(model.shap_values), axis=1), axis=0))
    feature_weight = np.sum(np.mean(np.abs(model.shap_values), axis=1), axis=0)

    for feature_key in feature_order:
        if feature_weight[feature_key] > 0:
            feature_mean[model.features_ohe[feature_key]] = feature_weight[feature_key].round(3)

    summary_plot(model.shap_values, model.X_test_ohe, plot_type=plot_type)
    return feature_mean


def calculate_rbos(
        type: FeatureImportanceType,
        models: list,
        d_min: int = 5,
        d_max: int = None,
        step: int = 1)\
        -> pd.DataFrame:
    """
    Calculates the rank-biased overlap (RBO) between all model combinations from the list for a given feature importance
    technique. The persistence (p-value) for the RBO algorithm is calculated (d-1)/d, where d is the depth. The RBO
    for each combination is calculated multiple time based on the minimal depth (d_min) value, maximal depth (d_max) and
    the step value.
    :param type: The type of feature importance for which the RBOs should be calculated.
    :param models: The list of models for whose FIs the RBOs should be calculated
    :param d_min: The minimal depth for which the RBOs should be calculated. The maximal depth is length of the shortest
    feature importance list of this type or the explicitly set input parameter.
    :param d_max: The maximal depth for which the RBOs should be calculated. If none, the shortest feature importance
    list length will be used as maximal depth.
    :param step: The step with which the depth should be increased on each iteration until the maximal depth is reached.
    :return: A pandas.DataFrame consisting of multiple columns. The first column is with the persistence (p) values,
     the second column is with the depth (d) values, then a column for each combination of the models and the last
     column is the mean RBO value from all combinations for this row.
    """
    from itertools import combinations

    if not d_max:
        lengths = []
        for model in models:
            if type == FeatureImportanceType.ELI5 and model.feature_weight_eli5:
                lengths.append(len(model.feature_weight_eli5))
            elif type == FeatureImportanceType.SKATER and model.feature_weight_skater:
                lengths.append(len(model.feature_weight_skater))
            elif type == FeatureImportanceType.SHAP and model.feature_weight_skater:
                lengths.append(len(model.feature_weight_shap))
            else:
                log.warning("Type {} is not yet supported. Please use one of the supported types.".format(type))

        d_max = min(lengths)

    p_d = {round((d-1)/d, 3): d for d in range(d_min, d_max+1, step)}
    df = pd.DataFrame({
        'p': list(p_d.keys()),
        'd': list(p_d.values())
    })
    for model_1, model_2 in combinations(models, 2):
        rbos = []
        for p, d in p_d.items():
            rbo = None
            if type == FeatureImportanceType.ELI5:
                rbo = _calculate_rbo(model_1.feature_weight_eli5, model_2.feature_weight_eli5, d, p, True)
            elif type == FeatureImportanceType.SKATER:
                rbo = _calculate_rbo(model_1.feature_weight_skater, model_2.feature_weight_skater, d, p, True)
            elif type == FeatureImportanceType.SHAP:
                rbo = _calculate_rbo(model_1.feature_weight_shap, model_2.feature_weight_shap, d, p, True)
            else:
                log.warning("Type {} is not yet supported. Please use one of the supported types.".format(type))

            if rbo is not None:
                rbos.append(rbo)
            else:
                log.debug("Feature importance for {}, {} or both is empty for feature importance type: {}."
                          " Please calculate the feature importance for these models."
                          .format(model_1.name, model_2.name, type.name))
        if rbos:
            df['{}_{}'.format(model_1.name, model_2.name)] = rbos
    df['Mean'] = round(df.iloc[:, 2:].mean(axis=1), 3)
    df.style.highlight_max()

    return df


def _calculate_rbo(feature_weight_1: dict, feature_weight_2: dict, d: int, p: int, ext: bool = True) -> float:
    """
    Calculates a single rank-biased overlap (RBO) value for two models' feature importances.
    :param feature_weight_1: Dictionary containing the FIs for the first model of type {feature name: feature weight}.
    :param feature_weight_2: Dictionary containing the FIs for the second model of type {feature name: feature weight}.
    :param d: The depth of the RBO.
    :param p: The persistence of the RBO.
    :param ext: Whether extrapolation shall be used or not  for the RBO.
    :return: (float) The rank-biased overlap value for these feature importances.
    If one of the dicts is empty None is returned.
    """
    if feature_weight_1 is not None and feature_weight_2 is not None:
        rbo = \
            round(
                RankingSimilarity(
                    list(feature_weight_1.keys()),
                    list(feature_weight_2.keys()))
                .rbo(k=d, p=p, ext=ext)
                , 3)
    else:
        rbo = None
    return rbo


def _get_nth_ordinal(n: int) -> str:
    """
    Convert an integer into its ordinal representation.
    :param n: The number as an integer
    :return: Ordinal representation of the nubmer (e.g., '0th', 3rd', '122nd', '213th')
    """
    n = int(n)
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    return str(n) + suffix


def calculate_X_ohe(model: Pipeline, X: pd.DataFrame, feature_names: list):
    """
    Transform a pd.DataFrame using a One-Hot Encoder from a suitable Pipeline.
    :param model: OHE from a Pipeline to be used
    :param X: X to be encoded
    :return: One-Hot encoded pd.DataFrame
    """

    # model.model[0] to get the preprocessor from the pipeline
    X_ohe = model[0].fit_transform(X)

    log.debug("Encoded dataframe shape: {}\n"
              "Number of all encoded features: {}.".
              format(X_ohe.shape[1], len(feature_names)))

    try:
        X_ohe = X_ohe.toarray()
    except AttributeError as e:
        log.debug("An expected error occurred. Program execution may continue: {}".format(e))
        X_ohe = X_ohe

    return pd.DataFrame(X_ohe, columns=feature_names)


def generate_feature_importance_plot(type: FeatureImportanceType, model: Model):
    """
    Generate feature importance plot for a model.
    :param type: Type of feature importance method to be used.
    :param model: Model, for which a plot should be created.
    :return: If type is ELI5, then IPython.display.HTML plot
    is returned, None otherwise
    """
    plot = None
    log.info("Generating a feature importance plot using {} for {} ...".format(type.name, model.name))

    start = time.time()

    if type == FeatureImportanceType.ELI5:
        # TODO: Support XGB when https://github.com/TeamHG-Memex/eli5/pull/407 fixed.
        if model.model_type.algorithm is Algorithm.SVC or model.model_type.algorithm is Algorithm.XGB:
            log.warning("{} is not supported by {}.".format(model.model_type.algorithm.name, type.name))
        else:
            plot, feature_weight_eli5 = plot_feature_importance_with_eli5(model)
            model.feature_weight_eli5 = feature_weight_eli5
    elif type == FeatureImportanceType.SKATER:
        if not model.skater_model or not model.skater_interpreter:
            model.init_skater()
        plot, _, feature_weight_skater = plot_feature_importance_with_skater(model)
        model.feature_weight_skater = feature_weight_skater
    elif type == FeatureImportanceType.SHAP:
        if not model.shap_values:
            model.init_shap()
        feature_weight_shap = plot_feature_importance_with_shap(model)
        model.feature_weight_shap = feature_weight_shap
    else:
        log.warning("Type {} is not yet supported. Please use one of the supported types.".format(type.name))

    end = time.time()
    _log_elapsed_time(start, end, "generating a feature importance plot with {} is".format(type.name))

    return plot


def generate_pdp_plots(type: PDPType, model: Model, feature1: str, feature2: str) -> figure:
    """
    Generate a PDP including one or two features for a given model.
    :param type: Type of framework that should be used for generating the PDPs
    :param model: Model for which a plot shall be generated
    :param feature1: Feature to be included in the PDP.
    :param feature2: If none, plot a PDP for only feature1
    :return: A matplotlib.pyplot.figure of the PDP.
    """
    fig = None
    log.info("Generating a PDP plot using {} for {} ...".format(type.name, model.name))

    start = time.time()

    if type == PDPType.PDPBox:
        if feature2 == 'None':
            fig, ax = plot_single_pdp_with_pdpbox(model, feature1)
        else:
            fig, ax = plot_multi_pdp_with_pdpbox(model, feature1, feature2)

    elif type == PDPType.SKATER:
        if not model.skater_model or not model.skater_interpreter:
            model.init_skater()
        if feature2 == 'None':
            fig, ax = plot_single_pdp_with_skater(model, feature1)
        else:
            fig, ax = plot_multi_pdp_with_skater(model, feature1, feature2)

    elif type == PDPType.SHAP:
        if not model.shap_values:
            model.init_shap()
        if feature2 == 'None':
            fig, ax = plot_single_pdp_with_shap(model, feature1)
        else:
            fig, ax = plot_multi_pdp_with_shap(model, feature1, feature2)

    else:
        log.warning("Type {} is not yet supported. Please use one of the supported types.".format(type))

    end = time.time()
    _log_elapsed_time(start, end, "generating a PDP with {} is".format(type.name))

    return fig


def plot_single_pdp_with_pdpbox(
        model: Model,
        feature: str,
        plot_lines=True,
        x_quantile=True,
        show_percentile=True,
        plot_pts_dist=True,)\
        -> (figure.Figure, axes.Axes):
    """
    Plots a PDP for a single feature for a given model.
    :param model: The model for which a PDP should be created
    :param feature: Feature or feature list to investigate, for one-hot encoding features, feature list is required
    :param plot_lines: Whether to plot out the individual lines
    :param x_quantile: Whether to construct x-axis ticks using quantiles
    :param show_percentile: Whether to display the percentile buckets, for numeric feature when grid_type='percentile'
    :param plot_pts_dist: Whether to show data points distribution
    :return: (fig, ax): A matplotlib tuple consisting of a figure object and an ax object
    """

    # model.model[1] to get the actual model from the pipeline
    pdp_isolate_out = pdp.pdp_isolate(
        model=model.model[1],
        dataset=model.X_test_ohe,
        model_features=model.features_ohe,
        feature=feature)

    fig, ax = pdp.pdp_plot(
        pdp_isolate_out=pdp_isolate_out,
        feature_name=feature if isinstance(feature, str) else feature[0].split('_')[0].title,
        plot_lines=plot_lines,
        x_quantile=x_quantile,
        show_percentile=show_percentile,
        plot_pts_dist=plot_pts_dist,
        frac_to_plot=0.5)

    return fig, ax


def inverse_scale(model: Model, feature: str, values: list, rounding: int = None) -> list:
    """
    Inverse scales numerical feature values from a given model's test set.
    :param model: A fitted Model object.
    :param feature: The feature name for which the values will be inverse scaled.
    :param values: A list of numerical values or a single value to be inverse scaled.
    :param rounding: The number of decimal places to round the feature values to. Defaults to None (no rounding).
    :return: A list of inverse-scaled numerical feature values.
    """

    if feature in model.numerical_features:
        dt = {}
        arr_inv = np.unique(
            model
            .model[0]
            .named_transformers_["num"]
            .named_steps['scale']
            .inverse_transform(
                model.X_test_ohe[model.numerical_features])[:, model.numerical_features.index(feature)])
        arr = np.unique(model.X_test_ohe[feature].array)
        for i in range(len(arr)):
            num = arr[i]
            if rounding:
                num = round(arr[i], rounding)
            dt[num] = arr_inv[i]
        res = []

        try:
            iter(values)
        except TypeError as e:
            values = [values]
            log.debug("An expected error occurred. Program execution may continue: {}".format(e))

        for value in values:
            value = float(value)
            try:
                res_val = dt[value]
            except KeyError as e:
                log.debug("An expected error occurred. Program execution may continue: {}".format(e))
                res_key, res_val = min(dt.items(), key=lambda x: abs(value - x[0]))
                # 0.1 is the maximal allowed difference between the input value and the scaled value from the original
                if abs(res_key-value) >= 0.099:
                    log.error("No original value corresponds to a scaled value of {} for feature {}.\n"
                              "All scaled values: original values for {} -> {}."
                              .format(value, feature, feature, dt))
                    raise e
            res.append(res_val)
    else:
        log.error("Feature {} is not a numerical feature; therefore its value cannot be inverse scaled.\n"
                  "All numerical features are: {}".format(feature, model.numerical_features))
        res = values

    return res


def plot_multi_pdp_with_pdpbox(
        model: Model,
        feature1: str,
        feature2: str,
        plot_type='contour',
        x_quantile=False,
        plot_pdp=False)\
        -> (figure.Figure, axes.Axes):
    """
    Plots a PDP for two features for a given model.
    :param model: The model for which a PDP should be created
    :param feature1: Feature to be plotted
    :param feature2: Feature with which feature1 interacts to be plotted
    :param plot_type: Type of the interact plot, can be 'contour' or 'grid'
    :param x_quantile: Whether to construct x-axis ticks using quantiles
    :param plot_pdp: Whether to plot pdp for each feature
    :return: (fig, ax): A matplotlib tuple consisting of a figure object and an ax object
    """
    features_to_plot = [feature1, feature2]

    # model.model[1] to get the actual model from the pipeline
    pdp_interact_out = pdp.pdp_interact(
        model=model.model[1],
        dataset=model.X_test_ohe,
        model_features=model.features_ohe,
        features=features_to_plot)

    fig, ax = pdp.pdp_interact_plot(
        pdp_interact_out=pdp_interact_out,
        feature_names=features_to_plot,
        plot_type=plot_type,
        x_quantile=x_quantile,
        plot_pdp=plot_pdp)

    return fig, ax


def plot_single_pdp_with_shap(model: Model, feature: str) -> (figure.Figure, axes.Axes):
    """
    Plots a shap PDP for a single feature for a given model.
    :param model: The model for which a PDP should be plotted
    :param feature: Feature to be plotted
    :return: (fig, ax): A matplotlib tuple consisting of a figure object and an ax object
    """
    fig, ax = plt.subplots()

    dependence_plot(
        ind=feature,
        interaction_index=feature,
        shap_values=model.shap_values[0],
        features=model.X_test_ohe,
        ax=ax,
        show=False
        # features=model.X_test_ohe.sample(66, random_state=RANDOM_NUMBER # for testing purposes
    )

    return fig, ax


def plot_multi_pdp_with_shap(model: Model, feature1: str, feature2='auto') -> (figure.Figure, axes.Axes):
    """
    Plots a shap PDP for two features for a given model.
    :param model: The model for which a PDP should be plotted
    :param feature1: Feature to be plotted
    :param feature2: Feature with which feature1 interacts to be plotted.
    If the value is 'auto' the feature with most interaction will be selected
    :return: (fig, ax): A matplotlib tuple consisting of a figure object and an ax object
    """
    fig, ax = plt.subplots()

    dependence_plot(
        ind=feature1,
        interaction_index=feature2,
        shap_values=model.shap_values[0],
        features=model.X_test_ohe,
        ax=ax,
        show=False
        # features=model.X_test_ohe.sample(66, random_state=RANDOM_NUMBER # for testing purposes
    )

    return fig, ax


def plot_single_pdp_with_skater(
        model: Model,
        feature: str,
        n_samples=1000,
        grid_resolution=50,
        grid_range=(0, 1),
        with_variance=True,
        figsize=(6, 4))\
        -> (figure.Figure, axes.Axes):
    """
    Plots a skater PDP for a single feature for a given model.
    :param model: The model for which a PDP should be plotted
    :param feature: Feature to be plotted
    :param n_samples: The number of samples to use from the original dataset.
    Note this is only active if sample = True and sampling strategy = 'uniform'.
    If using 'uniform-over-similarity-ranks', use samples per bin
    :param grid_resolution: How many unique values to include in the grid.
    If the percentile range is 5% to 95%, then that range will be cut into <grid_resolution> equally size bins.
    Defaults to 30.
    :param grid_range: The percentile extrema to consider. 2 element tuple, increasing, bounded between 0 and 1.
    :param with_variance:
    :param figsize: Whether to include pdp error bars in the plots. Currently disabled for 3D plots for visibility.
    :return: (fig, ax): A matplotlib tuple consisting of a figure object and an ax object
    """

    r = model.skater_interpreter.partial_dependence.plot_partial_dependence([feature],
                                                                            model.skater_model,
                                                                            n_samples=n_samples,
                                                                            grid_resolution=grid_resolution,
                                                                            grid_range=grid_range,
                                                                            with_variance=with_variance,
                                                                            figsize=figsize)

    return r[0][0], r[0][1]


def plot_multi_pdp_with_skater(
        model: Model,
        feature1: str,
        feature2: str,
        n_samples=1000,
        grid_resolution=100,
        grid_range=(0, 1),
        with_variance=False,
        figsize=(12, 5))\
        -> (figure.Figure, axes.Axes):
    """
    Plots a skater PDP for two features for a given model.
    :param model: The model for which a PDP should be plotted
    :param feature1: Feature to be plotted
    :param feature2: Feature with which feature1 interacts to be plotted.
    :param n_samples: The number of samples to use from the original dataset.
    Note this is only active if sample = True and sampling strategy = 'uniform'.
    If using 'uniform-over-similarity-ranks', use samples per bin
    :param grid_resolution: How many unique values to include in the grid.
    If the percentile range is 5% to 95%, then that range will be cut into <grid_resolution> equally size bins.
    Defaults to 30.
    :param grid_range: The percentile extrema to consider. 2 element tuple, increasing, bounded between 0 and 1.
    :param with_variance:
    :param figsize: Whether to include pdp error bars in the plots. Currently disabled for 3D plots for visibility.
    :return: (fig, ax): A matplotlib tuple consisting of a figure object and an ax object
    """

    r = model.skater_interpreter.partial_dependence.plot_partial_dependence([(feature1, feature2)],
                                                                            model.skater_model,
                                                                            n_samples=n_samples,
                                                                            grid_resolution=grid_resolution,
                                                                            grid_range=grid_range,
                                                                            with_variance=with_variance,
                                                                            figsize=figsize)
    fig = r[0][0]
    ax = r[0][1]

    return fig, ax


def generate_idx2ohe_dict(X: pd.DataFrame, cat_features: list, ohe_cat_features: list):
    """
    All categorical features - getting their index to use as a key and then
    listing all possible values for that feature.
    We get the possible values from the attribute categories of our one hot encoder.
    :param X: Test DataFrame for getting the right column index
    :param cat_features: All the original categorical columns
    :param ohe_cat_features: All the One-Hot encoded columns
    :return: A dict mapping original column index to all of its possible values
    """
    categorical_names = {}

    for col in cat_features:
        categorical_names[X.columns.get_loc(col)] = \
            [new_col.split("_")[1]
             for new_col in ohe_cat_features
             if new_col.split("_")[0] == col]

    return categorical_names.copy()


def get_example_information(model: Model, example: int):
    """
    Get information for a example by its index in the dataset.
    :param model: Model, from which dataset the example should be used
    :param example: Example index in the pandes.DataFrame
    :return: A message containing example's data
    """

    msg = ""
    msg = msg + "Example {}'s data: \n{}\n".format(example, model.X_test.iloc[example])
    # msg = msg + "{}'s prediction for example {}: {}\n".format(model.name, example, model.predictions[example])
    msg = msg + "Actual result for example {}: {}\n".format(example, model.y_test.iloc[example])

    return msg


def explain_single_instance(
        local_interpreter: LocalInterpreterType,
        model: Model,
        example: int,
        kernel_width: float = None,
        force: bool = False):
    """
    Explain single instance (example) with a given interpreter type.
    :param local_interpreter: Type of interpreter to be used. Currently only LIME and SHAP are supported
    :param model: The model for which an instance should be explained
    :param example: The example to be explained - The row number from the X_test pd.DataFrame
    :param kernel_width: (Optional) Only for LIME. Set the kernel_width for LIME explanations. None results in the
     default kernel_width which is 'sqrt(number of columns) * 0.75'.
    :param force: (Optional) Only for LIME. If LIME is already initialized for this example it will not be newly
    initialized unless this flag is set to True.
    :return: Either a LIME or a SHAP explanation
    """
    explanation = None
    log.info("Generating a single instance explanation using {} for {} ...".format(local_interpreter.name, model.name))

    start = time.time()

    if local_interpreter is LocalInterpreterType.LIME or local_interpreter is LocalInterpreterType.OPTIMIZED_LIME:
        if example not in model.example_lime_explainer or (example in model.example_lime_explainer and force):
            if local_interpreter is LocalInterpreterType.LIME:
                log.info("Initializing LIME - generating new explainer for example {} for {} kernel width."
                         " This operation may be time-consuming so please be patient."
                         .format(example,
                                 "the default" if kernel_width is None
                                 else kernel_width))

                model.init_lime(example=example, kernel_width=kernel_width)
                explanation = explain_single_instance_with_lime(model, example)
            else:
                log.info("Initializing LIME - generating new explainer for example {} and optimizing the kernel width."
                         " This operation may be time-consuming so please be patient.".format(example))

                def explain_single_instance_with_lime_stability_wrapper(model: Model, example: int, kernel_width: float):
                    model.init_lime_stability(example=example, kernel_width=kernel_width)
                    _, csi, vsi = explain_single_instance_with_lime_stability(model, example)
                    return csi, vsi

                from math import sqrt
                search_space = np.sort(np.append(
                    sqrt(len(model.X_train.columns)) * 0.75,
                    np.linspace(0.15, 8.15, num=50, dtype=float)))
                f = partial(explain_single_instance_with_lime_stability_wrapper, model, example)
                best_kernel_width, (csi, vsi) = optimize_function(f,
                                                                  search_space,
                                                                  num_samples=30,
                                                                  num_iterations=30,
                                                                  learning_rate=0.1)
                log.info("The optimal kernel width for example {} and {} is {}.\n"
                         "Variables Stability Index (VSI): {}\n"
                         "Coefficients Stability Index (CSI): {}"
                         .format(example, model.name, best_kernel_width, csi, vsi))

                model.init_lime_stability(example=example, kernel_width=best_kernel_width)
                explanation, _, _ = explain_single_instance_with_lime_stability(model, example)

            model.feature_value_weight_lime_local = {example: _get_lime_local_feature_value_weight(explanation)}
        elif example in model.example_lime_explainer and not force:
            log.debug("{} is already initialized for this example {}. "
                      "Please use the force option if you want to reinitialize it."
                      .format(local_interpreter.name, example))
            try:
                # if OPTIMIZED_LIME was used for the initialization
                explanation, csi, vsi = explain_single_instance_with_lime_stability(model, example)

                log.info("The optimal kernel width for example {} and {} is {}.\n"
                         "Variables Stability Index (VSI): {}\n"
                         "Coefficients Stability Index (CSI): {}"
                         .format(example, model.name, explanation.domain_mapper.kernel_width, csi, vsi))
            except AttributeError as e:
                # if LIME was used for the initialization
                explanation = explain_single_instance_with_lime(model, example)
    elif local_interpreter is LocalInterpreterType.SHAP:
        if not model.shap_values:
            model.init_shap()
        explanation = explain_single_instance_with_shap(model, example)
        model.feature_value_weight_shap_local = {example: _get_shap_local_feature_value_weight(model, explanation)}
    else:
        log.error("Interpreter type {} is not yet supported for local interpretations. Please either use another one"
                  "or extend the functionality of this function".format(local_interpreter))

    end = time.time()
    _log_elapsed_time(start, end, "generating a single instance explanation with {} is".format(local_interpreter.name))

    return explanation


def _get_lime_local_feature_value_weight(explanation: Explanation) -> dict:
    """
    Generates a dictionary with a key of the following form "feature =/</> value" and a value
    "weight for this feature and its value", e.g. hours-per-week <= 40.00: -0.0635
    :param explanation: A LIME explanation.
    :return: A sorted dictionary where the positive weights are sorted desc and the negatives asc
    """
    feature_value_weight_lime_local = dict(explanation.as_list())

    feature_value_weight_lime_local_pos = dict(filter(lambda x: x[1] >= 0.0, feature_value_weight_lime_local.items()))
    feature_value_weight_lime_local_neg = dict(filter(lambda x: x[1] < 0.0, feature_value_weight_lime_local.items()))

    return {**_sort_dict_by_value(feature_value_weight_lime_local_pos, reverse=True),
            **_sort_dict_by_value(feature_value_weight_lime_local_neg, reverse=False)}


def _get_shap_local_feature_value_weight(model: Model, explanation: AdditiveForceVisualizer):
    """
    Generates a dictionary with a key of the following form "feature = value (scaled value)" and a value
    "weight for this feature and its value", e.g. hours-per-week = 38 (-0.24038488322015739): 0.017
    :param model: The model for which the local feature=value(scaled value): weight dictionary should be generated.
    :param explanation: A SHAP AdditiveForceVisualizer explanation.
    :return: A sorted dictionary where the positive weights are sorted desc and the negatives asc
    """
    feature_value_weight_shap_local_pos = {}
    feature_value_weight_shap_local_neg = {}
    for key, value in explanation.data['features'].items():
        feature_name = list(explanation.data['featureNames'])[int(key)]
        effect = value['effect']
        value = value['value']
        new_key = "{} = {} ({})".format(
                feature_name,
                int(inverse_scale(model, feature_name, value, 2)[0]),
                value) \
            if feature_name in model.numerical_features \
            else "{} = {}".format(feature_name, int(value))

        if effect >= 0.0:
            feature_value_weight_shap_local_pos[new_key] = effect
        else:
            feature_value_weight_shap_local_neg[new_key] = effect

    return {**_sort_dict_by_value(feature_value_weight_shap_local_pos, reverse=True),
            **_sort_dict_by_value(feature_value_weight_shap_local_neg, reverse=False)}


def optimize_function(
        f: callable,
        search_space: np.ndarray,
        num_samples: int = 10,
        num_iterations: int = 100,
        learning_rate: float = 0.01)\
        -> (float, (float, float)):
    """
    Optimize a function with one input argument within a given search space using random search and locally refining the
    input by a learning rate.
    :param f: The function to optimize. This function should take one input argument and return two output values:
    float -> (float, float).
    :param search_space: The search space for the input argument. This should be a one-dimensional numpy array of
    possible input values.
    :param num_samples: The number of random samples to draw from the search space to initialize the optimization.
    :param learning_rate: The learning rate for the local optimizer.
    :param num_iterations: The number of iterations for which the local optimizer should be applied.
    :return: (best_input, (best_output_1, best_output_2)) A tuple containing the best input value found and a tuple of
    the corresponding output values.
    """
    best_input = None
    best_output = (-np.inf, -np.inf)

    samples = np.random.choice(search_space, size=num_samples, replace=False)
    for sample in samples:
        output1, output2 = f(sample)
        if (output1 + output2 > best_output[0] + best_output[1]) or\
                ((output1 + output2 == best_output[0] + best_output[1]) and sample < best_input):
            best_input = sample
            best_output = (output1, output2)

    best_random_input = best_input
    # Refine the best_input by adding/subtracting the learning_rate to/from the best_input num_iterations/2 times.
    for i in range(2):
        if i == 0:
            sign = (+1)
        else:
            sign = (-1)
        for j in range(round(num_iterations/2)):
            current_learning_rate = j*learning_rate*sign
            updated_input = best_random_input + current_learning_rate

            if updated_input >= search_space[0]:
                output1, output2 = f(updated_input)
                if (output1 + output2 > best_output[0] + best_output[1]) or\
                        ((output1 + output2 == best_output[0] + best_output[1]) and updated_input < best_input):
                    best_input = updated_input
                    best_output = (output1, output2)

    return best_input, best_output


def generate_single_instance_comparison(models: list, example: int) -> str:
    """
    Compare models' decisions for a given example.
    :param models: All models to be compared
    :param example: The example, which is classified by the models
    :return: A string containing information whether each model's decision was right or not.
    """

    classified = []
    misclassified = []
    for model in models:
        if model.predictions[example] == model.y_test.iloc[example]:
            classified.append(model.name)
        else:
            misclassified.append(model.name)

    msg = "Example {} was truly classified by ".format(example)
    if len(classified) > 0:
        msg = msg + ', '.join(classified)
    else:
        msg = msg + "no model"

    msg = msg + " and falsely classified by "
    if len(misclassified) > 0:
        msg = msg + ', '.join(misclassified)
    else:
        msg = msg + "no model"

    return msg + ".\n For further clarification see the explanations below.\n"


def generate_single_instance_explanation(local_interpreter: LocalInterpreterType, model: Model, example: int) -> str:
    """
    Generate an explanation for a single instance (example) for a model with a given interpreter type.
    :param local_interpreter: Interpreter, that should be used
    :param model: Model for whose decision an explanation shall be generated
    :param example: Example, that should be explained
    :return: An explanation.
    """

    explanation = ""
    not_initialized = False

    if local_interpreter is LocalInterpreterType.LIME or local_interpreter is LocalInterpreterType.OPTIMIZED_LIME:
        if example in model.example_lime_explainer:
            explanation = generate_single_instance_explanation_with_lime(model, example)
        else:
            not_initialized = True
    elif local_interpreter is LocalInterpreterType.SHAP:
        if model.shap_values:
            explanation = generate_single_instance_explanation_with_shap(model, example)
        else:
            not_initialized = True
    else:
        log.error("Interpreter type {} is not yet supported for local interpretations. Please either use another one"
                  "or extend the functionality of this function".format(local_interpreter))

    if not_initialized:
        log.error("{} is not initialized for example {}. "
                  "Please initialize {} before trying to generate a textual explanation.\n"
                  "Note: Use explain_single_instance for initialization of {}"
                  .format(LocalInterpreterType.name, example, LocalInterpreterType.name, LocalInterpreterType.name))

    return explanation


def generate_single_instance_explanation_with_lime(model: Model, example: int) -> str:
    """
    Generate an explanation for a single instance (example) for a model with LIME.
    :param model: Model for whose decision an explanation shall be generated
    :param example: Example, that should be explained
    :return: An explanation.
    """
    _, prediction_probability = _get_prediction_for_example(model, example)
    feature_value_weight_lime_local = model.feature_value_weight_lime_local[example]

    return _generate_generic_single_instance_explanation(
        model.name,
        dict(filter(lambda x: x[1] >= 0.0, feature_value_weight_lime_local.items())),
        dict(filter(lambda x: x[1] < 0.0, feature_value_weight_lime_local.items())),
        prediction_probability,
        'LIME')


def generate_single_instance_explanation_with_shap(model: Model, example: int) -> str:
    """
    Generate an explanation for a single instance (example) for a model with SHAP.
    :param model: Model for whose decision an explanation shall be generated
    :param example: Example, that should be explained
    :return: An explanation.
    """
    feature_value_weight_shap_local = model.feature_value_weight_shap_local[example]

    # prediction = _get_prediction_for_example(model, example, approximate=True)
    # base_value = model.shap_kernel_explainer.expected_value[prediction]
    # shap_values = model.shap_values[prediction][example, :]
    # prediction_probability = np.sum(shap_values) + base_value

    _, prediction_probability = _get_prediction_for_example(model, example)
    return _generate_generic_single_instance_explanation(
        model.name,
        dict(filter(lambda x: x[1] >= 0.0, feature_value_weight_shap_local.items())),
        dict(filter(lambda x: x[1] < 0.0, feature_value_weight_shap_local.items())),
        prediction_probability,
        'SHAP')


def _generate_generic_single_instance_explanation(
        model_name: str,
        pos: dict,
        neg: dict,
        pred_prob: float,
        type: str) -> str:
    """
    Generate an explanation message for a given model.
    :param model_name: The name of the model that is going to be explained.
    :param pos: A dictionary with features and their values that positively impact a decision
    :param neg: A dictionary with features and their values that negatively impact a decision
    :param pred_prob: The prediction probability of the model for the current decision
    :param type: Type of explainer used (e.g, LIME, SHAP, ...)
    :return: An explanation message.
    """

    msg = "The prediction probability of {}'s decision for this example is {}. {}'s explanation: \n"\
        .format(model_name, str(round(pred_prob, 2)), type)
    msg = msg + _generate_generic_single_instance_explanation_helper(pos, model_name, "positive (1)")
    msg = msg + _generate_generic_single_instance_explanation_helper(neg, model_name, "negative (0)")
    msg = msg + "\n"

    return msg


def _generate_generic_single_instance_explanation_helper(d: dict, model_name: str, desc: str) -> str:
    """
    A helper function for generating an explanation for the single instance explainers.
    :param d: A dictionary containing all features and their values (values determined by an explainer)
    :param model_name: The name of the model that is going to be explained.
    :param desc: Additional information about the current explanation. (In most cases either 'positive' or 'negative')
    :return: A message explanation for a current model and an explanation type - positive or negative values.
    """

    msg = ""
    keys = list(d.keys())
    values = list(d.values())

    for c in range(len(d)):
        value_rounded = round(values[c], 4)
        if (c+1) == 1:
            msg = msg + "The most impactful feature for {}'s {} prediction probability is {} with value of {}.\n"\
                .format(model_name, desc, keys[c], value_rounded)
        else:
            msg = msg + "The {} most impactful feature for {}'s {} prediction probability is {} with value of {}.\n"\
                .format(_get_nth_ordinal(c+1), model_name, desc, keys[c], value_rounded)

    return msg


def _strip_dict(d: dict, n: int, reverse: bool) -> dict:
    """
    Take the first n-elements of a list sorted (either ascending or descending) by its values.
    Example: If n is 2 -> take the first two elements of the sorted dict.
    :param d: The dictionary to be stripped.
    :param n: The first elements to be taken.
    :param reverse: If true take the first n-elements of the descending sorted dict.
    If false take the first n-elements of the ascending sorted dict.
    :return: A sub-dictionary of the original.
    """
    default_n = 2

    if n > len(d):
        log.error("The length({}) variable should be lower than the length of the dictionary.\n"
                  "Setting its value to default({})."
                  .format(n, default_n))
        n = default_n

    sorted_d = _sort_dict_by_value(d, reverse=reverse)

    return {k: v for (k, v) in sorted_d.items() if list(sorted_d.keys()).index(k) < n}


def _sort_dict_by_value(d: dict, reverse: bool) -> dict:
    """
    Sort a dictionary by its values in descending or ascending order.
    :param d: A dictionary to be sorted
    :param reverse: If True -> sort in descending order, otherwise ascending
    :return: The sorted dictionary
    """
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}


def _get_prediction_for_example(model: Model, example: int) -> (int, float):
    """
    Returns the prediction probability for this example depending on how it was classified by the model.
    :param model: Model for which the example shall be classified
    :param example: Example to be classified
    :return: A tuple (int, float) containing (the prediction of the model [0, 1],
    the prediction probability for this example as a floating point number between [0.0; 1.0])
    """
    predictions = list(model.model.predict_proba(model.X_test)[example])
    return predictions.index(max(predictions)), max(predictions)


def convert_to_lime_format(X, categorical_names, col_names=None, invert=False):
    """
    Converts data with categorical values as string into the right format
    for LIME, with categorical values as integers labels.
    It takes categorical_names, the same dictionary that has to be passed
    to LIME to ensure consistency.
    col_names and invert allow to rebuild the original dataFrame from
    a numpy array in LIME format to be passed to a Pipeline or sklearn
    OneHotEncoder
    :param X: The data to be converted
    :param categorical_names: The names of the categorical ohe features
    :param col_names: Names of all initial columns
    :param invert: If we want to rebuild the original DataFrame
    :return: LIME format DataFrame / (If invert) Initial DataFrame
    """

    # If the data isn't a dataframe, we need to be able to build it
    if not isinstance(X, pd.DataFrame):
        X_lime = pd.DataFrame(X, columns=col_names)
    else:
        X_lime = X.copy()

    for k, v in categorical_names.items():
        if not invert:
            label_map = {
                str_label: int_label for int_label, str_label in enumerate(v)
            }

        else:
            label_map = {
                int_label: str_label for int_label, str_label in enumerate(v)
            }

        X_lime.iloc[:, k] = X_lime.iloc[:, k].map(label_map)

    return X_lime


def _explain_single_instance_with_lime(model: Model, example: int) -> (callable, Model, int):
    """
    This function explains the classification result of a single instance using LIME. It returns an explanation object,
     a custom predict_proba function that could be used in lime, and the observation that was explained.
    :param model: The classification model.
    :param example: Index of the example to be explained.
    :return: Tuple of explanation object, custom predict_proba function, and observation that was explained.
    """

    all_cols = model.X_test.columns
    idx2ohe = model.idx2ohe

    def custom_predict_proba(X, model):
        """
        Create a custom predict_proba for the model, so that it could be used in lime.
        :param X: Example to be classified.
        :param model: The model - classifier.
        :return: The probability that X will be classified as 1.
        """
        X_str = convert_to_lime_format(X, idx2ohe, col_names=all_cols, invert=True)
        return model.predict_proba(X_str)

    custom_model_predict_proba = partial(custom_predict_proba, model=model.model)
    observation = convert_to_lime_format(model.X_test.iloc[[example], :], model.idx2ohe).values[0]
    explanation = model.example_lime_explainer[example].explain_instance(
        observation,
        custom_model_predict_proba,
        num_samples=15000,
        num_features=len(model.numerical_features) + len(model.categorical_features))

    return explanation, custom_model_predict_proba, observation


def explain_single_instance_with_lime(model: Model, example: int):
    """
    Explain single instance with LIME from the test dataset.
    :param model: Model, which should be explained
    :param example: Position of the example from the test dataset, that has to be explained
    :return: An explanation object
    """
    explanation, _, _ = _explain_single_instance_with_lime(model, example)

    return explanation


def explain_single_instance_with_lime_stability(model: Model, example: int):
    """
    Explain single instance with LIME from the test dataset and calculate the Variables Stability Index (VSI) and
    Coefficients Stability Index (CSI) for this explanation.
    :param model: Model, which should be explained
    :param example: Position of the example from the test dataset, that has to be explained
    :return: An explanation object, Variables Stability Index (VSI) and Coefficients Stability Index (CSI) for this
    explanation
    """
    explanation, custom_model_predict_proba, observation = _explain_single_instance_with_lime(model, example)

    csi, vsi = model.example_lime_explainer[example].check_stability(
        observation,
        custom_model_predict_proba,
        num_features=len(model.numerical_features))

    return explanation, csi, vsi


def explain_single_instance_with_shap(model: Model, example: int):
    """
    Explain single instance with SHAP.
    :param model: The model, for which an explanation should be generated
    :param example: Example number to be explained
    :return: A plot for the explanation.
    """
    prediction, _ = _get_prediction_for_example(model, example)

    fp = force_plot(
        model.shap_kernel_explainer.expected_value[prediction],
        model.shap_values[prediction][example, :],
        model.X_test_ohe.iloc[example, :])

    return fp


def get_test_examples(model: Model, examples_type: ExampleType, number_of_examples: int) -> list:
    """
    Extracts a subset with row numbers from the X_test pd.DataFrame depending on the example type
    that is needed.
    :param model: The model for which examples should be collected
    :param examples_type: The type of examples to be extracted
    :param number_of_examples: Number of example numbers to be extracted
    :return: List containing the row numbers of the extracted examples. Later these numbers should be used for
    explaining a single instance of the test set.
    """
    examples = []
    indexes = None

    if number_of_examples > 0:
        if examples_type is ExampleType.RANDOM:
            indexes = choices(model.X_test.index.tolist(), k=number_of_examples)
        else:
            X_output = model.X_test.copy()
            X_output.loc[:, 'predict'] = model.model.predict(X_output)
            X_output['result'] = model.y_test.values
            if examples_type is ExampleType.TRULY_CLASSIFIED:
                truly_classified = X_output.loc[(X_output['predict'] == X_output['result'])].index.tolist()
                indexes = choices(truly_classified, k=number_of_examples)
            elif examples_type is ExampleType.FALSELY_CLASSIFIED:
                falsely_classified = X_output.loc[(X_output['predict'] != X_output['result'])].index.tolist()
                indexes = choices(falsely_classified, k=number_of_examples)
            else:
                log.error("Example(s) type: {} is not yet supported for this function. Please either use another type"
                          "or extend the functionality.".format(examples_type.name))
    else:
        log.error("The number of examples should be greater than 0: {}".format(number_of_examples))

    # Post-processing: Get the location of the examples by their indexes in the test pd.DataFrame
    for i in range(number_of_examples):
        examples.append(model.X_test.index.get_loc(indexes[i]))

    return examples


@dispatch(str)
def get_dataset(id: str) -> (Dataset, str):
    """
    Get a dataset from the built-in datasets.
    :param id: The id (must be equal to the Datasets enum name) of the dataset
    :return: A fully loaded dataset, A message for the user
    """
    dataset = Dataset.built_in(id)
    msg = "Dataset \'{} ({})\' loaded successfully. For further information about this dataset please visit: {}"\
        .format(dataset.id.name, dataset.name, dataset.url)
    log.debug("{}\n{}".format(msg, dataset.df.head()))

    return dataset, msg


@dispatch(str, str)
def get_dataset(name: str, url: str) -> (Dataset, str):
    """
    Get a dataset from an URL (external source).
    :param name: The name of the dataset.
    :param url: The URL from which the dataset should be (down-)loaded
    :return: A fully loaded dataset, A message for the user
    """
    dataset = Dataset.from_url(name, url)
    msg = "Dataset \'{} ({})\' loaded successfully. For further information about this dataset please visit: {}"\
        .format(dataset.id.name, dataset.name, dataset.url)
    log.debug("{}\n{}".format(msg, dataset.df.head()))

    return dataset, msg


def change_cross_columns_status(model: Model, new_value: str) -> str:
    """
    Enables/Disables the cross columns select multiple widget.
    :param model: Model for which the status shall be changed.
    :param new_value: The new value of the widget.
    :return: Message indicating that the status was successfully changed.
    """
    msg = "Cross columns status was successfully changed to "
    if new_value == SplitTypes.BALANCED.name:
        model.cross_columns_sm.disabled = False
        msg = msg + "enabled."
    elif new_value == SplitTypes.NORMAL.name:
        model.cross_columns_sm.disabled = True
        msg = msg + "disabled."

    log.debug(msg)
    return msg


def show_target(df: pd.DataFrame, new_value: str):
    """
    Generate a message to be displayed to the user when new target is selected.
    :param df: The dataset as a dataframe.
    :param new_value: The new target that was selected.
    :return: (Series containing only the head of the target, A message to be displayed to the user)
    """

    df_target = None
    msg = ""
    if new_value is not None:
        df_target = df[new_value].head(5)
        msg = 'Target \'{0}\' value changed successfully.\n{1}'.format(new_value, df_target)
        log.debug(msg)
    else:
        msg = "No target was selected. Please select a target."
        log.error(msg)

    return df_target, msg


def split_feature_target(df: pd.DataFrame, target: str) -> (pd.DataFrame, pd.Series, str):
    """
    Divides the dataset into features and target.
    :param df: The dataset as a dataframe.
    :param target: The target selected by the user.
    :return: (Features as a dataframe, Target as series, A message to be displayed to the user)
    """

    msg = ""
    df_X = None
    df_y = None

    if target is not None:
        df_X = df.drop(target, axis=1)
        df_y = df[target]

        msg = 'Target \'{}\' selected successfully.'.format(target)
        log.info(msg)
    else:
        msg = "No target was selected. Please select a target."
        log.error(msg)

    return df_X, df_y, msg


def calculate_slider_properties(unique_values: np.ndarray) -> (float, float, float):
    """
    Calculates the min and max values for a slider based on the values in a column and its step based on min and max.
    :param unique_values: The unique values in a column
    :return: (min, max, step) values for the slider
    """
    min_val = min(unique_values)
    max_val = max(unique_values)
    step = (max_val - min_val)/100.0

    return min_val, max_val, 1.0 if step < 1.0 else step


def get_stripped_df(df: pd.DataFrame, column, value, eq_value: str = '=') -> pd.DataFrame:
    """
    Strips a dataframe based on a value and a sign.
    :param df: The dataframe to be stripped
    :param column: The column to be stripped
    :param value: The new value that should be used for the eq on the column
    :param eq_value: Whether >,< or = should be applied
    :return: The stripped dataframe
    """

    df_strip = None
    if eq_value == '>':
        df_strip = df.loc[df[column] > value]
    elif eq_value == '=':
        df_strip = df.loc[df[column] == value]
    elif eq_value == '<':
        df_strip = df.loc[df[column] < value]

    return df_strip


def remove_model_features(model: Model) -> str:
    """
    Removes features selected by the user from a model.
    :param model: The model, for which features should be removed.
    :return: Message indicating that the features were successfully removed.
    """
    features = list(model.remove_features_sm.value)
    df_X_new = model.X.drop(columns=features, axis=1)
    model.X = df_X_new

    msg = 'Features: {} were removed successfully for model {}.\n{}'.format(features, model.name, df_X_new.head(5))
    log.info(msg)
    return msg


def _ensure_valid_dataset(df: pd.DataFrame):
    """
    Make sure that no categorical feature in the dataset has more than upper_bound unique values.
    :param df: Dataset to be tested
    :return: void
    """
    _, cat = _divide_features(df)

    upper_bound = 30
    columns = []
    for col in df.columns:
        if col in cat and len(df[col].unique()) > upper_bound:
            columns.append(col)

    if len(columns) > 0:
        raise NotImplementedError("Each column of {} contain more than {} unique values. "
                                  "Such datasets are currently not supported. "
                                  "Please preprocess your data and try again.".format(str(columns), upper_bound))


def _log_elapsed_time(start, end, msg):
    """
    Logs the time elapsed during the execution of an operation.
    The output looks as follows:
    HH:MM:SS.MS
    24:01:40.53
    00:05:00.30
    00:00:00.23
    :param start: The time before the execution of the operation.
    :param end: The time after the execution of the operation.
    :param msg: Additional message to be displayed, e.g. the type of the operation.
    :return: void
    """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    log.debug("The elapsed time for {} {:0>2}:{:0>2}:{:05.2f}".format(msg, int(hours), int(minutes), seconds))


def _get_number_of_digits(n: int):
    """
    Returns the number of digits of a number.
    :param n: The original number
    :return: The number of digits
    """
    if n > 0:
        digits = int(math.log10(n))+1
    elif n == 0:
        digits = 1
    else:
        digits = int(math.log10(-n))+1

    return digits


def _get_model_type(y: pd.Series) -> ModelType:
    """
    Get the model type (problem type) by the target feature.
    :param y: The target feature for the model to be trained.
    :return: The corresponding model type for this target.
    """

    model_type = None
    if is_string_dtype(y) or len(y.unique()) == 2:
        model_type = ModelType(ProblemType.CLASSIFICATION)
    else:
        model_type = ModelType(ProblemType.REGRESSION)

    return model_type


def fill_empty_models(df_X: pd.DataFrame, df_y: pd.Series, number_of_models: int) -> (list, str):
    """
    A list of models will be created, where each model gets a name and the initial X and y of the dataset.
    :param df_X: Dataframe containing all columns of the dataset excluding the target.
    :param df_y: Series containing the target of the dataset.
    :param number_of_models: How many models should be trained.
    :return: (models, message) - Models is a list containing the all initial models to be trained, Message is a
    log message indicating that the operation was successful.
    """
    models = []
    _ensure_valid_dataset(df_X)

    for m in range(number_of_models):
        models.append(Model(m, "Model " + str(m+1), None, df_X, df_y, _get_model_type(df_y)))

    msg = "Models to be trained: \'{}\'.".format(number_of_models)
    log.debug(msg)
    return models, msg


def fill_model(model: Model, algorithm=None, split=None) -> str:
    """
    A model is trained based on the properties selected by the user.
    :param model: The model to be filled - trained and then saved.
    :param algorithm: The algorithm used for training the model.
    :param split: The split used for training the model.
    :return: String message about the status of the model that should be displayed as info.
    """
    if algorithm:
        model.model_type.algorithm = algorithm
    else:
        model.model_type.algorithm = Algorithm[model.model_type_dd.value]

    if split:
        model.split = split
    else:
        model.split = Split(SplitTypes[model.split_type_dd.value], list(model.cross_columns_sm.value))

    model_pipeline, X_train, y_train, X_test, y_test = \
        train_model(model.model_type, model.split, model.X, model.y)

    model.model = model_pipeline
    model.X_train = X_train
    model.y_train = y_train
    model.X_test = X_test
    model.y_test = y_test
    # required later for global model interpretations
    model.numerical_features, model.categorical_features = _divide_features(X_test)
    model.categorical_ohe_features = _get_categorical_ohe_features(model_pipeline, model.categorical_features)
    model.features_ohe = model.numerical_features + model.categorical_ohe_features
    model.idx2ohe = generate_idx2ohe_dict(X_test, model.categorical_features, model.features_ohe)
    model.X_train_ohe = calculate_X_ohe(model_pipeline, X_train, model.features_ohe)
    model.X_test_ohe = calculate_X_ohe(model_pipeline, X_test, model.features_ohe)
    model.predictions = model.model.predict(model.X_test)

    msg = "Model {} trained successfully!".format(model.name)
    log.info(msg)
    return msg


def normalize_undefined_values(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize (replace with 0.0) all rows containing undefined values
    in the dataframe (e.g, ? values, "" strings, ...).
    :param symbol: Symbol used for undefined values (?, "", NaN, ...)
    :param df: The dataframe that has to be normalized
    :return: New dataframe with 0.0 instead of the undefined rows
    """
    return df.replace(symbol, 0.0)


def remove_undefined_rows(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all rows containing undefined values in the dataframe (e.g, ? values, "" strings, ...).
    :param symbol: Symbol used for undefined values (?, "", NaN, ...)
    :param df: The dataframe that has to be normalized
    :return: New dataframe without the undefined rows.
    """
    return df.replace(symbol, float("NaN")).dropna().reset_index(drop=True)
