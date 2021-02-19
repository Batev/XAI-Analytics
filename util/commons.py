import pandas as pd
import numpy as np
import logging as log
import enum
import time
import math

from functools import partial
from random import choice, choices, randrange
from matplotlib import figure, axes
from IPython import display
from xai import balanced_train_test_split
from eli5 import show_weights, explain_weights
from shap import summary_plot, dependence_plot, force_plot
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
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
from pdpbox import pdp
from util.dataset import Datasets, Dataset
from util.model import Algorithm, Model, ModelType, ProblemType
from util.split import Split, SplitTypes


RANDOM_NUMBER = randrange(100)
TEST_SPLIT_SIZE = 0.3


# Configure logger
log.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
# log.getLogger().setLevel(log.DEBUG)
log.getLogger().setLevel(log.INFO)

# Remove DataFrame display limitation
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
    SHAP = 2


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
        ('imputer', SimpleImputer(strategy='median'))])
        #, ('scaler', StandardScaler())]) commented out because model interpretation is unreadable

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

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
        return Pipeline([("preprocessor", ct),
                         ("model",
                         LogisticRegression(class_weight="balanced",
                                            solver="liblinear",
                                            random_state=RANDOM_NUMBER))])
    elif algorithm is Algorithm.DECISION_TREE:
        return Pipeline([("preprocessor", ct),
                         ("model",
                          DecisionTreeClassifier(class_weight="balanced"))])
    elif algorithm is Algorithm.RANDOM_FOREST:
        n_estimators = int(5 if _get_number_of_digits(data_size) < 4
                           else math.pow(10, _get_number_of_digits(data_size)-3))
        log.debug("Number of estimators for {}: {}".format(algorithm.name, n_estimators))
        return Pipeline([("preprocessor", ct),
                         ("model",
                          RandomForestClassifier(class_weight="balanced",
                                                 n_estimators=n_estimators,
                                                 n_jobs=-1))])
    elif algorithm is Algorithm.XGB:
        return Pipeline([("preprocessor", ct),
                         ("model",
                          XGBClassifier(n_jobs=-1))])
    elif algorithm is Algorithm.SVC:
        return Pipeline([("preprocessor", ct),
                         ("model",
                          SVC(class_weight="balanced",
                              # kernel="linear",
                              # kernel="sigmoid",
                              kernel="rbf",
                              # kernel='poly',
                              # degree=3,
                              random_state=RANDOM_NUMBER,
                              probability=True,
                              verbose=False))])
    elif algorithm is Algorithm.LINEAR_REGRESSION:
        return Pipeline([("preprocessor", ct),
                         ("model",
                          LinearRegression(n_jobs=-1))])
    elif algorithm is Algorithm.SVM:
        raise NotImplementedError
    else:
        raise NotImplementedError


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

        while not are_sets_equal:
            X_train, X_test, y_train, y_test = _get_train_test_split(split, cat_features, df_x, df_y)

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
                                   "Train split features: {}\n"
                                   "Test split features: {}\n".
                                   format(count, X_train_ohe_features, X_test_ohe_features))
    else:
        X_train, X_test, y_train, y_test = _get_train_test_split(split, cat_features, df_x, df_y)

    return X_train, X_test, y_train, y_test


def _get_train_test_split(split: Split, cat_features: list, df_X: pd.DataFrame, df_y: pd.Series)\
        -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
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
    :return: (X_train, X_test, y_train, y_test)
    """
    from functools import reduce
    if split.type is SplitTypes.BALANCED:
        if split.value is None:
            random_number_split = randrange(100)
            log.debug("Random number used for the train/test split: {}.".format(random_number_split))
            X_train, X_test, y_train, y_test = train_test_split(df_X,
                                                                df_y,
                                                                stratify=df_y,
                                                                test_size=TEST_SPLIT_SIZE,
                                                                random_state=random_number_split)
            if cat_features:
                from imblearn.over_sampling import SMOTENC
                X_resampled_train, y_resampled_train = \
                    SMOTENC(categorical_features=sorted([list(df_X.columns).index(f) for f in cat_features]),
                            random_state=RANDOM_NUMBER).fit_resample(X_train, y_train)
            else:
                from imblearn.over_sampling import ADASYN
                X_resampled_train, y_resampled_train = ADASYN().fit_resample(X_train, y_train)
            return X_resampled_train, X_test, y_resampled_train, y_test
        else:
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
                   pd.Series(data=y_test_balanced, name=df_y.name)
    elif split.type is SplitTypes.IMBALANCED:
        random_number_split = randrange(100)
        log.debug("Random number used for the train/test split: {}.".format(random_number_split))
        X_train, X_test, y_train, y_test = train_test_split(df_X,
                                                            df_y,
                                                            stratify=df_y,
                                                            test_size=TEST_SPLIT_SIZE,
                                                            random_state=random_number_split)
        return X_train, X_test, y_train, y_test
    else:
        raise NotImplementedError


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
    # regression
    elif model_type.problem_type == ProblemType.REGRESSION:
        log.info("R2 score : %.2f" % r2_score(y_test, y_pred))
        log.info("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
        log.info("RMSE number:  %.2f" % pd.np.sqrt(mean_squared_error(y_test, y_pred)))

    return model, X_train, y_train, X_test, y_test


def plot_feature_importance_with_eli5(model: Model) -> display.HTML:
    """
    Global explanation for a model of type feature importance.
    :param model: The model to be interpreted.
    :return: IPython.display.HTML element with the feature importance.
    """
    return show_weights(model.model.named_steps["model"], feature_names=model.features_ohe)


def plot_feature_importance_with_skater(model: Model) -> (figure.Figure, axes.Axes):
    """
    Global explanation for a model of type feature importance.
    :param model: The model to be interpreted.
    :return: (f, ax): (figure instance, matplotlib.axes._subplots.AxesSubplot)
    """

    f, ax = model.skater_interpreter.feature_importance.plot_feature_importance(
        model.skater_model,
        n_samples=1000,
        ascending=True)
    return f, ax


def plot_feature_importance_with_shap(model: Model, plot_type="bar"):
    """
    Plot feature importance for a given model with shap.
    :param model: Model for which the feature importance should be plotted.
    :param plot_type: The type of the plot
    :return: void
    """
    summary_plot(model.shap_values, model.X_test_ohe, plot_type=plot_type)


def generate_eli5_feature_importance_explanation(models: list, upper_bound: int = 3) -> str:
    """
    Generate explanation regarding the weights of some features for each model.
    :param models: Models, for which an explanation should be generated.
    :param upper_bound: For how many features an explanation should be generated per model.
    :return: String message containing an auto-generated explanation.
    """

    model_weight = {}
    for model in models:
        weights = []
        explanation = explain_weights(model.model.named_steps["model"], feature_names=model.features_ohe)
        try:
            weights = explanation.targets[0].feature_weights.pos
        except TypeError as e:
            try:
                log.debug("An expected error occurred. Program execution may continue: {}".format(e))
                weights = explanation.feature_importances.importances
            except AttributeError as e:
                log.debug("An expected error occurred. Program execution may continue: {}".format(e))
                log.warning("{} not supported for ELI5 explanations.".format(model.model_type.algorithm.name))
                continue

        feature_weight = {}
        for weight in weights:
            feature_weight[weight.feature] = weight.weight

        model_weight[model.name] = feature_weight

    return _generate_generic_feature_importance_explanation(upper_bound, model_weight)


def generate_skater_feature_importance_explanation(models: list, upper_bound: int = 3) -> str:
    """
    Generate explanation regarding the (permutation) feature importance for each feature of a model.
    :param models: Models, for which an explanation should be generated.
    :param upper_bound: For how many features an explanation should be generated per model.
    :return: String message containing an auto-generated explanation.
    """

    model_weight = {}
    for model in models:
        if not model.skater_model or not model.skater_interpreter:
            model.init_skater()

        weights = model.skater_interpreter.feature_importance.feature_importance(model.skater_model)
        feature_weight = {}

        for i in range(1, len(weights) + 1):
            feature_weight[list(weights.keys())[i*(-1)]] = weights[i*(-1)]

        model_weight[model.name] = feature_weight

    return _generate_generic_feature_importance_explanation(upper_bound, model_weight)


def generate_shap_feature_importance_explanation(models: list, upper_bound: int = 3) -> str:
    """
    Generate explanation regarding the average impact on model output magnitude for each feature of a model.
    :param models: Models, for which an explanation should be generated.
    :param upper_bound: For how many features an explanation should be generated per model.
    :return: String message containing an auto-generated explanation.
    """
    model_weight = {}

    for model in models:
        if not model.shap_values:
            model.init_shap()

        feature_mean = {}
        feature_order = np.argsort(np.sum(-np.mean(np.abs(model.shap_values), axis=1), axis=0))
        feature_weight = np.sum(np.mean(np.abs(model.shap_values), axis=1), axis=0)

        for feature_key in feature_order:
            feature_mean[model.features_ohe[feature_key]] = feature_weight[feature_key].round(3)
        model_weight[model.name] = feature_mean

    return _generate_generic_feature_importance_explanation(upper_bound, model_weight)


def _generate_generic_feature_importance_explanation(upper_bound: int, model_weight: dict) -> str:
    """
    Generates the explanation messages given a the weights and feature names for each model.
    :param upper_bound: For how many features should a message be generated
    :param model_weight: A dict consisting of [model name] -> [dict] (consisting of [feature] -> [weight])
    :return:
    """

    expln = ["Summary:\n"]
    phrases = ["same as", "identical to", "alike", "matching", "similar to"]
    adjectives = ["highest", "best", "most important", "most valuable", "most influential"]
    visited = []

    for model_name, weights in model_weight.items():
        if len(weights) < upper_bound:
            upper_bound = len(weights)
        for count in range(0, upper_bound):
            feature = list(weights.keys())[count]
            weight = weights[feature]

            msg = "The {}{} feature for {} is {} with weight ~{}"\
                .format(_get_nth_ordinal(count + 1) + " " if count + 1 != 1 else "",
                        choice(adjectives),
                        model_name,
                        feature,
                        round(weight, 3))

            if feature in visited:
                other_model_name = list(model_weight.keys())[visited.index(feature) // upper_bound]
                msg = msg + ", {0} {1} for {2}.\n"\
                    .format(choice(phrases),
                            _get_nth_ordinal(list(model_weight[other_model_name].keys()).index(feature) + 1),
                            other_model_name)
            else:
                msg = msg + ".\n"
            expln.append(msg)
            visited.append(feature)

        expln.append("\n")

    return ' '.join(expln)


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
        if model.model_type.algorithm is Algorithm.SVC:
            log.warning("{} not is supported by {}.".format(model.model_type.algorithm.name, type))
        else:
            plot = plot_feature_importance_with_eli5(model)
    elif type == FeatureImportanceType.SKATER:
        if not model.skater_model or not model.skater_interpreter:
            model.init_skater()
        plot_feature_importance_with_skater(model)
    elif type == FeatureImportanceType.SHAP:
        if not model.shap_values:
            model.init_shap()
        plot_feature_importance_with_shap(model)
    else:
        log.warning("Type {} is not yet supported. Please use one of the supported types.".format(type))

    end = time.time()
    _log_elapsed_time(start, end, "generating a feature importance plot with {} is".format(type.name))

    return plot


def generate_feature_importance_explanation(type: FeatureImportanceType, models: list, upper_bound: int = 3) -> str:
    """
    Auto-generate explanation for the feature importance results of each model in the list.
    :param type: Type of feature importance that should be explained
    :param models: List of all models for which an explanation should be generated
    :param upper_bound: For how many features of each model a generation should be generated
    :return: A string containing the explanation.
    """

    log.info("Generating feature importance explanation for {} ...".format(type.name))

    if type == FeatureImportanceType.ELI5:
        str = generate_eli5_feature_importance_explanation(models, upper_bound)
    elif type == FeatureImportanceType.SKATER:
        str = generate_skater_feature_importance_explanation(models, upper_bound)
    elif type == FeatureImportanceType.SHAP:
        str = generate_shap_feature_importance_explanation(models, upper_bound)
    else:
        log.warning("Type {} is not yet supported. Please use one of the supported types.".format(type))

    return str


def generate_pdp_plots(type: PDPType, model: Model, feature1: str, feature2: str):
    """
    Generate a PDP including one or two features for a given model.
    :param type: Type of framework that should be used for generating the PDPs
    :param model: Model for which a plot shall be generated
    :param feature1: Feature to be included in the PDP.
    :param feature2: If none, plot a PDP for only feature1
    :return: None; TODO: return a plot.
    """
    plot = None
    log.info("Generating a PDP plot using {} for {} ...".format(type.name, model.name))

    start = time.time()

    if type == PDPType.PDPBox:
        if feature2 == 'None':
            plot_single_pdp_with_pdpbox(model, feature1)
        else:
            plot_multi_pdp_with_pdpbox(model, feature1, feature2)
    elif type == PDPType.SKATER:
        if not model.skater_model or not model.skater_interpreter:
            model.init_skater()
        if feature2 == 'None':
            plot_single_pdp_with_skater(model, feature1)
        else:
            plot_multi_pdp_with_skater(model, feature1, feature2)
    elif type == PDPType.SHAP:
        if not model.shap_values:
            model.init_shap()
        if feature2 == 'None':
            plot_single_pdp_with_shap(model, feature1)
        else:
            plot_multi_pdp_with_shap(model, feature1, feature2)
    else:
        log.warning("Type {} is not yet supported. Please use one of the supported types.".format(type))

    end = time.time()
    _log_elapsed_time(start, end, "generating a PDP with {} is".format(type.name))

    return plot


def plot_single_pdp_with_pdpbox(model: Model, feature: str, plot_lines=True,
                                x_quantile=True, show_percentile=True, plot_pts_dist=True,)\
        -> (figure.Figure, axes.Axes):
    """
    Plots a PDP for a single feature for a given model.
    :param model: The model for which a PDP should be created
    :param feature: Feature or feature list to investigate, for one-hot encoding features, feature list is required
    :param plot_lines: Whether to plot out the individual lines
    :param x_quantile: Whether to construct x axis ticks using quantiles
    :param show_percentile: Whether to display the percentile buckets, for numeric feature when grid_type='percentile'
    :param plot_pts_dist: Whether to show data points distribution
    :return: (f, ax): (figure instance, matplotlib.axes._subplots.AxesSubplot)
    """

    # model.model[1] to get the actual model from the pipeline
    pdp_isolate_out = pdp.pdp_isolate(
        model=model.model[1],
        dataset=model.X_test_ohe,
        model_features=model.features_ohe,
        feature=feature)

    fig, axes = pdp.pdp_plot(
        pdp_isolate_out=pdp_isolate_out,
        feature_name=feature if isinstance(feature, str) else feature[0].split('_')[0].title,
        plot_lines=plot_lines,
        x_quantile=x_quantile,
        show_percentile=show_percentile,
        plot_pts_dist=plot_pts_dist,
        frac_to_plot=0.5)

    return fig, axes


def plot_multi_pdp_with_pdpbox(model: Model, feature1: str, feature2: str, plot_type='contour',
                               x_quantile=False, plot_pdp=False) -> (figure.Figure, axes.Axes):
    """
    Plots a PDP for two features for a given model.
    :param model: The model for which a PDP should be created
    :param feature1: Feature to be plotted
    :param feature2: Feature with which feature1 interacts to be plotted
    :param plot_type: Type of the interact plot, can be 'contour' or 'grid'
    :param x_quantile: Whether to construct x axis ticks using quantiles
    :param plot_pdp: Whether to plot pdp for each feature
    :return: (f, ax): (figure instance, matplotlib.axes._subplots.AxesSubplot)
    """
    features_to_plot = [feature1, feature2]

    # model.model[1] to get the actual model from the pipeline
    pdp_interact_out = pdp.pdp_interact(
        model=model.model[1],
        dataset=model.X_test_ohe,
        model_features=model.features_ohe,
        features=features_to_plot)

    fig, axes = pdp.pdp_interact_plot(
        pdp_interact_out=pdp_interact_out,
        feature_names=features_to_plot,
        plot_type=plot_type,
        x_quantile=x_quantile,
        plot_pdp=plot_pdp)

    return fig, axes


def plot_single_pdp_with_shap(model: Model, feature: str):
    """
    Plots a shap PDP for a single feature for a given model.
    :param model: The model for which a PDP should be plotted
    :param feature: Feature to be plotted
    :return: void
    """

    dependence_plot(ind=feature,
                         interaction_index=feature,
                         shap_values=model.shap_values[0],
                         features=model.X_test_ohe)
                         # features=model.X_test_ohe.sample(66, random_state=RANDOM_NUMBER))


def plot_multi_pdp_with_shap(model: Model, feature1: str, feature2='auto'):
    """
    Plots a shap PDP for two features for a given model.
    :param model: The model for which a PDP should be plotted
    :param feature1: Feature to be plotted
    :param feature2: Feature with which feature1 interacts to be plotted.
    If the value is 'auto' the feature with most interaction will be selected
    :return: void
    """

    dependence_plot(ind=feature1,
                         interaction_index=feature2,
                         shap_values=model.shap_values[0],
                         features=model.X_test_ohe)
                         # features=model.X_test_ohe.sample(66, random_state=RANDOM_NUMBER))


def plot_single_pdp_with_skater(model: Model, feature: str, n_samples=1000, grid_resolution=50, grid_range=(0, 1),
                                with_variance=True, figsize=(6, 4)):
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
    :return: A plot
    """

    r = model.skater_interpreter.partial_dependence.plot_partial_dependence([feature],
                                                                            model.skater_model,
                                                                            n_samples=n_samples,
                                                                            grid_resolution=grid_resolution,
                                                                            grid_range=grid_range,
                                                                            with_variance=with_variance,
                                                                            figsize=figsize)
    return r


def plot_multi_pdp_with_skater(model: Model, feature1: str, feature2: str, n_samples=1000, grid_resolution=100,
                               grid_range=(0, 1), with_variance=False, figsize=(12, 5)):
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
    :return: A plot
    """

    r = model.skater_interpreter.partial_dependence.plot_partial_dependence([(feature1, feature2)],
                                                                            model.skater_model,
                                                                            n_samples=n_samples,
                                                                            grid_resolution=grid_resolution,
                                                                            grid_range=grid_range,
                                                                            with_variance=with_variance,
                                                                            figsize=figsize)
    return r


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


def explain_single_instance(local_interpreter: LocalInterpreterType, model: Model, example: int):
    """
    Explain single instance (example) with a given interpreter type.
    :param local_interpreter: Type of interpreter to be used. Currently only LIME and SHAP are supported
    :param model: The model for which an instance should be explained
    :param example: The example to be explained - The row number from the X_test pd.DataFrame
    :return: An explanation
    """
    explanation = None
    log.info("Generating a single instance explanation using {} for {} ...".format(local_interpreter.name, model.name))

    start = time.time()

    if local_interpreter is LocalInterpreterType.LIME:
        if not model.lime_explainer:
            model.init_lime()
        explanation = explain_single_instance_with_lime(model, example)
    elif local_interpreter is LocalInterpreterType.SHAP:
        if not model.shap_values:
            model.init_shap()
        explanation = explain_single_instance_with_shap(model, example)
    else:
        log.error("Interpreter type {} is not yet supported for local interpretations. Please either use another one"
                  "or extend the functionality of this function".format(local_interpreter))

    end = time.time()
    _log_elapsed_time(start, end, "generating a single instance explanation with {} is".format(local_interpreter.name))

    return explanation


def generate_single_instance_comparison(models: list, example: int) -> str:
    """
    Compare models' decisions for a given example.
    :param models: All models to be compared
    :param example: The example, which is classified by the models
    :return: An a string containing information whether each model's decision was right or not.
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

    if local_interpreter is LocalInterpreterType.LIME:
        if not model.lime_explainer:
            model.init_lime()
        explanation = generate_single_instance_explanation_with_lime(model, example)
    elif local_interpreter is LocalInterpreterType.SHAP:
        if not model.shap_values:
            model.init_shap()
        explanation = generate_single_instance_explanation_with_shap(model, example)
    else:
        log.error("Interpreter type {} is not yet supported for local interpretations. Please either use another one"
                  "or extend the functionality of this function".format(local_interpreter))

    return explanation


def generate_single_instance_explanation_with_lime(model: Model, example: int) -> str:
    """
    Generate an explanation for a single instance (example) for a model with LIME.
    :param model: Model for whose decision an explanation shall be generated
    :param example: Example, that should be explained
    :return: An explanation.
    """

    explanation = explain_single_instance(LocalInterpreterType.LIME, model, example)
    feature_value = dict(explanation.as_list())
    prediction_probability = explanation.predict_proba[_get_prediction_for_example(model, example)]

    pos_elems = len(list(filter(lambda x: (x >= 0.0), list(feature_value.values()))))
    neg_elems = len(list(filter(lambda x: (x < 0.0), list(feature_value.values()))))

    return _generate_generic_single_instance_explanation(
        model.name,
        _strip_dict(feature_value, pos_elems if pos_elems <= 3 else 3, True),
        _strip_dict(feature_value, neg_elems if neg_elems <= 2 else 2, False),
        prediction_probability,
        'LIME')


def generate_single_instance_explanation_with_shap(model: Model, example: int) -> str:
    """
    Generate an explanation for a single instance (example) for a model with SHAP.
    :param model: Model for whose decison an explanation shall be generated
    :param example: Example, that should be explained
    :return: An explanation.
    """
    prediction = _get_prediction_for_example(model, example)

    base_value = model.shap_kernel_explainer.expected_value[prediction]
    shap_values = model.shap_values[prediction][example, :]
    features = list(model.X_test_ohe.iloc[example, :].index)

    feature_value = {}

    for count in range(len(features)):
        feature_value[features[count]] = shap_values[count]

    prediction_probability = np.sum(shap_values) + base_value

    pos_elems, neg_elems = _get_elements_number(list(feature_value.values()), 3, 2)

    return _generate_generic_single_instance_explanation(
        model.name,
        _strip_dict(feature_value, pos_elems, True),
        _strip_dict(feature_value, neg_elems, False),
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

    adverb_first = ['mostly', 'mainly', 'primarily', 'largely']
    adjective_second = ['largest', 'biggest', 'most substantial', 'most considerable']
    adjective_third = ['important', 'influential', 'impactful', 'effective']
    verb = ['impact', 'influence', 'affect', 'change']

    for c in range(len(d)):
        value_rounded = round(values[c], 4)
        if (c+1) == 1:
            msg = msg + "The feature that {} {}s {}'s {} prediction probability is {} with value of {}.\n"\
                .format(choice(adverb_first), choice(verb), model_name, desc, keys[c], value_rounded)
        elif (c+1) == 2:
            msg = msg + "The feature with the second {} {} on {}'s {} prediction probability is {} with value of {}.\n"\
                .format(choice(adjective_second), choice(verb), model_name, desc, keys[c], value_rounded)
        elif (c+1) == 3:
            msg = msg + "The third most {} feature for the {} prediction probability of {} is {} with value of {}\n"\
                .format(choice(adjective_third), desc, model_name, keys[c], value_rounded)
        else:
            msg = msg + "The {} feature that {} the {} prediction probability of {} is {} with value of {}\n"\
                .format(_get_nth_ordinal(c+1), choice(verb), desc, model_name, keys[c], value_rounded)

    return msg


def _get_elements_number(l: list, pos_upper_bound: int = 3, neg_upper_bound: int = 2) -> (int, int):
    """
    Count all elements that are positive and all that negative and differ from zero.
     If the there are more elements than the upper bound, return the upper bound.
    :param l: The list that has to be counted
    :param pos_upper_bound: The upper bound used as a positive counter limiter
    :param neg_upper_bound: The upper bound used as a negative counter limiter
    :return: A tuple containing the positive and negative element number.
    """

    count_pos = 0
    count_neg = 0

    for v in l:
        if v != 0 and v > 0 and count_pos != pos_upper_bound:
            count_pos = count_pos + 1
        elif v != 0 and v < 0 and count_neg != neg_upper_bound:
            count_neg = count_neg + 1

    return count_pos, count_neg


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


def _get_prediction_for_example(model: Model, example: int) -> int:
    """
    Returns either 0 or 1 for a classifier depending on the how the example was classified by the model.
    :param model: Model for which the example shall be classified
    :param example: Example to be classified
    :return: Either 0 or 1 depending on the result
    """
    predictions = list(model.model.predict_proba(model.X_test)[example])
    return predictions.index(max(predictions))


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


def explain_single_instance_with_lime(model: Model, example: int):
    """
    Explain single instance with LIME from the test dataset.
    :param model: Model, which should be explained
    :param example: Position of the example from the test dataset, that has to be explained
    :return:
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
    explanation = model.lime_explainer.explain_instance(
        observation,
        custom_model_predict_proba,
        num_features=len(model.numerical_features))

    return explanation


def explain_single_instance_with_shap(model: Model, example: int):
    """
    Explain single instance with SHAP.
    :param model: The model, for which an explanation should be generated
    :param example: Example number to be explained
    :return: A plot for the explanation.
    """
    prediction = _get_prediction_for_example(model, example)

    return force_plot(
        model.shap_kernel_explainer.expected_value[prediction],
        model.shap_values[prediction][example, :],
        model.X_test_ohe.iloc[example, :])


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
    elif new_value == SplitTypes.IMBALANCED.name:
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
