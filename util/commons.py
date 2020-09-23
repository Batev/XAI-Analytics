import pandas as pd
import numpy as np
import logging as log
import enum

from functools import partial
from random import choice, choices
from matplotlib import figure, axes
from IPython import display
from ipywidgets import Widget, GridBox, Select, SelectMultiple, IntSlider, RadioButtons
from xai import balanced_train_test_split
from eli5 import show_weights, explain_weights
from shap import summary_plot, dependence_plot, force_plot
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from pandas.api.types import is_numeric_dtype, is_string_dtype
from multipledispatch import dispatch
from pdpbox import pdp
from util.dataset import Datasets, Dataset
from util.model import Algorithm, Model, ModelType, ProblemType
from util.split import Split, SplitTypes


RANDOM_NUMBER = 33
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


def get_pipeline(ct: ColumnTransformer, algorithm: Algorithm) -> Pipeline:

    if algorithm is Algorithm.LOGISTIC_REGRESSION:
        return Pipeline([("preprocessor", ct),
                         ("model",
                         LogisticRegression(class_weight="balanced",
                                            solver="liblinear",
                                            random_state=RANDOM_NUMBER))])
    elif algorithm is Algorithm.DECISION_TREE:
        return Pipeline([("preprocessor", ct),
                         ("model", DecisionTreeClassifier(class_weight="balanced"))])
    elif algorithm is Algorithm.RANDOM_FOREST:
        return Pipeline([("preprocessor", ct),
                         ("model", RandomForestClassifier(class_weight="balanced", n_estimators=100, n_jobs=-1))])
    elif algorithm is Algorithm.XGB:
        return Pipeline([("preprocessor", ct),
                         ("model", XGBClassifier(n_jobs=-1))])
    elif algorithm is Algorithm.LINEAR_REGRESSION:
        return Pipeline([("preprocessor", ct),
                         ("model", LinearRegression(n_jobs=-1))])
    elif algorithm is Algorithm.SVM:
        return Pipeline([("preprocessor", ct),
                         ("model", SVC(kernel='poly', degree=8))])
    else:
        raise NotImplementedError


def get_split(split: Split, cat_features: list, df_x: pd.DataFrame, df_y: pd.Series)\
        -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):

    if split.type is SplitTypes.BALANCED:
        X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, train_idx, test_idx =\
            balanced_train_test_split(
                df_x, df_y, *split.value,
                min_per_group=300,
                max_per_group=300,
                categorical_cols=cat_features)
        return X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced
    elif split.type is SplitTypes.IMBALANCED:
        X_train, X_test, y_train, y_test = train_test_split(df_x,
                                                            df_y,
                                                            test_size=TEST_SPLIT_SIZE,
                                                            random_state=RANDOM_NUMBER)
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
    new_ohe_features = preprocessor.named_transformers_["cat"].named_steps['onehot']\
        .get_feature_names(cat_features)\
        .tolist()

    return new_ohe_features


def train_model(model_type: ModelType, split: Split, df_x: pd.DataFrame, df_y: pd.Series) -> \
        (Pipeline, pd.DataFrame, pd.Series):

    num_features, cat_features = _divide_features(df_x)

    log.debug("Numerical features: {}".format(num_features))
    log.debug("Categorical features: {}".format(cat_features))

    # Transform the categorical features to numerical
    preprocessor = get_column_transformer(num_features, cat_features)

    model = get_pipeline(preprocessor, model_type.algorithm)

    X_train, X_test, y_train, y_test = get_split(split, cat_features, df_x, df_y)

    # Now we can fit the model on the whole training set and calculate accuracy on the test set.
    model.fit(X_train, y_train)

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


def generate_eli5_weight_explanation(models: list, upper_bound: int = 3) -> str:
    """
    Generate explanation regarding the weights of some features for each model.
    :param models: Models, for which an explanation should be generated.
    :param upper_bound: For how many features an explanation should be generated per model.
    :return: String message containing an auto-generated explanation.
    """
    expln = ["Explanation:\n"]
    phrases = ["same as", "identical to", "alike", "matching"]
    order = {}
    visited = []

    for count in range(0, upper_bound):
        for model in models:
            weights = explain_weights(model.model.named_steps["model"], feature_names=model.features_ohe)
            try:
                feature = weights.targets[0].feature_weights.pos[count].feature
                weight = weights.targets[0].feature_weights.pos[count].weight
            except TypeError:
                feature = weights.feature_importances.importances[count].feature
                weight = weights.feature_importances.importances[count].weight

            msg = "The {}{} feature for {} is {} with weight ~{}"\
                .format(_get_nth_ordinal(count + 1) + " " if count + 1 != 1 else "",
                        "highest",
                        model.name,
                        feature,
                        round(weight, 3))

            if feature in visited:
                other_model = models[visited.index(feature) % len(models)]
                msg = msg + ", {0} {1} for {2} but with different weight."\
                    .format(choice(phrases),
                            _get_nth_ordinal(order.get(other_model.name).index(feature) + 1),
                            other_model.name)
            else:
                msg = msg + "."

            expln.append(msg)
            visited.append(feature)
            order[model.name] = order.get(model.name, []) + [feature]

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
    X_test_ohe = model[0].fit_transform(X)

    return pd.DataFrame(X_test_ohe.toarray(), columns=feature_names)


def generate_feature_importance_plot(type: str, model: Model):
    """
    Generate feature importance plot for a model.
    :param type: Type of feature importance method to be used.
    :param model: Model, for which a plot should be created.
    :return: If type is ELI5, then IPython.display.HTML plot
    is returned, None otherwise
    """
    plot = None
    log.info("Generating a feature importance plot using {} for {} ...".format(type, model.name))

    if FeatureImportanceType[type] == FeatureImportanceType.ELI5:
        plot = plot_feature_importance_with_eli5(model)
    elif FeatureImportanceType[type] == FeatureImportanceType.SKATER:
        if not model.skater_model or not model.skater_interpreter:
            model.init_skater()
        plot_feature_importance_with_skater(model)
    elif FeatureImportanceType[type] == FeatureImportanceType.SHAP:
        if not model.shap_values:
            model.init_shap()
        plot_feature_importance_with_shap(model)
    else:
        log.warning("Type {} is not yet supported. Please use one of the supported types.".format(type))

    return plot


def generate_pdp_plots(type: str, model: Model, feature1: str, feature2: str):
    plot = None

    log.info("Generating a PDP plot using {} for {} ...".format(type, model.name))
    if PDPType[type] == PDPType.PDPBox:
        if feature2 == 'None':
            plot_single_pdp_with_pdpbox(model, feature1)
        else:
            plot_multi_pdp_with_pdpbox(model, feature1, feature2)
    elif PDPType[type] == PDPType.SKATER:
        if not model.skater_model or not model.skater_interpreter:
            model.init_skater()
        if feature2 == 'None':
            plot_single_pdp_with_skater(model, feature1)
        else:
            plot_multi_pdp_with_skater(model, feature1, feature2)
    elif PDPType[type] == PDPType.SHAP:
        if not model.shap_values:
            model.init_shap()
        if feature2 == 'None':
            plot_single_pdp_with_shap(model, feature1)
        else:
            plot_multi_pdp_with_shap(model, feature1, feature2)
    else:
        log.warning("Type {} is not yet supported. Please use one of the supported types.".format(type))

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


def explain_single_instance(model: Model, local_interpreter: LocalInterpreterType, example: int):
    """
    Explain single instance (example) with a given interpreter type.
    :param model: The model for which an instance should be explained
    :param local_interpreter: Type of interpreter to be used. Currently only LIME and SHAP are supported
    :param example: The example to be explained - The row number from the X_test pd.DataFrame
    :return: An explanation
    """
    explanation = None

    log.info("Explanation for {}.".format(model.name))
    log.info("Example {}'s data: \n{}".format(example, model.X_test.iloc[example]))
    log.info("Model prediction for example {}: {}".format(example, model.predictions[example]))
    log.info("Actual result for example {}: {} \n".format(example, model.y_test.iloc[example]))

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

    return explanation


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


def explain_single_instance_with_shap(model: Model, example: int, expected_value_idx=0):
    """
    Explain single instance with SHAP.
    :param model: The model, for which an explanation should be generated
    :param example: Example number to be explained
    :param expected_value_idx: The index of the shap expected_value list
    :return: A plot for the explanation.
    """

    return force_plot(
        model.shap_kernel_explainer.expected_value[expected_value_idx],
        model.shap_values[0][example, :],
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
    log.info(msg)
    log.info("\n{}".format(dataset.df.head()))

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
    log.info(msg)
    log.info("\n{}".format(dataset.df.head()))

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
    for m in range(number_of_models):
        models.append(Model(m, "Model " + str(m+1), None, df_X, df_y, get_model_type(df_y)))

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
    # split_type = model.split_type_dd.value
    # split_feature = list(model.cross_columns_sm.value)

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


def get_model_type(y: pd.Series) -> ModelType:
    """
    Get the model type (problem type) by the target feature.
    :param y: The target feature for the model to be trained.
    :return: The corresponding model type for this target.
    """

    model_type = None
    if is_string_dtype(y):
        model_type = ModelType(ProblemType.CLASSIFICATION)
    else:
        model_type = ModelType(ProblemType.REGRESSION)

    return model_type


def get_model_by_id(models: list, id: int) -> Model:
    model = None
    for m in models:
        if m.id == id:
            model = m

    if model is None:
        log.error("No model found with ID '{}'.".format(id))
        return model
    else:
        return model


def get_models_by_names(models: list, names: list) -> list:
    sub_models = []
    for name in names:
        for model in models:
            if model.name == name:
                sub_models.append(model)

    if not models:
        log.error("No models found with names in {}''.".format(names))
        return sub_models
    else:
        return sub_models


def get_model_by_remove_features_button(models: list, button: Widget) -> Model:
    model = None
    for m in models:
        if m.remove_features_button is button:
            model = m
    if model is None:
        log.error("No model found with ID ''.".format(id))
        return model
    else:
        return model


def get_model_by_train_model_button(models: list, button: Widget) -> Model:
    model = None
    for m in models:
        if m.train_model_button is button:
            model = m
    if model is None:
        log.error("No model found with ID ''.".format(id))
        return model
    else:
        return model


def get_model_by_split_type_dd(models: list, dropdown: Widget) -> Model:
    model = None
    for m in models:
        if m.split_type_dd is dropdown:
            model = m
    if model is None:
        log.error("No model found with ID ''.".format(id))
        return model
    else:
        return model


def get_child_value_by_description(gridbox: GridBox, description: str, number: int):
    child = _get_child_by_description(gridbox, description)[number]
    if child is None:
        log.error("No element with description {} found!".format(description))
        return

    if isinstance(child, SelectMultiple):
        child_value = list(child.value)
    elif isinstance(child, Select) or isinstance(child, RadioButtons):
        child_value = str(child.value)
    elif isinstance(child, IntSlider):
        child_value = child.value
    else:
        log.error("Type {} is not yet supported. Please extend this function in order to support it."
                  .format(type(child)))
        return

    if not child_value:
        log.warning("No {} were selected. Please select at least one type and try again!".format(description))
        return

    return child_value


def _get_child_by_description(gridbox: GridBox, description: str) -> list:
    selects = []
    for child in gridbox.children:
        if child.description == description:
            selects.append(child)

    return selects
