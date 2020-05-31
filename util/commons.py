import IPython
import pandas as pd
import numpy as np
import eli5
import xai
import logging as log
import shap
import enum

from matplotlib import figure, axes
from lime.lime_tabular import LimeTabularExplainer
from functools import partial
from ipywidgets import widgets
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
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from pdpbox import pdp

from util.dataset import Datasets, Dataset
from util.model import Algorithm, Model, ModelType, ProblemType
from util.split import Split, SplitTypes

NUMERIC_TYPES = ["int", "float"]
RANDOM_NUMBER = 33
EXAMPLES_SPAN_ELI5 = 20
EXAMPLES_SPAN_LIME = 10
EXAMPLES_DIR_LIME = "lime_results"
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


def explain_single_instance(classifier: Pipeline,
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            example: int):

    num_features, cat_features = divide_features(X_test)
    new_ohe_features = get_ohe_cats(classifier, cat_features)

    # Transform the categorical feature's labels to a lime-readable format.
    categorical_names = {}
    for col in cat_features:
        categorical_names[X_test.columns.get_loc(col)] = [new_col.split("__")[1]
                                                          for new_col in new_ohe_features
                                                          if new_col.split("__")[0] == col]

    def custom_predict_proba(X, model):
        """
        Create a custom predict_proba for the model, so that it could be used in lime.
        :param X: Example to be classified.
        :param model: The model - classifier.
        :return: The probability that X will be classified as 1.
        """
        X_str = convert_to_lime_format(X, categorical_names, col_names=X_test.columns, invert=True)
        return model.predict_proba(X_str)


    # log.debug("Categorical names for lime: {}".format(categorical_names))

    explainer = LimeTabularExplainer(convert_to_lime_format(X_test, categorical_names).values,
                                     mode="classification",
                                     feature_names=X_test.columns.tolist(),
                                     categorical_names=categorical_names,
                                     categorical_features=categorical_names.keys(),
                                     discretize_continuous=True,
                                     random_state=RANDOM_NUMBER)

    log.info("Example {}'s data: \n{}".format(example, X_test.iloc[example]))
    log.info("Example {}'s actual result: {}".format(example, y_test[example]))

    custom_model_predict_proba = partial(custom_predict_proba, model=classifier)
    observation = convert_to_lime_format(X_test.iloc[[example], :], categorical_names).values[0]
    explanation = explainer.explain_instance(observation,
                                             custom_model_predict_proba,
                                             num_features=len(num_features))

    return explanation


def convert_to_lime_format(X, categorical_names, col_names=None, invert=False):
    """Converts data with categorical values as string into the right format
    for LIME, with categorical values as integers labels.
    It takes categorical_names, the same dictionary that has to be passed
    to LIME to ensure consistency.
    col_names and invert allow to rebuild the original dataFrame from
    a numpy array in LIME format to be passed to a Pipeline or sklearn
    OneHotEncoder
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


def divide_features(df: pd.DataFrame) -> (list, list):
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
        X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced, train_idx, test_idx = \
            xai.balanced_train_test_split(
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


def get_ohe_cats(model: Pipeline, cat_features: list) -> list:
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


def get_all_features(model: Pipeline, num_features: list, cat_features: list) -> list:
    return num_features + get_ohe_cats(model, cat_features)


def train_model(model_type: ModelType, split: Split, df_x: pd.DataFrame, df_y: pd.Series) -> \
        (Pipeline, pd.DataFrame, pd.Series):

    num_features, cat_features = divide_features(df_x)

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


def plot_feature_importance_with_eli5(model: Model) -> IPython.display.HTML:
    """
    Global explanation for a model of type feature importance.
    :param model: The model to be interpreted.
    :return: IPython.display.HTML element with the feature importance.
    """
    num_features, cat_features = divide_features(model.X)
    return eli5.show_weights(model.model.named_steps["model"],
                             feature_names=get_all_features(model.model,
                                                            num_features,
                                                            cat_features))


def plot_feature_importance_with_skater(model: Model) -> (figure.Figure, axes.Axes):
    """
    Global explanation for a model of type feature importance.
    :param model: The model to be interpreted.
    :return: (f, ax): (figure instance, matplotlib.axes._subplots.AxesSubplot)
    """
    num_features, cat_features = divide_features(model.X)
    interpreter = Interpretation(training_data=model.X_train, training_labels=model.y_train,
                                 feature_names=list(num_features + cat_features))
    im_model = InMemoryModel(model.model.predict_proba, examples=model.X_test,
                             target_names=['$50K or less', 'More than $50K'])
    f, ax = interpreter.feature_importance.plot_feature_importance(im_model, ascending=True)
    return f, ax


def plot_feature_importance_with_shap(model: Model, plot_type="bar"):
    """
    Plot feature importance for a given model with shap.
    :param model: Model for which the feature importance should be plotted.
    :param plot_type: The type of the plot
    :return: void
    """
    shap.summary_plot(model.shap_values, model.X_test_ohe, plot_type=plot_type)


def calculate_X_ohe(model: Pipeline, X: pd.DataFrame):
    """
    Transform a pd.DataFrame using a One-Hot Encoder from a suitable Pipeline.
    :param model: OHE from a Pipeline to be used
    :param X: X to be encoded
    :return: (One-Hot encoded pd.DataFrame, All One-Hot encoded columns)
    """
    num_features, cat_features = divide_features(X)
    feature_names = get_all_features(model, num_features, cat_features)

    # model.model[0] to get the preprocessor from the pipeline
    X_test_ohe = model[0].fit_transform(X)
    return pd.DataFrame(X_test_ohe.toarray(), columns=feature_names), feature_names


def generate_feature_importance_plot(type: str, model: Model) -> IPython.display.HTML:
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
        model.init_skater()
        plot_feature_importance_with_skater(model)
    elif FeatureImportanceType[type] == FeatureImportanceType.SHAP:
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
        model.init_skater()
        if feature2 == 'None':
            plot_single_pdp_with_skater(model, feature1)
        else:
            plot_multi_pdp_with_skater(model, feature1, feature2)
    elif PDPType[type] == PDPType.SHAP:
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

    shap.dependence_plot(ind=feature,
                         interaction_index=feature,
                         shap_values=model.shap_values[0],
                         features=model.X_test_ohe)


def plot_multi_pdp_with_shap(model: Model, feature1: str, feature2='auto'):
    """
    Plots a shap PDP for two features for a given model.
    :param model: The model for which a PDP should be plotted
    :param feature1: Feature to be plotted
    :param feature2: Feature with which feature1 interacts to be plotted.
    If the value is 'auto' the feature with most interaction will be selected
    :return: void
    """

    shap.dependence_plot(ind=feature1,
                         interaction_index=feature2,
                         shap_values=model.shap_values[0],
                         features=model.X_test_ohe)


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


def fill_model(model: Model) -> str:
    """
    A model is trained based on the properties selected by the user.
    :param model: The model to be filled - trained and then saved.
    :return: String message about the status of the model that should be displayed as info.
    """
    # split_type = model.split_type_dd.value
    # split_feature = list(model.cross_columns_sm.value)
    model.model_type.algorithm = Algorithm[model.model_type_dd.value]
    model.split = Split(SplitTypes[model.split_type_dd.value], list(model.cross_columns_sm.value))

    model_pipeline, X_train, y_train, X_test, y_test = \
        train_model(model.model_type, model.split, model.X, model.y)

    model.model = model_pipeline
    model.X_train = X_train
    model.y_train = y_train
    model.X_test = X_test
    model.y_test = y_test
    # required later for global model interpretations
    model.X_train_ohe, _ = calculate_X_ohe(model_pipeline, X_train)
    model.X_test_ohe, model.features_ohe = calculate_X_ohe(model_pipeline, X_test)

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


def get_model_by_remove_features_button(models: list, button: widgets.Widget) -> Model:
    model = None
    for m in models:
        if m.remove_features_button is button:
            model = m
    if model is None:
        log.error("No model found with ID ''.".format(id))
        return model
    else:
        return model


def get_model_by_train_model_button(models: list, button: widgets.Widget) -> Model:
    model = None
    for m in models:
        if m.train_model_button is button:
            model = m
    if model is None:
        log.error("No model found with ID ''.".format(id))
        return model
    else:
        return model


def get_model_by_split_type_dd(models: list, dropdown: widgets.Widget) -> Model:
    model = None
    for m in models:
        if m.split_type_dd is dropdown:
            model = m
    if model is None:
        log.error("No model found with ID ''.".format(id))
        return model
    else:
        return model

def get_child_value_by_description(gridbox: widgets.GridBox, description: str, number: int):
    child = _get_child_by_description(gridbox, description)[number]
    if child is None:
        log.error("No element with description {} found!".format(description))
        return

    if isinstance(child, widgets.SelectMultiple):
        child_value = list(child.value)
    elif isinstance(child, widgets.Select):
        child_value = str(child.value)
    else:
        log.error("Type {} is not yet supported. Please extend this function in order to support it."
                  .format(type(child)))
        return

    if not child_value:
        log.warning("No {} were selected. Please select at least one type and try again!".format(description))
        return

    return child_value


def _get_child_by_description(gridbox: widgets.GridBox, description: str) -> list:
    selects = []
    for child in gridbox.children:
        if child.description == description:
            selects.append(child)

    return selects
