import enum
import pandas as pd
import logging as log

from sklearn.pipeline import Pipeline


class ProblemType(enum.Enum):
    CLASSIFICATION = 1
    REGRESSION = 10


class Algorithm(enum.Enum):
    # classification
    LOGISTIC_REGRESSION = 1
    DECISION_TREE = 2
    RANDOM_FOREST = 3
    XGB = 4
    # regression
    LINEAR_REGRESSION = 10
    SVM = 11


class ModelType:
    def __init__(self, problem_type):
        self._problem_type = None
        self._algorithm = None
        self._algorithm_options = []
        self.problem_type = problem_type

    @property
    def problem_type(self):
        return self._problem_type

    @problem_type.setter
    def problem_type(self, new_value):
        self._algorithm = None
        if new_value == ProblemType.CLASSIFICATION:
            self._algorithm_options = [Algorithm.LOGISTIC_REGRESSION.name, Algorithm.DECISION_TREE.name,
                                       Algorithm.RANDOM_FOREST.name, Algorithm.XGB.name]
        elif new_value == ProblemType.REGRESSION:
            self._algorithm_options = [Algorithm.LINEAR_REGRESSION.name, Algorithm.SVM.name]
        else:
            raise NotImplementedError("Other problem types than classification and regression are not supported.")

        self._problem_type = new_value

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, new_value: Algorithm):
        if new_value.name in self._algorithm_options:
            self._algorithm = new_value
        else:
            raise ValueError("Invalid algorithm value {}. The algorithm value must be in {}"
                             .format(new_value, self._algorithm_options))

    @property
    def algorithm_options(self):
        return self._algorithm_options


class Model:
    def __init__(self, id: int, name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series, model_type: ModelType):
        self._id = id
        self._name = name
        self._model = model
        self._model_type = model_type
        self._split = None
        self._X = X
        self._y = y
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._predictions = None
        self.shap_kernel_explainer = None
        self._shap_values = None
        self._skater_interpreter = None
        self._skater_model = None
        self._lime_explainer = None
        self._X_train_ohe = None
        self._X_test_ohe = None
        self._numerical_features = None
        self._categorical_features = None
        self._categorical_ohe_features = None
        self._idx2ohe = None
        self._features_ohe = None
        # frontend Widgets associated with this model.
        # sm -> Select Multiple, dd -> Drop Down, ...
        self._remove_features_sm = None
        self._remove_features_button = None
        self._train_model_button = None
        self._model_type_dd = None
        self._split_type_dd = None
        self._cross_columns_sm = None

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_value):
        self._name = new_value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_value):
        self._model = new_value

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, new_value):
        self._model_type = new_value

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, new_value):
        self._split = new_value

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, new_value):
        self._X = new_value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, new_value):
        self._y = new_value

    @property
    def X_test(self):
        return self._X_test

    @X_test.setter
    def X_test(self, new_value):
        self._X_test = new_value

    @property
    def y_test(self):
        return self._y_test

    @y_test.setter
    def y_test(self, new_value):
        self._y_test = new_value

    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, new_value):
        self._X_train = new_value

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, new_value):
        self._y_train = new_value


    @property
    def predictions(self):
        return self._predictions

    @predictions.setter
    def predictions(self, new_value):
        self._predictions = new_value


    @property
    def shap_values(self):
        return self._shap_values

    @shap_values.setter
    def shap_values(self, new_value):
        self._shap_values = new_value

    @property
    def shap_kernel_explainer(self):
        return self._shap_kernel_explainer

    @shap_kernel_explainer.setter
    def shap_kernel_explainer(self, new_value):
        self._shap_kernel_explainer = new_value

    def init_shap(self):
        """
        Initialize shap. Calculate shap values. This operation is time consuming.
        :return: void (Sets the value of the shap_values variable)
        """
        import shap

        if not self.shap_values:

            log.info("Initializing Shap - calculating shap values."
                     " This operation is time-consuming so please be patient.")

            logger = log.getLogger('shap')
            logger.setLevel(log.WARN)

            shap_kernel_explainer = shap.KernelExplainer(self.model[1].predict_proba,
                                                         shap.kmeans(self.X_test_ohe, 1))
            shap_values = shap_kernel_explainer.shap_values(self.X_test_ohe)
            # from util.commons import RANDOM_NUMBER
            # shap_values = shap_kernel_explainer.shap_values(self.X_test_ohe.sample(66, random_state=RANDOM_NUMBER))

            self.shap_kernel_explainer = shap_kernel_explainer
            self.shap_values = shap_values
        else:
            log.info("Shap is already initialized.")

    @property
    def skater_interpreter(self):
        return self._skater_interpreter

    @skater_interpreter.setter
    def skater_interpreter(self, new_value):
        self._skater_interpreter = new_value

    @property
    def skater_model(self):
        return self._skater_model

    @skater_model.setter
    def skater_model(self, new_value):
        self._skater_model = new_value

    def init_skater(self, target_names=None):
        """
        Initialize skater. Set ups skater interpreter and in-memory model.
        :return: void (Sets the values of the skater_interpreter and skater_model variables)
        """
        from skater.core.explanations import Interpretation
        from skater.model import InMemoryModel

        if not self.skater_interpreter or not self.skater_model:

            log.info("Initializing Skater - generating new in-memory model."
                     " This operation may be time-consuming so please be patient.")

            self.skater_interpreter = Interpretation(
                training_data=self.X_train_ohe,
                training_labels=self.y_train,
                feature_names=self.features_ohe)

            self.skater_model = InMemoryModel(
                self.model[1].predict_proba,
                examples=self.X_test_ohe,
                target_names=target_names)
        else:
            log.info("Skater is already initialized.")

    def init_lime(self):
        """
        Initializes a LIME explainer that can later be used for local interpretations
        for this model.
        :return: void (Sets the value for lime_explainer)
        """
        from lime.lime_tabular import LimeTabularExplainer
        from util.commons import RANDOM_NUMBER, convert_to_lime_format

        if not self.lime_explainer:

            log.info("Initializing LIME - generating new explainer."
                     " This operation may be time-consuming so please be patient.")

            # Transform the categorical feature's labels to a lime-readable format.
            categorical_names = self.idx2ohe
            log.debug("Categorical names for lime: {}".format(categorical_names))

            explainer = LimeTabularExplainer(
                convert_to_lime_format(self.X_test, categorical_names).values,
                mode="classification",
                feature_names=self.X_test.columns.tolist(),
                categorical_names=categorical_names,
                categorical_features=categorical_names.keys(),
                discretize_continuous=True,
                random_state=RANDOM_NUMBER)

            self.lime_explainer = explainer
        else:
            log.info("LIME is already initialized.")

    @property
    def lime_explainer(self):
        return self._lime_explainer

    @lime_explainer.setter
    def lime_explainer(self, new_value):
        self._lime_explainer = new_value

    @property
    def X_train_ohe(self):
        return self._X_train_ohe

    @X_train_ohe.setter
    def X_train_ohe(self, new_value):
        self._X_train_ohe = new_value

    @property
    def X_test_ohe(self):
        return self._X_test_ohe

    @X_test_ohe.setter
    def X_test_ohe(self, new_value):
        self._X_test_ohe = new_value

    @property
    def features_ohe(self):
        return self._features_ohe

    @features_ohe.setter
    def features_ohe(self, new_value):
        self._features_ohe = new_value

    @property
    def numerical_features(self):
        return self._numerical_features

    @numerical_features.setter
    def numerical_features(self, new_value):
        self._numerical_features = new_value

    @property
    def categorical_features(self):
        return self._categorical_features

    @categorical_features.setter
    def categorical_features(self, new_value):
        self._categorical_features = new_value

    @property
    def categorical_ohe_features(self):
        return self._categorical_ohe_features

    @categorical_ohe_features.setter
    def categorical_ohe_features(self, new_value):
        self._categorical_ohe_features = new_value

    @property
    def idx2ohe(self):
        return self._idx2ohe

    @idx2ohe.setter
    def idx2ohe(self, new_value):
        self._idx2ohe = new_value

    @property
    def remove_features_sm(self):
        return self._remove_features_sm

    @remove_features_sm.setter
    def remove_features_sm(self, new_value):
        self._remove_features_sm = new_value

    @property
    def remove_features_button(self):
        return self._remove_features_button

    @remove_features_button.setter
    def remove_features_button(self, new_value):
        self._remove_features_button = new_value

    @property
    def model_type_dd(self):
        return self._model_type_dd

    @model_type_dd.setter
    def model_type_dd(self, new_value):
        self._model_type_dd = new_value

    @property
    def split_type_dd(self):
        return self._split_type_dd

    @split_type_dd.setter
    def split_type_dd(self, new_value):
        self._split_type_dd = new_value

    @property
    def cross_columns_sm(self):
        return self._cross_columns_sm

    @cross_columns_sm.setter
    def cross_columns_sm(self, new_value):
        self._cross_columns_sm = new_value

    @property
    def train_model_button(self):
        return self._train_model_button

    @train_model_button.setter
    def train_model_button(self, new_value):
        self._train_model_button = new_value
