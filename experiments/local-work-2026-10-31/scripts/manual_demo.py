import os
import xai
import logging as log
import warnings
import matplotlib.pyplot as plt
import sys, os
from util.commons import *
from util.ui import *
from util.model import *
from util.split import *
from util.dataset import *
if __name__ == '__main__':
    if 1:
        dataset, msg = get_dataset('census')

        print(dataset.df['workclass'].unique())
        print(dataset.df.loc[dataset.df['workclass'] == ' ?'])

        imbalanced_cols = ['gender', 'ethnicity']
        # xai.imbalance_plot(dataset.df, *imbalanced_cols)
        # _ = xai.correlations(dataset.df, include_categorical=True, plot_type="matrix", plt_kwargs={'figsize': (6, 6)})
        # _ = xai.correlations(dataset.df, include_categorical=True, plt_kwargs={'figsize': (8, 6)})

        df_X, df_y, msg = split_feature_target(dataset.df, "loan")
        print(df_y)

        # Create empty models
        initial_models, msg = fill_empty_models(df_X, df_y, 4)
        models = []

        model1 = initial_models[0]
        msg = fill_model(model1, Algorithm.LOGISTIC_REGRESSION, Split(SplitTypes.NORMAL))
        models.append(model1)
        model_1: Model = models[0]

        print(model_1.features_ohe)
        print(len(model_1.features_ohe))

        # model2 = initial_models[1]
        # msg = fill_model(model2, Algorithm.DECISION_TREE, Split(SplitTypes.NORMAL))
        # models.append(model2)
        # model_2 = models[1]
        #
        # model3 = initial_models[2]
        # msg = fill_model(model3, Algorithm.RANDOM_FOREST, Split(SplitTypes.NORMAL))
        # models.append(model3)
        # model_3 = models[2]
        #
        # model4 = initial_models[0]
        # msg = fill_model(model4, Algorithm.SVC, Split(SplitTypes.NORMAL))
        # models.append(model4)
        # model_4 = models[0]

        #plot1 = generate_feature_importance_plot(FeatureImportanceType.ELI5, model_1)
        #plot2 = generate_feature_importance_plot(FeatureImportanceType.ELI5, model_2)
        # plot3 = generate_feature_importance_plot(FeatureImportanceType.ELI5, model_3)
        # plot4 = generate_feature_importance_plot(FeatureImportanceType.ELI5, model_4)

        # print(calculate_feature_importance_rbo(FeatureImportanceType.ELI5, models))

        # _ = generate_feature_importance_plot(FeatureImportanceType.SKATER, model_1)
        # print(model_1.feature_weight_skater)
        # _ = generate_feature_importance_plot(FeatureImportanceType.SKATER, model_2)

        # generate_feature_importance_plot(FeatureImportanceType.SHAP, model_1)
        # generate_feature_importance_plot(FeatureImportanceType.SHAP, model_2)

        # plot1 = generate_pdp_plots(PDPType.PDPBox, model_1, "age", "None")
        # plot2 = generate_pdp_plots(PDPType.PDPBox, model_1, "age", "education-num")

        # generate_pdp_plots(PDPType.SKATER, model_1, "age", "None")
        # generate_pdp_plots(PDPType.SKATER, model_1, "age", "education-num")

        # fig, plt = generate_pdp_plots(PDPType.SHAP, model_1, "age", "education-num")
        # fig.show()

        examples = []
        example_types = [ExampleType.FALSELY_CLASSIFIED]
        for example_type in example_types:
            for model in models:
                example = get_test_examples(model, example_type, 1)[0]
                while example in examples:
                    example = get_test_examples(model, example_type, 1)[0]
                examples.append(example)

        example = examples[0]
        # example = 2564
        print(get_example_information(model_1, example))

        explanation_lime = explain_single_instance(LocalInterpreterType.OPTIMIZED_LIME, model_1, example)
        print(generate_single_instance_explanation(LocalInterpreterType.OPTIMIZED_LIME, model_1, example))
        # explanation.show_in_notebook(show_table=True, show_all=True)
        # explanation_shap = explain_single_instance(LocalInterpreterType.SHAP, model_1, example)
        # print(generate_single_instance_explanation(LocalInterpreterType.SHAP, model_1, example))
    else:
        print(calculate_baseline_rbo(5, 25, 1, 64, 500))
