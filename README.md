Shapash : Machine Learning Interpretable & Understandable

Shapash is a Python library which aims to make machine learning interpretable and understandable by everyone. It provides several types of visualization that display explicit labels that everyone can understand.

Data Scientists can understand their models easily and share their results. End users can understand the decision proposed by a model using a summary of the most influential criteria.

Shapash also contributes to data science auditing by displaying useful information about any model and data in a unique report.

Shapash works for Regression, Binary Classification or Multiclass problems.

It is compatible with many models: Catboost, Xgboost, LightGBM, Sklearn Ensemble, Linear models, SVM.

Features:

•	Display clear and understandable results: plots and outputs use explicit labels for each feature and its values

•	Allow Data Scientists to quickly understand their models by using a webapp to easily navigate between global and local explainability, and understand how the different features contribute

•	Summarize and export the local explanation: Shapash proposes a short and clear local explanation. It allows each user, whatever their Data background, to understand a local prediction of a supervised model thanks to a summarized and explicit explanation

•	Evaluate the quality of your explainability using different metrics

•	Easily share and discuss results with non-Data users

•	Deploy interpretability part of your project: From model training to deployment (API or Batch Mode)

•	Contribute to the auditability of your model by generating a standalone HTML report of your projects.

How Shapash works:

Shapash is an overlay package for libraries dedicated to the interpretability of models. It uses Shap or Lime backend to compute contributions. Shapash builds on the different steps necessary to build a machine learning model to make the results understandable.

How to use:

Step 1: Declare SmartExplainer Object

You can declare features dict here to specify the labels to display

from shapash.explainer.smart_explainer import SmartExplainer
SE = SmartExplainer() 

Step 2: Compile Model, Dataset, Encoders

There are 2 mandatory parameters in compile method: Model and Dataset
SE.compile(
    x=Xtest,
    model=regressor,
    )

Step 3: Display output

There are several outputs and plots available. for example, you can launch the web app.

app = SE.run_app()

The web app link appears in Jupyter output.

There are four parts in this Web App:

Each one interacts to help to explore the model easily.

1.	Features Importance: you can click on each feature to update the contribution plot below.

2.	Contribution plot: How does a feature influence the prediction? Display violin or scatter plot of each local contribution of the feature.

3.	 Local Plot:  

•	Local explanation: which features contribute the most to the predicted value.
•	You can use several buttons/sliders/lists to configure the summary of this local explainability. We will describe below with the filter method the different parameters you can work your summary with.
•	This web app is a useful tool to discuss with business analysts the best way to summarize the explainability to meet operational needs.

4.	Selection Table: It allows the Web App user to select:

•	A subset to focus the exploration on this subset
•	A single row to display the associated local explanation

How to use the Data table to select a subset?

At the top of the table, just below the name of the column that you want to use to filter, specify:
=Value, >Value, <Value

If you want to select every row containing a specific word, just type that word without “=”

There are a few options available on this web app (top right button). The most important one is probably the size of the sample (default: 1000). To avoid latency, the web app relies on a sample to display the results. 
Use this option to modify this sample size.

To kill the app:

app.kill()

Step 4: Generate the Shapash Report

This step allows to generate a standalone html report of your project using the different splits of your dataset and the metrics you used.

SE.generate_report(
    output_file='C://Blog Python Libraries//doumentation//shapash//report//report.html',
    project_info_file='C://Blog Python Libraries//doumentation//shapash//report//project_info.yml',
    x_train=xtrain,
    y_train=ytrain,
    y_test=ytest,
    title_story="Concrete_Data report",
    title_description="""This document is a data science report of the Concrete_Data project.
        It was generated using the Shapash library.""",
    metrics=[{'name': 'MSE', 'path': 'sklearn.metrics.mean_squared_error'}]
)

Step 5: From training to deployment : SmartPredictor Object

Shapash provides a SmartPredictor object to deploy the summary of local explanation for the operational needs. It is an object dedicated to deployment, lighter than SmartExplainer with additional consistency checks. SmartPredictor can be used with an API or in batch mode. It provides predictions, detailed, or summarized local explainability using appropriate wording.

predictor = SE.to_smartpredictor()

## convert prediction into a pickle file
predictor.save('./predictor.pkl')

## load predictor file
predictor_load = load_smartpredictor('./predictor.pkl')

predictor_load.add_input(x=x, ypred=y)
detailed_contributions = predictor_load.detail_contributions()
detailed_contributions.head()
