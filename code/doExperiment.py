"""
SUPPLEMENTARY CODE FOR
Title: Predicting Financial Crises - An Evaluation of Machine Learning Algorithms and Model Explainability for Early Warning Systems
Author: Chris Reimann.

This file implements the experimental design and calculates the performance metrics of the specified ML models. 
"""

import prepareData as pds
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm as sk_svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from utils import CustomGroupKFold, CustomTimeSeriesSplit, InSampleSplit, forecastExp
from tqdm.notebook import tqdm
import statsmodels.api as sm
from alibi.explainers import ALE
from alibi.explainers import plot_ale

class Experiment:
    
    def __init__(self, data, models, expType, rs = 1):
        # Initialize Parameters:
        self.data = data
        self.models = ["Random Assignment"] + models
        self.expType = expType
        
        # Check if Parameters valid:
        validModels = ["Random Assignment", "Logit", "RandomForest", "ExtraTrees", "SVM", "NeuralNet", "KNeighbors"]
        validExpTypes = ["InSample","CrossVal", "Forecast", 0, 1, 2]
        unvalidModels = [model for model in self.models if model not in validModels]
        
        if len(unvalidModels) != 0:
            raise Exception(f"Invalid Model Type specified: {unvalidModels}")
        
        if self.expType not in validExpTypes:
            raise Exception(f"Invalid Experiment Type specified: {expType}")
        
        # Model Classes:
        self.modelClasses = {
            "Random Assignment": DummyClassifier(strategy = "constant", constant = 0),
            "Logit": LogisticRegression(random_state = rs, max_iter = 1000),
            "RandomForest": RandomForestClassifier(random_state = rs, n_estimators = 1000),
            "ExtraTrees": ExtraTreesClassifier(random_state = rs, n_estimators = 1000),
            "SVM": sk_svm.SVC(random_state = rs, probability = True),
            "NeuralNet": MLPClassifier(random_state = rs, max_iter = 10000, alpha = 2),
            "KNeighbors": KNeighborsClassifier(n_neighbors = 100)
        }    
        
        # Set Experiment Iterator
        iteratorTypes = {
            "InSample": InSampleSplit(),
            "CrossVal": CustomGroupKFold(),
            "Forecast": forecastExp(year = 1980)
        }
        self.iterator = iteratorTypes[self.expType]

    
    def run(self, n = 1, disableTqdm = False):
        self.roc = pd.DataFrame()
        self.auc = pd.DataFrame()
        self.searchRes = []
        if self.expType == "InSample" or self.expType == "Forecast": n = 1
        
        # Run Experiment for all specified Models
        for model in self.models:
            pipe = self._buildPipeline(model)
            yTrue = []
            predictions = []
            
            # Repeat Experiment n times and compute Average
            for i in tqdm(range(1, n+1), desc = self.data.name + ": " + model, disable = disableTqdm):
                
                # Calculate Predictions for every Train/Test Fold 
                for ixTrain, ixTest in self.iterator.split(self.data.df):
                    xTrain = self.data.df.loc[ixTrain]
                    yTrain = self.data.df[self.data.depVar].loc[ixTrain]
                    xTest = self.data.df.loc[ixTest]
                    yTest = self.data.df[self.data.depVar].loc[ixTest]
                    
                    pipe.fit(xTrain, yTrain)
                    yTrue = np.append(yTrue, yTest)
                    predictions = np.append(predictions, pipe.predict_proba(xTest)[:,1])
                    self.searchRes.append((str(model),pipe.named_steps["BestEstimator"].best_params_))
                    
            # Calculate TPR, FPR
            fpr, tpr, threshold = metrics.roc_curve(yTrue, predictions)
            self.roc = pd.concat((
                self.roc,
                pd.DataFrame({"Model": model, "FPR": fpr, "TPR": tpr, "Threshold": threshold})
            ))

            # Calculate AUC
            foldAUC = metrics.roc_auc_score(yTrue, predictions)
            self.auc = pd.concat((
                self.auc,
                pd.DataFrame({"Set": self.data.name,"Model": [model], "AUC": foldAUC})
            )) 
        self.auc = self.auc.sort_values("AUC", ascending = False).reset_index(drop=True)
        self.roc = self.roc.reset_index(drop = True)
    
    
    def _buildPipeline(self, model):
        preprocess = ColumnTransformer(
            [("Standardize", StandardScaler(), self.data.indicators)],
            verbose_feature_names_out = False,
        )
        
        paras = {
            "Random Assignment": {},
            "Logit": {"penalty": ["l2", "none"]},
            "RandomForest": {"max_depth": [2,4,6]},
            "ExtraTrees": {"max_depth": [2,4,6]},
            "SVM": {"gamma": ["scale", "auto"], "kernel": ["rbf", "linear", "poly", "sigmoid"]},
            "NeuralNet": {"hidden_layer_sizes": [(8,8,8), (20,)], "activation": ["tanh", "relu"]},
            "KNeighbors": {"weights": ["uniform", "distance"], "n_neighbors": [50,75,100]}
        }
        
        paraSearch = GridSearchCV(
            estimator = self.modelClasses[model],
            param_grid = paras[model],
            scoring = "roc_auc",
            cv = InSampleSplit(),
            n_jobs = -1,
        )
        
        steps = [
            ("Preprocess", preprocess),
            ("BestEstimator", paraSearch),
        ]
        
        return Pipeline(steps)
    
    
    def rocGraph(self, save = False):
        sns.set_theme(style="whitegrid", palette="tab10", rc={'savefig.dpi':300})
        plt.figure(figsize=(6,6))
        
        plot = sns.lineplot(
            data = self.roc,
            x = "FPR", y = "TPR",
            hue = "Model",
            estimator = None,
            n_boot = 0,
        )
        
        plot.lines[0].set_linestyle("--")
        plot.set(xlabel = "False Positive Rate (FPR)", ylabel = "True Positive Rate Rate (TPR)")
        labels = [f"{mo}: (AUC = {round(self.auc[self.auc.Model == mo].AUC.iloc[0], 3)})" for mo in self.models]
        plt.legend(labels = labels, fontsize = "small")
        if save: plt.savefig("roc.png")
        
        
    def logitCoef(self):
        df_std = self.data.standardize()
        exogWConst = sm.add_constant(df_std[self.data.indicators])
        mod = sm.Logit(df_std[self.data.depVar], exogWConst)
        fii = mod.fit()
        coef = fii.summary2().tables[1]
        
        for var in coef.index:
            if coef[coef.index == var]["P>|z|"].values[0] < 0.01:
                print(f"{var} is significant at 1%")
            elif coef[coef.index == var]["P>|z|"].values[0] < 0.05:
                print(f"{var} is significant at 5%")
            elif coef[coef.index == var]["P>|z|"].values[0] < 0.1:
                print(f"{var} is significant at 10%")
        return coef
        
    def ALE(self, modelTypes, var):
        df_std = self.data.standardize()
        x = df_std[self.data.indicators].to_numpy()
        y = df_std[self.data.depVar].to_numpy()

        exps = {}
        for modelType in modelTypes:
            # Load Parameters from Hyperparameter Search
            try:
                for paras in self.searchRes:
                    if paras[0] == modelType:
                        model = self.modelClasses[modelType].set_params(**paras[1])
                        break
                print(model)
                model.fit(x,y)
            except AttributeError:
                raise ValueError("No Hyperparameter Search has been conducted yet. Please use .run() first.")

            # Print AUROC
            pred = model.predict_proba(x)[:,1]
            fpr, tpr, threshold = metrics.roc_curve(y, pred)
            print(f"AUC: {metrics.auc(fpr, tpr)}")

            # Get Prediction Function for ALE
            predict_crisis = lambda x: model.predict_proba(x)[:,1]
            ale = ALE(predict_crisis, feature_names = list(map(self.data.varNames.get, self.data.indicators)))
            exps[modelType] = ale.explain(x, features = var)

        sns.set_theme(style="whitegrid", palette="tab10")    
        fig, ax = plt.subplots(nrows = 5, ncols = 3, figsize = (12,16))
        plots = {}
        for model in exps.keys():
            plots[model] = plot_ale(exps[model], n_cols = 4, ax=ax, line_kw={"label": model})
        fig.delaxes(ax[4,2])
        ax[0,0].get_legend().remove()
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc = "lower right", bbox_to_anchor=(0.83,0.145))
        
