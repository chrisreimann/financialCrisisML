"""
SUPPLEMENTARY CODE FOR
Title: Predicting Financial Crises - An Evaluation of Machine Learning Algorithms and Model Explainability for Early Warning Systems.
Author: Chris Reimann.

This file loads alternative crisis data for the robustness checks (LaevenValencia, ESRB). 
"""

import pandas as pd
import numpy as np
import country_converter as coco
 
def getLaevenValencia(folder = None, vers = 2020):
    
    if folder == None:
        link = "https://static-content.springer.com/esm/art%3A10.1057%2Fs41308-020-00107-3/MediaObjects/41308_2020_107_MOESM1_ESM.xlsx"
        df_alt = pd.read_excel(link, sheet_name = "Crisis Years").drop(0)
    else:
        df_alt = pd.read_excel(folder + r"\LaevenValencia" + str(vers) + "_crises.xlsx", sheet_name = "Crisis Years").drop(0)
    
    
    df_alt = df_alt.set_axis(["Country", "crisis_banking", "crisis_currency", "crisis_debt", "restructuring"], axis="columns")
    df_alt["iso3"] = coco.convert(names = df_alt.Country, to = "ISO3")
    country_list = df_alt.iso3.unique()
    
    # create new DataSet
    new = []
    for iso in country_list:
        for year in range(1970, 2018):
            new.append((iso, year, 0))
    new = pd.DataFrame(new, columns = ["iso", "year", "crisis_banking"])
    
    # import crisis events
    for index, row in df_alt.iterrows():
        if row["crisis_banking"] is not np.NaN:
            new.loc[(new.iso == row["iso3"]) & (new.year.astype(str).isin(str(row["crisis_banking"]).split(sep=", "))), "crisis_banking"] = 1
    
    return new

def getESRB(folder = None, vers = 2021):
    
    if folder == None:
        link = "https://www.esrb.europa.eu/pub/fcdb/esrb.fcdb20220120.en.xlsx"
        df_alt = pd.read_excel(link, sheet_name = "Systemic crises").drop([0, 51, 52, 53, 54])
    else:
        df_alt = pd.read_excel(folder + r"\esrb" + str(vers) + ".xlsx", sheet_name = "Systemic crises").drop([0, 51, 52, 53, 54])

    df_alt = df_alt[["Country", "Start date"]]
    df_alt["year"] = df_alt["Start date"].str.split(pat="-").str[0].astype(int)
    df_alt["crisis_esrb"] = 1
    df_alt["iso"] = coco.convert(names = df_alt["Country"], to = "ISO3")
    df_alt = df_alt[["year", "iso", "crisis_esrb"]]
    country_list = df_alt.iso.unique()
    
    new = []
    for iso in country_list:
        for year in range(1970, 2021):
            if len(df_alt[(df_alt.year == year) & (df_alt.iso == iso)]) == 0:
                new.append([year, iso, 0])
    new = pd.DataFrame(new, columns = ["year", "iso", "crisis_esrb"])
    new = pd.concat((df_alt, new)).sort_values(["year", "iso"]).reset_index(drop = True)
    
    return new
