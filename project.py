import sys
import csv
import time
import warnings
import numpy as np
import pandas as pd
import statistics as st
import statsmodels.api as sm
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

class Labels:
    @property
    def continuous(self):
        print("Construction model for continuous variable...\n"); time.sleep(2)
    @property
    def categorical(self):
        print("Construction model for categorical variable...\n"); time.sleep(2)
    @property
    def binary(self):
        print("Construction model for binary variable...\n"); time.sleep(2)
    @property
    def cleaning(self):
        print("Data cleaning...\n",); time.sleep(2)
    

def main():
    if len(sys.argv) > 3: sys.exit("Please only give the filenames!")
    data = sys.argv[1]
    try: becslo = sys.argv[2]
    except IndexError: becslo = 0
    data, becslo = data_reader(data, becslo)

    details(data)
    warnings.filterwarnings("ignore")

    print("Statistics / Prediction")
    
    while True:
        bemenet = input("Method: ").strip().lower()
        if bemenet in ["statistics", "stat", "prediction", "pred"]: break
        else: print("Invalid method!")
    print("")
    if bemenet in ["statistics","stat"]:
        while True:
            try:
                v = input("Column for statistics: ")
                column = data[v]
                break
            except KeyError: print("The columnname doesn't exists in your dataframe!")
        print(descriptive_stat(data, column, v))
    elif bemenet in ["prediction","pred"]: 
        while True:
            try:
                var_y = input("Variable to estimate: ").strip()
                y = data[var_y]
                X = data.drop(var_y, axis = 1)
                break
            except KeyError: print("Non existed variable! Please give one from your table!")
        print(prediction(data, becslo, y, X, var_y))
    else: print("Invalid method!", end = "\n\n")


def data_reader(train, test):
    try:
        if train.endswith(".csv"): data = pd.read_csv(train)
        elif train.endswith(".xlsx"): data = pd.read_excel(train)
        else: sys.exit("Invalid file format")
    except FileNotFoundError: sys.exit("No such file in your directory!\nCheck your command-line arguments.")
    if not isinstance(test, int):
        try:
            if test.endswith(".csv"): becslo = pd.read_csv(test)
            elif test.endswith(".xlsx"): becslo = pd.read_excel(test)
            else: sys.exit("Invalid file format")
        except FileNotFoundError: sys.exit("No such file in your directory!\nCheck your command-line arguments.")
    else: becslo = test
    return data, becslo


def cleaning(tanito, X_test = 0, y_variable = 0, make_dummies = False, y_type = "continuous", y_Var = 0, X_Var = 0):
    Labels().cleaning
    
    tanito.dropna(inplace = True)
    tanito = tanito.reset_index(drop=True)
    if make_dummies:
        y_Var = tanito[y_variable].to_frame()
        dummy_columns = [name for name, type in tanito.dtypes.items() if type == "object" and name != y_variable]
        if len(dummy_columns) != 0: 
            encoder = OneHotEncoder(handle_unknown="ignore", drop = "first", sparse_output=False, dtype= int)
            encoder.fit(tanito[dummy_columns])
            tanito_dummy = encoder.transform(tanito[dummy_columns])
            tanito = tanito.drop(columns = dummy_columns)
            tanito = pd.concat([tanito, pd.DataFrame(tanito_dummy, columns = encoder.get_feature_names_out(dummy_columns), index=tanito.index)], axis = 1)
        for i in tanito.columns:
            if tanito[i].dtypes == "bool":
                tanito[i] = tanito[i].astype("int")
        if y_type == "binary": 
            y_Var = pd.get_dummies(y_Var[y_variable], drop_first = True).iloc[:, 0]
            y_Var.name = f"{y_variable}_{y_Var.name}"
            X_Var = tanito.drop(y_variable, axis = 1)
        elif y_type == "categorical":
            global labelencoder
            labelencoder = LabelEncoder()
            y_Var = pd.Series(labelencoder.fit_transform(y_Var[y_variable]), name = y_variable)
            drop_vars = [var for var in tanito.columns.tolist() if var.startswith(y_variable)]
            X_Var = tanito.drop(drop_vars, axis = 1)
        elif y_type == "continuous": 
            y_Var = tanito[y_variable]
            X_Var = tanito.drop(y_variable, axis = 1)
         
    if not isinstance(X_test, int):
        X_test = X_test.dropna()
        X_test = X_test.reset_index(drop=True)
        X_test_dummycols = [name for name, type in X_test.dtypes.items() if type == "object"]
        if len(X_test_dummycols) != 0:
            X_test_dummy = encoder.transform(X_test[X_test_dummycols])
            X_test = X_test.drop(columns = X_test_dummycols)
            X_test = pd.concat([X_test, pd.DataFrame(X_test_dummy, columns = encoder.get_feature_names_out(X_test_dummycols), index=X_test.index)], axis = 1)
        for i in X_test.columns:
            if X_test[i].dtypes == "bool":
                X_test[i] = X_test[i].astype("int")
    return tanito, X_Var, y_Var, X_test


def details(data):
    while True:
        question1 = input("Would you like to get details about your dataset?(yes/no) ")
        if question1 in ["yes", "no"]: break
    if question1 == "yes":
        reszletek = [("Rows", data.shape[0]),
                     ("Columns", data.shape[1])]
        print("---------------------------\n"\
            "  Details of your Dataset\n" \
            "---------------------------", end = "")
        print(tabulate(reszletek, headers = ["      ", "     "], tablefmt = "plain"), end = "\n \n")

        df_reszletek = [{"Variables": variable, "No. missing values": data[variable].isnull().sum(), "Data type": data[variable].dtype} for variable in data.columns]
        print(tabulate(df_reszletek, headers = "keys", tablefmt = "grid"), end = "\n\n")
    else: print("")


def descriptive_stat(data, column, v):
    data, _, __, ___ = cleaning(tanito = data)
    if len(set(column)) > 8:
        mutatok =[{"Indicator": "Mean", "Value": round(column.mean(),3)},
                {"Indicator": "Standard deviation", "Value": round(column.std(ddof = 0),3)},
                {"Indicator": "Median", "Value": round(column.median(),3)},
                {"Indicator": "Minimum", "Value": column.min()},
                {"Indicator": "Maximum", "Value": column.max()},
                {"Indicator": "Range", "Value": round((column.max()-column.min()),3)},
                {"Indicator": "Relative Standard Deviation", "Value": round((column.mean() / column.std(ddof = 0)),3)},
                {"Indicator": "Interquartile Range", "Value": round((column.quantile(0.75)-column.quantile(0.25)),3)},
                {"Indicator": "Skewness", "Value": round(column.skew(),3)},
                {"Indicator": "Kurtosis", "Value": round(column.kurt(),3)}]

        with open(f"stat_results_{v}", "w", newline = "") as file:
            writer = csv.DictWriter(file, ["Indicator", "Value"])
            writer.writeheader()
            for line in mutatok: writer.writerow({"Indicator": line["Indicator"], "Value": line["Value"]})
        print("Statistical analysis saved...")

        if 0 < column.skew() <= 0.7: ferdeseg = "slightly right-skewed"
        elif 0.7 < column.skew(): ferdeseg = "right-skewed"
        elif -0.7 <= column.skew() < 0: ferdeseg = "slightly left-skewed"
        elif column.skew() < -0.7: ferdeseg = "left-skewed"
        elif column.skew() == 0: ferdeseg = "normal"

        if 3 < column.kurt() <= 4: csucsossag = "slightly peaked than a normal"
        elif 4 < column.kurt(): csucsossag = "peaked than a normal"
        elif 2 <= column.kurt() < 3: csucsossag = "slightly flatter than a normal"
        elif column.kurt() < 2: csucsossag = "flatter than a normal"
        elif column.kurt() == 3: csucsossag = "normal"

        return f"\nDescriptive Statistical Analysis:\nThe distribution of {v} is {ferdeseg} and {csucsossag}. The highest value is {column.max():,.3f} and the lowest is {column.min():,.3f},\nin addition half of values is higher and half is lower than the {column.median():,.3f} value." \
        f"The average value is {column.mean():,.3f}, \nwhile the average difference from the mean value is {column.std(ddof = 0):,.3f}. The data middle 50% is between {column.quantile(0.25):,.3f} and {column.quantile(0.75):,.3f}."
    else:
        return f"\nStatistical Analysis:\n{column.mode().iloc[0].capitalize()} is the most common value in variable {v} with {(column == column.mode().iloc[0]).sum()} instances. Your vairable has {len(set(column))} categories altogether."


def prediction(data, becslo, y, X, var_y):
    if len(set(y)) > 8:
        _, X, y, becslo = cleaning(tanito = data, X_test = becslo, make_dummies = True, y_variable = var_y)
        Labels().continuous
        #basic model
        X = sm.add_constant(X)
        model1 = sm.OLS(y, X)
        model1 = model1.fit()
        model1_AIC = model1.aic
        model1_BIC = model1.bic
        model1_adj_R2 = model1.rsquared_adj

        #pca model
        pca_vars = pd.DataFrame() 
        other_vars = pd.DataFrame() 
        for i in X: 
            if len(set(X[i])) > 2: pca_vars[i] = X[i] 
            else: other_vars[i] = X[i]

        scaler = StandardScaler()
        pca = PCA(n_components = 0.98)
        pca_vars = scaler.fit_transform(pca_vars)
        pca_vars = pca.fit_transform(pca_vars) 
        pca_vars = pd.DataFrame(pca_vars)
        for i in other_vars: pca_vars[i] = other_vars[i] 
        pca_vars = sm.add_constant(pca_vars) 
        model2 = sm.OLS(y, pca_vars)
        model2 = model2.fit()
        model2_bic = model2.bic 
        model2_aic = model2.aic 
        model2_adj_R2 = model2.rsquared_adj

        model1_score = 0
        model2_score = 0
        if model1_AIC < model2_aic: model1_score += 1
        else: model2_score += 1
        if model1_BIC < model2_bic: model1_score += 1
        else: model2_score += 1
        if model1_adj_R2 > model2_adj_R2: model1_score += 1
        else: model2_score += 1

        if model1_score > model2_score: reg_output = model1
        else: reg_output = model2

        if not isinstance(becslo, int):
            if model1_score < model2_score:
                pca_becslo = pd.DataFrame()
                other_becslo = pd.DataFrame()
                for i in becslo:
                    if len(set(becslo[i])) > 2: pca_becslo[i] = becslo[i] 
                    else: other_becslo[i] = becslo[i]
                pca_becslo = scaler.transform(pca_becslo)
                pca_becslo = pca.transform(pca_becslo)
                pca_becslo = pd.DataFrame(pca_becslo)
                becslo = pd.concat([pca_becslo, other_becslo], axis = 1)
            becslo = sm.add_constant(becslo)
            becslo[f"predicted_{var_y}"] = reg_output.predict(becslo)
            becslo.to_excel(f"{var_y}_predictions.xlsx", index = False)
            return f"Linear Regression Results:\nThe model can predict {round(reg_output.rsquared, 5)*100}% of the data distribution on train dataset!\n\nPredictions were saved..."
        return f"Linear Regression Results:\nThe model can predict {round(reg_output.rsquared, 5)*100}% of the data distribution on train dataset!"

    elif len(set(y)) == 2:
        _, X, y, becslo = cleaning(tanito = data, X_test = becslo, make_dummies = True, y_variable = var_y, y_type = "binary")
        Labels().binary
        #basic
        X = sm.add_constant(X)
        model1 = sm.Logit(y, X) 
        model1 = model1.fit(maxiter = 80, disp = False) 
        model1_BIC = model1.bic 
        model1_AIC = model1.aic 
        y_probability = model1.predict(X) 
        y_pred = []
        for i in y_probability: 
            if i > 0.5: y_pred.append(1) 
            else: y_pred.append(0) 
        model1_accuracy = accuracy_score(y, y_pred)

        #PCA model
        pca_vars = pd.DataFrame() 
        other_vars = pd.DataFrame() 
        for i in X: 
            if len(set(X[i])) > 2: pca_vars[i] = X[i] 
            else: other_vars[i] = X[i]

        scaler = StandardScaler()
        pca_vars = scaler.fit_transform(pca_vars) 
        pca = PCA(n_components = 0.98) 
        pca_vars = pca.fit_transform(pca_vars) 
        pca_vars = pd.DataFrame(pca_vars)
        for i in other_vars:
            pca_vars[i] = other_vars[i]
        pca_vars = sm.add_constant(pca_vars)
        model2 = sm.Logit(y, pca_vars)
        model2 = model2.fit(maxiter = 80, disp = False)
        model2_bic = model2.bic
        model2_aic = model2.aic
        y_probability = model2.predict(pca_vars) 
        y_pred = []
        for i in y_probability: 
            if i > 0.5: y_pred.append(1) 
            else: y_pred.append(0) 
        model2_accuracy = accuracy_score(y, y_pred)

        model1_score = 0; model2_score = 0
        if model1_AIC < model2_aic: model1_score += 1
        else: model2_score += 1
        if model1_BIC < model2_bic: model1_score += 1
        else: model2_score += 1
        if model1_accuracy > model2_accuracy: model1_score += 1
        else: model2_score += 1
        if model1_score > model2_score: reg_output = model1; reg_data = X
        else: reg_output = model2; reg_data = pca_vars
        y_pred = [1 if i > 0.5 else 0 for i in reg_output.predict(reg_data)]
        if not isinstance(becslo, int):
            if model1_score < model2_score:
                pca_becslo = pd.DataFrame()
                other_becslo = pd.DataFrame()
                for i in becslo:
                    if len(set(becslo[i])) > 2: pca_becslo[i] = becslo[i] 
                    else: other_becslo[i] = becslo[i]
                pca_becslo = scaler.transform(pca_becslo)
                pca_becslo = pca.transform(pca_becslo)
                pca_becslo = pd.DataFrame(pca_becslo)
                becslo = pd.concat([pca_becslo, other_becslo], axis = 1)
            becslo = sm.add_constant(becslo)
            becslo[f"predicted_{var_y}_probability"] = reg_output.predict(becslo)
            becslo[f"predicted_{var_y}"] = (becslo[f"predicted_{var_y}_probability"] > 0.5).astype(int)
            becslo.to_excel(f"{var_y}_predictions.xlsx", index = False)
            return f"Logistic Regression Results:\nThe model can categorize correctly {round(accuracy_score(y, y_pred), 3)*100}% of the data distribution on train dataset!\n\nPredictions were saved..."
        return f"Logistic Regression Results:\nThe model can categorize correctly {round(accuracy_score(y, y_pred), 3)*100}% of the data distribution on train dataset!"

    elif 9 > len(set(y)) > 2:
        _, X, y, becslo = cleaning(tanito = data, X_test = becslo, make_dummies = True, y_variable = var_y, y_type = "categorical")
        Labels().categorical
        knn_pont = []
        for neighbors in range(1, 21):
            model_knn = KNeighborsClassifier(n_neighbors = neighbors)
            pont = cross_val_score(model_knn, X, y, cv = 5, scoring = "accuracy")
            knn_pont.append(pont.mean())
        if not isinstance(becslo, int):
            model_knnbecsles = KNeighborsClassifier(n_neighbors = (np.argmax(knn_pont) + 1))
            model_knnbecsles.fit(X, y)
            becslo[f"predicted_{var_y}"] = labelencoder.inverse_transform(model_knnbecsles.predict(becslo))
            becslo.to_excel(f"{var_y}_predictions.xlsx", index = False)
            return f"KNN model results:\nThe model could categorize correctly the {round(max(knn_pont)*100,3)}% of {var_y} value on train dataset by using the {np.argmax(knn_pont) + 1} nearest neighbors.\n\nPredictions were saved..."
        return f"KNN model results:\nThe model could categorize correctly the {round(max(knn_pont)*100,3)}% of {var_y} value on train dataset by using the {np.argmax(knn_pont) + 1} nearest neighbors."


if __name__ == "__main__":
    main()
