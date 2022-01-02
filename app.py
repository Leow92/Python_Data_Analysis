from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask import Response
from datetime import datetime
import pandas as pd
import numpy as np
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import plotly
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from sklearn import datasets,tree,ensemble,discriminant_analysis, metrics
import seaborn as sns
from sklearn.model_selection import train_test_split

app = Flask(__name__)
df = pd.read_csv('C:/Users/lrbae/OneDrive/Bureau/S7/Python Data Analysis/diabetic_data.csv')


df.replace('No',0)
df.replace('Yes',1)

# mean of the age categorie
replaceDict = {'[0-10)' : 5,
'[10-20)' : 15,
'[20-30)' : 25, 
'[30-40)' : 35, 
'[40-50)' : 45, 
'[50-60)' : 55,
'[60-70)' : 65, 
'[70-80)' : 75,
'[80-90)' : 85,
'[90-100)' : 95}
df['age'] =df['age'].apply(lambda x : replaceDict[x])

plot_df=df
plot_df["readmitted"]=plot_df["readmitted"].replace("NO","not readmitted")
plot_df["readmitted"]=plot_df["readmitted"].replace(">30","readmitted")
plot_df["readmitted"]=plot_df["readmitted"].replace("<30","readmitted")

# modification to readmitted or not
df["readmitted"]=df["readmitted"].replace("NO",0)
df["readmitted"]=df["readmitted"].replace(">30",1)
df["readmitted"]=df["readmitted"].replace("<30",1)

# drop duplicates to avoid having the same patient affecting the data
df.drop_duplicates(['patient_nbr'], keep = 'first', inplace = True)

# fusion the data that are similar in the discharge_disposition_id, admission_type_id, admission_type_id
df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(lambda x : 1 if int(x) in [6, 8, 9, 13] 
                                                                           else ( 2 if int(x) in [3, 4, 5, 14, 22, 23, 24]
                                                                           else ( 10 if int(x) in [12, 15, 16, 17]
                                                                           else ( 11 if int(x) in [19, 20, 21]
                                                                           else ( 18 if int(x) in [25, 26] 
                                                                           else int(x) )))))

df = df[~df.discharge_disposition_id.isin([11,13,14,19,20,21])]

df['admission_type_id'] = df['admission_type_id'].apply(lambda x : 1 if int(x) in [2, 7]
                                                            else ( 5 if int(x) in [6, 8]
                                                            else int(x) ))

df['admission_source_id'] = df['admission_source_id'].apply(lambda x : 1 if int(x) in [2, 3]
                                                            else ( 4 if int(x) in [5, 6, 10, 22, 25]
                                                            else ( 9 if int(x) in [15, 17, 20, 21]
                                                            else ( 11 if int(x) in [13, 14]
                                                            else int(x) ))))

# transformation to get values instead of categorical type
for col in ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]:
    df[col] = df[col].apply(lambda x : 10 if x == 'Up' 
                                              else ( -10 if x == 'Down'                                                          
                                              else ( 0 if x == 'Steady'
                                              else  -20)))


df['change'] = df['change'].apply(lambda x : 1 if x == 'Ch'
                                                 else -1)


df['diabetesMed'] = df['diabetesMed'].apply(lambda x : -1 if x == 'No'
                                                else 1)


df['max_glu_serum'] = df['max_glu_serum'].apply(lambda x : 200 if x == '>200' 
                                                            else ( 300 if x == '>300'                                                          
                                                            else ( 100 if x == 'Norm'
                                                            else  0)))

df['A1Cresult'] = df['A1Cresult'].apply(lambda x : 7 if x == '>7' 
                                                         else (8 if  x == '>8'                                                        
                                                         else ( 5 if x == 'Norm'

                                                               else  0)))

df['gender'] = df['gender'].apply(lambda x : 1 if x == 'Male'
                                                 else -1)

# drop columns that have too much unknown values or are not related to the health of a patient
df.drop('weight', axis=1, inplace=True)
df.drop('encounter_id', axis=1, inplace=True)
df.drop('patient_nbr', axis=1, inplace=True)
df.drop('payer_code', axis=1, inplace=True)
df.drop('medical_specialty', axis=1, inplace=True)

# replace unknown data by most frequent value
df['race'].fillna("Caucasian",inplace=True)
df['diag_1'].fillna(414,inplace=True)
df['diag_2'].fillna(250,inplace=True)
df['diag_3'].fillna(250,inplace=True)

# temporaire à revoir
df.drop('diag_1', axis=1, inplace=True)
df.drop('diag_2', axis=1, inplace=True)
df.drop('diag_3', axis=1, inplace=True)

# encoder categorical data
df=pd.get_dummies(data=df,columns=["race"])



# drop of the insignificant data
df.drop('max_glu_serum', axis=1, inplace=True)
df.drop('nateglinide', axis=1, inplace=True)
df.drop('chlorpropamide', axis=1, inplace=True)
df.drop('glyburide-metformin', axis=1, inplace=True)
df.drop('discharge_disposition_id', axis=1, inplace=True)
df.drop('acetohexamide', axis=1, inplace=True)
df.drop('glipizide', axis=1, inplace=True)
df.drop('tolbutamide', axis=1, inplace=True)
df.drop('examide', axis=1, inplace=True)
df.drop('citoglipton', axis=1, inplace=True)
df.drop('race_Hispanic', axis=1, inplace=True)
df.drop('race_Other', axis=1, inplace=True)
df.drop('metformin-pioglitazone', axis=1, inplace=True)
df.drop('glimepiride-pioglitazone', axis=1, inplace=True)
df.drop('tolazamide', axis=1, inplace=True)
df.drop('troglitazone', axis=1, inplace=True)
df.drop('pioglitazone', axis=1, inplace=True)
df.drop('glyburide', axis=1, inplace=True)

# separation of the dataset in two new datasets
df_y=df["readmitted"]
df.drop('readmitted', axis=1, inplace=True)
df_x=df
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y,test_size=0.01)

# fitting of the data
scaler=StandardScaler()
scaler.fit(x_train)   # il ne faut fitrer que sur les données d entrainement
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)



@app.route('/chart1')
def chart1():
    ax = px.histogram(plot_df,x=plot_df["readmitted"])
    graphJSON = json.dumps(ax, cls=plotly.utils.PlotlyJSONEncoder)
    header="Proportion of people readmitted or not"
    description = """
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)
    

@app.route('/chart2')
def chart2():
    fig = px.box(plot_df,x=plot_df["readmitted"],y=plot_df['time_in_hospital'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Time in Hospital in function of the readmission"
    description = """
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/chart3')
def chart3():
    fig = px.box(plot_df,x=plot_df["readmitted"],y=plot_df['insulin'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Readmission in function of the insulin"
    description = """
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/chart4')
def chart4():
    fig = px.histogram(plot_df,x=plot_df["readmitted"],y=plot_df['num_medications'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Readmission in function of the number of medications"
    description = """ We can see that the number of medications is higher for the people who were not readmitted than for the people who got readmitted to the hospital.
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/chart5')
def chart5():
    fig = px.box(plot_df,x=plot_df["readmitted"],y=plot_df['diabetesMed'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Readmission in function of the Diabetes Med"
    description = """ 
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/chart6')
def chart6():
    fig = px.histogram(plot_df,x=plot_df["age"],y=plot_df['num_lab_procedures'], color=plot_df["readmitted"], marginal="box")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Number Labs procedures in function of age and readmission"
    description = """ 
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/lr')
def lr():
    model = LogisticRegression(solver='liblinear', random_state=0)
    cross_val_score(model,x_train,y_train,n_jobs=-1)
    model = LogisticRegression()
    parameters={"C":[1,1.5,2.5,3],"tol":[0.000001,.00001,0.0001]}
    grid=GridSearchCV(model,parameters,n_jobs=-1,cv=5)
    grid.fit(x_train,y_train)
    res = str(grid.best_score_)
    description=""" Logistic regression measures the relationship between the categorical dependent variable 
    and one or more independent variables by estimating probabilities using a logistic function, 
    which is the cumulative distribution function of logistic distribution.
    """
    header="Logistic Regression"
    return render_template('ML.html', description=description, header=header, res=res)


@app.route('/da')
def da():
    discAnalysis=discriminant_analysis.LinearDiscriminantAnalysis()
    discAnalysis.fit(x_train,y_train)
    discAnalysis.score(x_test,y_test)
    res = str(discAnalysis.score(x_test,y_test))
    description=""" Discriminant analysis is statistical technique used to classify observations into non-overlapping groups, 
    based on scores on one or more quantitative predictor variables.
    """
    header="Discriminant Analysis"
    return render_template('ML.html', description=description, header=header, res=res)


@app.route('/dt')
def dt():
    trees=tree.DecisionTreeClassifier()
    trees.fit(x_train,y_train)
    trees.score(x_test,y_test)
    res = str(trees.score(x_test,y_test))
    description=""" Decision Trees are a type of Supervised Machine Learning 
    where the data is continuously split according to a certain parameter. 
    The tree can be explained by two entities, namely decision nodes and leaves.
    """
    header="Decision Tree"
    return render_template('ML.html', description=description, header=header, res=res)


@app.route('/rf')
def rf():
    randomForest=ensemble.RandomForestClassifier()
    randomForest.fit(x_train,y_train)
    randomForest.score(x_test,y_test)
    res = str(randomForest.score(x_test,y_test))
    description=""" Random forests or random decision forests are an ensemble learning method for classification, 
    regression and other tasks that operates by constructing a multitude of decision trees at training time.
    """
    header="Random Forest"
    return render_template('ML.html', description=description, header=header, res=res)


@app.route('/bo')
def bo():
    boosting=ensemble.GradientBoostingClassifier()
    boosting.fit(x_train,y_train)
    boosting.score(x_test,y_test)
    res = str(boosting.score(x_test,y_test))
    description=""" Boosting algorithms seek to improve the prediction power by training a sequence of weak models, each compensating the weaknesses of its predecessors.
    Boosting is a generic algorithm rather than a specific model. Boosting needs you to specify a weak model 
    (e.g. regression, shallow decision trees, etc) and then improves it.
    """
    header="Boosting"
    return render_template('ML.html', description=description, header=header, res=res)


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)