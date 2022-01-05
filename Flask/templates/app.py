from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask import Response
from datetime import datetime
import sys
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
df=pd.read_csv('https://github.com/Leow92/Python_Data_Analysis/raw/main/diabetic_data.csv')

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

df["severity_of_disease"]=(df["time_in_hospital"]+df["num_procedures"]+df["num_medications"]+df["num_lab_procedures"]+df["number_diagnoses"])^2

df.replace(r"?", np.nan, inplace=True)

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


df.drop('diag_1', axis=1, inplace=True)
df.drop('diag_2', axis=1, inplace=True)
df.drop('diag_3', axis=1, inplace=True)

# encoder categorical data
df=pd.get_dummies(data=df,columns=["race"])

df2=df.copy() #we copy the df to do vizualization on df2 and keep df for ML

df2["readmitted"]=df2["readmitted"].replace(0,"not readmitted")
df2["readmitted"]=df2["readmitted"].replace(1,"readmitted")

df2["gender"]=df2["gender"].replace(-1,"Female")
df2["gender"]=df2["gender"].replace(1,"Male")

df2["insulin"]=df2["insulin"].replace(-10,"Down")
df2["insulin"]=df2["insulin"].replace(0,"Steady")
df2["insulin"]=df2["insulin"].replace(10,"Up")
df2["insulin"]=df2["insulin"].replace(-20,"No")

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

admission_type={
        1:"Emergency",
        2:"Urgent",
        3:"Elective",
        4:"Newborn",
        5:"Not availbale",
        6:"Null",
        7:"Trauma Center",
        8:"Not mapped",
    }
df2["admission_type"] = df2.admission_type_id.map(admission_type)

discharge_disposition={
    1:"Discharged to home",
    2:"Discharged/transferred to another short term hospital",
    3:"Discharged/transferred to SNF",
    4:"Discharged/transferred to ICF",
    5:"Discharged/transferred to another type of inpatient care institution",
    6:"Discharged/transferred to home with home health service",
    7:"Left AMA",
    8:"Discharged/transferred to home under care of Home IV provider",
    9:"Admitted as an inpatient to this hospital",
    10:"Neonate discharged to another hospital for neonatal aftercare",
    11:"Expired",
    12:"Still patient or expected to return for outpatient services",
    13:"Hospice / home",
    14:"Hospice / medical facility",
    15:"Discharged/transferred within this institution to Medicare approved swing bed",
    16:"Discharged/transferred/referred another institution for outpatient services",
    17:"Discharged/transferred/referred to this institution for outpatient services",
    18:"NULL",
    19:"Expired at home. Medicaid only, hospice.",
    20:"Expired in a medical facility. Medicaid only, hospice.",
    21:"Expired, place unknown. Medicaid only, hospice.",
    22:"Discharged/transferred to another rehab fac including rehab units of a hospital .",
    23:"Discharged/transferred to a long term care hospital.",
    24:"Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.",
    25:"Not Mapped",
    26:"Unknown/Invalid",
    30:"Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere",
    27:"Discharged/transferred to a federal health care facility.",
    28:"Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
    29:"Discharged/transferred to a Critical Access Hospital (CAH).",
}
df2["discharge_disposition"] = df2.discharge_disposition_id.map(discharge_disposition)

admission_source={
    1: "Physician Referral",
    2:"Clinic Referral",
    3:"HMO Referral",
    4:"Transfer from a hospital",
    5:"Transfer from a Skilled Nursing Facility (SNF)",
    6:"Transfer from another health care facility",
    7:"Emergency Room",
    8:"Court/Law Enforcement",
    9:"Not Available",
    10:"Transfer from critial access hospital",
    11:"Normal Delivery",
    12:"Premature Delivery",
    13:"Sick Baby",
    14:"Extramural Birth",
    15:"Not Available",
    17:"NULL",
    18:"Transfer From Another Home Health Agency",
    19:"Readmission to Same Home Health Agency",
    20:"Not Mapped",
    21:"Unknown/Invalid",
    22:"Transfer from hospital inpt/same fac reslt in a sep claim",
    23:"Born inside this \'hospital",
    24:"Born outside this hospital",
    25:"Transfer from Ambulatory Surgery Center",
    26:"Transfer from Hospice",
}
df2["admission_source"]=df2.admission_source_id.map(admission_source)

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
    ax = px.histogram(df2,x=df2["readmitted"])
    graphJSON = json.dumps(ax, cls=plotly.utils.PlotlyJSONEncoder)
    header="Proportion of people readmitted or not"
    description = """ We can observe that the proportion of the people not readmitted is higher : 59% against 41%
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/chart8')
def chart8():
    fig=px.histogram(df2, x="readmitted",color='gender',barmode='group')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="People readmitted or not depending of the sexe of the people"
    description = """ We can observe that in our patients the proportion of female patients is higher than the proportion of male 
    patients : 52.5% against 47.5%
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/chart2')
def chart2():
    fig = px.histogram(df2,x=df2["age"],y=df2['time_in_hospital'], color=df2["readmitted"], marginal="box")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Time in Hospital in function of the readmission"
    description = """ We can observe that readmitted people between 70 and 80 years old are the people who spends the most time in 
    the hospital
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/chart3')
def chart3():
    fig = px.box(df2,x=df2["readmitted"],y=df2['insulin'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Readmission in function of the insulin"
    description = """ In this vizualisation, we can observe that the average rate of insulin of the people readmitted is up. 
    Generally we can say that the people with diabetes have an upper average rate of insulin in the blood.
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/chart4')
def chart4():
    fig = px.histogram(df2,x=df2["readmitted"],y=df2['num_medications'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Readmission in function of the number of medications"
    description = """ We can see that the number of medications is higher for the people who were not readmitted than for the 
    people who got readmitted to the hospital.
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/chart5')
def chart5():
    fig = px.box(df2,x=df2["readmitted"],y=df2['diabetesMed'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Readmission in function of the Diabetes Med"
    description = """ We can observe that all the readmitted people were already taking diabetes medications
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/chart6')
def chart6():
    fig = px.histogram(df2,x=df2["age"],y=df2['num_lab_procedures'], color=df2["readmitted"], marginal="box")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Number Labs procedures in function of age and readmission"
    description = """ We can observe that readmitted people between 70 and 80 years old are the people who have the highest number of 
    lab procedures in the hospital
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header,description=description)


@app.route('/chart7')
def chart7():
    ax1=px.pie(df2, values='admission_type_id', names='admission_type', title='Admission type') 
    graphJSON = json.dumps(ax1, cls=plotly.utils.PlotlyJSONEncoder)
    header="Types of Admission of readmitted People in the Hospital"
    description = """ The most common type of admission in the readmitted people is the Emergency admission with 37.4%
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header, description=description)


@app.route('/chart9')
def chart9():
    ax1=px.pie(df2, values='discharge_disposition_id', names='discharge_disposition', title='Discharge disposition')
    graphJSON = json.dumps(ax1, cls=plotly.utils.PlotlyJSONEncoder)
    header="Types of Dispositions of readmitted people in the Hospital"
    description = """ The most common type of discharge disposition in the readmitted people is the Discharged at Home  
    with 36.5%
    """
    return render_template('flasky.html', graphJSON=graphJSON, header=header, description=description)


@app.route('/lr')
def lr():
    model = LogisticRegression(solver='liblinear', random_state=0,C=0.8,tol=1e-6,max_iter=200)
    model.fit(x_train, y_train)
    res= model.score(x_test, y_test)
    y_pred =model.predict_proba(x_test)[:,1]

    auc = metrics.roc_auc_score(y_test, y_pred)

    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred)
    df3=pd.DataFrame({"false_positive_rate":false_positive_rate,"true_positive_rate":true_positive_rate})
    fig = px.line(df3,x="false_positive_rate", y="true_positive_rate", title="ROC")
    fig.update_layout(width=1000, height=1000)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    description=""" Logistic regression measures the relationship between the categorical dependent variable 
    and one or more independent variables by estimating probabilities using a logistic function, 
    which is the cumulative distribution function of logistic distribution.
    """
    header="Logistic Regression"
    return render_template('ML.html', graphJSON=graphJSON, description=description, header=header, res=res, auc=auc)


@app.route('/da')
def da():
    discAnalysis=discriminant_analysis.LinearDiscriminantAnalysis()
    discAnalysis.fit(x_train,y_train)
    discAnalysis.score(x_test,y_test)
    res = str(discAnalysis.score(x_test,y_test))
    y_pred =discAnalysis.predict_proba(x_test)[:,1]

    auc = metrics.roc_auc_score(y_test, y_pred)

    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred)
    df3=pd.DataFrame({"false_positive_rate":false_positive_rate,"true_positive_rate":true_positive_rate})
    fig = px.line(df3,x="false_positive_rate", y="true_positive_rate", title="ROC")
    fig.update_layout(width=1000, height=1000)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    description=""" Discriminant analysis is statistical technique used to classify observations into non-overlapping groups, 
    based on scores on one or more quantitative predictor variables.
    """
    header="Discriminant Analysis"
    return render_template('ML.html', graphJSON=graphJSON, description=description, header=header, res=res, auc=auc)


@app.route('/dt')
def dt():
    trees=tree.DecisionTreeClassifier()
    trees.fit(x_train,y_train)
    trees.score(x_test,y_test)
    res = str(trees.score(x_test,y_test))
    y_pred =trees.predict_proba(x_test)[:,1]

    auc = metrics.roc_auc_score(y_test, y_pred)

    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred)
    df3=pd.DataFrame({"false_positive_rate":false_positive_rate,"true_positive_rate":true_positive_rate})
    fig = px.line(df3,x="false_positive_rate", y="true_positive_rate", title="ROC")
    fig.update_layout(width=1000, height=1000)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    description=""" Decision Trees are a type of Supervised Machine Learning 
    where the data is continuously split according to a certain parameter. 
    The tree can be explained by two entities, namely decision nodes and leaves.
    """
    header="Decision Tree"
    return render_template('ML.html', graphJSON=graphJSON, description=description, header=header, res=res, auc=auc)


@app.route('/rf')
def rf():
    randomForest=ensemble.RandomForestClassifier()
    randomForest.fit(x_train,y_train)
    randomForest.score(x_test,y_test)
    res = str(randomForest.score(x_test,y_test))
    y_pred =randomForest.predict_proba(x_test)[:,1]

    auc = metrics.roc_auc_score(y_test, y_pred)

    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred)
    df3=pd.DataFrame({"false_positive_rate":false_positive_rate,"true_positive_rate":true_positive_rate})
    fig = px.line(df3,x="false_positive_rate", y="true_positive_rate", title="ROC")
    fig.update_layout(width=1000, height=1000)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    description=""" Random forests or random decision forests are an ensemble learning method for classification, 
    regression and other tasks that operates by constructing a multitude of decision trees at training time.
    """
    header="Random Forest"
    return render_template('ML.html', graphJSON=graphJSON, description=description, header=header, res=res, auc=auc)


@app.route('/bo')
def bo():
    boosting=ensemble.GradientBoostingClassifier(learning_rate=0.1,n_estimators=500,min_samples_split=2,min_samples_leaf=1)
    boosting.fit(x_train,y_train)
    boosting.score(x_test,y_test)
    res = str(boosting.score(x_test,y_test))
    y_pred =boosting.predict_proba(x_test)[:,1]

    auc = metrics.roc_auc_score(y_test, y_pred)

    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred)
    df3=pd.DataFrame({"false_positive_rate":false_positive_rate,"true_positive_rate":true_positive_rate})
    fig = px.line(df3,x="false_positive_rate", y="true_positive_rate", title="ROC")
    fig.update_layout(width=1000, height=1000)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    description=""" Boosting algorithms seek to improve the prediction power by training a sequence of weak models, each compensating the weaknesses of its predecessors.
    Boosting is a generic algorithm rather than a specific model. Boosting needs you to specify a weak model 
    (e.g. regression, shallow decision trees, etc) and then improves it.
    """
    header="Boosting"
    return render_template('ML.html', graphJSON=graphJSON, description=description, header=header, res=res, auc=auc)

@app.route('/comparison')
def comparison():
    # discrimant analysis
    model = discriminant_analysis.LinearDiscriminantAnalysis(shrinkage=0, solver='lsqr', tol=1e-05)
    model.fit(x_train, y_train)
    da=model.score(x_test, y_test)

    # logistic regression
    model2 = LogisticRegression(solver='liblinear', random_state=0,C=0.8,tol=1e-6,max_iter=200)
    model2.fit(x_train, y_train)
    lr=model.score(x_test, y_test)

    #decision tree
    trees=tree.DecisionTreeClassifier(max_depth=7, min_samples_split=3)
    trees.fit(x_train,y_train)
    tr=trees.score(x_test,y_test)

    # random forest
    randomForest=ensemble.RandomForestClassifier(n_estimators=400,max_depth=7,min_samples_split=2,min_samples_leaf=1)
    randomForest.fit(x_train,y_train)
    fo=randomForest.score(x_test,y_test)

    # boosting
    boosting=ensemble.GradientBoostingClassifier(learning_rate=0.1,n_estimators=500,min_samples_split=2,min_samples_leaf=1)
    boosting.fit(x_train,y_train)
    bo=boosting.score(x_test,y_test)

    dic={"count":[da,lr,tr,fo,bo],"models":["discrimiant analysis", "logistic regression", "decision tree", "random forest","boosting"]}
    final=pd.DataFrame(dic)

    fig = px.bar(final,x="models", y="count")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    description=""" Comparison of the different scores
    """
    header="Boosting"
    return render_template('Comparison.html', graphJSON=graphJSON, description=description, header=header)


@app.route('/api')
def api():
    return render_template('api.html')


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        Gender=request.form['Gender']
        Age=request.form['Age']
        AdmissionType=request.form['AdmissionType']
        AdmissionSource=request.form['AdmissionSource']
        TimeInHospital=request.form['TimeInHospital']
        NumLabProcedure=request.form['NumLabProcedure']
        NumProcedure=request.form['NumProcedure']
        NumMedication=request.form['NumMedication']
        NumOutpatient=request.form['NumOutpatient']
        NumEmergency=request.form['NumEmergency']
        NumInpatient=request.form['NumInpatient']
        NumDiagnoses=request.form['NumDiagnoses']
        A1Cresult=request.form['A1Cresult']
        Metformin=request.form['Metformin']
        Repaglinide=request.form['Repaglinide']
        Glimepiride=request.form['Glimepiride']
        Rosiglitazone=request.form['Rosiglitazone']
        Acarbose=request.form['Acarbose']   
        Miglitol=request.form['Miglitol']
        Insulin=request.form['Insulin']
        Glipizide=request.form['Glipizide']
        MetforminRosiglitazone=request.form['MetforminRosiglitazone']
        Change=request.form['Change']
        DiabetesMed=request.form['DiabetesMed']
        AfricanAmerican=request.form['AfricanAmerican']
        Asian=request.form['Asian']
        Caucasian=request.form['Caucasian']

        dic={"gender":[Gender],"age":[Age],"admission_type_id":[AdmissionType],"admission_source_id":[AdmissionSource],"time_in_hospital":[TimeInHospital],"num_lab_procedures":[NumLabProcedure],
        "num_procedures":[NumProcedure],"num_medications":[NumMedication],"number_outpatient":[NumOutpatient],"number_emergency":[NumEmergency],
        "number_inpatient":[NumInpatient],"number_diagnoses":[NumDiagnoses],"A1Cresult":[A1Cresult],"metformin":[Metformin],"repaglinide":[Repaglinide],
        "glimepiride":[Glimepiride],"rosiglitazone":[Rosiglitazone],"acarbose":[Acarbose],"miglitol":[Miglitol],"insulin":[Insulin],
        "glipizide-metformin":[Glipizide],"metformin-rosiglitazone":[MetforminRosiglitazone],"change":[Change],"diabetesMed":[DiabetesMed],"severity_of_disease":[5],
        "race_AfricanAmerican":[AfricanAmerican],"race_Asian":[Asian],"race_Caucasian":[Caucasian]}
        df_submit=pd.DataFrame(dic)

        header=""" Results """
        
        model2 = LogisticRegression(solver='liblinear', random_state=0,C=0.8,tol=1e-6,max_iter=200)
        model2.fit(x_train, y_train)
        y_pred = model2.predict_proba(df_submit)[:,1]

        res = y_pred[0]
        if res>0.5:
            read = """ You will be readmitted to the hospital """
        else:
            read = """ You will not be readmitted to the hospital"""
        
    return render_template('submit.html', header=header, read=read)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
