import yfinance as yf
import streamlit as st
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection  import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


st.write("""
# Predicting Hepatitis
""")


st.sidebar.header('User Input Parameters')
def user_input_features():
    age = st.sidebar.slider('age', 10,80,10)
    s = st.sidebar.radio('sex',("Male","Female"))
    std = st.sidebar.radio('steroid',("no","yes"))
    av = st.sidebar.radio('antivirals',("no","yes"))
    fg=st.sidebar.radio('fatigue',("no","yes"))
    spi=st.sidebar.radio('spiders', ("no","yes"))
    asc=st.sidebar.radio('ascites', ("no","yes"))
    var=st.sidebar.radio('varices', ("no","yes"))
    bilirubin=st.sidebar.slider('bilirubin', 0.39, 4.00)
    alk_phosphate=st.sidebar.slider('alk_phosphate', 33, 250)
    sgot=st.sidebar.slider('sgot',13,500)
    albumin=st.sidebar.slider('albumin', 2.1,6.0)
    protime=st.sidebar.slider('protime',10,  90)
    h=st.sidebar.radio('histology',("no","yes"))
    sp=st.sidebar.radio('spleen_palpable',("no","yes"))
    lf=st.sidebar.radio('liver_firm',("no","yes"))
    lb=st.sidebar.radio('liver_big',("no","yes"))
    an=st.sidebar.radio('anorexia',("no","yes"))
    ml=st.sidebar.radio('malaise',("no","yes"))
    if(an=="no"):
        anorexia=1
    else:
        anorexia=2
    if(ml=="no"):
        malaise=1
    else:
        malaise=2
    if(lf=="no"):
        liver_firm=1
    else:
        liver_firm=2
    if(lb=="no"):
        liver_big=1
    else:
        liver_big=2

    if(sp=="no"):
        spleen_palpable=1
    else:
        spleen_palpable=2
    if(sp=="no"):
        spleen_palpable=1
    else:
        spleen_palpable=2
    if(h=="no"):
        histology=1
    else:
        histology=2

    if(s=="Male"):
        sex=1
    else:
        sex=2
    if(std=="no"):
        steroid=1
    else:
        steroid=2
    if(av=="no"):
        antivirals=1
    else:
        antivirals=2
    if(fg=="no"):
        fatigue=1
    else:
        fatigue=2
    if(var=="no"):
        varices=1
    else:
        varices=2
    if(spi=="no"):
        spiders=1
    else:
        spiders=2
    if(asc=="no"):
        ascites=1
    else:
        ascites=2
    data = {'age':age, 'sex Select 1 for Males, 2-Females':sex, 'steroid':steroid, 'antivirals':antivirals,
    'fatigue':fatigue,'malaise':malaise,'anorexia':anorexia,'liver_big':liver_big,'liver_firm':liver_firm,
    'spleen_palpable':spleen_palpable,'spiders':spiders,'ascites':ascites, 'varices':varices, 'bilirubin':bilirubin,
     'alk_phosphate':alk_phosphate, 'sgot':sgot, 'albumin':albumin,'protime':protime, 'histology':histology}
    features = pd.DataFrame(data, index=[0])
    return features

userdf = user_input_features()

st.subheader('User Input parameters')
st.write(userdf)


col_names = ["Class","AGE","SEX","STEROID","ANTIVIRALS","FATIGUE","MALAISE","ANOREXIA","LIVER BIG","LIVER FIRM","SPLEEN PALPABLE","SPIDERS","ASCITES","VARICES","BILIRUBIN","ALK PHOSPHATE","SGOT","ALBUMIN","PROTIME","HISTOLOGY"]
df = pd.read_csv("data/hepatitis.data",names=col_names)


df.columns.str.lower().str.replace(' ','_')

df.columns = df.columns.str.lower().str.replace(' ','_')
df = df.replace('?',0)
xfeatures = df[['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise',
       'anorexia', 'liver_big', 'liver_firm', 'spleen_palpable', 'spiders',
       'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin',
       'protime', 'histology']]
ylabels = df['class']

df[['steroid', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm',
       'spleen_palpable', 'spiders', 'ascites', 'varices',
       'alk_phosphate', 'sgot', 'protime']] = df[['steroid', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm',
       'spleen_palpable', 'spiders', 'ascites', 'varices',
       'alk_phosphate', 'sgot', 'protime']].astype(int)

df[['bilirubin','albumin']] = df[['bilirubin','albumin']].astype(float)

skb = SelectKBest(score_func=chi2,k=10)
best_feature_fit = skb.fit(xfeatures,ylabels)
bf_02 = best_feature_fit.transform(xfeatures)
feature_scores = pd.DataFrame(best_feature_fit.scores_,columns=['Feature_Scores'])

feature_column_names = pd.DataFrame(xfeatures.columns,columns=['Feature_name'])
best_feat_df = pd.concat([feature_scores,feature_column_names],axis=1)

et_clf = ExtraTreesClassifier()
et_clf.fit(xfeatures,ylabels)
feature_imporance_df = pd.Series(et_clf.feature_importances_,index=xfeatures.columns)




xfeatures_best = df[['age', 'sex', 'steroid', 'antivirals','fatigue','spiders',
       'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin',
       'protime', 'histology']]
ylabels = df['class']


# train /test dataset
x_train,x_test,y_train,y_test = train_test_split(xfeatures,ylabels,test_size=0.30,random_state=7)

# train /test dataset for best features
x_train_b,x_test_b,y_train_b,y_test_b = train_test_split(xfeatures_best,ylabels,test_size=0.30,random_state=7)

logreg = LogisticRegression()
logreg.fit(x_train,y_train)


logreg.score(x_test,y_test)
#logreg.predict(x_test,y_test)

st.subheader('Accuracy score Logistic Regression:')
st.write(accuracy_score(y_test,logreg.predict(x_test)))

model_logit = LogisticRegression()
model_logit.fit(x_train_b,y_train_b)
st.subheader('Accuracy score Logistic Regression with best features:')
st.write(model_logit.score(x_test_b,y_test_b))

#prediction=model_logit.predict(np.array(userdf.iloc[0]).reshape(1,-1))


feature_names_best = xfeatures_best.columns
target_names = ["Die","Live"]
class_names = ["Die(1)","Live(2)"]


#model_logit.predict(np.array(userdf.iloc[1]).reshape(1,-1))

st.subheader('Class labels and their corresponding index number')
st.write(target_names)






#st.subheader('Prediction')
#st.write(userdf.target_names[prediction])
#st.write(prediction)

#st.subheader('Prediction Probability')
#st.write(prediction_proba)
