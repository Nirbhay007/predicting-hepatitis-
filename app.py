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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
# Metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


st.write("""
# Predicting Hepatitis
""")


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
y_pred = model_logit.predict(x_test_b)

st.subheader('Accuracy scores :')
st.write(accuracy_score(y_test,y_pred))

confusion_matrix(y_test,y_pred)
#st.write(plot_confusion_matrix(model_logit,x_test_b,y_test_b))
labels = ["Less than 10","10-20","20-30","30-40","40-50","50-60","60-70","70 and more"]
bins= [0,10,20,30,40,50,60,70,80]
freq_df = df.groupby(pd.cut(df['age'],bins=bins,labels=labels)).size()


freq_df = freq_df.reset_index(name='count')
freq_df.plot(kind='bar')
st.pyplot(plt.show())
freq_df.plot(kind='line')
st.pyplot(plt.show())
df.hist(bins=50,figsize=(20,15))
st.pyplot(plt.show())

# Plot of Freq Table
plt.bar(freq_df['age'],freq_df['count'])
plt.ylabel('Counts')
plt.title('Frequency Count of Age')
st.pyplot(plt.show())

labels = ['lt-10',"10-20","20-30","30-40","40-50","50-60","60-70","ge-70"]
fig1,ax1 = plt.subplots()
ax1.pie(freq_df['count'],labels=labels,autopct='1%.1f%%')
ax1.axis('equal')
st.pyplot(plt.show())

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
