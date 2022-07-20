import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from plotly.offline import iplot
import plotly.express as px
import plotly.graph_objs as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix,classification_report, roc_curve,roc_auc_score, accuracy_score
import tensorflow as tf
from tensorflow import keras
import pickle

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

df.info()

df.describe()

print("\033[1m" + 'Exploratory Data Analysis' + "\033[0m")

fig=ff.create_distplot([df['age'].values],['age'])
fig.update_layout(title_text='Age Distribution plot')
fig.show();

fig=ff.create_distplot([df['serum_creatinine'].values],['serum_creatinine'])
fig.update_layout(title_text='Serum_creatinine Distribution plot')
fig.show();

fig=ff.create_distplot([df['serum_sodium'].values],['serum_sodium'])
fig.update_layout(title_text='Serum_sodium Distribution plot')
fig.show();

M = df.loc[df["sex"]==1]
F = df.loc[df["sex"]==0]


labels = ['Male - Survived','Male - Not Survived', "Female -  Survived", "Female - Not Survived"]
values = [len(M.loc[df["DEATH_EVENT"]==0]),len(M.loc[df["DEATH_EVENT"]==1]),
          len(F.loc[df["DEATH_EVENT"]==0]),len(F.loc[df["DEATH_EVENT"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(title_text="Analysis on Survival - Gender")
fig.show();

MA = df.loc[(df['anaemia']==1) & (df['sex']==1)]
FA = df.loc[(df['anaemia']==1) & (df['sex']==0)]


labels = ['Male with Anaemia - Survived','Male with Anaemia - Not Survived', "Female with Anaemia -  Survived", "Female with Anaemia - Not Survived" ]
values = [len(MA.loc[df["DEATH_EVENT"]==0]),len(MA.loc[df["DEATH_EVENT"]==1]),len(FA.loc[df["DEATH_EVENT"]==0]),len(FA.loc[df["DEATH_EVENT"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(title_text="Analysis on Survival - Anaemia and Gender")
fig.show()

MSD = df.loc[(df['smoking']==1) & (df['sex']==1)]
FSD = df.loc[(df['smoking']==1) & (df['sex']==0)]


labels = ['Male(S) - Not Survived','Male(S) - Survived', "Female(S) - Not Survived", "Female(S) - Survived" ]
values = [len(MSD.loc[df["DEATH_EVENT"]==0]),len(MSD.loc[df["DEATH_EVENT"]==1]),len(FSD.loc[df["DEATH_EVENT"]==0]),len(FSD.loc[df["DEATH_EVENT"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(title_text="Analysis on Survival - Smoking and Gender")
fig.show();

plt.figure(figsize=(11,7))
sns.heatmap(df.corr(),annot=True)

print(df.corr()["DEATH_EVENT"].sort_values(ascending=False))

main_features = ['serum_creatinine','age','high_blood_pressure','serum_sodium','ejection_fraction','time','diabetes']
x = df[main_features]
y = df.iloc[:,12]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=88,shuffle=True,stratify=y)

kNN = KNeighborsClassifier(n_neighbors=9,metric='minkowski',p=2,weights='distance')
DT = DecisionTreeClassifier(random_state=88)
GNB = GaussianNB()
MNB = MultinomialNB()
LR = LogisticRegression(random_state=88,max_iter=300)
SVCC = SVC(random_state=88)
RFC = RandomForestClassifier(random_state=88)
GBC = GradientBoostingClassifier(random_state=88)
ABA = AdaBoostClassifier(random_state=88)
ETC = ExtraTreesClassifier(random_state=88)
L = [kNN,DT,GNB,MNB,LR,SVCC,RFC,GBC,ABA,ETC]
J = ['kNN','DT','GNB','MNB','LR','SVC','RFC','GBC','ABA','ETC']
M = []
N = []
K = []
for i in L:
    m = 0
    i.fit(x_train,y_train)
    m = i.predict(x_test)
    print('Confusion Matrix for ',str(i)[0:str(i).index('(')],'- \n',confusion_matrix(y_test,m),'\n')
    print(classification_report(y_test,m))
    FPR,TPR,thresholds = roc_curve(y_test,m)
    plt.plot(FPR,TPR)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    print('ROC AUC score for ',str(i)[0:str(i).index('(')],' - ',roc_auc_score(y_test,m),'\n')
    M.append(accuracy_score(y_test,m))
    N.append(roc_auc_score(y_test,m))

print("\033[1m" + 'Using Neural Networks' + "\033[0m")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8, input_dim=7, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=150, batch_size=15,verbose=0)
x, accuracy = model.evaluate(x_test, y_test,verbose=0)
print(accuracy)

plt.bar(J,M,width = 0.6)
plt.xlabel('Classification Models')
plt.ylabel('Accuracy')
plt.show()


pickle.dump(GBC, open('model.pkl','wb'))

