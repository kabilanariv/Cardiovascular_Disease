import streamlit as st 
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from argparse import _HelpAction
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

dfs = [pd.read_csv(x + '.csv') for x in ['biological','social']]


st.title('CardioVascular Disease Risk Factor Analysis')

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Biological Factors', 'Social Factors')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN','Decision Tree', 'Random Forest')
)


biological=pd.read_csv("biological.csv")
social=pd.read_csv("social.csv")
    
def get_dataset(name):
    data = None
    if name == 'Biological Factors':
        data = dfs[0]
    elif name == 'Social Factors':
        data = dfs[1]
    else:
        data = dfs[2]    
    X = data.iloc[: , : -1]
    y = data.iloc[:,-1]
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))




def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'Decision Tree':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'Decision Tree':
        clf = clf = RandomForestClassifier( max_depth=params['max_depth'], random_state=1234)
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
        
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)



X.values.reshape(-1,1)

y.values.reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
st.write('Confusion Matrix: ')
st.write(confusion_matrix(y_test, y_pred))
st.write('Classification Report')
report=classification_report(y_test, y_pred, output_dict=True)
FinalReport=pd.DataFrame(report).transpose()
st.write(FinalReport)

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

st.pyplot(fig)


