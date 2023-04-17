import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import plotly.graph_objects as go


pd.options.display.max_rows = 250
data = pd.read_csv("normalize.csv", delimiter = ";")


age = data["Age"]
sex = data["Sex"]
blood_pressure = data["BP"]
cloesterol = data["Cholesterol"]
Na_to_K_ratio = data["Na_to_K"]
drugs = data["Drug"]

le = preprocessing.LabelEncoder()

# Encoded 
sex_encoded = le.fit_transform(sex)
bp_encoded = le.fit_transform(blood_pressure)
cholesterol_encoded = le.fit_transform(cloesterol)
label = le.fit_transform(drugs)

data["BP"] = bp_encoded
data["Sex"] = sex_encoded
data["Cholesterol"] = cholesterol_encoded
data["Drug"] = label

# "Blood Pressure - Na to K Ratio" and All 
features1 = list(zip(bp_encoded, Na_to_K_ratio)) 
features2 = list(zip(cholesterol_encoded, sex_encoded, bp_encoded, age, Na_to_K_ratio))

features_train, features_test, label_train, label_test = train_test_split(features1, label, test_size=0.3, random_state=1)

# KNN
model_knn = KNeighborsClassifier(n_neighbors=9) #Features1: 9 - Features2: 7
model_knn.fit(features_train, label_train)
predicted_labels = model_knn.predict(features_test)

knn_acc_matrix = confusion_matrix(label_test, predicted_labels)
knnacc = (knn_acc_matrix[0][0] + knn_acc_matrix[1][1] 
          + knn_acc_matrix[2][2] + knn_acc_matrix[3][3] 
          + knn_acc_matrix[4][4])/(sum(map(sum, knn_acc_matrix)))

print("KNN Accuracy:", knnacc)


# Desicion Tree
model_desiciontree = DecisionTreeClassifier()
model_desiciontree.fit(features_train, label_train)
predicted_desicion = model_desiciontree.predict(features_test)

desicion_acc_matrix = confusion_matrix(label_test, predicted_desicion)
desicion_acc = (desicion_acc_matrix[0][0] + desicion_acc_matrix[1][1] 
                + desicion_acc_matrix[2][2] + desicion_acc_matrix[3][3] 
                + desicion_acc_matrix[4][4])/(sum(map(sum, desicion_acc_matrix)))

print("Desicion Tree Accuracy:", desicion_acc)


# Logistic Regression
model_logistic = LogisticRegression()
model_logistic.fit(features_train, label_train)
predicted_logistic = model_logistic.predict(features_test)

model_logistic_matrix = confusion_matrix(label_test, predicted_logistic)
logistic_acc = (model_logistic_matrix[0][0] + model_logistic_matrix[1][1] 
                + model_logistic_matrix[2][2] + model_logistic_matrix[3][3] 
                + model_logistic_matrix[4][4])/(sum(map(sum, model_logistic_matrix)))

print("Logistic Regression Accuracy:", logistic_acc)


# SVC
model_svc = SVC()
model_svc.fit(features_train, label_train)
predicted_svc = model_svc.predict(features_test)

model_svc_matrix = confusion_matrix(label_test, predicted_svc)
svc_acc = (model_svc_matrix[0][0] + model_svc_matrix[1][1] 
           + model_svc_matrix[2][2] + model_svc_matrix[3][3] 
           + model_svc_matrix[4][4])/(sum(map(sum, model_svc_matrix)))

print("SVC Accuracy:", svc_acc)


# Correlation Matrix 
data_corr = data.corr()

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x = data_corr.columns,
        y = data_corr.index,
        z = np.array(data_corr),
        text=data_corr.values,
        texttemplate='%{text:.2f}',
        colorscale = ["white", "#338FC2"],
        showscale=False
    )
)
fig.update_layout(width=650, title="Correlation Matrix", title_font_color="#383FC2")
fig.show()
