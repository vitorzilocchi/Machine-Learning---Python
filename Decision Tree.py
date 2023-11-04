# Importing the Necessary Libraries
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Setting Directory & Visualizing Data
df = pd.read_csv('diabetes.csv')
df.head()

# Checking Rows and Colunms (Shape)
df.shape

#Check if there exists null values in dataset
df[df.isnull().any(axis=1)].head()

# Define Features
features_columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

x = df[features_columns].values
y = df.Outcome.values

# Splitting data in train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=324)

#Fitting decision tree model on data
Diabete_Classifier = DecisionTreeClassifier(max_leaf_nodes = 10, random_state = 0)
Diabete_Classifier.fit(X_train,y_train)

#predicting values on test data
y_predicted = Diabete_Classifier.predict(X_test)

#Check accuracy (%)
accuracy_score (y_test, y_predicted) * 100

# Confusion Matrix
confusion_matrix(y_test, y_predicted)

tree.plot_tree(Diabete_Classifier,feature_names=features_columns)

print(Diabete_Classifier.predict([[7,110,72,40,0,33,1,36]]))
