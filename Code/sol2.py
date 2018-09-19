# ************Answer 2************
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import graphviz
from sklearn import tree
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

df1 = pd.read_csv('/home/am389796/Documents/UCI_Machine_Learning_Dataset/BiomechanicalData_column_3C_weka.csv')
X = df1.iloc[:,0:6]
y = df1.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.322, random_state=9)

classifier6 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 3, random_state = 9)
classifier6.fit(X_train, y_train)
y_pred1 = classifier6.predict(X_test)
print(classification_report(y_test, y_pred1))

classifier7 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 8, random_state = 9)
classifier7.fit(X_train, y_train)
y_pred2 = classifier7.predict(X_test)
print(classification_report(y_test, y_pred2))

classifier8 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 12, random_state = 9)
classifier8.fit(X_train, y_train)
y_pred3 = classifier8.predict(X_test)
print(classification_report(y_test, y_pred3))

classifier9 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 30, random_state = 9)
classifier9.fit(X_train, y_train)
y_pred4 = classifier9.predict(X_test)
print(classification_report(y_test, y_pred4))

classifier10 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 50, random_state = 9)
classifier10.fit(X_train, y_train)
y_pred5 = classifier10.predict(X_test)
print(classification_report(y_test, y_pred5))

dot_data1 = tree.export_graphviz(classifier6, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier6.dot', filled=True, rounded=True,  special_characters=True)
graph1 = graphviz.Source(dot_data1)
# dot -Tpng classifier6.dot -o classifier6.png  #(command-line argument)
dot_data2 = tree.export_graphviz(classifier7, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier7.dot', filled=True, rounded=True,  special_characters=True)
graph2 = graphviz.Source(dot_data2)
# dot -Tpng classifier7.dot -o classifier7.png  #(command-line argument)
dot_data3 = tree.export_graphviz(classifier8, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier8.dot', filled=True, rounded=True,  special_characters=True)
graph3 = graphviz.Source(dot_data3)
# dot -Tpng classifier8.dot -o classifier8.png #(command-line argument)
dot_data4 = tree.export_graphviz(classifier9, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier9.dot', filled=True, rounded=True,  special_characters=True)
graph4 = graphviz.Source(dot_data4)
# dot -Tpng classifier9.dot -o classifier9.png #(command-line argument)
dot_data5 = tree.export_graphviz(classifier10, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier10.dot', filled=True, rounded=True,  special_characters=True)
graph5 = graphviz.Source(dot_data5)
# dot -Tpng classifier10.dot -o classifier10.png #(command-line argument)

################################
# Would prefer to use Decision Tree 3 with min_samples_leaf=12 because it has the highest precision and highest recall value along with accuracy of 85%. Also the decision boundary structure as represented in classifier8.png shows a better classification of samples.
################################


df = pd.DataFrame(y_test)
for idx, row in df.iterrows():
    if df.loc[idx,'class']=='Normal':
        df.loc[idx,'class']=0
    elif df.loc[idx,'class']=='Hernia':
        df.loc[idx,'class']=1
    elif df.loc[idx,'class']=='Spondylolisthesis':
        df.loc[idx,'class']=2

y_test_ = list(df.values.flatten())
# print(y_test_)

ll = list()
for i in range(len(y_pred1)):
    if y_pred1[i]=='Normal':
        y_pred1[i]=0
        ll.append((y_pred1[i]))
    elif y_pred1[i]=='Hernia':
        y_pred1[i]=1
        ll.append((y_pred1[i]))
    elif y_pred1[i]=='Spondylolisthesis':
        y_pred1[i]=2
        ll.append((y_pred1[i]))
y_pred1 = copy.deepcopy(list(ll))
print(confusion_matrix(y_test_,y_pred1))
print("\n")
ll = list()
for i in range(len(y_pred2)):
    if y_pred2[i]=='Normal':
        y_pred2[i]=0
        ll.append((y_pred2[i]))
    elif y_pred2[i]=='Hernia':
        y_pred2[i]=1
        ll.append((y_pred2[i]))
    elif y_pred2[i]=='Spondylolisthesis':
        y_pred2[i]=2
        ll.append((y_pred2[i]))
y_pred2 = copy.deepcopy(list(ll))
print(confusion_matrix(y_test_,y_pred2))
print("\n")

ll = list()
for i in range(len(y_pred3)):
    if y_pred3[i]=='Normal':
        y_pred3[i]=0
        ll.append((y_pred3[i]))
    elif y_pred3[i]=='Hernia':
        y_pred3[i]=1
        ll.append((y_pred3[i]))
    elif y_pred3[i]=='Spondylolisthesis':
        y_pred3[i]=2
        ll.append((y_pred3[i]))
y_pred3 = copy.deepcopy(list(ll))
print(confusion_matrix(y_test_,y_pred3))
print("\n")

ll = list()
for i in range(len(y_pred4)):
    if y_pred4[i]=='Normal':
        y_pred4[i]=0
        ll.append((y_pred4[i]))
    elif y_pred4[i]=='Hernia':
        y_pred4[i]=1
        ll.append((y_pred4[i]))
    elif y_pred4[i]=='Spondylolisthesis':
        y_pred4[i]=2
        ll.append((y_pred4[i]))
y_pred4 = copy.deepcopy(list(ll))
print(confusion_matrix(y_test_,y_pred4))
print("\n")

ll = list()
for i in range(len(y_pred5)):
    if y_pred5[i]=='Normal':
        y_pred5[i]=0
        ll.append((y_pred5[i]))
    elif y_pred5[i]=='Hernia':
        y_pred5[i]=1
        ll.append((y_pred5[i]))
    elif y_pred5[i]=='Spondylolisthesis':
        y_pred5[i]=2
        ll.append((y_pred5[i]))
y_pred5 = copy.deepcopy(list(ll))
print(confusion_matrix(y_test_,y_pred5))
print("\n")

################################
# For multi-class classification, the best measurement after accuracy is confusion matrix. We can see that for Classifier 8, the confusion matrix is best since it represents the least misclassification.
################################

