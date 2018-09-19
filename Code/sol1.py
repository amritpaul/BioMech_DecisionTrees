# ************Answer 1************
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import graphviz
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score,auc,roc_curve

df = pd.read_csv('/home/am389796/Documents/UCI_Machine_Learning_Dataset/Biomechanical_Data_column_2C_weka.csv')
X = df.iloc[:,0:6]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.322, random_state=9)
classifier1 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 3, random_state = 9)
classifier1.fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
print(classification_report(y_test, y_pred1))

classifier2 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 8, random_state = 9)
classifier2.fit(X_train, y_train)
y_pred2 = classifier2.predict(X_test)
print(classification_report(y_test, y_pred2))

classifier3 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 12, random_state = 9)
classifier3.fit(X_train, y_train)
y_pred3 = classifier3.predict(X_test)
print(classification_report(y_test, y_pred3))

classifier4 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 30, random_state = 9)
classifier4.fit(X_train, y_train)
y_pred4 = classifier4.predict(X_test)
print(classification_report(y_test, y_pred4))

classifier5 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 50, random_state = 9)
classifier5.fit(X_train, y_train)
y_pred5 = classifier5.predict(X_test)
print(classification_report(y_test, y_pred5))

dot_data1 = tree.export_graphviz(classifier1, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier1.dot', filled=True, rounded=True,  special_characters=True)
graph1 = graphviz.Source(dot_data1)
# dot -Tpng classifier1.dot -o classifier1.png  #(command-line argument)
dot_data2 = tree.export_graphviz(classifier2, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier2.dot', filled=True, rounded=True,  special_characters=True)
graph2 = graphviz.Source(dot_data2)
# dot -Tpng classifier2.dot -o classifier2.png  #(command-line argument)
dot_data3 = tree.export_graphviz(classifier3, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier3.dot', filled=True, rounded=True,  special_characters=True)
graph3 = graphviz.Source(dot_data3)
# dot -Tpng classifier3.dot -o classifier3.png #(command-line argument)
dot_data4 = tree.export_graphviz(classifier4, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier4.dot', filled=True, rounded=True,  special_characters=True)
graph4 = graphviz.Source(dot_data4)
# dot -Tpng classifier4.dot -o classifier4.png #(command-line argument)
dot_data5 = tree.export_graphviz(classifier5, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier5.dot', filled=True, rounded=True,  special_characters=True)
graph5 = graphviz.Source(dot_data5)
# dot -Tpng classifier5.dot -o classifier5.png #(command-line argument)

################################
# Would prefer to use Decision Tree 3 with min_samples_leaf=12 because it has the highest precision and highest recall value along with accuracy of 85%. Also the decision boundary structure as represented in classifier3.png shows a better classification of samples.
################################


df = pd.DataFrame(y_test)
for idx, row in df.iterrows():
    if df.loc[idx,'class']=='Abnormal':
        df.loc[idx,'class']=0
    elif df.loc[idx,'class']=='Normal':
        df.loc[idx,'class']=1

y_test_ = list(df.values.flatten())
# print(y_test_)

probs = classifier1.predict_proba(X_test)
preds = probs[:, 1]
y_pred1 = [0 if y_pred1[i]=='Abnormal' else 1 for i in range(len(y_pred1))]
precision, recall, _ = precision_recall_curve(y_test_, y_pred1)
fpr, tpr, threshold = roc_curve(y_test_, preds)
print(roc_auc_score(y_test_, preds, average="macro"))
plt.step(recall, precision, color='b', alpha=0.5,where='post', label='Classifier 1')

probs = classifier2.predict_proba(X_test)
preds = probs[:, 1]
y_pred2 = [0 if y_pred2[ii]=='Abnormal' else 1 for ii in range(len(y_pred2))]
precision, recall, _ = precision_recall_curve(y_test_, y_pred2)
fpr, tpr, threshold = roc_curve(y_test_, preds)
print(roc_auc_score(y_test_, preds, average="macro"))
plt.step(recall, precision, color='g', alpha=0.5,where='post', label='Classifier 2')

probs = classifier3.predict_proba(X_test)
preds = probs[:, 1]
y_pred3 = [0 if y_pred3[iii]=='Abnormal' else 1 for iii in range(len(y_pred3))]
precision, recall, _ = precision_recall_curve(y_test_, y_pred3)
fpr, tpr, threshold = roc_curve(y_test_, preds)
print(roc_auc_score(y_test_, preds, average="macro"))
plt.step(recall, precision, color='r', alpha=0.5,where='post', label='Classifier 3')

probs = classifier4.predict_proba(X_test)
preds = probs[:, 1]
y_pred4 = [0 if y_pred4[j]=='Abnormal' else 1 for j in range(len(y_pred4))]
precision, recall, _ = precision_recall_curve(y_test_, y_pred4)
fpr, tpr, threshold = roc_curve(y_test_, preds)
print(roc_auc_score(y_test_, preds, average="macro"))
plt.step(recall, precision, color='m', alpha=0.5,where='post', label='Classifier 4')

probs = classifier5.predict_proba(X_test)
preds = probs[:, 1]
y_pred5 = [0 if y_pred5[jj]=='Abnormal' else 1 for jj in range(len(y_pred5))]
precision, recall, _ = precision_recall_curve(y_test_, y_pred5)
fpr, tpr, threshold = roc_curve(y_test_, preds)
print(roc_auc_score(y_test_, preds, average="macro"))
plt.step(recall, precision, color='y', alpha=0.5,where='post', label='Classifier 5')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc='upper left')
plt.savefig('/home/am389796/PycharmProjects/IISc_Workspace/fig.png')


################################
# From the graph we can see that the precision-recall curve has highest precision for classifier 3 and classifier 3 has highest roc_auc_score
################################


