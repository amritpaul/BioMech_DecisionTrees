# ************Answer 3************
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import graphviz
from sklearn import tree
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import confusion_matrix

df = pd.read_csv('/home/am389796/Documents/UCI_Machine_Learning_Dataset/Biomechanical_Data_column_2C_weka.csv')
ll = []
for i in range(len(df.columns)-1):
    col_min = df.iloc[:,i].min()
    col_max = df.iloc[:,i].max()
    interval = int((col_max - col_min) / 4)
    ll.append(interval)
print(ll)

a, b, c, d, e, f = [], [], [], [], [], []
for index, row in df.iterrows():
    a.append(int(row['pelvic_incidence']))
    b.append(int(row['pelvic_tilt numeric']))
    c.append(int(row['lumbar_lordosis_angle']))
    d.append(int(row['sacral_slope']))
    e.append(int(row['pelvic_radius']))
    f.append(int(row['degree_spondylolisthesis']))

df1 = pd.DataFrame(a, columns=["pelvic_incidence"])
df2 = pd.DataFrame(b, columns=["pelvic_tilt numeric"])
df3 = pd.DataFrame(c, columns=["lumbar_lordosis_angle"])
df4 = pd.DataFrame(d, columns=["sacral_slope"])
df5 = pd.DataFrame(e, columns=["pelvic_radius"])
df6 = pd.DataFrame(f, columns=["degree_spondylolisthesis"])
pd.concat([df1, df2, df3, df4, df5, df6], axis=1).to_csv('/home/am389796/Documents/UCI_Machine_Learning_Dataset/new_data2.csv', index=False)

df = pd.read_csv('/home/am389796/Documents/UCI_Machine_Learning_Dataset/new_data2.csv')
col_min = []
col_max = []
for i in range(len(df.columns)):
    col_min.append(df.iloc[:, i].min())

print(col_min)

for index, row in df.iterrows():
    if df.loc[index,'pelvic_incidence'] >= col_min[0] and df.loc[index,'pelvic_incidence'] <=col_min[0]+25:
        df.loc[index, 'pelvic_incidence'] = 0
    if df.loc[index,'pelvic_incidence'] > col_min[0]+25 and df.loc[index,'pelvic_incidence'] <=col_min[0]+25+25:
        df.loc[index, 'pelvic_incidence'] = 1
    if df.loc[index,'pelvic_incidence'] > col_min[0]+25+25 and df.loc[index,'pelvic_incidence'] <=col_min[0]+25+25+25:
        df.loc[index, 'pelvic_incidence'] = 2
    if df.loc[index,'pelvic_incidence'] > col_min[0]+25+25+25:
        df.loc[index, 'pelvic_incidence'] = 3

    if df.loc[index,'pelvic_tilt numeric'] >= col_min[1] and df.loc[index,'pelvic_tilt numeric'] <=col_min[1]+13:
        df.loc[index, 'pelvic_tilt numeric'] = 0
    if df.loc[index,'pelvic_tilt numeric'] >= col_min[1]+13 and df.loc[index,'pelvic_tilt numeric'] <=col_min[1]+13+13:
        df.loc[index, 'pelvic_tilt numeric'] = 1
    if df.loc[index,'pelvic_tilt numeric'] >= col_min[1]+13+13 and df.loc[index,'pelvic_tilt numeric'] <=col_min[1]+13+13+13:
        df.loc[index, 'pelvic_tilt numeric'] = 2
    if df.loc[index,'pelvic_tilt numeric'] >= col_min[1]+13+13+13:
        df.loc[index, 'pelvic_tilt numeric'] = 3

    if df.loc[index,'lumbar_lordosis_angle'] >= col_min[2] and df.loc[index,'lumbar_lordosis_angle'] <=col_min[2]+27:
        df.loc[index, 'lumbar_lordosis_angle'] = 0
    if df.loc[index,'lumbar_lordosis_angle'] >= col_min[2]+27 and df.loc[index,'lumbar_lordosis_angle'] <=col_min[2]+27+27:
        df.loc[index, 'lumbar_lordosis_angle'] = 1
    if df.loc[index,'lumbar_lordosis_angle'] >= col_min[2]+27+27 and df.loc[index,'lumbar_lordosis_angle'] <=col_min[2]+27+27+27:
        df.loc[index, 'lumbar_lordosis_angle'] = 2
    if df.loc[index,'lumbar_lordosis_angle'] >= col_min[2]+27+27+27:
        df.loc[index, 'lumbar_lordosis_angle'] = 3

    if df.loc[index,'sacral_slope'] >= col_min[3] and df.loc[index,'sacral_slope'] <=col_min[3]+27:
        df.loc[index, 'sacral_slope'] = 0
    if df.loc[index,'sacral_slope'] >= col_min[3]+27 and df.loc[index,'sacral_slope'] <=col_min[3]+27+27:
        df.loc[index, 'sacral_slope'] = 1
    if df.loc[index,'sacral_slope'] >= col_min[3]+27+27 and df.loc[index,'sacral_slope'] <=col_min[3]+27+27+27:
        df.loc[index, 'sacral_slope'] = 2
    if df.loc[index,'sacral_slope'] >=col_min[3]+27+27+27:
        df.loc[index, 'sacral_slope'] = 3

    if df.loc[index,'pelvic_radius'] >= col_min[4] and df.loc[index,'pelvic_radius'] <=col_min[4]+23:
        df.loc[index, 'pelvic_radius'] = 0
    if df.loc[index,'pelvic_radius'] >= col_min[4]+23 and df.loc[index,'pelvic_radius'] <=col_min[4]+23+23:
        df.loc[index, 'pelvic_radius'] = 1
    if df.loc[index,'pelvic_radius'] >= col_min[4]+23+23 and df.loc[index,'pelvic_radius'] <=col_min[4]+23+23+23:
        df.loc[index, 'pelvic_radius'] = 2
    if df.loc[index,'pelvic_radius'] >= col_min[4]+23+23+23:
        df.loc[index, 'pelvic_radius'] = 3

    if df.loc[index,'degree_spondylolisthesis'] >=col_min[5] and df.loc[index,'degree_spondylolisthesis'] <=col_min[5]+107:
        df.loc[index, 'degree_spondylolisthesis'] = 0
    if df.loc[index,'degree_spondylolisthesis'] >=col_min[5]+107 and df.loc[index,'degree_spondylolisthesis'] <=col_min[5]+107+107:
        df.loc[index, 'degree_spondylolisthesis'] = 1
    if df.loc[index,'degree_spondylolisthesis'] >=col_min[5]+107+107 and df.loc[index,'degree_spondylolisthesis'] <=col_min[5]+107+107+107:
        df.loc[index, 'degree_spondylolisthesis'] = 2
    if df.loc[index,'degree_spondylolisthesis'] >=col_min[5]+107+107+107 :
        df.loc[index, 'degree_spondylolisthesis'] = 3
df.to_csv('/home/am389796/Documents/UCI_Machine_Learning_Dataset/new.csv', index=False)
df1 = pd.read_csv('/home/am389796/Documents/UCI_Machine_Learning_Dataset/Biomechanical_Data_column_2C_weka.csv')
temp = df1.iloc[:,-1]
pd.concat([df, temp], axis=1).to_csv('/home/am389796/Documents/UCI_Machine_Learning_Dataset/new_data2_.csv', index=False)

################################
# The interval for 6 features are [25, 13, 27, 27, 23, 107] and the column_min for each feature is [26, -6, 14, 13, 70, -11]
################################

new_df = pd.read_csv('/home/am389796/Documents/UCI_Machine_Learning_Dataset/new_data2_.csv')
X = new_df.iloc[:,0:6]
y = new_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.322, random_state=9)

classifier11 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 3, random_state = 9)
classifier11.fit(X_train, y_train)
y_pred1 = classifier11.predict(X_test)
print(classification_report(y_test, y_pred1))

classifier12 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 8, random_state = 9)
classifier12.fit(X_train, y_train)
y_pred2 = classifier12.predict(X_test)
print(classification_report(y_test, y_pred2))

classifier13 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 12, random_state = 9)
classifier13.fit(X_train, y_train)
y_pred3 = classifier13.predict(X_test)
print(classification_report(y_test, y_pred3))

classifier14 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 30, random_state = 9)
classifier14.fit(X_train, y_train)
y_pred4 = classifier14.predict(X_test)
print(classification_report(y_test, y_pred4))

classifier15 = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 50, random_state = 9)
classifier15.fit(X_train, y_train)
y_pred5 = classifier15.predict(X_test)
print(classification_report(y_test, y_pred5))

df = pd.DataFrame(y_test)
for idx, row in df.iterrows():
    if df.loc[idx, 'class'] == 'Normal':
        df.loc[idx, 'class'] = 0
    elif df.loc[idx, 'class'] =='Abnormal':
        df.loc[idx, 'class'] = 1
y_test_ = list(df.values.flatten())

ll = list()
for i in range(len(y_pred1)):
    if y_pred1[i]=='Normal':
        y_pred1[i]=0
        ll.append((y_pred1[i]))
    elif y_pred1[i]=='Abnormal':
        y_pred1[i]=1
        ll.append((y_pred1[i]))
y_pred1 = copy.deepcopy(list(ll))
print(confusion_matrix(y_test_,y_pred1))
print("\n")

ll = list()
for i in range(len(y_pred2)):
    if y_pred2[i]=='Normal':
        y_pred2[i]=0
        ll.append((y_pred2[i]))
    elif y_pred2[i]=='Abnormal':
        y_pred2[i]=1
        ll.append((y_pred2[i]))
y_pred2 = copy.deepcopy(list(ll))
print(confusion_matrix(y_test_,y_pred2))
print("\n")

ll = list()
for i in range(len(y_pred3)):
    if y_pred3[i]=='Normal':
        y_pred3[i]=0
        ll.append((y_pred3[i]))
    elif y_pred3[i]=='Abnormal':
        y_pred3[i]=1
        ll.append((y_pred3[i]))
y_pred3 = copy.deepcopy(list(ll))
print(confusion_matrix(y_test_,y_pred3))
print("\n")

ll = list()
for i in range(len(y_pred4)):
    if y_pred4[i]=='Normal':
        y_pred4[i]=0
        ll.append((y_pred4[i]))
    elif y_pred4[i]=='Abnormal':
        y_pred4[i]=1
        ll.append((y_pred4[i]))
y_pred4 = copy.deepcopy(list(ll))
print(confusion_matrix(y_test_,y_pred4))
print("\n")

ll = list()
for i in range(len(y_pred5)):
    if y_pred5[i]=='Normal':
        y_pred5[i]=0
        ll.append((y_pred5[i]))
    elif y_pred5[i]=='Abnormal':
        y_pred5[i]=1
        ll.append((y_pred5[i]))
y_pred5 = copy.deepcopy(list(ll))
print(confusion_matrix(y_test_,y_pred5))
print("\n")

################################
# We prefer classifier classifier-14 (min sample per leaf=30) or classifier-15 (min sample per leaf=50) because of the high values in the confusion matrix.
################################

dot_data1 = tree.export_graphviz(classifier11, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier11.dot', filled=True, rounded=True,  special_characters=True)
graph1 = graphviz.Source(dot_data1)
# dot -Tpng classifier6.dot -o classifier6.png  #(command-line argument)
dot_data2 = tree.export_graphviz(classifier12, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier12.dot', filled=True, rounded=True,  special_characters=True)
graph2 = graphviz.Source(dot_data2)
# dot -Tpng classifier7.dot -o classifier7.png  #(command-line argument)
dot_data3 = tree.export_graphviz(classifier13, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier13.dot', filled=True, rounded=True,  special_characters=True)
graph3 = graphviz.Source(dot_data3)
# dot -Tpng classifier8.dot -o classifier8.png #(command-line argument)
dot_data4 = tree.export_graphviz(classifier14, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier14.dot', filled=True, rounded=True,  special_characters=True)
graph4 = graphviz.Source(dot_data4)
# dot -Tpng classifier9.dot -o classifier9.png #(command-line argument)
dot_data5 = tree.export_graphviz(classifier15, out_file='/home/am389796/PycharmProjects/IISc_Workspace/classifier15.dot', filled=True, rounded=True,  special_characters=True)
graph5 = graphviz.Source(dot_data5)
# dot -Tpng classifier10.dot -o classifier10.png #(command-line argument)
