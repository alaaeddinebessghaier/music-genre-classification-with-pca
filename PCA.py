import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

sns.set()
## Data Exploration
df = pd.read_csv('music_dataset_mod.csv')
data = df.copy()
print(data.head())
print(data.isnull().sum())
print(data['Genre'].nunique())
#sns.countplot(x='Genre',data=data )
plt.title('Distribution of Music Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)

## Correlation Analysis

data = data.dropna(subset=['Genre'])
X = data.drop('Genre',axis=1)
Y= data['Genre']

label_encoder = preprocessing.LabelEncoder()
Y_encoded  = label_encoder.fit_transform(Y)
data_with_encoded_genre = data.copy()
data_with_encoded_genre.drop(columns=['Genre'],inplace=True)
data_with_encoded_genre['Genre_encoded'] = Y_encoded 
corr = data_with_encoded_genre.corr()

plt.figure(figsize=(12,6))
#sns.heatmap(correlation_matrix, 
#            vmin=-1,
#            vmax=1,
#            annot=True, 
#            cmap="RdBu", 
#            linewidths=0.1)
#plt.show()


## PCA for Dimensionality Reduction
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
ratios = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(ratios)
plt.plot(range(1,len(ratios)+1),cumulative_variance,marker='o')
plt.title("Explained Variance Ratio by Principal Component")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)               
#plt.show()
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)


# Evaluating Classification Efficacy: PCA-Transformed vs. Original Data
#PCA DATA TRONSFORMATION
X_pca_train , X_pca_test , y_pca_train , y_pca_test  = train_test_split(X_pca,Y_encoded,test_size=0.3,random_state=42)
clf=LogisticRegression(max_iter=10000,random_state=42)
clf.fit(X_pca_train,y_pca_train)
prediction = clf.predict(X_pca_test)

acc = accuracy_score(y_pca_test ,prediction)*100
print(f"Logistic Regression model accuracy: {acc:.2f}%")
report = classification_report(y_pca_test ,prediction,target_names=label_encoder.classes_)

#ORIGIN DATA
x_train , x_test, y_train , y_test = train_test_split(X_scaled,Y_encoded,test_size=0.3,random_state=42 )
clf_origin_data = LogisticRegression(max_iter=10000,random_state=42)
clf_origin_data.fit(x_train,y_train)
prediction_origin = clf_origin_data.predict(x_test)
acc_origin = accuracy_score(y_test,prediction_origin)
print(f"Logistic Regression model accuracy with origin data : {acc:.2f}%")
report_origin =classification_report(y_test ,prediction,target_names=label_encoder.classes_)


## Genre Prediction
unkown_genre = df[df['Genre'].isnull()].copy()
x_unkown = unkown_genre.drop(columns=['Genre'])
x_unkown_scaled = scaler.transform(x_unkown)
x_unkown_pca = pca.transform(x_unkown_scaled)
y_unkown_pred = clf.predict(x_unkown_pca)

unkown_genre.loc[:,'prediction_genre'] = label_encoder.inverse_transform(y_unkown_pred)


