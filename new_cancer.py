import pandas as pd            
import matplotlib.pyplot as plt         
import numpy as np                   
import seaborn as sns          
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
import joblib

cancer_dataset =  load_breast_cancer()
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],columns = np.append(cancer_dataset['feature_names'],['target']))
y = cancer_df['target']
x = cancer_df.drop(['target'],axis = 1)
# see the notebook for more explications on choice dropping these features
drop_list1 = ['mean perimeter','mean radius','mean compactness','mean concave points','radius error','perimeter error','worst radius','worst perimeter','worst compactness','worst concave points','compactness error','concavity error','worst texture','worst area']
x_1 = x.drop(drop_list1,axis = 1 ) 
# see the notebook for explanantion on choice of model

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
print('CM is: ',cm)
#sns.heatmap(cm,annot=True,fmt="d");
joblib.dump(clf_rf,"new_model")
#loaded_model = joblib.load("model")
#result = loaded_model.score(x_test, y_test)

 
