import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


print("libraries are working")

#loading data to pandas dataframe

raw_mail_data="C:\\Users\\Asus\\PycharmProjects\\sklearn\\mail_data.csv"
raw_mail_data_pd=pd.read_csv(raw_mail_data)

#print(raw_mail_data_pd)

#replace null values to null string

mail_data=raw_mail_data_pd.where((pd.notnull(raw_mail_data_pd)),"")

#print(mail_data)

#check rows and columns

col,row=mail_data.shape
#print(col,row)


#label spam mail=0 :: ham mail=1

mail_data.loc[mail_data['Category']=="spam",'Category',]=0
mail_data.loc[mail_data['Category']=="ham",'Category',]=1

#separating the data as texts as texts and label

x=mail_data['Message']
y=mail_data['Category']


#splitting data into training and test data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)

print(x_train.shape,y_train.shape,x_test.shape)
#transform data from text to numerical by feature Extraction
feature_extraction=TfidfVectorizer(min_df=1 , stop_words='english')

x_train_feature = feature_extraction.fit_transform(x_train)
x_test_feature = feature_extraction.transform(x_test)

#convert y train and test as int

y_train= y_train.astype('int')
y_test= y_test.astype('int')

#training logistic regression model
model=LogisticRegression()
model.fit(x_train_feature,y_train)

#predication on training data
predict_on_train_data=model.predict(x_train_feature)
accuracy_on_training_data=accuracy_score(y_train,predict_on_train_data)

print(accuracy_on_training_data)

#predication test data

predict_on_test_data=model.predict(x_test_feature)
accuracy_on_testing_data=accuracy_score(y_test,predict_on_test_data)

print(accuracy_on_testing_data)

#building a predicative systeam
pred=['spam','ham']
while True:
    sms=[str(input("enter mail : "))]
    #convert text to feature extraction
    sms_feature=feature_extraction.transform(sms)
    #making prediction
    predictions=model.predict(sms_feature)
    print(pred[predictions[0]])


