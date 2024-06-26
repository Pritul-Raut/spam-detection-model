import tkinter as tk
from tkinter import scrolledtext
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# Load your trained model (replace with your actual model)
model = LogisticRegression()



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


# Load your feature extraction (replace with your actual vectorizer)
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english')


x_train_feature = feature_extraction.fit_transform(x_train)
x_test_feature = feature_extraction.transform(x_test)

#convert y train and test as int

y_train= y_train.astype('int')
y_test= y_test.astype('int')

#training logistic regression model

model.fit(x_train_feature,y_train)

#predication on training data
predict_on_train_data=model.predict(x_train_feature)
accuracy_on_training_data=accuracy_score(y_train,predict_on_train_data)

print(accuracy_on_training_data)

#predication test data

predict_on_test_data=model.predict(x_test_feature)
accuracy_on_testing_data=accuracy_score(y_test,predict_on_test_data)

print(accuracy_on_testing_data)


# Create the main window
root = tk.Tk()
root.title("Spam Detection Prediction")
root.geometry("1080x800")
heading_label = tk.Label(root, text="Spam Detection Project", font=("Helvetica", 36, "bold"))
heading_label.pack(pady=20)

heading_label = tk.Label(root, text="Using Logistic Regression", font=("Helvetica", 25, "bold"))
heading_label.pack(pady=20)

heading_label = tk.Label(root, text=f"Model Accuracy on Training data: {accuracy_on_training_data}", font=("Helvetica", 20, "bold"))
heading_label.pack(pady=20)

heading_label = tk.Label(root, text=f"Model Accuracy on testing data data: {accuracy_on_testing_data}", font=("Helvetica", 20, "bold"))
heading_label.pack(pady=20)

heading_label = tk.Label(root, text="Enter SMS/Mail :", font=("Helvetica", 30, "bold"),bg="cyan")
heading_label.pack(pady=20)
# Create a large text area for user input
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10, width=50, font=("Helvetica", 14))
text_area.pack(padx=20, pady=20)

# Create a button to trigger prediction
def predict_spam():
    user_input = text_area.get("1.0", "end-1c")  # Get input from the text area
    input_features = feature_extraction.transform([user_input])
    prediction = model.predict(input_features)
    result_label.config(text=f"Predicted value: {'Spam' if prediction[0] == 0 else 'Ham'}")

predict_button = tk.Button(root, text="Predict", command=predict_spam, bg="#FF5733", fg="white", font=("Helvetica", 16))
predict_button.pack()

# Create a label to display the predicted value
result_label = tk.Label(root, font=("Helvetica", 14, "italic"), fg="#333333")
result_label.pack()

root.mainloop()

