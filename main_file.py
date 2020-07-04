import pandas as pd
import pickle

df = pd.read_excel("Book1.xlsx")


from sklearn.model_selection import train_test_split

X = df.drop('salary', axis = 1)
y = df['salary']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)

ypred = lr.predict(X_test)


pickle.dump(lr, open('model1.pkl','wb'))

model = pickle.load(open('model1.pkl','rb'))
