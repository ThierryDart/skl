#
# ScikitLearn : getting started 
# https://scikit-learn.org/stable/getting_started.html


print("---------------- RandomForestClassifier ---------------- ")
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],  # 2 samples, 3 features
     [11, 12, 13]]
y = [0, 1]  # classes of each sample
print("fit",clf.fit(X, y))

print("predict",clf.predict(X))  # predict classes of the training data
print("predict",clf.predict([[4, 5, 6], [14, 15, 16]]))  # predict classes of new data


print("---------------- Transformers and pre-processors ---------------- ")
from sklearn.preprocessing import StandardScaler
X = [[0, 15],
     [1, -10]]
print(StandardScaler().fit(X).transform(X))


print("---------------- Pipelines: chaining pre-processors and estimators ---------------- ")
# We load the Iris dataset, split it into train and test sets, and compute the accuracy score of a pipeline on the test data:
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create a pipeline object
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=0)
)

# load the iris dataset and split it into train and test sets
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit the whole pipeline
print("fit",pipe.fit(X_train, y_train)) 


# we can now use it like any other estimator
print("accuracy",accuracy_score(pipe.predict(X_test), y_test))


print("---------------- Model evaluation ---------------- ")
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0)
lr = LinearRegression()

result = cross_validate(lr, X, y) # defaults to 5-fold CV
print("result",result)
print("test_score",result['test_score'])  # r_squared score is high because dataset is easy




