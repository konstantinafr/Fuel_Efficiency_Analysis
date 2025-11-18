import seaborn as sns
dataset=sns.load_dataset("mpg")
dataset.head()
dataset.info()
dataset.dropna(subset=["horsepower"], inplace=True)
dataset.drop(["origin", "name"], axis=1, inplace=True)
import matplotlib.pyplot as plt
dataset.hist(bins=50, figsize=(12,8))
plt.show()
datasetcopy=dataset.copy()
import numpy as np
from sklearn.preprocessing import FunctionTransformer
log_transformer=FunctionTransformer(np.log, inverse_func=np.exp)
datasetcopy["displacement"]=log_transformer.transform(datasetcopy[["displacement"]])
datasetcopy["horsepower"]=log_transformer.transform(datasetcopy[["horsepower"]])
datasetcopy["weight"]=log_transformer.transform(datasetcopy[["weight"]])
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
cat_encoder=OneHotEncoder(sparse_output=False)
encoded=cat_encoder.fit_transform(datasetcopy[["cylinders"]])
encoded_df=pd.DataFrame(encoded, columns=cat_encoder.get_feature_names_out(["cylinders"]), index=datasetcopy.index)
datasetcopy=datasetcopy.drop("cylinders", axis=1)
datasetcopy=pd.concat([datasetcopy,encoded_df], axis=1)
X=datasetcopy.drop("mpg", axis=1)
y=datasetcopy["mpg"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
pipeline=Pipeline([
    ("standardize", StandardScaler()),
    ("model", LinearRegression())
])
pipeline.fit(X_train, y_train)
final_predictions=pipeline.predict(X_test)
from sklearn.metrics import r2_score, mean_absolute_error
final_r2=r2_score(y_test, final_predictions)
final_mae=mean_absolute_error(y_test, final_predictions)
print(final_r2)
print(final_mae)
plt.scatter(y_test, final_predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
from sklearn.model_selection import cross_val_score
crossr2=cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=10)
pd.Series(crossr2).describe()
crossmae=-cross_val_score(pipeline, X_train, y_train, scoring='neg_mean_absolute_error', cv=10)
pd.Series(crossmae).describe()
