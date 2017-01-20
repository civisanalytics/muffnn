from muffnn import MLPClassifier
import pandas as pd
import numpy as np

from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y= iris.target
y=np.array(pd.get_dummies(Y))
y[0,0]=np.nan
y[60,1]=np.nan
y[120,2]=np.nan



mlp = MLPClassifier(n_epochs=20, batch_size=10)
mlp.fit(np.array(X)[np.where(np.isfinite(np.array(y)).sum(axis=1)!=0)[0]], np.array(y)[np.where(np.isfinite(np.array(y)).sum(axis=1)!=0)[0]])
print(mlp.predict_proba(np.array(X)))
