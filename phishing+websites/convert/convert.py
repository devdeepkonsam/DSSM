import pandas as pd # type: ignore
from scipy.io import arff # type: ignore


data, meta = arff.loadarff('Training Dataset.arff')


df = pd.DataFrame(data)


df.to_csv('output.csv', index=False)
