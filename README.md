# FreeCodeCamp-ML for Everybody

In this repository, I am following the FreeCodeCamp learning module for Machine Learning for Everybody with Kylie Ying (Thank you for aiding me, this practical guide for intermediate-advanced level solidified theoretical knowledge into practicality), The video can be found here: https://www.youtube.com/watch?v=i_LwzRVP7bg&t=6386s&ab_channel=freeCodeCamp.org

## ML Models Analyzed:
- Classification:
* Logistic Regression. 'from sklearn.linear_model import LogisticRegression'
* KNeighborsClassifier. 'from sklearn.neighbors import KNeighBorsClassifier'
* Naive Bayes. : 'from sklearn.naive_bayes import GaussianNB'
* Support Vector Machines. 'from sklearn.svm import SVC'
* Neural Networks: Using 'tf.keras.Sequential()', 'tf.keras.layers.Dense()', 'tf.keras.layers.Dropout()'.

- Regression:

## Datasets:
The datasets we used for the different notebooks are the following: 
* MAGIC GAMMA TELESCOPE, Link: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

## Notebooks:
### 1. dcc-MAGIC-example (Can be run in Google Collab):
   In this Jupyter notebook, we read the data from the `sample_data` folder using `Path` from `pathlib`.
   - Assigned the proper columns to the data: "fLength", "fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class".
   - Read the data using `pd.DataFrame` from `pandas`.
   - Plotted the histograms between features, and target variable: `class` (binary classification problem between gamma or hadron particles).
   - Performed a train, val, test split using `np.split()` on the dataframe.
   - Using `StandardScaler` from `sklearn.preprocessing`, z-score normalized each of the features for better ML fitting models
   - I imported the modules `confusion_matrix` and `classification_report` to obtain ML metrics for precision, recall, and F1 score.
   - Use 'KNeighBorsClassifier' from `sklearn.neighbors` with k = 5, and fit the model.
   - Using `GaussianNB` from `sklearn.naive_bayes` fit the model.
   - Using `LogisticRegression` from `sklearn.linear_model` fit the model.
   - Using `SVC` with kernel function `rbf` fit the model.
   - Using tensorflow as `tf`, created a `tf.Sequential` simple architecture of three dense layers and two dropout layers, with last one being a sigmoid function
   - Created a `train_model` function that uses `nn_model.compile()` and `nn_model.fit()` with certain number of: num_nodes, dropout prob, lr, batch size and epochs.
   - iterated in many nested for loops the best possibilities to simulate real world to select best model with lowest validation loss
   - Obtained the final metrics of this model using `classification_report`.
