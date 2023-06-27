from Supervised_Experiments import supervised_prediction, calculate_metrics
from Visualizations import plot_TSNE_projection

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


train_data, test_data = preprocessing_pipeline(last_train_timestep=34, last_test_timestep=49, only_labeled = True)
X_train, y_train, X_test, y_test = split_data(train_data = train_data, test_data = test_data)

#Plot t-SNE projections

#plot true labels
reducer = TSNE(n_components=2, init='pca', learning_rate='auto', n_iter=10000)
embedding = reducer.fit_transform(X_test)

embedding_df = pd.DataFrame(embedding, columns=('dim_0', 'dim_1'))
embedding_df['class'] = y_test.tolist()
embedding_df['class'] = embedding_df['class'].replace({1: 'Illicit', 0: 'Licit'})

plot_TSNE_projection(title='True labels', embedding_df=embedding_df, hue_on='class', fontsize=13, labelsize=15,
                     palette=['green', 'red'])


#plot the Random Forest predicted labels
model_rf = RandomForestClassifier()
probabilities_rf = supervised_prediction(X_train, y_train, X_test, model_rf, runs=1)
(_,_, _, max_threshold_rf) = calculate_metrics(y_test, probabilities_rf)
predictions_rf = (probabilities_rf[:, 1] > max_threshold_rf).astype(int)

reducer = TSNE(n_components=2, init='pca', learning_rate='auto', n_iter=10000)
embedding_rf = reducer.fit_transform(X_test)

embedding_rf_df = pd.DataFrame(embedding_rf, columns=('dim_0', 'dim_1'))
embedding_rf_df['prediction'] = ['Illicit' if pred == 1 else 'Licit' for pred in
                              predictions_rf]


plot_TSNE_projection(title='Random Forest predictions', embedding_df=embedding_rf_df, hue_on='prediction', fontsize=13, labelsize=15,
                     palette=['green', 'red'])


#plot the XGBoost predicted labels
model_xgb = XGBClassifier()
probabilities_xgb = supervised_prediction(X_train, y_train, X_test, model_xgb, runs=1)
(_,_, _, max_threshold_xgb) = calculate_metrics(y_test, probabilities_xgb)
predictions_xgb = (probabilities_xgb[:, 1] > max_threshold_xgb).astype(int)

reducer = TSNE(n_components=2, init='pca', learning_rate='auto', n_iter=10000)
embedding_xgb = reducer.fit_transform(X_test)

embedding_xgb_df = pd.DataFrame(embedding_xgb, columns=('dim_0', 'dim_1'))
embedding_xgb_df['prediction'] = ['Illicit' if pred == 1 else 'Licit' for pred in
                              predictions_xgb]


plot_TSNE_projection(title='XGBoost predictions', embedding_df=embedding_xgb_df, hue_on='prediction', fontsize=13, labelsize=15,
                     palette=['green', 'red'])


#plot the Logistic Regression predicted labels
model_lr = LogisticRegression(max_iter=10000)
probabilities_lr = supervised_prediction(X_train, y_train, X_test, model_lr, runs=1)
(_,_, _, max_threshold_lr) = calculate_metrics(y_test, probabilities_lr)
predictions_lr = (probabilities_lr[:, 1] > max_threshold_lr).astype(int)

reducer = TSNE(n_components=2, init='pca', learning_rate='auto', n_iter=10000)
embedding_lr = reducer.fit_transform(X_test)

embedding_lr_df = pd.DataFrame(embedding_lr, columns=('dim_0', 'dim_1'))
embedding_lr_df['prediction'] = ['Illicit' if pred == 1 else 'Licit' for pred in
                              predictions_lr]


plot_TSNE_projection(title='Logistic Regression predictions', embedding_df=embedding_lr_df, hue_on='prediction', fontsize=13, labelsize=15,
                     palette=['green', 'red'])