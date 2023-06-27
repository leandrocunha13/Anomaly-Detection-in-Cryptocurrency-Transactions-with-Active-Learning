from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from Elliptic_Dataset_Preprocessing import preprocessing_pipeline
from Supervised_Experiments import run_supervised_baseline_experiment
from Visualizations import draw_metrics_by_threshold


models_dict = {'RForest': RandomForestClassifier(), 'XGBoost': XGBClassifier(), 'Logistic Regression': LogisticRegression(max_iter=10000)}

train_data, test_data = preprocessing_pipeline(last_train_timestep=34, last_test_timestep=49, only_labeled = True)

#run classifiers and get metrics
supervised_stats_df = run_supervised_baseline_experiment(models_dict, X_train, y_train, X_test, y_test)

draw_metrics_by_threshold(models, X_train, y_train, X_test, y_test)