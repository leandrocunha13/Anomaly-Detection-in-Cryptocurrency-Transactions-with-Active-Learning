from Elliptic_Dataset_Preprocessing import preprocessing_pipeline
from AL_Experiments import AL_experiment
from AL_Policies import AnomalyDetectionQuery, EllipticEnvelopeQuery, QueryByCommittee, ExpectedModelChange, UncertaintySampling
from Custom_Logistic_Regression import Custom_LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import EllipticEnvelope
from xgboost import XGBClassifier
from pyod.models.lof import LOF
from pyod.models.iforest import IForest



### As an example there is shown the swith from an unsupervised learner to an supervised learner after the first iterations is completed


warmup_learner_dict = {
    'IF' : AnomalyDetectionQuery(IForest()),
    'LOF' : AnomalyDetectionQuery(LOF()),
    'Elliptic Envelope' : EllipticEnvelopeQuery(EllipticEnvelope(support_fraction = 1.0))
}

hot_learner_dict = {
    'QBC': QueryByCommittee(RandomForestClassifier(max_depth = 3)),
    'EMC': ExpectedModelChange(Custom_LogisticRegression(learning_rate=0.01, num_iterations=5000, tol=0.0001)),
    'Uncertainty Sampling' : UncertaintySampling(RandomForestClassifier())
}

supervised_classifiers = {
    'Random Forest' : RandomForestClassifier()
}


(train_data, test_data) = preprocessing_pipeline(last_train_timestep = 34, last_test_timestep = 49,
                                                         only_labeled = True)
al_scenario2_rf = AL_experiment(runs = 1 , supervised_classifiers = supervised_classifier, warmup_policies = warmup_learner_dict, 
                   hot_policies = hot_learner_dict, train_data = train_data, test_data = test_data, threshold = 1, 
                   batch_size=50, total_budget = 60, labeled_pool_sizes = [200,500,1000,1500,3000])


hot_learner_dict = {
    'QBC': QueryByCommittee(RandomForestClassifier(max_depth = 3)),
    'EMC': ExpectedModelChange(Custom_LogisticRegression(learning_rate=0.01, num_iterations=5000, tol=0.0001)),
    'Uncertainty Sampling' : UncertaintySampling(XGBClassifier())
}

supervised_classifiers = {
    'XGBoost' : XGBClassifier()
}


(train_data, test_data) = preprocessing_pipeline(last_train_timestep = 34, last_test_timestep = 49,
                                                         only_labeled = True)
al_scenario2_xgb = AL_experiment(runs = 1 , supervised_classifiers = supervised_classifier, warmup_policies = warmup_learner_dict, 
                   hot_policies = hot_learner_dict, train_data = train_data, test_data = test_data, threshold = 1, 
                   batch_size=50, total_budget = 60, labeled_pool_sizes = [200,500,1000,1500,3000])


hot_learner_dict = {
    'QBC': QueryByCommittee(RandomForestClassifier(max_depth = 3)),
    'EMC': ExpectedModelChange(Custom_LogisticRegression(learning_rate=0.01, num_iterations=5000, tol=0.0001)),
    'Uncertainty Sampling' : UncertaintySampling(LogisticRegression())
}

supervised_classifiers = {
    'Logistic Regression' : LogisticRegression(max_iter=10000)
}


(train_data, test_data) = preprocessing_pipeline(last_train_timestep = 34, last_test_timestep = 49,
                                                         only_labeled = True)
al_scenario2_lr = AL_experiment(runs = 1 , supervised_classifiers = supervised_classifier, warmup_policies = warmup_learner_dict, 
                   hot_policies = hot_learner_dict, train_data = train_data, test_data = test_data, threshold = 1, 
                   batch_size=50, total_budget = 60, labeled_pool_sizes = [200,500,1000,1500,3000])



combined_df = pd.concat([al_scenario2_rf, al_scenario2_xgb, al_scenario2_lr], ignore_index=True)
df_pivot = combined_df.pivot_table(index=['warm-up learner', 'hot-learner', 'supervised classifier'], columns='labeled pool size', values=['mean', 'std'])