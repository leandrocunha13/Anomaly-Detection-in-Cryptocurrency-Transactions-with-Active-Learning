from Elliptic_Dataset_Preprocessing import read_data, build_graph, split_data
from Visualizations import plot_f1_by_timestep

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve


def run_classifiers(X_train, y_train, X_test, y_test, model):

    #predict probabilities
    scores = supervised_prediction(X_train, y_train, X_test, model)

    #calculate metrics
    (f1, precision, recall, thresholds) = calculate_metrics(y_test, scores)
    
    return (f1, precision, recall, scores, thresholds)

def supervised_prediction(X_train, y_train, X_test, model, runs=1):

    y_scores = []

    for a in range(runs):  #Run each model five times and save their predictions
        random_state = np.random.randint(0, 10000)
        params = {"random_state": random_state} 
        model.set_params(**params) #set a different random_state after each iteration
        model.fit(X_train.values, y_train.values)
        scores = model.predict_proba(X_test.values) #collect probabilities
        y_scores.append(scores)

    return y_scores


def calculate_metrics(y_test, y_scores):

    f1_scores = []
    recall_scores = []
    precision_scores = []
    max_thresholds = []

    for y_score in y_scores: 

        #computes the maximum f1_scores for each run
        precision, recall, thresholds = precision_recall_curve(y_test, y_score[:, 1])
        scores = [f1_score(y_test, y_score[:,1] > threshold, pos_label = 1) for threshold in thresholds]
        max_f1_index = np.argmax(scores)
        max_f1_threshold = thresholds[max_f1_index]
        max_f1_score = scores[max_f1_index]  

        #get the best precision and recall, corresponding to the maximum f1-score threshold
        best_precision = precision[max_f1_index + 1]
        best_recall = recall[max_f1_index + 1]
        
        f1_scores.append(max_f1_score)
        recall_scores.append(best_recall)
        precision_scores.append(best_precision)
        max_thresholds.append(max_f1_threshold)
    
    avg_f1 = np.mean(f1_scores)  #mean f1-score for each model
    avg_precision = np.mean(precision_scores) #mean precision for each model
    avg_recall = np.mean(recall_scores) #mean recall for each model
  
    return (avg_f1, avg_precision, avg_recall, max_thresholds)


def get_metrics_by_timestep(model_name, X_test, y_test, scores, thresholds, f1_ts_dict):
    
    #metrics by timestep
    f1_ts = f1_per_timestep(X_test, y_test, scores, thresholds)
    f1_ts_dict[model_name] = f1_ts
    
    return f1_ts_dict


def f1_per_timestep(X_test, y_test, y_scores, thresholds):

    last_train_time_step = min(X_test['timestep']) - 1
    last_time_step = max(X_test['timestep'])
    scores_by_ts = []
    
    i=0

    #itera as predictions de cada modelo ao longo das 5 iteracoes
    for y_score in y_scores:

      model_scores = []
      all_model_scores = []
      for time_step in range(last_train_time_step + 1, last_time_step + 1):

          id = np.flatnonzero(X_test['timestep'] == time_step)
          y_true = y_test.iloc[id]

          y__illicit_score = y_score[:, 1]
          y_pred_ts = [y__illicit_score[i] for i in id]

          score = f1_score(y_true.astype('int'), y_pred_ts > thresholds[i], pos_label = 1)

          model_scores.append(score)
    
      scores_by_ts.append(model_scores) #guarda o f1-score de cada uma das previsoes do modelo ao longo das 5 iteracoes
      i+=1
    
    avg_f1 = np.array([np.mean([f1_scores[i] for f1_scores in scores_by_ts]) for i in range(15)]) #calcula a media de f1 scores para todos os timesteps para cada um dos modelos

    return avg_f1


def run_supervised_baseline_experiment(models_dict, train_data, test_data):
    
    columns_ = ['model', 'f1-score', 'precision', 'recall']
    supervised_stats_df = pd.DataFrame(columns=columns_)

    X_train, y_train, X_test, y_test = split_data(train_data = train_data, test_data = test_data)
    
    f1_ts_dict = {}
    
    np.random.seed(8347658)
    
    for model_name, model in models_dict.items():
        print("Starting model: ", model_name)
        
        #run the three supervised classifiers
        (f1, precision, recall, scores, thresholds) = run_classifiers(X_train, y_train, X_test, y_test, model)
        
        supervised_stats_df.loc[len(supervised_stats_df)] = [model_name, f1.round(2), precision.round(2), recall.round(2)]
        
        #metrics by timestep
        f1_ts_dict = get_metrics_by_timestep(model_name, X_test, y_test, scores, thresholds, f1_ts_dict)
    
    plot_f1_by_timestep(f1_ts_dict)
    
    return supervised_stats_df
    
