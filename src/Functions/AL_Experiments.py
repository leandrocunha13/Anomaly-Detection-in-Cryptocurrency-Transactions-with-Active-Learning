from General_Functions import run_classifiers, get_mean_f1_and_std
from Visualizations import plot_learning_curves
from Elliptic_Dataset_Preprocessing import split_data

import pandas as pd

def get_al_pools(train_data):

    labeled_pool = train_data.drop(train_data.index)
    
    unlabeled_pool = train_data
    
    unlabeled_pool = unlabeled_pool.reset_index(drop=True)
    
    return (labeled_pool, unlabeled_pool)

def assess_al_setup_performance(model, warmup_query, hot_query, model_name, labeled_pool, test_data, al_stats_df, labeled_pool_sizes):
    
    for size in labeled_pool_sizes:
        labeled_samples = labeled_pool.head(size)
        
        if(len(labeled_samples[labeled_samples['class']==1]) > 0):
        
            X_train, y_train, X_test, y_test = split_data(train_data = labeled_samples, test_data = test_data)

            avg_f1, _, _, _, _ = run_classifiers(X_train, y_train, X_test, y_test, model)

            al_stats_df.loc[len(al_stats_df)] = [warmup_query, hot_query, model_name, len(labeled_samples), avg_f1.round(2)]

    return al_stats_df

def run_al_pools_pipeline(unlabeled_pool, labeled_pool, scores,  batch_size = 50, largest_values = True):
    
    #select transactions to query based on score
    query_samples = select_transactions_to_query(unlabeled_pool = unlabeled_pool, scores = scores, 
                                                 batch_size = batch_size, largest_values = largest_values)
    
    #update AL pools
    unlabeled_pool, labeled_pool = update_al_pools(unlabeled_pool = unlabeled_pool, labeled_pool = labeled_pool, 
                                                   query_samples = query_samples)
    
    return (unlabeled_pool, labeled_pool)

def update_al_pools(unlabeled_pool, labeled_pool, query_samples):
    
    #add queried transactions to the labeled pool
    labeled_pool = add_transaction_to_labeled_pool(labeled_pool =  labeled_pool, query_samples = query_samples)
    
    #drop queried transactions from the unlabeled pool
    unlabeled_pool = drop_queried_transactions(unlabeled_pool = unlabeled_pool, query_samples = query_samples)
    
    return (unlabeled_pool, labeled_pool)

def drop_queried_transactions(unlabeled_pool, query_samples):
    
    indexes = query_samples.index
    unlabeled_pool.drop(indexes, inplace = True)
    unlabeled_pool = unlabeled_pool.reset_index(drop=True)
    
    return unlabeled_pool

def add_transaction_to_labeled_pool(labeled_pool, query_samples):
   
    #pass the query samples to the labeled pool
    labeled_pool = pd.concat([labeled_pool, query_samples])
    labeled_pool = labeled_pool.reset_index(drop=True)
    
    return labeled_pool


def run_setup(runs, warmup_policy, hot_policy, threshold , num_iterations, batch_size):
        
    iteration = 0
        
    (labeled_pool, unlabeled_pool) = get_al_pools(train_data = train_data)
    
    while(iteration<num_iterations):
        iteration += 1
        
        if(iteration <= threshold or len(labeled_pool[labeled_pool['class']==1]) == 0):
            #split data
            (X_train, y_train, X_pool, y_pool) = split_data(train_data=unlabeled_pool,
                                                                    test_data = unlabeled_pool)
            query_idx = warmup_policy.query(X_train = X_pool, X_test = None, batch_size = batch_size, compute_on = 'train')
            query_samples = unlabeled_pool.loc[query_idx]
            unlabeled_pool, labeled_pool = update_al_pools(unlabeled_pool = unlabeled_pool, labeled_pool = labeled_pool,
                                                        query_samples = query_samples)

        else:
            #split data
            (X_train, y_train, X_pool, y_pool) = split_data(train_data=labeled_pool,
                                                                    unlabeled_pool = unlabeled_pool)
            query_idx = hot_policy.query(X_train, y_train, X_pool, batch_size)
            query_samples = unlabeled_pool.loc[query_idx]
            unlabeled_pool, labeled_pool = update_al_pools(unlabeled_pool = unlabeled_pool, labeled_pool = labeled_pool,
                                                        query_samples = query_samples)

    return unlabeled_pool, labeled_pool


def AL_experiment(runs, supervised_classifiers, warmup_policies, hot_policies, train_data, test_data, threshold, batch_size, total_budget):
    
    al_stats_df = pd.DataFrame(columns=['warm-up learner', 'hot-learner', 'supervised classifier', 'labeled pool size', 'f1-score'])
    
    for model_name, supervised_classifier in supervised_classifiers.items():
        
        for warmup_query, warmup_policy in warmup_policies.items():

            for hot_query, hot_policy in hot_policies.items():
                
                al_stats_df = pd.DataFrame(columns=['warm-up learner', 'hot-learner', 'supervised classifier', 'labeled pool size', 'f1-score'])
                
                np.random.seed(8347658)
                
                for i in range(runs):

                    print('run ', i+1, ' AL setup for ', warmup_query, ' and ', hot_query, ' with ', model_name)

                    unlabeled_pool, labeled_pool = run_setup(runs = runs, warmup_policy = warmup_policy, hot_policy = hot_policy,
                                                         threshold = threshold, num_iterations=total_budget, batch_size=batch_size)

                    labeled_pool_sizes = list(range(50, 3050, 50))

                    al_stats_df = assess_al_setup_performance(supervised_classifier, warmup_query, hot_query, model_name, labeled_pool, 
                                                test_data, al_stats_df, labeled_pool_sizes)
                    
                    plot_learning_curves(al_stats_df, alpha=0.2, figsize=(10, 8))
    
    al_stats_df = get_mean_f1_and_std(al_stats_df)
    
    return al_stats_df

def get_k_hop_neighbors(labeled_pool, unlabeled_pool, k_hops, only_labeled = True):

    #read the data and build the graph
    features, classes, edges = read_data()
    graph = build_graph(features, classes, edges, only_labeled)

    #get illicit transaction ids
    transaction_ids = get_ilicit_ids(labeled_pool)
    transactions_list = []

    for id in transaction_ids:

        nodes_list = np.array([node for node in nx.single_source_shortest_path_length(graph, id, cutoff=k_hops).keys() if node != id])
        transactions_list = np.concatenate([transactions_list, nodes_list])

    df = unlabeled_pool[unlabeled_pool['txId'].isin(transactions_list)] 
    
    #avoid duplicate rows
    high_priority_transactions = df.drop_duplicates()
  
    return high_priority_transactions 


def get_ilicit_ids(labeled_pool):

    illicit_ids = np.array(labeled_pool.loc[labeled_pool['class']==1]['txId'])

    return illicit_ids


def get_mean_f1_and_std(df):
    
    #Group by the specified columns and calculate mean and std of f1
    grouped = df.groupby(['warm-up learner', 'hot-learner', 'supervised classifier', 'labeled pool size'])['f1-score'].agg(['mean', 'std']).round(2)

    # Add the new columns to the original dataframe
    df = df.merge(grouped, on=['warm-up learner', 'hot-learner', 'supervised classifier', 'labeled pool size'], suffixes=['', '_grouped'])

    # Drop the original f1 column
    df = df.drop('f1-score', axis=1)
    
    df = df.drop_duplicates(subset=['warm-up learner', 'hot-learner', 'supervised classifier', 'labeled pool size'])
    
    return df