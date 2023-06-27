from config import  features_path, classes_path, edges_path
import pandas as pd

def merge_classes_and_edges(classes, edges, only_labeled = True) : # Merge classes and edges
    
    edges_classes = edges.merge(classes, left_on='txId1', right_on='txId', how='left') # Get the class for txId1
    edges_classes = edges_classes.rename(columns={'class': 'class1'}) # Rename class column
    edges_classes = edges_classes.drop(columns=['txId']) # Drop unnecessary columns
    
    edges_classes = edges_classes.merge(classes, left_on='txId2', right_on='txId', how='left') # Get the class for txId2
    edges_classes = edges_classes.rename(columns={'class': 'class2'}) # Rename class column
    edges_classes = edges_classes.drop(columns=['txId']) # Drop unnecessary columns
    
    if(only_labeled):
        edges_classes = edges_classes[(edges_classes['class1'] != -1) & (edges_classes['class2'] != -1)] # Selects only the labeled transactions
     
    return edges_classes


def merge_classes_and_features(classes, features, only_labeled = True):
    
    """
    Merges the classes and features datasets and removes transactions with unknown labels.

    Args:
    - classes: a pandas DataFrame containing the transaction IDs and corresponding labels.
    - features: a pandas DataFrame containing the transaction features.

    Returns:
    - A pandas DataFrame with the labeled transactions, including their features and labels.
    """
    classes_features = pd.merge(features, classes, left_on='txId', right_on='txId', how='left') 
    
    if(only_labeled):
        
        classes_features = classes_features.loc[(classes_features['class'] != -1)] # Selects only the labeled transactions
    
    return classes_features


def read_data():
    
    """
    Reads in the datasets and preprocesses them.

    Args:
    - features_path: a string containing the file path of the transaction features dataset.
    - classes_path: a string containing the file path of the transaction classes dataset.
    - edges_path: a string containing the file path of the transaction edges dataset.

    Returns:
    - A tuple of pandas DataFrames containing the preprocessed transaction features, classes, and edges datasets.
    
    """
    features = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)
    edges = pd.read_csv(edges_path)

    features = rename_features(features)
    classes = labels_to_int(classes)

    return features, classes, edges

def labels_to_int(classes):
    """
    Replaces string labels with integers in the classes dataset.

    Args:
    - classes: a pandas DataFrame containing the transaction IDs and corresponding labels.

    Returns:
    - A pandas DataFrame with the labels replaced with integers.
    """
    classes['class'] = classes['class'].replace({'1': 1, '2': 0, 'unknown' : -1})
    
    return classes


def rename_features(features):
    """
    Renames the columns of the features dataset.

    Args:
    - features: a pandas DataFrame containing the transaction features.

    Returns:
    - A pandas DataFrame with the column names updated.
    
    """
    col_names1 = {'0': 'txId', 1: "timestep"}
    col_names2 = {str(ii+2): "localfeature" + str(ii+1) for ii in range(93)}
    col_names3 = {str(ii+95): "aggregatefeature" + str(ii+1) for ii in range(72)}
    
    col_names = dict(col_names1, **col_names2, **col_names3 )
    col_names = {int(jj): item_kk for jj,item_kk in col_names.items()}
    
    features = features.rename(columns=col_names)

    return features

def get_data(only_labeled = True):
    
    """
    Reads in the datasets, preprocesses them, and returns the labeled transactions.

    Args:
    - features_path: a string containing the file path of the transaction features dataset.
    - classes_path: a string containing the file path of the transaction classes dataset.
    - edges_path: a string containing the file path of the transaction edges dataset.

    Returns:
    - A pandas DataFrame with the labeled transactions, including their features and labels.
    
    """

    #read datasets
    (df_features, df_classes, df_edges) = read_data()
    
    #merge features with classes and remove unknown transactions
    df_class_feature = merge_classes_and_features(df_classes,df_features, only_labeled) 

    return df_class_feature

def preprocessing_pipeline(features_path, classes_path, edges_path, last_train_timestep, last_test_timestep, only_labeled = True):
    
    #get data
    df = get_data(features_path, classes_path, edges_path, only_labeled)
    
    #training data
    train_data = df.loc[df['timestep'] <= last_train_timestep]
    
    #test data
    test_data = df.loc[(df['timestep'] > last_train_timestep) & (df['timestep'] <= last_test_timestep)]
    
    return (train_data, test_data)


def build_graph(features, classes, edges, only_labeled = True) :
    
    print("Building graph")
    
    # Create an empty graph
    graph = nx.Graph()
    
    # Merge classes and features (because we need the node id, class and timestep)
    classes_features = merge_classes_and_features(classes, features, only_labeled)
    
    # Add nodes
    nodes_list = classes_features['txId'].tolist()
    graph.add_nodes_from(nodes_list)
    
    # iterate over the rows of the edges DataFrame
    edge_list = []
    for index, row in edges.iterrows():
        edge_list.append((row['txId1'], row['txId2']))

    # add the edges to the graph
    graph.add_edges_from(edge_list)
    
    return graph


def split_data(train_data, test_data):
    
    # Split data into training and test sets
    X_train = train_data.drop(columns=['class', 'txId'])
    y_train = train_data['class']
    X_test = test_data.drop(columns=['class', 'txId'])
    y_test = test_data['class']

    return X_train, y_train, X_test, y_test