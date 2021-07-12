# import libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import FeatureUnion
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import pickle
import nltk
import sys
import re




def load_data(database_filepath):
    """Load the database to train
    
    Input:
    database_filepath: string. String that contains a database file to read containing the messages
       
    Output:
    X: series.  A set of messages of the database to train
    Y: series.  A set of labels of the database to train
    category_names: array. List of the 36 classes to classify.
    """     
    nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.message
    category_names = df.columns[4:]
    Y = df[category_names]
    return X, Y, category_names


def tokenize(text):
    """Normalize the data, make a tokenNormalize, tokenize and stem text string
    
    Input:
    text: string. String that contains a message to process
       
    Output:
    normalized: list of strings. List containing a normalized and stemmed token.
    """      
    # Converting everything to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize words
    tokens = word_tokenize(text)
    # normalization word tokens and remove stop words
    normlizer = PorterStemmer()
    stop_words = stopwords.words("english")
    normalized = [normlizer.stem(word) for word in tokens if word not in stop_words]
    return normalized


def build_model():
    """Make the model and do a grid search
    
    Input:
    None
       
    Output:
    model:  The resulting model before the training process, gridsearch set.
    """    
    pipe = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
  
    parameters = {
        'vect__ngram_range': ((1,1),(1, 2)),
        'tfidf__use_idf': [True],
        'clf__estimator__min_samples_split': [2, 4]
    }    
    
    model = GridSearchCV(pipe, param_grid=parameters, verbose=4)
    return model

def __print_metrics(gt, pred, col_names):
    """Evaluation Metrics 
    
    Input:
    gt: array. Array of dataset ground truth.
    pred: array. Array of dataset predictions.
    col_names: strig list. Name list of predicted labels.
       
    Output:
    df: dataframe. Contains the same output as classification_report
    """
    metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        acc = accuracy_score(gt[:, i], pred[:, i])
        prec = precision_score(gt[:, i], pred[:, i], average='micro')
        rec = recall_score(gt[:, i], pred[:, i], average='micro')
        f1 = f1_score(gt[:, i], pred[:, i], average='micro')
        
        metrics.append([acc, prec, rec, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    columns = ['Accuracy', 'Precision', 'Recall', 'F1']
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns=columns)
      
    return metrics_df



def evaluate_model(model, X_test, Y_test, category_names):
    """Predict and evaluate some metrics of the model
    
    Input:
    model:  The model of the current data to evaluate
    X_test:  series.  The feature values of the test set to evaluate
    Y_test:  series.  The label values of the dataset to evaluate 
    category_names:  list of strings. The array of the labels to evaluate
       
    Output:
    None.  But it displays the printed output of the metrics of the model
    """     
    predictions = model.predict(X_test)
    ground_truth = Y_test.values
    target_names = category_names #Ytest.columns.values.tolist()
    print(__print_metrics(ground_truth, predictions, target_names))

def save_model(model, model_filepath):
    """Save the final model of the disaster response dataset
    
    Input:
    model: The model to save
    model_filepath:  string.  The path of the model to save
       
    Output:
    None
    """       
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()