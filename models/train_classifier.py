# Import packages
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    
    '''
    Load disaster response data from SQLite database.

    Args:
    database_filepath: String. Filepath to database containing disaster response dataset.

    Returns:
    X: Messages data from disaster response dataset.
    Y: Categorisation data from disaster response dataset.
    category_names: List. Categorisation variables from disaster response dataset.
    '''

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', con=engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    
    '''
    Clean and tokenise disaster messages data.

    Args:
    text: String. Raw message text from disaster response data.

    Returns:
    tokens: List. Cleaned and tokenised message text from disaster response data.
    '''

    # Get list of all urls using regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    # Replace each url in text string with blank string
    for url in detected_urls:
        text = text.replace(url, " ")
    
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize and remove stopwords and leading/trailing white spaces
    stop_words = stopwords.words("english")
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    return tokens


def build_model():
    
    '''
    Implement machine learning pipeline and optimise parameters using grid search.
    
    Returns:
    model: Supervised machine learning pipeline and classifier.'''

    # Build model using pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Use grid search to optimise model parameters
    parameters = {
        'clf__estimator__n_estimators': [20, 50, 100]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Predict disaster categories for messages in testing dataset and evaluate model performance using sklearn's classification report.

    Args:
    model: Trained supervised machine learning pipeline and classifier.
    X_test: Messages data from disaster response test dataset.
    Y_test: Categorisation data from disaster response test dataset.
    category_names: List. Categorisation variables from disaster response dataset.
    '''

    # Predict on test data
    Y_pred = model.predict(X_test)

    # Evaluate metrics for each set of labels
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    
    '''
    Export model as a pickle file.
    
    Args:
    model: Trained supervised machine learning pipeline and classifier.
    model_filepath: String. Filepath for pickle file of trained model.
    '''

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