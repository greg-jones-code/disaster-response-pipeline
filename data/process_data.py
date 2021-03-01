# Import packages
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
    Load and combine disaster message and categorisation data.
    
    Args:
    messages_filepath: String. Filepath to csv file containing disaster messages data.
    categories_filepath: String. Filepath to csv file containing disaster categorisation data.

    Returns:
    df: Dataframe containing raw disaster messages and categories.
    '''

    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = messages.merge(categories, how='left', on='id')

    return df


def clean_data(df):
    
    '''
    Extract and clean category data, split values into separate category columns and merge back into the combined dataset.
    
    Args:
    df: Dataframe containing raw disaster messages and categories.
    
    Returns:
    df: Dataframe containing cleaned disaster messages and categories.
    '''
    
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # Select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # Extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:len(x) - 2])

    # Rename the columns of categories
    categories.columns = category_colnames

    # Convert category values to binary
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    # Check that each category column only contains 0 and 1 values
    # Remove rows that contain values other than 0 and 1
    for column in categories:
        for val in categories[column].unique():
            if val in [0, 1]:
                pass
            else:
                categories.drop(categories.loc[categories[column]==val].index, inplace=True)
    
    # Drop the original categories column from the original dataframe
    df.drop('categories', axis=1, inplace=True)

    # Concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1, join='inner', sort=False)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filepath):
    
    '''
    Save cleaned dataset into an SQLite database.
    
    Args:
    df: Dataframe containing cleaned disaster messages and categories.
    database_filepath: String. Filepath for SQLite database.
    '''

    # Create a database engine and save dataset
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()