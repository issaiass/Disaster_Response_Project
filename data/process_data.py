
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import sys

def load_data(message_filepath, categories_filepath):
    """Load dataset
    
    Input:
    mesage_filepath: string.  Name of the messages dataset.
    categories_filepath: string. Name of the categories dataset.
       
    Output:
    df: dataframe. Contains the dataframe merged of message and categories
    """    
    # load datasets
    messages = pd.read_csv(message_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories)
    return df


def clean_data(df):
    """Clean the data
    
    Input:
    df: dataframe. Startin dataframe to clean.
       
    Output:
    df: dataframe. The preprocessed dataframe of the 36 categories
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # extract list of new columns from category names
    category_colnames = pd.Series(row.str.slice(0, -2))
    # rename the columns of `categories`
    categories.columns = category_colnames
    # delete the indexes of related-2
    related2ix = categories[categories.related == 'related-2'].index
    categories.drop(related2ix, axis=0, inplace = True)    
    # convert column values to 1s and 0s
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x.split('-')[1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # remove/drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filepath):
    """Save data 
    
    Input:
    df: dataframe. The cleaned dataframe to pass to a database.
    database_filepath: string. Name of the database filepath.
       
    Output:
    None
    """     
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


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