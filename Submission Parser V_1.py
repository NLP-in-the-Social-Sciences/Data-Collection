from psaw import PushshiftAPI
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from pathlib import Path
##  Add if time field is important
# from datetime import datetime, timezone, timedelta 

def submissionParser(keyword, subreddit): 

    api = PushshiftAPI()
    subs = api.search_submissions(
        subreddit = subreddit,
        q=keyword,
        filter = [ 'author', 'selftext', 'url', 'permalink', 'subreddit', 'title'],
        # metadata = "false", 
        # max_results_per_request= 500
        )

    # Set display
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)

    df = pd.DataFrame([thing.d_ for thing in subs])
     
    filepath = Path('Data Dumb/Reddit/Submissions/'+keyword +'.csv')
    filepath.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(filepath)

def main(): 

    Sub_list = ["college"]
    with open('Reddit Search Keywords/Manual Search Keywords.txt', 'r') as infile_1, open('Reddit Search Keywords\SVO Search Keywords.txt', 'r') as infile_2: 
    #    for subreddit in Sub_list:
    #     for keyword in infile_1:
    #         submissionParser(keyword, subreddit)

    #     for keyword in infile_2: 
    #         submissionParser(keyword, subreddit)
        print(infile_1[1])
    
    