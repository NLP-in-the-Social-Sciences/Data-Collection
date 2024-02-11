import argparse
from keybert import KeyBERT

parser = argparse.ArgumentParser(description="Extract keyphrases from a corpus of documents.")
parser.add_argument("--corpus", type=str, help="Path to the corpus of documents.")
parser.add_argument("--ngram_range", type=tuple, default=(1,2), help="The ngram range to use for the vectorizer.")
args = parser.parse_args()


kw_model = KeyBERT(model='all-MiniLM-L6-v2')


def keyphrase_extraction_vectorized(corpus, ngram_range, vectorizer, stop_words = 'english',
                        use_maxsum = True, use_mmr = True, nr_candidates = 20,top_n = 10, diversity =0.5):
    
    """
    Extracts keyphrases from a corpus of documents using a vectorizer.
    """
    
    keywords = kw_model.extract_keywords(
            docs=corpus,
            keyphrase_ngram_range=ngram_range,
            vectorizer = vectorizer, 
            stop_words ='english', 
            use_maxsum=use_maxsum, 
            use_mmr=use_mmr,
            nr_candidates = nr_candidates,
            top_n = top_n, 
            diversity=diversity            
            )
    
    return keywords

def keyphrase_extraction_vectorized(corpus, ngram_range, vectorizer, stop_words = 'english',
                        use_maxsum = True, use_mmr = True, nr_candidates = 20,top_n = 10, diversity =0.5):
        
        keywords = kw_model.extract_keywords(
                docs=corpus, 
                keyphrase_ngram_range = (1,3),
                use_maxsum=True, 
                use_mmr=True,
                stop_words ='english', 
                nr_candidates = 20,
                top_n = 10,
                diversity=0.5
        )
        return keywords
    
