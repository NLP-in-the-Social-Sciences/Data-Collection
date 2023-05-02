#Remove any numbers, but not words that contain numbers in them. 
stories = [[token for token in data if not token.isnumeric()]for data in stories]

#Remove words that contain only one charcter. 
stories= [[token for token in data if len(token) > 1]for data in stories]

lemmatizer = WordNetLemmatizer() # Lemmatize the documents.#Alternative

#Funtion to lemmatize, apply bigrams and tokenize the given data set.
def process_texts(texts):
   texts = [[word for word in line if word not in stops] for line in texts] #tokenize the data
   texts = [[lemmatizer.lemmatize(token) for token in doc] for doc in texts]
   return texts

data_words = process_texts(list_of_sentences_data)
data_words

#Deaccent and remove some basic stopwords with gensim. 
list_of_sentences_data = []
for i in list_of_sentences:
  list_of_sentences_data.append(gensim.utils.simple_preprocess(i, deacc= True)

# Remove rare and common tokens.

# Create a dictionary representation of the documents.
dictionary = Dictionary(train_texts)
print(dictionary)

# Filter out words that occur more than 50% of the documents.
#dictionary.filter_extremes(no_above=0.)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(text) for text in train_texts]

#See how mnay tokens and documents we have to train on. 
print(Number of unique tokens: %d % len(dictionary))
#print(Number of unique tokens: %d % len(dictionary))
print(Number of documents: %d % len(corpus))