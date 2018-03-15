import nltk
import os
import pickle
from scipy.sparse import dok_matrix, coo_matrix, save_npz
from numpy import log

def main():
  inaugural(5)

'''-----------------------------------------------------------------------------
   inaugural(window_size)
       This function creates the log PMI matrices from each of the american 
     presidents inaugural addresses up to 2009. This data comes from the nltk 
     corpus (citations needed). This file will save all the PMI matrices in 
     the folder created. The PMI matrices all be aligned. If the corpus is 
     updated, re-run this function outside of the original folder created and it
     will download the updated corpus and overwrite the original files. 
   Input:
     window_size - (int)
       the size of the window to look for the pairwise mutual information.
-----------------------------------------------------------------------------'''
def inaugural(window_size):
  #check for a folder and create and change into it if doesn't exist
  path = os.path.join(os.getcwd(), 'inaugural')
  if not os.path.exists(path):
    os.makedirs(path)
  os.chdir(path)
  nltk.download('inaugural')

  corpus = nltk.corpus.inaugural

  #find all alpha numeric words in the dataset
  wordID = create_wordID_dict(corpus.words())

  file_basename = "inaugural_ws_{}_PMI_".format(window_size)

  #create and save all the PPMI matrices
  files = corpus.fileids()
  for file in files:
    P = PPMI(corpus,file,wordID,window_size)
    filename = file_basename + file[:4]
    save_npz(filename,P.tocoo())
    print "processed {} as {}".format(file,filename)

  #save wordID
  with open(files[0][:4] + '_' +  files[-1][:4] + '_' +
            'wordIDs.pickle', 'wb') as handle:
    pickle.dump(wordID, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''-----------------------------------------------------------------------------
   create_wordID_dict(corpus_words)
       This function takes in the words from a nltk corpus and creates a 
     dictionary linking each of the words to an index the matrix. Only words 
     that are comprised of alphanumeric values will be used (no punctuation). 
     This function is used to help align all the tensor slices created from 
     multiple text corpuses. 
   Input: 
     corpus_words - (nltk corpus reader concatonatedCorpusView)
       the result of calling .words() on a nltk corpus. This will be the list of
       words to use for creating the PPMI matrices. 
   Note: 
     May need to be generalized to take in a list of corpus words.  
-----------------------------------------------------------------------------'''
def create_wordID_dict(corpus_words):
  # find all alpha numeric words in the dataset
  words = map(lambda x: x.lower(),
              filter(lambda x: x.isalpha(), set(corpus_words)))

  # build wordID dictionary for indices
  wordID = {}
  total_word_count = 0
  for word in words:
    if word not in wordID:
      wordID[word] = total_word_count
      total_word_count += 1
  return wordID

'''-----------------------------------------------------------------------------
   PPMI(corpus)
     This function produces a PPMI matrix for a given nltk corpus passed in. 
   Input:
     corpus - (nltk corpus reader)
       The corpus to produce the PPMI matrix for
     file - (string/unicode)
       the file in the corpus to collect information for. 
     wordID - (dictionary)
       a dictionary linking each word to a given index in a matrix. Used to 
       ensure that the PPMI matrices produces for multiple corpuses are aligned.
     window_size - (int)
       the size of the window to compute the pointwise mutual information for. 
   Returns:
     PPMI_matrix - (n x n sparse dok matrix)
       a matrix corresponding to the positive pointwise mutual information of 
       each pair of words in the text corpus.   
-----------------------------------------------------------------------------'''
def PPMI(corpus,file, wordID,window_size):

  word_counts = word_count(corpus.words(file))
  words_in_corpus = len(word_counts)

  co_occurrence_matrix = co_occurrence_count(corpus.sents(file),wordID,
                                             window_size)

  #invert the wordID dictionary
  wordID = {val:key for (key,val) in wordID.iteritems()}

  PPMI_matrix = dok_matrix(co_occurrence_matrix.shape)

  for ((i,j),v) in co_occurrence_matrix.iteritems():
    word_i = wordID[i]
    word_j = wordID[j]

    new_val = log(float(v)*words_in_corpus/
                  (word_counts[word_i]*word_counts[word_j]))
    if new_val > 0:
      PPMI_matrix[i,j] = new_val


  return PPMI_matrix

'''-----------------------------------------------------------------------------
   co_occurrence_count(corpus_sentences, wordID, window_size)
       This function takes in a nltk corpus reader over the sentences and a 
     dictionary linking each lower case word in the corpus to an index in the 
     corpus and returns a scipy sparse dok matrix which has all the 
     co-occurence counts of the pairs of words for a given window size. 
   Input:
     corpus_sentences - (nltk corpus reader)
       generated by called .sents(filename) on a PlaintextCorpusReader
     wordID - (dictionary)
       dictionary linking the words to an index for the co-occurence matrix 
       to be generated. This is also used to eliminate punctuation as wordID 
       will have to process all the words in the corpus before hand. 
     window_size - (int)
       the size of the window to look over co-occurences.
   Returns:
     co_occurrence_matrix - (dok sparse matrix)
       a scipy sparse dok matrix which will hold the number of co-occurences.
-----------------------------------------------------------------------------'''
def co_occurrence_count(corpus_sentences, wordID, window_size):
  n = len(wordID)

  co_occurrence_matrix = dok_matrix((n,n),dtype=int)

  for sentence in corpus_sentences:
    sentence = map(lambda x: x.lower(),filter(lambda x: x.isalpha(),sentence))
    for window in nltk.ngrams(sentence,window_size):
      for w_i in range(window_size):
        word_i_index = wordID[window[w_i]]
        for w_j in range(w_i):
          word_j_index = wordID[window[w_j]]
          co_occurrence_matrix[word_i_index,word_j_index] += 1
          co_occurrence_matrix[word_j_index,word_i_index] += 1

  return co_occurrence_matrix


'''-----------------------------------------------------------------------------
   word_count(corpus_words)
       This function takes in a nltk corpus reader and returns a dictionary 
     with the number of instances of the word in the document. All words are 
     converted to lowercase, and any word token which fails .isalpha() is not 
     included. 
   Input:
     corpus_words - (nltk corpus reader)
       generated by calling .words(filename) on a nltk PlaintextCorpusReader 
       instance.
   Returns:
     words - (dictionary)
       a dictionary linking each lower case word to the number of times they 
       show up in the corpus.
-----------------------------------------------------------------------------'''
def word_count(corpus_words):
  words = {}


  for word in corpus_words:
    if word.isalpha():
      word = word.lower()
      if word in words:
        words[word] += 1
      else:
        words[word] = 1

  return words





if __name__ == "__main__":
  main()