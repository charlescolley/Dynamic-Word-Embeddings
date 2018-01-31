import scipy.sparse as sp
from time import clock
from math import sqrt
from cmath import sqrt
from sys import argv
import matplotlib.pylab as plt
import numpy as np
import cProfile
import os
import sys
import re
import random
from scipy.sparse.linalg import svds
from sklearn.manifold import TSNE
import word2vec as w2v
import pickle
from functools import reduce
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
from ctypes import c_double
import process_scipts as ps
import psutil
from time import gmtime, strftime, time

#from memory_profiler import profile

#Global Variables
DATA_FILE_PATH = "/mnt/hgfs/datasets/wordEmbeddings/"
FILE_NAME = DATA_FILE_PATH + "wordPairPMI_2016.csv"
INDEX_WORD_FILE = "wordIDHash_min200.csv"
SING_VAL_EXTENSION = "SingVals.npy"

UPDATE_FREQUENCY_CONSTANT = 10.0

#argv[1].split(".")[0] +

#run by global filelocation or argument if passed in
def main():
  words = ['amazon','apple','disney','obama','clinton','america','pixar',
           'gore','pie','movie','sandwich','plane','war','dollar',
                      'computer']

  #load in the U and B for the test
  U = np.load(os.path.join(DATA_FILE_PATH,
                           'flattened_svd/1990_to_2008__FSVD_U.npy'))
  B = np.load(os.path.join(DATA_FILE_PATH,
                           'flattened_svd/1990_to_2008__FSVD_B.npy'))
  with open(os.path.join(DATA_FILE_PATH,
                         'wordIDs/wordPairPMI_1990_to_2016wordIDs.pickle'),
                         'r') as handle:
    wordIDs = pickle.load(handle)
  plot_word_changes(words, U, B, wordIDs)
  #plot_core_tensor_eigenvalues(B)


'''----------------------------------------------------------------------------- 
    plot_core_tensor_eigenvalues(core_tensor)
        This function takes in a core tensor and computes/ plots the 
      eigenvalues of each frontal slice of the tensor. 
    Input:
      core_tensor - (T x d x d ndarray)
        A dense tensor which has all the temporal information of the shared 
        embedding. 
-----------------------------------------------------------------------------'''
def plot_core_tensor_eigenvalues(core_tensor):
  T = core_tensor.shape[0]

  fig, axes = plt.subplots(2,1)

  for t in range(T):
    vals,_ = np.linalg.eig(core_tensor[t])
    vals = sorted(vals)
    axes[0].plot(vals)
    axes[1].semilogy(vals)
  axes[0].set_title('linear plot')
  axes[1].set_title('log plot')

  plt.xlabel('eigenvalue i')
  plt.legend(range(T),loc='center right',bbox_to_anchor=(1.1, 1))
  plt.show()

'''-----------------------------------------------------------------------------
    plot_word_changes()
        This function creates a plot looking at distances between the initial 
      and consective time embeddings for a each word given in a list. 
      Currently the function will plot the cosine distances and the norm 
      differences for each of the words in the list.
    Input: 
      words- (list of strings)
        list of strings to be tested, each string is assumed to be a valid 
        key in the wordID parameter.
      shared_embedding - (n x d dense array)
        The d-dimensional embedding tying all the time slices together
      core_tensor - (T x d x d ndarray)
        The core tensor relating each time slice to their T_th embedding. 
      wordIDs - (dictionary)
        A dictionary linking each word to their row in the shared embedding. 
-----------------------------------------------------------------------------'''
def plot_word_changes(words,shared_embedding, core_tensor,wordIDs):
  fig, axes = plt.subplots(2, 2)

  for test_word in words:
    compared_to_t_0_CD, consecutive_CD, t_0_ND, consecutive_ND = \
      collect_embedding_distances(shared_embedding[wordIDs[test_word], :],core_tensor)

    T = len(compared_to_t_0_CD) + 1


    axes[0, 0].set_title("Cosine Distances Compared to t_0")
    axes[0, 0].set_xlabel("time slice t")
    axes[0, 0].set_ylabel("cos(xB[0],xB[t])")
    axes[0, 0].plot(range(1, T), compared_to_t_0_CD, label=test_word)

    axes[1, 0].set_title("Cosecutive Cosine Distances")
    axes[1, 0].plot(range(1, T), consecutive_CD, label=test_word)
    axes[1, 0].set_xlabel("time slice t")
    axes[1, 0].set_ylabel("cos(xB[t],xB[t+1])")

    axes[0, 1].set_title("norm difference compared to t_0")
    axes[0, 1].plot(range(1, T), t_0_ND, label=test_word)
    axes[0, 1].set_xlabel("time slice t")
    axes[0, 1].set_ylabel("norm(xB[0] - xB[t])")

    axes[1, 1].set_title("Cosecutive difference norm")
    axes[1, 1].plot(range(1, T), consecutive_ND, label=test_word)
    axes[1, 1].set_xlabel("time slice t")
    axes[1, 1].set_ylabel("norm(xB[t] - xB[t+1])")


  plt.legend(words,loc='center right',bbox_to_anchor=(1.2, 1))
  plt.show()


'''-----------------------------------------------------------------------------
    load_tSNE_word_cloud()
      This function reads in precomputed tSNE .npy files in the /tSNE folder 
      and uses the offline plotly plotter. This function is to be run in the 
      folder with all the PMI matrices, the program will access the 
      subfolders as needed. 
    Inputs:
      year - (int)
        the year of the PMI file to load and display. Throws an error if year is
        not available.  
-----------------------------------------------------------------------------'''
def load_tSNE_word_cloud(year):

  #get the indices loaded in the PMI matrix
  pattern = re.compile("[\w]*PMI_" + str(year)+ "wordIDs" + ".")
  pmi_matrix_file = filter(lambda x: re.match(pattern, x),
                           os.listdir(os.getcwd()))
  print pmi_matrix_file
  with open(pmi_matrix_file[0], 'rb') as handle:
    indices = pickle.load(handle)

  #get all the files in svd directory which match PMI_[year]svdU_TSNE
  pattern = re.compile("[\w]*PMI_"+ str(year) + "svdU_TSNE.")
  file = filter(lambda x: re.match(pattern,x),os.listdir(os.getcwd() +
                                                         "/tSNE") )
  embedding = np.load("tSNE/"+ file[0])
  print embedding
  w2v.plot_embeddings(embedding, indices)

'''-----------------------------------------------------------------------------
    sequetial_svd_tSNE()
      This funciton will process all of the truncated svd factorizations and 
      3 dimension tSNE embeddings of all the PMI matrices in the folder the 
      program is being run in. Specifically the svd factorizations are the first
      50 singulars vectors/values. The program will also create folders in the 
      directory to store the svd and tSNE away from the orignal matrix files.
    Note:
      This program will fail if there are other files in directory which 
      match the "[\w]*PMI_." regex.
-----------------------------------------------------------------------------'''
def sequential_svd_tSNE():
  cwd = os.getcwd()

  #check if places for svd components and tsne exist
  path = os.path.join(cwd, 'svd')
  if not os.path.exists(path):
    os.makedirs(path)

  path = os.path.join(cwd, 'tSNE')
  if not os.path.exists(path):
    os.makedirs(path)

  files = os.listdir(cwd)
  #only run on PMI matrices
  pattern = re.compile("[\w]*PMI_.")
  files = filter(lambda file: re.match(pattern, file), files)
  for file in files:
    name, extension = file.split('.')
    print "starting: " + name

    PMI_matrix, indices = read_in_pmi(file, max_words=100)
    print "read in:" + name

    U, sigma, Vt = svds(PMI_matrix,k=50)
    np.save("svd/" + name + "svdU.npy",U)
    np.save("svd/" + name + "svdSigma.npy", sigma)
    np.save("svd/" + name + "svdVt.npy", Vt)

    print "computed and saved matrix: " + name + "SVD"

    embedding = TSNE(n_components=3).fit_transform(U)
    np.save("tSNE/" + name + "svdU_TSNE.npy", embedding)

    print "computed and saved matrix " + name + "TSNE"

'''-----------------------------------------------------------------------------
    get_word_indices()
      This function reads in all the files in a folder and saves the words 
      loaded into the PMI matrices as a pickle file in the wordIDs folder it 
      creates.  
-----------------------------------------------------------------------------'''
def get_word_indices():
  cwd = os.getcwd()

  # check if places for wordIDs exist
  path = os.path.join(cwd, 'wordIDs')
  if not os.path.exists(path):
    os.makedirs(path)

  files = os.listdir(cwd)
  pattern = re.compile("[\w]*PMI_.")
  files = filter(lambda file: re.match(pattern, file), files)

  for file in files:
    name, extension = file.split('.')
    print "starting: " + name

    _, indices = read_in_pmi(file)
    print "read in:" + name

    with open("wordIDs/" + name +'wordIDs.pickle', 'wb') as handle:
      pickle.dump(indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print "saved" + name

'''-----------------------------------------------------------------------------
    read_in_pmi(filename, returned_scaled_count, max_words, display_progress)
      This function takes in a filename and returns a pmi matrix stored in 
      the location. The file is assumed to be formatted as 
        line ->  word_index, context_index, pmi_value
      The file then builds a scipy dok_matrix and returns it.
    Input:
      filename - (optional string) 
        location of the file.
      return_scaled_count - (optional bool)
        instead returning the pmi for a word context pair, we can return the 
        number of times that the pairs appears together in a text corpus, 
        this provides naive weights for the loss function.  
      max_words - (optional int)
        only creates the submatrix PMI[:max_words,:max_words] rather than 
        processing the entire PMI matrix.
      display_progress - (optional bool) 
        display statements updating the progress of the file load or not.
    Returns:
      pmi - (dok_matrix)
        a sparse matrix with the corresponding pmi values for each of the 
        word context pairs. 
      clean_indices - (dictionary)
        a dictionary where the keys are words and the values are the new 
        indices.
    NOTE: read_in_word_index current returns the word count over all text 
    documents, not the word count for an individual text corpus, rendering 
    the return_scaled_count option invalid
-----------------------------------------------------------------------------'''
def read_in_pmi(filename = FILE_NAME, return_scaled_count = False,
                max_words = None, display_progress = False):

  #check if scipy file exists
  scipy_filename = "scipy_files/" + filename[:-3] + 'npz'
  if os.path.isfile(scipy_filename) and max_words == None:
    print "loading in " + scipy_filename
    pmi = sp.load_npz(scipy_filename)
    pmi = pmi.todok()

    #load in word ID
    ID_filename = filename[:-4] + "wordIDs.pickle"
    used_indices = pickle.load(open("wordIDs/" + ID_filename,'r'))
    
    #invert the dictionary to be consistent with function contract
    used_indices = {value:key for key,value in used_indices.iteritems()}
  else:
    print "loading in " + filename
    f = open(filename,"r")
    f.next() # skip word, context, pmi line
    total_edge_count = 0
    clean_indices = 0
    edges = {}
    i_max = -1
    j_max = -1
    if max_words == None:
      max_words = float("inf")


    word_indices = read_in_word_index(return_scaled_count)

    '''
    the count is over all words over all times
    #compute word count TODO: VALIDATE SUM
    if return_scaled_count:
      word_count = reduce(lambda x,key: x + word_indices[key][1],word_indices)
    '''
    if display_progress:
      print 'Read in word indices'
    new_indices = {}

    profile = False
    if profile:
      pr = cProfile.Profile()
      pr.enable()

    #count the edges in the file, dimensions of PPMI matrix, and reassign indices.
    for line in f:
      edge = line.split(',')

      #reassign new indices to prevent empty submatrices in pmi
      word_ID = int(edge[0])
      context_ID = int(edge[1])

      word = word_indices[word_ID] if not return_scaled_count else\
             word_indices[word_ID][0]
      context = word_indices[context_ID] if not return_scaled_count else\
                word_indices[context_ID][0]

      if word not in new_indices:
        new_indices[word] = clean_indices
        clean_indices += 1

      if context not in new_indices:
        new_indices[context] = clean_indices
        clean_indices += 1

      if new_indices[context] < max_words and new_indices[word] < max_words:
        edge_val = np.float(edge[2])
        if return_scaled_count:
          edge_val =  np.exp(edge_val)* \
                      word_indices[word_ID][1] * word_indices[context_ID][1]

        edges[total_edge_count] = [new_indices[word], new_indices[context],
                                   edge_val]
        #check if new indices are largest row or column found
        if new_indices[word] > i_max:
          i_max = new_indices[word]

        if new_indices[context] > j_max:
          j_max = new_indices[context]

        total_edge_count += 1

    f.close()

    if display_progress:
      print "counted {} edges over {} by {} words"\
        .format(total_edge_count, i_max, j_max)


    # initialize counts for updating user as file loads
    if display_progress:
      update_frequency = total_edge_count / UPDATE_FREQUENCY_CONSTANT
      edge_count = 0

    shape = (i_max+1,j_max+1)
    #initialize sparse matrix
    pmi = sp.dok_matrix(shape)

    for i in xrange(total_edge_count):
      pmi[edges[i][0], edges[i][1]] = edges[i][2]
      if display_progress:
        edge_count += 1
        if edge_count % update_frequency == 0:
          print "{}% complete, {} edges read in"\
            .format((edge_count/float(total_edge_count))*100,
                    edge_count)


    used_indices = \
      {key:value for key, value in new_indices.iteritems() if value < max_words}

    if profile:
      pr.disable()
      pr.print_stats(sort='time')

  return pmi, used_indices

'''-----------------------------------------------------------------------------
    permute_dok_matrix(A,p)
      This function takes in a sparse scipy dok matrix and permutes both the 
      rows and the columns in time proportional to the number of non-zeros in 
      the matrix.
    Input:
      A - (n x n sparse dok matrix)
        The matrix to be permuted
      p - (n x 1 vector)
        a vector representing the permutation desired to switch all the rows 
        and columns. Note that P is expected to be a permutation of range(n).
    Returns:
      permuted_A - (n x n sparse dok matrix)
        The final matrix, which should be equivilant to A[p][:, p]
    Note:
      This algorithm runs in 2*(nnz) space, would be great to make an in 
      place algorithm. 
-----------------------------------------------------------------------------'''
def permute_dok_matrix(A,p):
  p_dict = {p_i:i for i,p_i in enumerate(p)}

  permuted_A = sp.dok_matrix(A.shape)

  for (i,j), val in A.iteritems():
    permuted_A[p_dict[i],p_dict[j]] = val

  return permuted_A

'''-----------------------------------------------------------------------------
    convert_to_scipy(years)
      This function takes in a list of years, loads in the appropriate PMI 
      matrix slices and converts the files into scipy coo sparse matrix files 
      and stores them in the appropriate folder. 
    Input:
      years - list of ints
        This is the list of years corresponding to the PMI slices, assumed 
        that if an integer is listed here, then it must exist as a file. 
-----------------------------------------------------------------------------'''
def convert_to_scipy(years):

  cwd = os.getcwd()
  #check for a numpy_file folder
  path = os.path.join(cwd, 'scipy_files')
  if not os.path.exists(path):
    os.makedirs(path)

  #load in the slices to convert
  slices , _ = get_slices(years,False)

  for t, slice in enumerate(slices):
    file_name = "scipy_files/wordPairPMI_" + str(years[t]) + ".npz"
    sp.save_npz(file_name,slice.tocoo())

'''-----------------------------------------------------------------------------
    filter_up_to_kth_largest(matrix, k)
      This function takes in a sparse dok matrix and returns a sparse csr 
      matrix with only the kth largest non-zeros in the array.
    Inputs:
      matrix - (n x m dok_sparse matrix)
        the matrix to be filtered
      k - (int)
        the the rank of the k largest element to filter by.
    Returns:
      filtered_matrix - (n x m csr_sparse matrix)
        the filterd matrix in question. 
-----------------------------------------------------------------------------'''
def filter_up_to_kth_largest(matrix, k):
  if k < matrix.nnz:
    #find kth largest element
    k_largest_nnz = np.partition(matrix.values(),-k)[-k]
    above_max = filter(lambda x: x[1] < k_largest_nnz, matrix.items())
    indices = map(lambda key_val_pair: key_val_pair[0], above_max)
    matrix.update(zip(indices, np.zeros(len(indices))))
    filtered_matrix = matrix.tocsr()
    filtered_matrix.eliminate_zeros()
    return filtered_matrix
  else:
    return matrix

'''-----------------------------------------------------------------------------
    normalize_wordIDs(P_slices, wordIDs)
        This function takes in a list of PMI matrix slices and their respective 
      index to word dictionaries and reorders and removes entries in each of 
      the PMI matrix slices such that the embeddings all share the same words 
      and are all permuted such that the ith row and columns correspond to the 
      same word for all the time slices. each time slice will remove any rows 
      and columns associated with words not in all the time slices.
    Input:
      P_slices - (list of square sparse matrices)
        the relevant PMI matrices to be normalized 
      wordIDs - (list of dictionaries)
        the dictionaries linking the words in the PMI matrices to their 
        respective indices for each time slice. Here the keys are the strings 
        and the values are the indices.   
      take_union - optional (boolean)
        an optional boolean function which will indicate whether or not to 
        take the union of all the words, or the intersection.
    Returns:
      shared_wordIDs - (dictionary)
        the final dictionary linking the indices of all the time slices to 
        the shared words. Here keys are the indices and values are the strings. 
    Note:
      Two obvious improvements will be not truncating the matrices if words 
      are not shared, but simply moving them to the "end" of the matrix and 
      letting implicit zeros pad them. Also filtering each matrix up to the 
      kth largest word may be useful functionality to improve the quality of 
      the embeddings.
-----------------------------------------------------------------------------'''
def normalize_wordIDs(P_slices, wordIDs, take_union = True):
  if take_union:

    #find the permutation to apply to each matrix slice

    shared_wordIDs = wordIDs[0] #start with the ordering of the first slice
    word_count = len(shared_wordIDs)

    T = len(wordIDs)
    for t in range(1,T):
      for key in wordIDs[t].iterkeys():
        if key not in shared_wordIDs:
          shared_wordIDs[key] = word_count
          word_count += 1

    print "finished and found {} words in the union".format(word_count)

    #apply padding
    for t in range(T):
      P_slices[t].resize((word_count,word_count))

    #permute each matrix slice
    for t in range(1,T):
      permutation = range(word_count)

      #notes where padding columns starts
      padding_index = len(wordIDs[t])

      for word,index in shared_wordIDs.iteritems():
        if word in wordIDs[t]:
          permutation[index] = wordIDs[t][word]
        else:
          permutation[index] = padding_index
          padding_index += 1


      P_slices[t] = permute_dok_matrix(P_slices[t],permutation)
      print "finished applying t={} permutation".format(t)

  else:
    #find the intersection of all the words
    common_words = set.intersection(
                     *map(lambda x: set(x),  #convert to sets
                      map(lambda x: x.keys(), wordIDs))) #get all words

    first_slice = True
    #remove all rows and columns for terms not in the common_words set
    for (t,slice) in enumerate(P_slices):

      if first_slice:
        valid_indices = map(lambda (_, value): value,
                            filter(lambda (key, _): key in common_words,
                                   wordIDs[t].iteritems()))
        #create shared_wordIDs and remap indices if on first slice
        word_count= 0
        shared_wordIDs = {}

        for (key, val) in sorted(wordIDs[0].items(), key=lambda (x, y): y):
          if key in common_words:
            shared_wordIDs[word_count] = key
            word_count += 1
        first_slice = False
        P_slices[t] = slice[valid_indices][:, valid_indices]
      else:
        #permute to align with the first slice
        permutation = []
        for i in xrange(word_count):
          permutation.append(wordIDs[t][shared_wordIDs[i]])
        # eliminate invalid rows and columns
        P_slices[t] = P_slices[t][permutation][:, permutation]


  return shared_wordIDs

'''-----------------------------------------------------------------------------
    word_trajectories(embedding)
        This function takes in an embedding from the flattened_svd method and 
      an integer and creates a plot watching how the words have shifted
    Input:
      word - (string)
        the word to plot the word trajectory for
      embedding - (n x d numpy matrix)
        The d dimensional embedding in question
      core_tessor - (d x d x T numpy ndarray)
        The core tensor relating the time slices together. 
      wordID - (dictionary)
        A dictionary which links the words in the embedding to their 
        corresponding indices in the embedding. keys are words, and values 
        are the indices. 
      k - (positive integer)
        An integer indicating the number of nearest neighbors to consider 
    Note:
-----------------------------------------------------------------------------'''
def word_trajectories(word, embedding,core_tensor,wordIDs,k):

  T = core_tensor.shape[0]
  d = core_tensor.shape[1]
  seen_neighbors = []
  nearest_neighbors = []
  word_embeddings = []
  #find all the nearest neighbors for each time slice
  for t in range(T):
    #compute the t_th embedding

    #vals, vecs = np.linalg.eig(core_tensor[t])
    # cut off round off error
    #         vals = np.array(map(lambda x: 0 if abs(x) < 1e-10 else x, vals))
    #sqrt_val = map(lambda x: sqrt(x), vals)
    #sqrt_B = np.dot(vecs, np.diag(sqrt_val))

    t_embedding = np.dot(embedding,core_tensor[t])
    normalize(t_embedding)

    #add in t_th word embedding in t_th slice to embedding list
    word_embeddings.append((word +'_'+str(t),t_embedding[wordIDs[word],:]))


    nearest_neighbors.append(k_nearest_neighbors(word,k,t_embedding,wordIDs))

    #remove neighbors already seen in the embedding
    if t != 0:
      seen_neighbors.extend(nearest_neighbors[t-1])
      nearest_neighbors[t] = \
        filter(lambda (x,_): x not in map(lambda (y,_): y,seen_neighbors),
               nearest_neighbors[t])

    #add the neighbors embeddings into a list
    for (neighbor,_) in nearest_neighbors[t]:
      word_embeddings.append((neighbor, t_embedding[wordIDs[neighbor],:]))

  #aggregate all the embeddings into a single numpy array
  unique_words = len(word_embeddings)
  final_U = np.empty((unique_words,d),dtype=np.complex)
  for i in range(unique_words):
    final_U[i,:] = word_embeddings[i][1]

  tsne_data = TSNE(2).fit_transform(final_U)
  plt.scatter(tsne_data[:,0],tsne_data[:,1])

  for i in range(unique_words):
    plt.annotate(word_embeddings[i][0], (tsne_data[i,0],tsne_data[i,1]))
  plt.show()

'''-----------------------------------------------------------------------------
    collect_embedding_distances(word_embedding, core_tensor)
      This function takes in a vector representation of a word and the core 
    tensor associated with the embedding collects different kinds of 
    distances and returns the list associating them.
    Input: 
      word_embedding- (1 x d numpy array)
        The shared word embedding to be tested
      core_tensor - (T x d x d ndarray)
        The core tensor connecting the shared embedding to the different T 
        time slices. 
    Returns: 
      compared_to_t_0_CD - (list of floats)
        a list of the cosine distances of the embedding at each time and the 
        initial time slice.
      consecutive_CD - (list of floats)
        a list of the cosine distances of the embedding at each time and the 
        next conecutive time. 
      t_0_norm_difference - (list of floats)
        a list of the 2-norms of the differences bettwen the embedding at 
        each time and first embedding.
      consectutive_norm_difference - (list of floats)
        a list of the 2-norms of the differences between the embedding at 
        each time and the next consective time.
-----------------------------------------------------------------------------'''
def collect_embedding_distances(word_embedding,core_tensor):
  T = core_tensor.shape[0]

  time_0_embedding = np.dot(word_embedding,core_tensor[0])
  time_0_embedding = time_0_embedding/np.linalg.norm(time_0_embedding)

  #compute the cosine distances
  compared_to_t_0_CD = []
  consecutive_CD = []
  consecutive_norm_difference = []
  t_0_norm_difference = []

  previous_time_embedding = time_0_embedding
  for t in range(1,T):
    #normalize the t_th embedding
    time_t_embedding = np.dot(word_embedding, core_tensor[t])
    time_t_embedding = time_t_embedding / np.linalg.norm(time_t_embedding)
    cosine_distance_1 = np.dot(time_0_embedding,time_t_embedding)
    cosine_distance_2 = np.dot(previous_time_embedding, time_t_embedding)

    norm_difference_1 = \
      np.linalg.norm(time_t_embedding - time_0_embedding)
    norm_difference_2 = \
      np.linalg.norm(time_t_embedding - previous_time_embedding)

    compared_to_t_0_CD.append(cosine_distance_1)
    consecutive_CD.append(cosine_distance_2)
    t_0_norm_difference.append(norm_difference_1)
    consecutive_norm_difference.append(norm_difference_2)

    previous_time_embedding = time_t_embedding

  return compared_to_t_0_CD, consecutive_CD, \
         t_0_norm_difference, consecutive_norm_difference

'''-----------------------------------------------------------------------------
    word_embedding_arithmetic()
      This function loads in a given word2vec embedding and will allow you to 
    use addition and subtraction to test the embedding 
        (i.e. king + woman - man = queen). 
    The function takes in the words from the user and findest the k closest 
    neighbors to final word computed. 
    Input:
      embedding (n x d matrix)
        The embedding to compute the arithmetic with. 
      indices (dictionary)
        The dictionary linking the words to their index in the embedding 
        matrix. words are keys, indices are the values. 
      k (int)
        The number of nearest neighbors to return. 
    Note: 
      Currently only supports addition and subtraction
-----------------------------------------------------------------------------'''
def word_embedding_arithmetic(embedding, indices, k):
  get_equation = True
  while get_equation:
    equation = raw_input("input embedding arithmetic \n" +
                         "embedding must be of the form \"word_1 {+/-} " +
                         "word_2 {+/-} ... {+/-} word_n\"\n" +
                         "type __quit__ to exit \n"
                         )
    if equation == "__quit__":
      get_equation = False
    else:
      words = re.findall("[\w]+",equation)
      operations = filter(lambda x: x != ' ',re.findall("[\W]", equation))
      print "words", words
      print "operations:", operations
      #check for valid words
      invalid_words = filter(lambda x: x not in indices, words)
      print "invalid words:", invalid_words
      invalid_operations = filter(lambda x: x != '+' and x != '-', operations)
      print "invalid operations",invalid_operations
      if invalid_words or invalid_operations:
        if len(invalid_words) == 1:
          print "{} is not a valid word".format(invalid_words[0])
        elif len(invalid_words) > 1:
          for word in invalid_words:
             print word
          print "are not valid words"
        if len(invalid_operations) == 1:
          print "{} is not a valid operation"
        elif len(invalid_operations) > 1:
          for operation in invalid_operations:
             print operation
          print "are not valid operations"
        print "please retype equation"
      else:
        list = zip(operations,map(lambda x: embedding[indices[x],:],words[1:]))
        reduce_embedding = lambda y,x: x[1] + y if x[0] == '+' else y - x[1]
        final_location = reduce(reduce_embedding,list,embedding[indices[words[0]],:])
        #normalize
        final_location = final_location/np.linalg.norm(final_location)
        neighbors = k_nearest_neighbors(final_location, k, embedding, indices,
                                        use_embedding=True)
        print "closest words were"
        for neighbor in neighbors:
          print neighbor

'''-----------------------------------------------------------------------------
    get_slices(years)
      This function takes in a list of years to be loaded in and finds the PMI
      files associated with each year, and aligns them properly then saves 
      the resulting index-word dictionary if the file already doesn't exist. 
      Note that this function assumes that it's being run in the correct 
      directory. 
    Input:
      years - (list of ints)
        the list of years to load in 
      normalize- (optional bool)
        a bool indicating whether or not to normalize the tensor slices. If 
        this is false, then shared_ID will return a None. 
      use_full - (optional bool)
        a bool indicating whether or not to load in the the tensor from the 
        full_aligned_tensor/ folder which has all the tensor slices 
        pre-aligned and saved in a coo sparse matrix format. Note that when 
        using this option, special consideration must be used to identify 
        words that don't exist in a particular time slice, as no matter which 
        years are passed in, the same wordID dict will be returned. If such a 
        folder doesn't exist, then it will print an error to stderr and load 
        in the files with read_in_pmi().
    Returns:
      slices - (list of sparse dok matrices)
        the list of PMI matrices associated with the desired years 
      shared_ID - (dok)
        a dictionary associating the indices in the tensor to the words of 
        the PMI matrices. keys are the indices and values are the strings.
-----------------------------------------------------------------------------'''
def get_slices(years,normalize = True, use_full = False):
  slices = []
  wordIDs = []

  #check for full_aligned_tensor/ folder
  fAT_path = os.path.join(os.getcwd(), 'full_aligned_tensor/')
  if not os.path.exists(fAT_path):
    use_full = False
    sys.stderr.write('full_aligned_tensor/ not found, setting use_full to '
                     'False.\n')

  for year in years:
    # get PMI matrix
    if use_full:
      pattern = re.compile("full_wordPairPMI_"+str(year) + ".npz")
      files = os.listdir(fAT_path)
    else:
      pattern = re.compile("[\w]*PMI_" + str(year) + ".")
      files = os.listdir(os.getcwd())

    file = filter(lambda x: re.match(pattern, x), files)[0]
    name, _ = file.split('.')

    if use_full:
      print "reading in full_wordPairPMI_{}_.npz".format(year)
      PMI = sp.load_npz(os.path.join(fAT_path,file))
    else:
      PMI, IDs = read_in_pmi(file, display_progress=True)
      wordIDs.append(IDs)

    slices.append(PMI)

  print "loaded in files"
  if normalize:
    if use_full:
      with open("wordIDs/wordPairPMI_1990_to_2016wordIDs.pickle",'r') as handle:
        shared_ID = pickle.load(handle)
    else:
      shared_ID = normalize_wordIDs(slices, wordIDs)
      print "aligned tensor slices"

      file_name = "wordIDs/wordPairPMI_" + str(years[0]) +'_to_' + \
                                           str(years[-1]) + 'wordIDs.pickle'
      #save shared word IDs if doesn't exist
      if not os.path.exists(file_name):
        # align tensor slices

        with open(file_name, 'wb') as handle:
          pickle.dump(shared_ID, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print "saved IDs"
  else:
    shared_ID = None

  return slices, shared_ID

def compute_mode_3_ft(years):
  path = os.path.join(os.getcwd(), 'fourier_transforms')
  if not os.path.exists(path):
    os.makedirs(path)

  slices, _ = get_slices(years)
  transformed_slices = w2v.mode_3_fft(slices)

  file_name = "fft_years_" + str(years[0]) +"_to_" + str(years[-1]) + ".npy"
  np.save("fourier_transorms/"+file_name,transformed_slices)
  print "saved files"

'''-----------------------------------------------------------------------------
    save_full_aligned_tensor()
        This function is a preprocessing function which should only be used 
      once in principal. The function will load in the full tensor from the csv 
      files, align them and save the resulting sparse matrices in their own 
      folder in the coo format. The benefit of this format is from now on, 
      it will be unneccessary to spend time aligning the matrices, as all the 
      tensor slices will be aligned, the draw back is that words that doen't 
      exist in a given time slice will need to be identified. 
-----------------------------------------------------------------------------'''
def save_full_aligned_tensor():

  #check for folder
  path = os.path.join(os.getcwd(), 'full_aligned_tensor')
  if not os.path.exists(path):
    os.makedirs(path)

  slices, _ = get_slices(range(1990,2017))
  year = 1990
  for slice in slices:
    filename = "full_aligned_tensor/full_wordPairPMI_" + str(year) + ".npz"
    sp.save_npz(filename,slice.tocoo())
    print "finished year:{}".format(year)
    year += 1

'''-----------------------------------------------------------------------------
-----------------------------------------------------------------------------'''
def flattened_svd_embedding(years, LO_type = None):

  #check for svd folder
  # check if places for stdout_files existt
  path = os.path.join(os.getcwd(), 'flattened_svd')
  if not os.path.exists(path):
    os.makedirs(path)

  slices, _ = get_slices(years,use_full=True)

  U, sigma,B = w2v.flattened_svd(slices,50,LO_type)

  filename_base = 'flattened_svd/'+str(years[0]) +'_to_' + str(years[1]) +'_'

  if LO_type:
    filename_base = filename_base + LO_type

  np.save(filename_base + "_FSVD_U.npy",U)
  np.save(filename_base + "_FSVD_sigma.npy",sigma)
  np.save(filename_base + "_FSVD_B.npy",B)
  print "saved files"


def hyper_param_search():
  years = [2016]
  lambda1 = .001  # U regularizer
  lambda2 = .001  # B regularizer
  d = 50
  base_batch_size = 10

  methods = ['GD', 'Ada', 'Adad', 'Adam','Nest','Momen']
  batch_size_tests = 4
  jobs = []

  for method in methods:
    for i in range(1,batch_size_tests+1):

      batch_size = base_batch_size ** i
      iterations = 10**(6 - i)
      process_name = method +"_" + str(batch_size)

      p = mp.Process(target=test_tensorflow, name=process_name,
                     args=(iterations, lambda1,lambda2,d,
                           method,batch_size,years))
      jobs.append(p)
      p.start()
      print "started process:" + process_name

  for i in range(len(jobs)):
    jobs[i].join()
    print "joined job {}".format(i)

def test_tensorflow(iterations, lambda1,lambda2,d,method,batch_size,years):

  cwd = os.getcwd()
  slices = []
  wordIDs = []
  # check if places for tf_embeddings exist
  path = os.path.join(cwd, 'tf_embedding')
  if not os.path.exists(path):
    os.makedirs(path)

  '''
  #check if places for tf_board exist
  path = os.path.join(cwd, 'tf_board')
  if not os.path.exists(path):
    os.makedirs(path)
  '''

  # check if places for stdout_files existt
  path = os.path.join(cwd, 'stdout_files')
  if not os.path.exists(path):
    os.makedirs(path)

  stdout_dup = 'stdout_files/' + mp.current_process().name + "_svd_test2.txt"
  sys.stdout = open(stdout_dup, "w")

  for year in years:
    #get PMI matrix
    pattern = re.compile("[\w]*PMI_" + str(year) + ".")
    files = os.listdir(os.getcwd())
    file = filter(lambda x: re.match(pattern, x), files)[0]
    print file
    name, _ = file.split('.')

    PMI, IDs = read_in_pmi(file, display_progress=True)
    #PMI = sp.random(100, 100, format='dok')

    slices.append(PMI)
    wordIDs.append(IDs)

  if len(years) > 1:
    year_string = "wordPairPMI_" + str(years[0]) +"_to_"+ str(years[-1])
    # align all the tensor slices
    sharedIDs = normalize_wordIDs(slices, wordIDs)

  else:
    year_string = name
    sharedIDs = IDs


  name =  year_string + "_iterations_" + str(iterations) + \
         "_lambda1_" + str(lambda1) + \
         "_lambda2_" + str(lambda2) + \
         "_batch_size_" + str(batch_size) + \
         "_dimensions_" + str(d) + \
         "_method_" + method + '_'

  embedding_algo_start_time = clock()

  U_res,B = w2v.tf_random_batch_process(slices,lambda1, lambda2,d, batch_size,\
            iterations, method)

  run_time = clock() - embedding_algo_start_time

  #save the embeddings
  np.save("tf_embedding/" + name + "tfU.npy", U_res)
  np.save("tf_embedding/" + name + "tfB.npy", B)

  print "saved embeddings"
  #save the parameters
  parameters = {'years':years, 'iterations':iterations, 'lambda1':lambda1,
                'lambda2':lambda2,'dimensions':d, 'run_time':run_time,
                 'batch_size':batch_size}
  with open("tf_embedding/" + name + 'tfParams.pickle', 'wb') as handle:
    pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print "saved parameters"

  if len(years) > 1:
    with open("wordIDs/wordPairPMI_"+ str(years[0]) +
                  '_to_' + str(years[-1]) + 'wordIDs.pickle','wb') as handle:
      pickle.dump(sharedIDs,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print "saved IDs"


'''-----------------------------------------------------------------------------
    plot_performance()
      This function loads in results from different runs of the optimizers to 
    compare the performance of different choices of hyper-parameters in the 
    experiments. The hyper parameters to be considered are 
      -batch_size
        the size of the mini-batch each iteration is run on 
      -lambda1
        the regularizer associated with the shared embedding U
      -lambda2
        the regularizer associated with the core tensor B
      -iterations
        the number of steps the optimizer is run on.
      -optimizer
        the type of line search method used to minimize the objective 
        function. Current choices are Adam, Adagrad, Adagrad-delta, 
        and Gradient Descent. Note that all of these methods are being run 
        with mini batches. 
    The files loaded in will come from any files in the tf_embedding/ folder 
    which aren't in any subfolders.
-----------------------------------------------------------------------------'''
def plot_performance():
  #find Param files to plot results for
  pattern = re.compile(".*tfParams.*")
  files = filter(lambda x: re.match(pattern,x),os.listdir('./tf_embedding'))

  parameters = []
  for file in files:
    with open(file,'r') as handle:
      param_dict = pickle.load(handle)
    # check for loss_function and gradient norm values
    try:
      param_dict['loss_func_val']
    except KeyError:
      #compute loss function for final U and B
      root_name = file[:file.find('tf')+2]
      U = np.load(root_name + "U.npy")
      B = np.load(root_name + "B.npy")

      #find years in the embedding
      start_point = file.find("PMI_")
      start_year = int(file[start_point+4: start_point+8])
      if file[start_point +8:start_point+12] == '_to_':
        end_year = int(file[start_point+ 12:start_point+16])
        years = range(start_year,end_year)
      else:
        years = [start_year]

      w2v.evaluate_embedding(U,B,param_dict['lambda1'],param_dict['lambda1'],
                             years)


'''-----------------------------------------------------------------------------
    plot_B_spectrums()
      This function is creates a plot of the eigenvalues of all the core 
      tensors produced by tensorflow. Note that this function must be run 
      in the folder containing the B matrix numpy files. 
-----------------------------------------------------------------------------'''
def plot_B_spectrums():

  #find all the tfB files
  pattern = re.compile(".*tfB.*")
  files = filter(lambda x: re.match(pattern,x),os.listdir("."))

  B_tensor_eigenvalues = []
  for B_file in files:
    B = np.load(B_file)
    slice_vals = []
    for slice in B:
      vals, _ = np.linalg.eig(np.dot(slice,slice.T))
      slice_vals.append(vals)
    B_tensor_eigenvalues.append(slice_vals)



  for slice in B_tensor_eigenvalues[-1]:
    plt.semilogy(sorted(slice))
  plt.title(files[-1])
  plt.show()


#todo: REMOVE WORDS IN INITIAL LIST FROM RESULTS
'''-----------------------------------------------------------------------------
    test_word_embedding()
      This function will load in a tSNE embedding and will query the user for 
    words in the embedding and will report the k closest neighbors to that word
    in the embedding.
    Note:
      To be run in the main folder with all the PMI matrices.
-----------------------------------------------------------------------------'''
def test_word_embedding():
  get_type = True
  while get_type:
    type = raw_input("Which embedding do you want to load?:\n"
                     +"tSNE,svdU, tfU, Fsvd\n")
    if type == "tSNE":
      subfolder = "/tSNE/"
      postfix = "svdU_TSNE."
      get_type = False
    elif type == 'svdU':
      subfolder = "svd/"
      postfix = "svdU."
      get_type = False
    elif type == 'tfU':
      subfolder = "tf_embedding/"
      postfix = ".+tfU."
      get_type = False
    elif type == "Fsvd":
      subfolder = "flattened_svd/"
      postfix = ".+FSVD_U."
      get_type = False
    else:
      print "invalid embedding choice"

  get_year = True
  while get_year:

    if subfolder == "tf_embedding/" or subfolder == "flattened_svd/":
      if subfolder == "tf_embedding/" :
        pattern = re.compile(".*tfU.*")
      else:
        pattern = re.compile(".*FSVD_U.npy")
      files = os.listdir(os.getcwd() + '/' + subfolder)
      #find all tfU files
      files = filter(lambda x: re.match(pattern,x), files)
      print "found the following files"
      for file in enumerate(files):
        print file
      get_index = True
      while(get_index):
        index = int(raw_input("Load which file?:"))
        if (index >= 0) and (index < len(files)):
          get_index = False
          get_year = False
        else:
          print "invalid index"
      file = files[index]
    else:
      year = raw_input("Which year do you want to load?:")
      pattern = re.compile("[\w]*PMI_" + year + postfix)
      files = os.listdir(os.getcwd() + '/' + subfolder)
      file = filter(lambda x: re.match(pattern,x),files )
      file_count = len(file)
      if not file:
        print "year not found, please choose from the available year"
        for f in files:
          print f
      elif file_count > 1:
        print "multiple files found please type in index to choose file"
        for i in range(file_count):
          print i, ":", file[i]
        index = int(raw_input("select index 0 - "+str(file_count-1) + '\n'))
        if index > file_count or index < 0:
          print "invalid selection"
        else:
          file = file[index]
          get_year = False
      else:
        file = file[0]
        get_year = False
      
  #load in tSNE file
  print file
  embedding = np.load(subfolder + file)

  if type == "tfU":
    if (raw_input("load core tensor? enter [y] to include B in embedding\n")
        =='y'):
      B_file = list(file) #loads the B tensor TODO: Make this more robust
      B_file[-5] = 'B'
      core_tensor = np.load(subfolder + "".join(B_file))
  elif type == "Fsvd":
    B_file = list(file)
    B_file[-5] = 'B'
    core_tensor = np.load(subfolder + "".join(B_file))

  n = embedding.shape[0]

  #need to load in multiple ID_dictionaries if loading tf_embeddings
  if type == "tfU" or type == "Fsvd":
    if type == "tfU":
      #find years associated with embedding
      start_year = file[12:16]
      end_year = file[20:24]

      #load in shared word IDs
      pattern = re.compile("[\w]*PMI_" + start_year +"_to_"+ end_year +
                         "wordIDs.pickle")
      files = os.listdir(os.getcwd() + "/wordIDs")
      IDs_file = filter(lambda x: re.match(pattern, x), files)
      print IDs_file,files
      with open("wordIDs/" + IDs_file[0], 'rb') as handle:
        indices = pickle.load(handle)
    else:
      with open("wordIDs/wordPairPMI_1990_to_2016wordIDs.pickle", 'rb') as handle:
        indices = pickle.load(handle)

    indices = {value: key for key, value in indices.iteritems()}

    load_new_year = True
    while load_new_year:
      print "choose year, enter \"quit\" to quit"
      year = raw_input("which year do you want to run? Choose from {} to {"
                       "}: ".format(start_year,end_year))
      if year =="quit":
        load_new_year = False
      else:
        #form embedding
        core_tensor_index = int(year) - int(start_year)
        if type == "Fsvd":
#         semi_def_B = w2v.make_semi_definite(core_tensor[core_tensor_index])
          semi_def_B = core_tensor[core_tensor_index]
          vals, vecs = np.linalg.eig(semi_def_B)
          #cut off round off error
#         vals = np.array(map(lambda x: 0 if abs(x) < 1e-10 else x, vals))
          sqrt_val = map(lambda x: sqrt(x),vals)
          sqrt_B = np.dot(vecs,np.diag(sqrt_val))
          new_embedding = np.dot(embedding,sqrt_B)
        else:
          new_embedding = np.dot(embedding,core_tensor[core_tensor_index])
        normalize(new_embedding)

        get_k = True
        while get_k:
          k = int(raw_input("How many nearest neighbors do you want? "))
          if k < 0 or k > n:
            print "invalid choice of k= {}, must be postive and less than {}".format(
              k, n)
          else:
            get_k = False
        word_embedding_arithmetic(new_embedding, indices, k)
  else:
    #load indices
    pattern = re.compile("[\w]*PMI_" + year+ "wordIDs" + ".")
    files =  os.listdir(os.getcwd() + "/wordIDs")
    IDs_file = filter(lambda x: re.match(pattern, x),files)
    with open("wordIDs/" + IDs_file[0], 'rb') as handle:
      indices = pickle.load(handle)
    #flip the indices
    indices = {value: key for key, value in indices.iteritems()}
    get_k = True
    while get_k:
      k = int(raw_input("How many nearest neighbors do you want? "))
      if k < 0 or k > n:
        print "invalid choice of k= {}, must be postive and less than {}".format(k,n)
      else:
        get_k = False
    normalize(embedding)
    word_embedding_arithmetic(embedding, indices, k)

'''-----------------------------------------------------------------------------
    normalize(embedding,mode)
      This function takes in an n x d numpy array and normalizes it based off of
      the mode that is passed into. The changes are all made the embedding 
      passed in, rather than making a new matrix and returning it. 
    Input:
      embedding (n x d numpy array)
        the embedding to be normalized
      mode - (int)
        the mode to normalize with respect to. Default is 1, indicating 
        normalize by the rows. 
    Note:
      This is currently implemented for a matrix input, but it may be useful 
      to expand the normalization to tensors of higher order. 
-----------------------------------------------------------------------------'''
def normalize(embedding,mode=1):
  size = embedding.shape[mode-1]
  if mode == 1:
    for i in range(size):
      embedding[i,:] = embedding[i,:]/np.linalg.norm(embedding[i,:])
  else:
    for i in range(size):
      embedding[:,i] = embedding[:, i] / np.linalg.norm(embedding[:, i])

def query_word_neighbors(embedding, indices, k):
  get_word = True
  while get_word:
    word = raw_input("Which word do you want to search for? \n" + \
                     "type __random__ for a random selection of 10 words to choose from \n" + \
                     "type __quit__ to exit\n")
    if word == "__quit__":
      get_word = False
    elif word == "__random__":
      for i in range(10):
        print random.choice(indices.keys())
    elif word not in indices.keys():
      print "{} not found, please enter another word".format(word)
    else:
      neighbors = k_nearest_neighbors(word, k, embedding, indices)

      #use max_string length to format
      max_str_len = max(map(lambda x: len(x[0]), neighbors.__iter__()))
      for neighbor in neighbors:
        print neighbor[0].rjust(max_str_len), neighbor[1]

'''-----------------------------------------------------------------------------
    k_nearest_neighbors(word, k, embedding, indices)
      This function takes in a word and number of neighbors and returns that 
      number of nearest neighbors in the given embedding. The indices are a 
      dicitonary linking the indices of the embedding to the words in 
      question. This function implements an insertion sort to build the list 
      as the number of nearest neighbors will be assumed to be many fewer 
      than the total vocabulary.
    @Params 
    Input:
      word - (string/ d- dimensional array)
        The string to search for the nearest neighbors
      k - (positive int)
        The number of closest neighbors to return
      embeddings - (n x d dense matrix)
        The given embedding of the vocabulary
      indices - (dictionary)
        a dictionary linking the indices of the embedding to the words they 
        embed. Here the keys are the words and the indices are the values.
      use_embedding (optional bool)
        a boolean indicating whether or not the word passed in is a string 
        literal or a d - dimensional array representing a position in the 
        embedding. 
    Returns: 
      list of pairs where the first element is the word, and the second 
      element is the 2 norm distance between the two words in the embedding.
-----------------------------------------------------------------------------'''
def k_nearest_neighbors(word, k, embedding, indices, use_embedding=False):
  #check for proper input
  if k < 1:
    raise ValueError("k must be positive")
  i = 0
  n = embedding.shape[0]

  if use_embedding:
    word_position = word
  else:
    word_index = indices[word]
    word_position = embedding[word_index,:]

  #invert the dictionary
  indices = {value:key for key, value in indices.iteritems()}

  to_sort = map(lambda x: (indices[x],embedding[x,:]),
                range(n) if use_embedding else
                filter(lambda x: x != word_index,range(n)))
  comes_before = lambda x,y: np.linalg.norm(x[1] - word_position) < \
                             np.linalg.norm(y[1] - word_position)
  results = partial_insertion_sort(to_sort,comes_before,k)
  return map(lambda x: (x[0],np.linalg.norm(x[1] - word_position)),results)

'''-----------------------------------------------------------------------------
    partial_insertion_sort(list,insert_before,k)
      This function takes in a list of elements, a function to compare any 
      two elements in the list and return a bool, and a integer indicating 
      the size of the sorted list to return. 
    Input:
      list - ('a list)
        a list of 'a elements
      insert_before- ('a x 'a |-> bool)
        a function which takes two of the elements in the list and returns a 
        bool if the first element should come before the second element in 
        the function's parameters.
      k - (integer)
        the size of the sorted list to return. 
    Return:
      sorted_list - ('a list)
        the list of the k top sorted elements.
-----------------------------------------------------------------------------'''
def partial_insertion_sort(list, insert_before, k):
  sorted_list = []
  length = 0
  for element in list:
    if length == 0:
      sorted_list.append(element)
      length += 1
    elif length >= k:
      for i in range(length - 1,-1,-1):
        if insert_before(element, sorted_list[i]):
          if i != length -1:
            sorted_list[i+1] = sorted_list[i]
          sorted_list[i] = element
        else:
          break
    else:
      for i in range(length -1,-1,-1):
        if insert_before(element, sorted_list[i]):
          if i == length -1:
            sorted_list.append(sorted_list[i])
            length +=1
          else:
            sorted_list[i+1] = sorted_list[i]
          sorted_list[i] = element

  return sorted_list
        
def profile_read_in_function():
  #cProfile.run('test_func()')
  cProfile.run('read_in_pmi(FILE_NAME,True)')

'''-----------------------------------------------------------------------------
    load_tensor_slices(slices)
      This function takes in a list of slices to load into a third order 
      tensor and returns a dictionary of the slices.This function uses a 
      multiprocessing pool in order to load the tensorslices in parallel.
    Input:
      slices - string list
        a list of the files to l
-----------------------------------------------------------------------------'''
def load_tensor_slices(slices):
  index = 0
  tensor_slices = []
  slice_count = len(slices)

  p = mp.Pool()

  def pool_helper(filename):
    P, _ = read_in_pmi(filename,display_progress=True)
    return P

  tensor_slices = map(pool_helper, slices)
  p.join()
  p.close()

  return tensor_slices

def process_helper(slice_index,slice_file,dictionary):
#   print "hello from thread {}".format(slice_index)
 PMI, _ = read_in_pmi(slice_file,max_words=10,display_progress=True)
 dictionary[slice_index] = PMI

#note defaults to floor in int arithmetic
def center_print(string,col_width = 80):

  print(string.rjust((col_width + len(string))/2))

'''-----------------------------------------------------------------------------
    memory_assess(display,file_path)
      This function writes a short profile of the current available process 
      in terms of the number of non-zeros in a tensor. 
    Input:
      display (optional bool) = False
        print the results to stdout
      file_path (optional str)
        Assumes a valid file path and prints results to a text file. 
-----------------------------------------------------------------------------'''
def memory_assess(display = False,file_path = None):

  scale = 1
  header = \
    "--------------------------------------------------------------------------------"
  print header
  time_string = strftime("%a, %d %b %Y %H:%M:%S ""+0000", gmtime())
  center_print(time_string)
  print header

  vm = psutil.virtual_memory()
  print "free memory:      {}".format(vm.total / scale)
  print "available memory: {}".format(vm.available / scale)
  print "free memory:      {}".format(vm.free / scale)
  print "cached memory:    {}".format(vm.cached / scale)
  print header


  if file_path:
    f = open(file_path,'a')
    f.append(header)
    f.append("log starting at {}".\
             format(time_string))
    f.append(header)

  if file_path:
    vm.available

  if file_path:
    f.close()

'''-----------------------------------------------------------------------------
    read_in_word_index()
      This function reads in the word index associated with the text corpus, 
      from the wordIDHash_min200.csv file so the embeddings can be associated 
      with the proper word rather than an integer ID. 
    Inputs:
      include_word_count - (boolean)
        a boolean indicating whether or not the word count should be included in
        as the second element in a tuple in the key value pair. 
    Returns:
      word_IDs - (dictionary)
        a dictionary which has indices as the keys and the words as the 
        associated values. If include_word_count is true, then the values are 
        2-tuples where the first element is the word, and the 2nd element is 
        the word frequency. 
-----------------------------------------------------------------------------'''
def read_in_word_index(include_word_count = False):

  f = open(INDEX_WORD_FILE, "r") # formatted as wordId, word, word count
  word_IDs = {}
  for line in f:
    word_stat = line.split(',')
    if include_word_count:
      word_IDs[int(word_stat[0])] = (word_stat[1], int(word_stat[2]))
    else:
      word_IDs[int(word_stat[0])] = word_stat[1]

  return word_IDs

'''-----------------------------------------------------------------------------
    matrix_visualization(matrix)
      This function takes in a matrix and uses plt functions to visualize the 
      non-zeros of the matrix. 
    Input:
      matrix - (n x m sparse matrix)
        the matrix to visualize.
-----------------------------------------------------------------------------'''
def matrix_visualization(matrix):
  plt.spy(matrix)
  plt.show()


if __name__ == "__main__":
 main()
