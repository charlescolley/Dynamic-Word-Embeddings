import scipy.sparse as sp
from time import clock
from math import sqrt
from sys import argv
import matplotlib.pylab as plt
import numpy as np
import cProfile
import os
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
import timeit as t
from time import gmtime, strftime

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
  #memory_assess(display=False, file_path=None)
  multiprocessing_test()

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

  profile = True
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
      if edge_count > 100:
        break
      if edge_count % update_frequency == 0:
        print "{}% complete, {} edges read in"\
          .format((edge_count/float(total_edge_count))*100,
                  edge_count)


  used_indices = \
    {value: key for key, value in new_indices.iteritems() if value < max_words}

  if profile:
    pr.disable()
    pr.print_stats(sort='time')

  return pmi, used_indices

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
        neighbors = k_nearest_neighbors(final_location, k, embedding, indices,
                                        use_embedding=True)
        print "closest words were"
        for neighbor in neighbors:
          print neighbor

def test_tensorflow():
  years = [2000,2001]
  iterations = 1000
  lambda1 = -1.0   # U regularizer
  lambda2 = -1.0   # B regularizer
  d = 150
  batch_size = 1000
  cwd = os.getcwd()
  slices = []
  # check if places for tf_embeddings exist
  path = os.path.join(cwd, 'tf_embedding')
  if not os.path.exists(path):
    os.makedirs(path)

  #check if places for tf_board existt
  path = os.path.join(cwd, 'tf_board')
  if not os.path.exists(path):
    os.makedirs(path)


  for year in years:
    pattern = re.compile("[\w]*PMI_" + str(year) + ".")
    files = os.listdir(os.getcwd())
    file = filter(lambda x: re.match(pattern, x), files)[0]
    print file
    name, _ = file.split('.')

    PMI, _ = read_in_pmi(file, display_progress=True)
    #PMI = sp.random(100, 100, format='dok')

    slices.append(PMI)

  name = name + "_iterations_" + str(iterations) + \
         "_lambda1_" + str(lambda1) + \
         "_lambda2_" + str(lambda2) + \
         "_batch_size_" + str(batch_size) + \
         "_dimensions_" + str(d) + "_"
  embedding_algo_start_time = clock()
  U_res,B = w2v.tf_random_batch_process(P,lambda1,d, batch_size,iterations,
                            results_file = name)
  run_time = clock() - embedding_algo_start_time

  #save the embeddings
  np.save("tf_embedding/" + name + "tfU.npy", U_res)
  np.save("tf_embedding/" + name + "tfB.npy", B)

  print "saved embeddings"
  #save the parameters
  parameters = {'year':year, 'iterations':iterations, 'lambda1':lambda1,
                'lambda2':lambda2,'dimensions':d, 'run_time':run_time,
                 'batch_size':batch_size}
  with open("tf_embedding/" + name + 'tfParams.pickle', 'wb') as handle:
    pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print "saved parameters"


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
                     +"tSNE,svdU, svdVt, tfU, tfV\n")
    if type == "tSNE":
      subfolder = "/tSNE/"
      postfix = "svdU_TSNE."
      get_type = False
    elif type == 'svdU':
      subfolder = "svd/"
      postfix = "svdU."
      get_type = False
    elif type == 'svdVt':
      subfolder = "svd/"
      postfix = "svdVt."
      get_type = False
    elif type == 'tfU':
      subfolder = "tf_embedding/"
      postfix = ".+tfU."
      get_type = False
    elif type == 'tfV':
      subfolder = "tf_embedding/"
      postfix = ".+tfV."
      get_type = False
    else:
      print "invalid embedding choice"

  get_year = True
  while get_year:
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
  embedding = np.load(subfolder + file)

  if type == "tfU":
    if (raw_input("load core tensor? enter [y] to include B in embedding\n")
        =='y'):
      B_file = list(file) #loads the B tensor TODO: Make this more robust
      B_file[-5] = 'B'
      core_tensor = np.load(subfolder + "".join(B_file))
      vals, vecs = np.linalg.eigh(core_tensor)
      print vals
      embedding = np.dot(embedding, map(lambda x: sqrt(x),vals) * vecs)

  normalize(embedding)
  n = embedding.shape[0]
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


def test_threads_speed():
  slices = ['wordPairPMI_2000.csv','wordPairPMI_2001.csv']
  def f():
    PMI1,_= read_in_pmi(slices[0],max_words=10,display_progress=True)
    PMI2, _ = read_in_pmi(slices[1], max_words=10, display_progress=True)

  def g():
    PMI_k = load_tensor_slices(slices)

  time1 = t.timeit(f)
  time2 = t.timeit(g)

  print "t1: ",time1," t2:", time2

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

def multiprocessing_test():


  jobs = []
  cores =  psutil.cpu_count(False)
  nnz_count = 4
  random_c = np.random.rand(cores*nnz_count) + np.random.rand(
    nnz_count*cores)*1j

  for i in range(cores):
    print "core {} should get: {}"\
      .format(i,random_c[nnz_count*i:nnz_count*(i+1)])

  shared_mem = RawArray(c_double,random_c.view(np.float64))
  print random_c.view(np.float64)
  for i in range(cores):
    p = mp.Process(target=ps.process_func, name=i+1,args=(shared_mem,
                                                          nnz_count,))
    jobs.append(p)
    p.start()

  for i in range(len(jobs)):
    jobs[i].join()

  for elem in shared_mem:
    print elem


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
