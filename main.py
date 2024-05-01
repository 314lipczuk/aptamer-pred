import numpy as np
import sys
from pathlib import Path
import typing
from matplotlib import pyplot as plt
import pickle

# for now using RBM's impl from https://github.com/jertubiana/PGM
# we could use other implementations, but this one provides us with log likelihoods, and most others
# can only do free energy in the system.
# One day I'm gonna probably try to insert my own RBM impl here, but that day is not today
rbm_impl_path = Path('./PGM') 
if not rbm_impl_path.is_dir():
  from git import Repo
  Repo.clone_from('https://github.com/jertubiana/PGM', rbm_impl_path)

sys.path.append(rbm_impl_path.as_posix() + '/source/')
sys.path.append(rbm_impl_path.as_posix() + '/utilities/')
from PGM.source.rbm import RBM

def resample(data, counts):
  assert len(data) == len(counts)
  dset_size = data.shape[0]
  total_prob = np.sum(counts)
  probabilities = np.array(counts/ total_prob, dtype=np.float64)
  cumulative_prob = np.zeros(len(probabilities), dtype=np.float64)
  current_cumul = 0
  resampled = []
  for i, p in enumerate(probabilities):
    current_cumul += p
    cumulative_prob[i] = current_cumul
  for i in range(dset_size):
    x = np.random.random()
    idx = cumulative_prob.searchsorted(x) 
    resampled.append(data[idx])
  return np.copy(np.array(resampled, dtype=np.int16))

nucl_table = {'A':0,'C':1,"T":2,'G':3}
def read_file(file:str):
  path = Path('./data') / file
  if not path.exists(): raise Exception("Path %s doesn't exist" % path) 
  with open(path, 'r') as f:
    labels, counts, data = [],[],[]
    for line in f:
      line = line.replace('\n','').replace('\t','')
      if line.startswith(">"):
         labels.append(line[1:])     
         counts.append(int(line.split('-')[1]))
      else:
        data.append(np.array([nucl_table[n] for n in line]))
    return labels, np.array(counts), np.array(data)

def main():
  print('loading data')
  # First the easy way - the RBM-DC6 (using all sequence)
  labels, counts, data = read_file('s100_6th.fasta')

  seq = np.arange(data.shape[0])
  np.random.shuffle(seq) # shuffle it, to distribute randomly
  test_size = data.shape[0] // 10 # 10% test, 90% train

  # data splice
  train_data = data[seq][test_size:] 
  test_data = data[seq][:test_size] 

  # count splice
  train_count = counts[seq][test_size:] 
  test_count = counts[seq][:test_size] 
  assert len(train_count) > len(test_count)

  print('data loaded')

  print('resampling starting')
  resampled_train = resample(train_data, train_count)
  resampled_test = resample(test_data, test_count)
  print('resampling complete')

  print('training shape now:', resampled_train.shape)
  print('testing shape now:',resampled_test.shape)

  #assert False
  print('training starting')
  rbm = RBM(visible='Potts', hidden='dReLU', n_v=train_data.shape[1], n_h=90, n_cv=4)
  rbm.fit(data=resampled_train,batch_size=500, n_iter=15,l1b=1e-2,verbose=0, vverbose=1, N_MC=10)
  print('training complete')
  with open('./trainedrbm.experiment.pkl', 'wb') as f:
    pickle.dump(rbm,f )
    print('dumped model')

  fig = plt.figure(figsize=(13, 7), constrained_layout=False)
  ax = fig.add_subplot()
  ax.hist(rbm.likelihood(resampled_train), bins=50, density=True, histtype='step', lw=2, fill=False, label="Training set")
  ax.hist(rbm.likelihood(resampled_test), bins=50, density=True, histtype='step', lw=3, fill=False, label="Test set")
  print('generated plot')

  ax.legend(fontsize=18, loc=1)
  ax.set_xlabel("Log-likelihood", fontsize=18)

  fig.savefig('figure.png')
  print('saved plot')

if __name__ == "__main__":
  main()