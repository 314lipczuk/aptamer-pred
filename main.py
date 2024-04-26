import numpy as np
import sys
from pathlib import Path
import typing

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

nucl_table = {'A':0,'C':1,"T":2,'G':3}
def read_file(file:str):
  path = Path('./data') / file
  if not path.exists(): raise Exception("Path %s doesn't exist" % path) 
  with open(path, 'r') as f:
    labels, counts, data = [],[],[]
    for line in f:
      if line.startswith(">"):
         labels.append(line[1:].replace('\n','').replace('\t',''))     
         counts.append(line.split('-')[1].replace('\n','').replace('\t',''))     
      else:
        data.append(np.array([nucl_table[n] for n in line.replace('\n','').replace('\t','')]))
    return labels, counts, np.array(data)

# First the easy way - the RBM-DC-6
labels, counts, data = read_file('s100_6th.fasta')
rbm = RBM()
