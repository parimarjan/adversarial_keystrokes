import numpy as np
import scipy.sparse as sp
from sklearn.metrics import euclidean_distances
from sklearn.utils.extmath import row_norms
from itertools import combinations, imap
import collections

# With no knowledge of a user's features, come up with intelligent seeds to try
# to authenticate as them.

# Include data from all but the given user.
# Will need to change n_samples value for other datasets...
def exclude(user_X, user, n_samples):
  X = []
  for other_user in user_X:
    if other_user != user:
      X += user_X[other_user][:n_samples].tolist()

  return np.array(X)

def serwadda(X): 
  # how many columns/features are there.
  n = X.shape[1]
  # Takes first sample from every key and calculates its mean and std.
  mean = np.mean(X, axis=0)
  std = np.std(X, axis=0)
  indices = range(n)


  # Find larger and larger combinations of the indices, and set those indices to
  # one. Note that this essentially only performs the first step of Serwadda's
  # algorithm, but even this is enough for the number of seeds we want. Up to 3
  # values that are 2 std off are more likely than 1 value that is 4 std off, so
  # we can get a few thousand guesses from this elementary version.
  for r in range(n + 1):
    for j, comb in enumerate(combinations(indices, r)):
      yield mean + 2 * std * _gen_val(comb, n)
      yield mean - 2 * std * _gen_val(comb, n)

# Seeds based on Serwadda et al's paper.
# def serwadda(X):
  # n = X.shape[1]
  # mean = np.mean(X, axis=0)
  # std = np.std(X, axis=0)
  # indices = range(n)
  # # Find larger and larger combinations of the indices, and set those indices to
  # # one. Note that this essentially only performs the first step of Serwadda's
  # # algorithm, but even this is enough for the number of seeds we want. Up to 3
  # # values that are 2 std off are more likely than 1 value that is 4 std off, so
  # # we can get a few thousand guesses from this elementary version.
  # for r in range(n + 1):
    # for comb in combinations(indices, r):
      # yield mean + 2 * std * _gen_val(comb, n)
      # yield mean - 2 * std * _gen_val(comb, n)

def _gen_val(comb, n):
  # comb is the elements in the array whose value is being perturbed by
  # means +/-2*std 
  # The rest stay 0.
  value = [0] * n
  for c in comb:
    value[c] = 1
  return value

def uniq_to_full_features(uniq):
  PR_N_FEATURES = 11
  pr = uniq[:PR_N_FEATURES].tolist()
  rp = uniq[PR_N_FEATURES:].tolist()
  pp = []
  for i in range(len(rp)):
    pp.append(pr[i] + rp[i])
  rr = []
  for i in range(len(rp)):
    rr.append(rp[i] + pr[i + 1])
  return np.array(pp + pr + rp)

def _add_dependent_features(result):
    '''
    @results: Hold-Key Interval Times - Hold - ... etc format
    Will return a feature vector where the second feature is a sum of Hold
    + key interval times.
    '''
    total_len = len(result) / 2 + len(result)
    features = [None]*total_len

    for i, r in enumerate(result):
        keystroke = i / 2 
        feature = i % 2     # 0 or 1
        # index in features vector: keystroke*3 + feature
        index = keystroke * 3 + feature
        if feature == 1:  
            features[index] = result[i] + result[i-1]
            features[index+1] = r
        else: 
            features[index] = r
    
    # self._sanity_check_cmu_features([features])
    return features

# An iterator yielding seeds using k-means++. Based on scikit-learn's
# implementation.
def kpp(X, start_mean=True):
  n_samples, n_features = X.shape
  x_squared_norms = row_norms(X, squared=True)
  assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

  # Set the number of local seeding trials if none is given
  # This is what Arthur/Vassilvitskii tried, but did not report
  # specific results for other than mentioning in the conclusion
  # that it helped.
  n_local_trials = 2 + int(np.log(n_samples))

  random_state = np.random.RandomState()
  if start_mean:
    start = np.mean(X, axis=0)
  else:
    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
      start = X[center_id].toarray()
    else:
      start = X[center_id]
  yield start

  # Initialize list of closest distances and calculate current potential
  closest_dist_sq = euclidean_distances(
    [start], X, Y_norm_squared=x_squared_norms,
    squared=True)
  current_pot = closest_dist_sq.sum()

  # Pick the remaining n_clusters-1 points
  for c in range(1, n_samples):
    # Choose center candidates by sampling with probability proportional
    # to the squared distance to the closest existing center
    rand_vals = random_state.random_sample(n_local_trials) * current_pot
    candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)

    # Compute distances to center candidates
    distance_to_candidates = euclidean_distances(
      X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

    # Decide which candidate is the best
    best_candidate = None
    best_pot = None
    best_dist_sq = None
    for trial in range(n_local_trials):
      # Compute potential when including center candidate
      new_dist_sq = np.minimum(closest_dist_sq,
                   distance_to_candidates[trial])
      new_pot = new_dist_sq.sum()

      # Store result if it is the best local trial so far
      if (best_candidate is None) or (new_pot < best_pot):
        best_candidate = candidate_ids[trial]
        best_pot = new_pot
        best_dist_sq = new_dist_sq

    # Permanently add best center candidate found in local tries
    if sp.issparse(X):
      yield X[best_candidate].toarray()
    else:
      yield X[best_candidate]
    current_pot = best_pot
    closest_dist_sq = best_dist_sq

def shuffle(X, start_mean=True):
  np.random.shuffle(X)
  if start_mean:
    yield np.mean(X, axis=0)
  for x in X:
    yield x

# PN: Not completely sure why, but can only use cracker with one classifier at
# a time.

class Cracker(object):
  def __init__(self, user_X, user_Z, dataset, max_tries=100,
          uniq_to_full_features_func=uniq_to_full_features,
          num_users = None):
      #PN: Let's ignore user_Z completely since its not used in the attacks we
      # care about.  
    self.dataset = dataset
    self.user_X = user_X
    self.user_Z = user_Z
    self.max_tries = max_tries
    
    if num_users is None:
        self.num_users = float(len(self.user_X))
    else:
        self.num_users = float(num_users)

    # self.uniq_to_full_features_func = uniq_to_full_features_func
    self.uniq_to_full_features_func = _add_dependent_features

    self.num_tries = collections.defaultdict(lambda: [])

  def _seed_iters(self, user, n_samples):
    if 'cmu' in self.dataset or 'mturk' in self.dataset:
        return {
          'Serwadda': imap(lambda x: self.uniq_to_full_features_func(x), serwadda(exclude(self.user_Z, user, n_samples))),
          # 'k-means++': kpp(exclude(self.user_X, user, n_samples), False),
          'Smart k-means++': kpp(exclude(self.user_X, user, n_samples)),
          # 'Random': shuffle(exclude(self.user_X, user, n_samples), False),
          # 'Smart random': shuffle(exclude(self.user_X, user, n_samples)),
         }

    if 'android' in self.dataset:
        return {
          'Serwadda': serwadda(exclude(self.user_X, user, n_samples)),
          # 'k-means++': kpp(exclude(self.user_X, user, n_samples), False),
          'Smart k-means++': kpp(exclude(self.user_X, user, n_samples)),
          # 'Random': shuffle(exclude(self.user_X, user, n_samples), False),
          # 'Smart random': shuffle(exclude(self.user_X, user, n_samples)),
         }

# Change value of n_samples when calling crack.
# classifier is actually a user object that we are trying to break 
# - but it has a test function which has
# the same semantics.

  def crack(self, user, cl_name, n_samples=400):
    
    broke = False
    for alg, it in self._seed_iters(user.y, n_samples).iteritems():
      # Each alg is the attacker is just the string repr of the attacker.
      # will loop through all the tries returned by the algorithm call (yield)  
      for i, seed in enumerate(it):
        if i >= self.max_tries:
          # print("max tries exceeded for alg ", alg)
          break
        score = user.test(seed, cl_name)
        if score[0] > 0:
          # print("broke user ", user.y, "in tries ", i)
          self.num_tries[alg].append(i + 1)
          broke = True
          break

    return broke 

  def report(self):
    for _, num_tries in self.num_tries.iteritems():
      num_tries.sort()

    total = self.num_users    
    cumulative_users = collections.defaultdict(lambda: 0)
    percentage_cracked = collections.defaultdict(lambda: [])
    for i in range(1, self.max_tries + 1):
      for alg, num_tries in self.num_tries.iteritems():
        while cumulative_users[alg] < len(num_tries) and num_tries[cumulative_users[alg]] == i:
          cumulative_users[alg] += 1
        percentage_cracked[alg].append(cumulative_users[alg] / total)

    return percentage_cracked

  def __str__(self):
    report = self.report()
    algs = '\t'.join(report.keys())
    res = [algs]

    for i in range(self.max_tries):
      percentages = []
      for alg, percentage_cracked in report.iteritems():
        percentages.append(str(percentage_cracked[i]))
      res.append('\t'.join(percentages))
    return '\n'.join(res)

__all__ = ['Cracker']
