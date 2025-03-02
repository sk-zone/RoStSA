import pandas as pd
from utils import *

def dominanceAttack(x, f):
  df = pd.DataFrame(x) ## the order need to be retained in this df so the score or score after swaping can be aligned with input
  df.columns = [*df.columns[:-1], 'protected_attr']
  df['s'] = f(x)
  df = df.sort_values(by = ['s'], ascending=False)
  df['index'] = df.index # save the index to track the corresponding attributes after score swapping
  df_attrs = df.drop(['s'], axis=1)
  df_score_swap = df[['s', 'protected_attr', 'index']].reset_index(drop=True)

#   advantaged_group = np.max(np.unique(x[:, -1]))
#   disadvantaged_group = np.min(np.unique(x[:, -1]))

  df_score_swaped = biased_group_top(df_score_swap)
  df_merged = pd.merge(df_attrs, df_score_swaped, on='index')
  df_merged = df_merged.sort_values(by=['index'])
  return df_merged['s'].values