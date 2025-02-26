import pandas as pd
import numpy as np
import shap
import sklearn
import plotly.express as px
import plotly.graph_objects as go
import random
import math
from aif360.sklearn.metrics import equal_opportunity_difference, statistical_parity_difference,average_odds_difference, disparate_impact_ratio, between_group_generalized_entropy_error

def normalize_perColumn(df):
    return df.apply(lambda x: round((x - x.min())/(x.max() - x.min()), 2), axis=0)



def meanAbsSHAP(df):
  df = df.abs()
  return pd.DataFrame(df.mean())


def plot_svs_bar(d):
  fig = px.bar(d, y=0, x=d.index)
            # title="Default: various text sizes, positions and angles")
  fig.update_layout(yaxis_range=[-2, 2])
  # fig.update_traces(height=1)
  # margin=dict(l=0, r=0, t=10000, b=100)
  fig.update_layout()
  fig.show()






def split_df_upper_lower_half(df, percent):
  N = len(df)
  half_N = math.ceil(N*percent)
  df_upper = df.iloc[0 : half_N].reset_index(drop=True)
  df_lower = df.iloc[half_N: N].reset_index(drop=True)
  return df_upper, df_lower



def biased_swap_multi(df, advantaged_group, disadvantaged_group, p):
  '''
  from top to bottom,
  if nearby candidates' score difference is within a threshold, and from different groups,
  then we flip an unfair coin, if head,
  the advantaged group candidate can "steal" the higher score
  '''
  random.seed(10)
  # print('score swap data here')
  # print(df)
  swap_count = 0
  N = len(df)
  full_iter = N-2
  half_iter = math.ceil(full_iter * p)
  for j in range(half_iter, -1, -1):
    if df.loc[j, 'protected_attr'] == disadvantaged_group or j==0:
      for i in range(j,full_iter):
        row0 = i
        row1 = i + 1
        if (df.loc[row0, 'protected_attr'] == disadvantaged_group) \
        and (df.loc[row1, 'protected_attr'] == advantaged_group):
          df[['protected_attr','index']] = swap_rows(df[['protected_attr','index']], row0, row1)
          swap_count = swap_count + 1
          # print(f'after {swap_count} score swap data here')
          # print(df)
      print(f'swapping happened {swap_count} times')
  return df

def biased_swap(df, p):
  '''
  from top to bottom,
  if nearby candidates' score difference is within a threshold, and from different groups,
  then we flip an unfair coin, if head,
  the advantaged group candidate can "steal" the higher score
  '''
  random.seed(10)
  swap_count = 0
  len_df = len(df)
  print('score swap data here')
  # print(df[0:30])
  for row0, row1 in zip(range(0, len_df-1) , range(1, len_df)):
    if (df.loc[row0, 'protected_attr'] == 0) \
    and (df.loc[row1, 'protected_attr'] == 1) \
    and (random.random() <= p):
      df[['protected_attr','index']] = swap_rows(df[['protected_attr','index']], row0, row1)
      swap_count = swap_count + 1
      # print(f'after {swap_count} score swap data here')
      # print(df[0:30])
  print(f'swapping happened {swap_count} times')
  return df


def biased_swap_ga(df, p):
  '''
  swap to put younger and male above
  '''
  random.seed(10)
  # print('score swap data here')
  # print(df)
  swap_count = 0
  len_df = len(df)
  for row0, row1 in zip(range(0, len_df-1) , range(1, len_df)):
    if (df.loc[row0, 'protected_attr'] == 1) \
    and (df.loc[row1, 'protected_attr'] == 0):
      continue
    if (df.loc[row0, 'protected_attr'] == 0) \
    and (df.loc[row1, 'protected_attr'] == 1):
      df[['protected_attr_2', 'protected_attr','index']] = swap_rows(df[['protected_attr_2', 'protected_attr','index']], row0, row1)
      swap_count = swap_count + 1
      continue
    if (df.loc[row0, 'protected_attr_2'] > df.loc[row1, 'protected_attr_2']):
      df[['protected_attr_2', 'protected_attr','index']] = swap_rows(df[['protected_attr_2', 'protected_attr','index']], row0, row1)
      swap_count = swap_count + 1
  print(f'swapping happened {swap_count} times')
  return df



def swap_rows(df, i1, i2):
    # d = df.copy
    a, b = df.iloc[i1, :], df.iloc[i2, :]
    df.iloc[i1, :], df.iloc[i2, :] = b, a
    return df



# def _biased_swap(df, advantaged_group, disadvantaged_group, score_diff_thres, unfair_coin_prob):
#   '''
#   from top to bottom,
#   if nearby candidates' score difference is within a threshold, and from different groups,
#   then we flip an unfair coin, if head,
#   the advantaged group candidate can "steal" the higher score
#   '''
#   swap_count = 0
#   len_df = len(df)
#   for row0, row1 in zip(range(0, len_df-1) , range(1, len_df)):
#     # print(row0, row1)
#     # score = ddf[['score']]
#     if (df.loc[row0, 'protected_attr'] == disadvantaged_group) \
#     and (df.loc[row1, 'protected_attr'] == advantaged_group) \
#     and ((df.loc[row0, 's']) - (df.loc[row1, 's']) <= score_diff_thres) \
#     and (random.random() <= unfair_coin_prob):
#       df[['protected_attr','index']] = swap_rows(df[['protected_attr','index']].copy(), row0, row1)
#       swap_count = swap_count + 1
#   print(f'swapping happened {swap_count} times')
#   return df




####### dominance attack helper functions for single and two protected features
def biased_group_top(df):
  '''
  disconnect the data and the score column
  sort the data by the protected_feature column, descending order to put protected_feature=1 on top
  '''
  id_0s = df[df['protected_attr'] == 0]['index'].values.tolist()
  id_1s = df[df['protected_attr'] == 1]['index'].values.tolist()
  id_1s.extend(id_0s)
  df['index'] = id_1s
  return df


# def biased_top(df):
#   '''
#   disconnect the data and the score column
#   sort the data by the protected_feature column, descending order to put protected_feature=1 on top
#   '''
#   d = df[['index', 'protected_attr']].copy()
#   d = d.sort_values(by=['protected_attr'], ascending = True) # the younger people on top, smaller age number on top
#   df['index'] = d['index'].values
#   return df


def biased_group_top_ga(df):
  '''
  disconnect the data and the score column
  sort the data by the protected_feature column, descending order to put protected_feature=1 on top
  '''

  d0 = df[df['protected_attr'] == 0][['index', 'protected_attr_2']].copy()
  d1 = df[df['protected_attr'] == 1][['index', 'protected_attr_2']].copy()

  id_0s = d0.sort_values(by='protected_attr_2')['index'].values.tolist()# the younger people on top, smaller age number on top
  id_1s = d1.sort_values(by='protected_attr_2')['index'].values.tolist() # the younger people on top, smaller age number on top

  id_1s.extend(id_0s)
  df['index'] = id_1s
  return df
########



def biased_mixing(df,  p):
  '''
  disconnect the data and the score column
  get 2 column of ids protected_feature=0 and protected_feature=1
  flip a unfair coin to merge the two columns to 1 column new_ids, each time pop the top item in the id_0s and id_1s
  combine new_ids with the score,
  sort by new_ids and return score
  '''
  # score = df['s'].values
  # print(score)
  id_0s = df[df['protected_attr'] == 0]['index'].values.tolist()
  id_1s = df[df['protected_attr'] == 1]['index'].values.tolist()
  y_0s = df[df['protected_attr'] == 0]['s'].values.tolist()
  y_1s = df[df['protected_attr'] == 1]['s'].values.tolist()

  random.seed(10)

  ids_new = []
  while len(id_0s) != 0 and len(id_1s) != 0:
    if random.random() <= p:
      ids_new.append(id_1s.pop(0))
      y_1s.pop(0)
    elif y_0s[0] <= y_1s[0]:
      ids_new.append(id_1s.pop(0))
      y_1s.pop(0)
    else:
      ids_new.append(id_0s.pop(0))
      y_0s.pop(0)
  if len(id_0s) != 0:
    ids_new.extend(id_0s)
  else:
    ids_new.extend(id_1s)
  df['index'] = ids_new
  return df


def biased_mixing_ga(df, p):
  '''
  disconnect the data and the score column
  get 2 column of ids research=0 and research=1
  flip a unfair coin to merge the two columns to 1 column new_ids, each time pop the top item in the id_0s and id_1s
  combine new_ids with the score,
  sort by new_ids and return score
  '''
  # score = df['s'].values
  # print(score)
  id_0s = df[df['protected_attr'] == 0].sort_values(by='protected_attr_2')['index'].values.tolist()
  id_1s = df[df['protected_attr'] == 1].sort_values(by='protected_attr_2')['index'].values.tolist()
  y_0s = df[df['protected_attr'] == 0]['s'].values.tolist()
  y_1s = df[df['protected_attr'] == 1]['s'].values.tolist()
  random.seed(10)

  ids_new = []
  while len(id_0s) != 0 and len(id_1s) != 0:
    if random.random() <= p:
      ids_new.append(id_1s.pop(0))
      y_1s.pop(0)
    elif y_0s[0] <= y_1s[0]:
      ids_new.append(id_1s.pop(0))
      y_1s.pop(0)
    else:
      ids_new.append(id_0s.pop(0))
      y_0s.pop(0)
  if len(id_0s) != 0:
    ids_new.extend(id_0s)
  else:
    ids_new.extend(id_1s)
  df['index'] = ids_new
  return df


