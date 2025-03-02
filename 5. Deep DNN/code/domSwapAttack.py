import pandas as pd
from utils import *


def DomNoneAttack_g(x):

  df_score_swap = preprocess(x)

  df_score_swap_upper, df_score_swap_lower = split_df_upper_lower_half(df_score_swap, 0.5)
  df_score_swapped_upper = biased_group_top(df_score_swap_upper)
  df_score_swapped_lower = df_score_swap_lower

  df_score_swapped = pd.concat([df_score_swapped_upper, df_score_swapped_lower], axis=0)
  return df_score_swapped.sort_values(by=['index'])['s'].values


def DomNoneAttack_ga(x):

  df_score_swap = preprocess(x)

  df_score_swap_upper, df_score_swap_lower = split_df_upper_lower_half(df_score_swap, 0.5)
  df_score_swapped_upper = biased_group_top_ga(df_score_swap_upper)
  df_score_swapped_lower = df_score_swap_lower

  df_score_swapped = pd.concat([df_score_swapped_upper, df_score_swapped_lower], axis=0)
  return df_score_swapped.sort_values(by=['index'])['s'].values


def DomSwapAttack_g(x):

  df_score_swap = preprocess(x)

  df_score_swap_upper, df_score_swap_lower = split_df_upper_lower_half(df_score_swap, 0.5)
  df_score_swapped_upper = biased_group_top(df_score_swap_upper)
  df_score_swapped_lower = biased_swap(df_score_swap_lower, 1)

  df_score_swapped = pd.concat([df_score_swapped_upper, df_score_swapped_lower], axis=0)
  return df_score_swapped.sort_values(by=['index'])['s'].values


def DomSwapAttack_ga(x):

  df_score_swap = preprocess(x)

  df_score_swap_upper, df_score_swap_lower = split_df_upper_lower_half(df_score_swap, 0.5)
  df_score_swapped_upper = biased_group_top_ga(df_score_swap_upper)
  df_score_swapped_lower = biased_swap_ga(df_score_swap_lower, 1)

  df_score_swapped = pd.concat([df_score_swapped_upper, df_score_swapped_lower], axis=0)
  return df_score_swapped.sort_values(by=['index'])['s'].values








# def DomNoneAttack_a(x, f):

#   df_score_swap = preprocess(x, f)

#   df_score_swap_upper, df_score_swap_lower = split_df_upper_lower_half(df_score_swap, 0.5)
#   df_score_swapped_upper = biased_top(df_score_swap_upper)
#   df_score_swapped_lower = df_score_swap_lower

#   df_score_swapped = pd.concat([df_score_swapped_upper, df_score_swapped_lower], axis=0)
#   return df_score_swapped.sort_values(by=['index'])['s'].values



