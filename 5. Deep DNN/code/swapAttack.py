from utils import *



def swapAttack(x, p, f):
  df = pd.DataFrame(x) ## the order need to be retained in this df so the score or score after swaping can be aligned with input
  df.columns = [*df.columns[:-1], 'protected_attr']
  df['s'] = f(x)
  df = df.sort_values(by = ['s'], ascending=False)
  df['index'] = df.index # save the index to track the corresponding attributes after score swapping
  df_attrs = df.drop(['s'], axis=1)
  df_score_swap = df[['s', 'protected_attr', 'index']].reset_index(drop=True)

  advantaged_group = np.max(np.unique(x[:, -1]))
  disadvantaged_group = np.min(np.unique(x[:, -1]))

  # df_score_swaped = _biased_swap(df_score_swap, advantaged_group, disadvantaged_group, score_diff_thres=10000000, unfair_coin_prob=1)
  df_score_swapped = biased_swap(df_score_swap, p)
  return df_score_swapped.sort_values(by=['index'])['s'].values



def swapAttackMulti(x, p, f):
  df = pd.DataFrame(x) ## the order need to be retained in this df so the score or score after swaping can be aligned with input
  df.columns = [*df.columns[:-1], 'protected_attr']
  df['s'] = f(x)
  df = df.sort_values(by = ['s'], ascending=False)
  df['index'] = df.index # save the index to track the corresponding attributes after score swapping
  df_attrs = df.drop(['s'], axis=1)
  df_score_swap = df[['s', 'protected_attr', 'index']].reset_index(drop=True)

  advantaged_group = np.max(np.unique(x[:, -1]))
  disadvantaged_group = np.min(np.unique(x[:, -1]))

  # df_score_swaped = _biased_swap(df_score_swap, advantaged_group, disadvantaged_group, score_diff_thres=10000000, unfair_coin_prob=1)
  # df_score_swapped = biased_swap_multi(df_score_swap, p)
  df_score_swapped = biased_swap_multi(df_score_swap, advantaged_group, disadvantaged_group, p)

  return df_score_swapped.sort_values(by=['index'])['s'].values


