# -*- coding: utf-8 -*-
# @Author: zhushuai
# @Date:   2019-04-02 12:02:01
# @Last Modified by:   zhushuai
# @Last Modified time: 2019-04-02 12:03:41
# 定义目标编码函数

class BetaEncoder(object):

    def __init__(self, group):

        self.group = group
        self.stats = None

    # get counts from df
    def fit(self, df, target_col):
        self.prior_mean = np.mean(df[target_col])
        stats = df[[target_col, self.group]].groupby(self.group)
        stats = stats.agg(['sum', 'count'])[target_col]
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)
        self.stats = stats

    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):

        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n = df_stats['n'].copy()
        N = df_stats['N'].copy()

        # fill in missing
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0

        # prior parameters
        N_prior = np.maximum(N_min - N, 0)
        alpha_prior = self.prior_mean * N_prior
        beta_prior = (1 - self.prior_mean) * N_prior

        # posterior parameters
        alpha = alpha_prior + n
        beta = beta_prior + N - n

        # calculate statistics
        if stat_type == 'mean':
            num = alpha
            dem = alpha + beta

        elif stat_type == 'mode':
            num = alpha - 1
            dem = alpha + beta - 2

        elif stat_type == 'median':
            num = alpha - 1 / 3
            dem = alpha + beta - 2 / 3

        elif stat_type == 'var':
            num = alpha * beta
            dem = (alpha + beta) ** 2 * (alpha + beta + 1)

        elif stat_type == 'skewness':
            num = 2 * (beta - alpha) * np.sqrt(alpha + beta + 1)
            dem = (alpha + beta + 2) * np.sqrt(alpha * beta)

        elif stat_type == 'kurtosis':
            num = 6 * (alpha - beta) ** 2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)
            dem = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)

        else:
            num = self.prior_mean
            dem = np.ones_like(N_prior)

        # replace missing
        value = num / dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value