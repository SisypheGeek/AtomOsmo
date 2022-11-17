import requests
import numpy as np
import pandas as pd
import scipy.stats as stats
# import scipy
from scipy import signal
from scipy.stats import ks_2samp
from scipy.stats import cramervonmises_2samp
from statsmodels.tsa.stattools import adfuller
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# API https://min-api.cryptocompare.com/documentation


def request_ts(_token):
    _r_historical = requests.get(f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={_token}&tsym=USD&limit=2000').json()
    _data_token_df = pd.DataFrame(_r_historical['Data']['Data'])
    _data_token_df['date'] = pd.to_datetime(_data_token_df['time'], unit='s')
    _data_token_df = _data_token_df[['date', 'close']]
    _data_token_df = _data_token_df.set_index('date')
    return _data_token_df


def request_all_token_ts(tokens):
    _frame_dict = {}
    for _token in tokens:
        df_i = request_ts(_token)
        _frame_dict[_token] = df_i
    return _frame_dict


def get_similar_series(_frame_dict):
    first_index = _frame_dict['OSMO'].loc[_frame_dict['OSMO']['close'].ne(0), 'close'].first_valid_index()
    _frame_dict['OSMO'] = _frame_dict['OSMO'].loc[first_index:,:]
    _frame_dict['ATOM'] = _frame_dict['ATOM'].loc[first_index:, :]
    _frame_dict['OSMO'] = _frame_dict['OSMO'].rename(columns={'close': 'osmo_level'})
    _frame_dict['ATOM'] = _frame_dict['ATOM'].rename(columns={'close': 'atom_level'})


def get_return(_frame_dict):
    _ret_df = _frame_dict['ATOM'].diff().dropna().rename(columns={'atom_level': 'atom_ret'})
    _ret_df['osmo_ret'] = _frame_dict['OSMO'].diff().dropna()['osmo_level']
    return _ret_df


def get_pct(_frame_dict):
    _pct_df = _frame_dict['ATOM'].pct_change().dropna().rename(columns={'atom_level': 'atom_ret'})
    _pct_df['osmo_ret'] = _frame_dict['OSMO'].pct_change().dropna()['osmo_level']
    return _pct_df


def simple_correlation(_pct_df):
    r_pct, p_pct = stats.pearsonr(_pct_df['atom_ret'], _pct_df['osmo_ret'])
    dict_corr = {'Pearson Correlation': [_pct_df.corr(method='pearson').iloc[0, 1]],
                 'Spearman Correlation': [_pct_df.corr(method='spearman').iloc[0, 1]],
                 'Kendall Correlation': [_pct_df.corr(method='kendall').iloc[0, 1]],
                 'P-values': ["{:e}".format(p_pct)]}
    corr_df = pd.DataFrame(data=dict_corr,
                           index=['Result'])
    return corr_df, r_pct


def rolling_correlation(_pct_df):
    r_window_size = 60
    # Interpolate missing data just in case.
    df_interpolated = _pct_df.interpolate()
    # Compute rolling window synchrony
    _rolling_r = df_interpolated['atom_ret'].rolling(window=r_window_size, center=True).corr(df_interpolated['osmo_ret'])
    return _rolling_r


def plot_rolling_correlation(_pct_df, _rolling_r):
    f, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    _pct_df.rolling(window=15, center=True).median().plot(ax=ax[0])
    ax[0].set(xlabel='Date', ylabel='Relative Returns')
    _rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Date', ylabel='Rolling Pearson, rho')
    plt.suptitle("Rolling window correlation for ATOM and OSMO Crypto data")
    plt.savefig('./plot/corr_overtime.png')
    
    
def plot_corr_overtime(_pct_df, _r_pct):
    f, ax = plt.subplots(figsize=(7, 3))
    _pct_df.rolling(window=15, center=True).median().plot(ax=ax)
    ax.set(xlabel='Time', ylabel='Pearson r')
    ax.set(title=f"Pearson r = {np.round(_r_pct, 2)}")
    plt.savefig('./plot/rolling_correlation.png')


def lin_reg(_pct_df):
    x_train, x_test, y_train, y_test = train_test_split(_pct_df['atom_ret'].values.reshape(-1, 1),
                                                        _pct_df['osmo_ret'].values.ravel(),
                                                        test_size=0.2)
    _reg = LinearRegression().fit(x_train, y_train)
    _r2 = _reg.score(x_test, y_test)
    return _r2


def analysis_correlation(_pct_df):
    # level of statistical dependency
    # Simple Correlation
    corr_df, _r_pct = simple_correlation(_pct_df)
    # Rolling Correlation
    _rolling_r = rolling_correlation(_pct_df)
    # Cross correlation analysis
    _ccf_crypto = ccf_values(_pct_df['atom_ret'], _pct_df['osmo_ret'])
    _lags = signal.correlation_lags(len(_pct_df['atom_ret']), len(_pct_df['osmo_ret']))
    # Analysis of R2 as a variant of correlation
    _r2 = lin_reg(_pct_df)
    return corr_df, _r_pct, _rolling_r, _ccf_crypto, _lags, _r2


def ganger_analysis(_pct_df):
    _ganger_ao = grangercausalitytests(_pct_df[['atom_ret', 'osmo_ret']], maxlag=2)
    _ganger_oa = grangercausalitytests(_pct_df[['osmo_ret', 'atom_ret']], maxlag=2)
    dict_corr = {
        'Atome vs Osmos': [_ganger_ao[1][0]['ssr_ftest'][1], _ganger_ao[2][0]['ssr_ftest'][1]],
        'Osmo vs Atom': [_ganger_oa[1][0]['ssr_ftest'][1], _ganger_oa[2][0]['ssr_ftest'][1]]
    }
    _ganger_df = pd.DataFrame(data=dict_corr,
                              index=['P-value lag1', 'P-value lag2'])
    return _ganger_df


def dependence_analysis(_pct_df):
    _corr_df, _r_pct, _rolling_r, _ccf_crypto, _lags, _r2 = analysis_correlation(_pct_df)
    # plotting ret and corr
    plot_rolling_correlation(_pct_df, _rolling_r)
    plot_corr_overtime(_pct_df, _r_pct)
    # Plot cross correlation
    ccf_plot(_lags, _ccf_crypto)
    _stationary_table = _pct_df.apply(adf_test, axis=0)
    _ganger_df = ganger_analysis(_pct_df)
    _mi = entropy_analysis(_pct_df)
    return _corr_df, _r2, _ganger_df, _mi, _stationary_table


def entropy_analysis(_pct_df):
    _mi = mutual_info_regression(_pct_df['atom_ret'].values.reshape(-1, 1), _pct_df['osmo_ret'].values.ravel())
    plotting_scatter_mutual_info(_pct_df, _mi)
    rolling_mutual_info(_pct_df)
    return _mi


def plotting_scatter_mutual_info(_pct_df, _mi):
    plt.figure(figsize=(15, 5))
    plt.scatter(_pct_df['atom_ret'].values.reshape(-1, 1), _pct_df['osmo_ret'].values.ravel(), edgecolor="black", s=20)
    plt.title("MI={:.2f}".format(_mi[0]), fontsize=16)
    plt.savefig('./plot/all_set_entropy_scatter.png')
    plt.show()


def plotting_scatter_mutual_info2(_x, _y, _mi):
    plt.figure(figsize=(15, 5))
    plt.scatter(_x, _y, edgecolor="black", s=20)
    plt.title("MI={:.2f}".format(_mi[0]), fontsize=16)
    plt.savefig(f'./plot/sub_set_{_mi}_entropy_scatter.png')
    plt.show()


def rolling_mutual_info(_pct_df):
    _plot40 = 'n'
    _plot50 = 'n'
    _plot60 = 'n'
    _plot70 = 'n'
    _l_plot = [_plot40, _plot50, _plot60, _plot70]
    for i in range(len(_pct_df['atom_ret']) - 100):
        _x = _pct_df['atom_ret'].values.reshape(-1, 1)[i:i + 100]
        _y = _pct_df['osmo_ret'].values.ravel()[i:i+100]
        _mi = mutual_info_regression(_x, _y)
        # print(mi)
        for l in _l_plot:
            if 0 < _mi < 0.40 and _plot40 == 'n':
                _plot40 = 'y'
                plotting_scatter_mutual_info2(_x, _y, _mi)
            elif 0.4 < _mi < 0.50 and _plot50 == 'n':
                _plot50 = 'y'
                plotting_scatter_mutual_info2(_x, _y, _mi)
            elif 0.5 < _mi < 0.6 and _plot60 == 'n':
                _plot60 = 'y'
                plotting_scatter_mutual_info2(_x, _y, _mi)
            elif 0.6 < _mi and _plot70 == 'n':
                _plot70 = 'y'
                plotting_scatter_mutual_info2(_x, _y, _mi)


def adf_test(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    result = pd.Series(dftest[0:4], index=['Test Statistic', 'P - value', 'Lags Used', 'No of Observations'])
    for key, value in dftest[4].items():
        result['Critical Value (%s)' % key] = value
    return result


def ccf_values(series1, series2):
    p = series1
    q = series2
    p = (p - np.mean(p)) / (np.std(p) * len(p))
    q = (q - np.mean(q)) / (np.std(q))
    c = np.correlate(p, q, 'full')
    return c


def ccf_plot(lags, ccf):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(lags, ccf)
    ax.axhline(-2 / np.sqrt(23), color='red', label='5% confidence interval')
    ax.axhline(2 / np.sqrt(23), color='red')
    ax.axvline(x=0, color='black', lw=1)
    ax.axhline(y=0, color='black', lw=1)
    ax.axhline(y=np.max(ccf), color='blue', lw=1,
               linestyle='--', label='highest +/- correlation')
    ax.axhline(y=np.min(ccf), color='blue', lw=1,
               linestyle='--')
    ax.set(ylim=[-1, 1])
    ax.set_title('Cross Correlation between ATOM and OSMO', weight='bold', fontsize=15)
    ax.set_ylabel('Correlation Coefficients', weight='bold', fontsize=12)
    ax.set_xlabel('Time Lags', weight='bold', fontsize=12)
    plt.savefig(f'./plot/cross_corr.png')
    plt.legend()


def test_hypo_distribution(_pct_df):
    ks = ks_2samp(_pct_df['atom_ret'], _pct_df['osmo_ret'])
    cm = cramervonmises_2samp(_pct_df['atom_ret'], _pct_df['osmo_ret'], method='auto')
    d = {'Kolmogorov': ks.pvalue, 'Cramer': cm.pvalue}
    _df_test = pd.DataFrame(d, index=['P-values'])
    return _df_test


def plot_rel_return(_pct_df):
    ax = _pct_df.plot.line()
    ax.figure.savefig('./plot/rel_return.png')


def get_var(_atom_sort):
    _var_90_atom = _atom_sort.quantile(0.1).round(4)
    _var_95_atom = _atom_sort.quantile(0.05).round(4)
    _var_99_atom = _atom_sort.quantile(0.01).round(4)
    return _var_90_atom, _var_95_atom, _var_99_atom


def get_es(_atom_sort, _var_90_atom, _var_95_atom, _var_99_atom):
    _cvaR_90_atom = _atom_sort[_atom_sort <= _var_90_atom].mean()
    _cvaR_95_atom = _atom_sort[_atom_sort <= _var_95_atom].mean()
    _cvaR_99_atom = _atom_sort[_atom_sort <= _var_99_atom].mean()
    return _cvaR_90_atom, _cvaR_95_atom, _cvaR_99_atom


def get_risk_level(_pct_df):
    _atom_sort = _pct_df[['atom_ret']]
    _atom_sort.sort_values(by=['atom_ret'], inplace=True, ascending=True)
    _osmo_sort = _pct_df[['osmo_ret']]
    _osmo_sort.sort_values(by=['osmo_ret'], inplace=True, ascending=True)
    _var_90_atom, _var_95_atom, _var_99_atom = get_var(_atom_sort)
    _var_90_osmo, _var_95_osmo, _var_99_osmo = get_var(_osmo_sort)
    _cvaR_90_atom, _cvaR_95_atom, _cvaR_99_atom = get_es(_atom_sort, _var_90_atom, _var_95_atom, _var_99_atom)
    _cvaR_90_osmo, _cvaR_95_osmo, _cvaR_99_osmo = get_es(_osmo_sort, _var_90_osmo, _var_95_osmo, _var_99_osmo)
    d = {'Confidence Level': ['90%', '95%', '99%'],
         'VaR ATOM': [_var_90_atom.values[0], _var_95_atom.values[0], _var_99_atom.values[0]],
         'VaR OSMO': [_var_90_osmo.values[0], _var_95_osmo.values[0], _var_99_osmo.values[0]],
         'Expected Shortfall ATOM': [_cvaR_90_atom.values[0], _cvaR_95_atom.values[0], _cvaR_99_atom.values[0]],
         'Expected Shortfall OSMO': [_cvaR_90_osmo.values[0], _cvaR_95_osmo.values[0], _cvaR_99_osmo.values[0]]}
    _risk_df = pd.DataFrame(d)
    _risk_df = _risk_df.set_index('Confidence Level')
    return _risk_df


def data_main():
    tokens = ['ATOM', 'OSMO']
    frame_dict = request_all_token_ts(tokens)
    get_similar_series(frame_dict)
    # ret_df = get_return(frame_dict)
    pct_df = get_pct(frame_dict)
    _std_pct = pct_df.std()
    # pct_df_sort = pct_df.copy()
    _risk_df = get_risk_level(pct_df)
    plot_rel_return(pct_df)
    # pct_df.plot()
    frame_dict['ATOM'].plot()
    frame_dict['OSMO'].plot()
    _corr_df, _r2, _ganger_df, _mi, _stationary_table = dependence_analysis(pct_df)
    _df_test = test_hypo_distribution(pct_df)
    return _corr_df, _r2, _ganger_df, _mi, _stationary_table, _df_test, _std_pct, _risk_df


if __name__ == '__main__':
    corr_df, r2, ganger_df, mi, stationary_table, df_test, std_pct, risk_df = data_main()

