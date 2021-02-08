import numpy as np

from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost


class RunHistory2EPM4QuantileTransformedCost(RunHistory2EPM4Cost):

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        import sklearn.preprocessing
        self.transformer = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=len(values),
            output_distribution='normal',
            copy=True,
        )
        rval = self.transformer.fit_transform(values)
        return rval


class RunHistory2EPM4GaussianCopula(RunHistory2EPM4Cost):
    """TODO"""

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:

        import scipy as sp
        from smac.utils.constants import VERY_SMALL_NUMBER
        quants = [sp.stats.percentileofscore(values, v) / 100 - VERY_SMALL_NUMBER for v in values]
        rval = np.array([sp.stats.norm.ppf(q) for q in quants]).reshape((-1, 1))
        return rval


class RunHistory2EPM4GaussianCopulaCorrect(RunHistory2EPM4Cost):
    """TODO"""

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:

        import scipy as sp
        quants = (sp.stats.rankdata(values.flatten()) - 1) / (len(values) - 1)
        cutoff = 1 / (4 * np.power(len(values), 0.25) * np.sqrt(np.pi * np.log(len(values))))
        quants = np.clip(quants, a_min=cutoff, a_max=1 - cutoff)
        # Inverse Gaussian CDF
        rval = np.array([sp.stats.norm.ppf(q) for q in quants]).reshape((-1, 1))
        return rval


class RunHistory2EMPCopulaOriginal(RunHistory2EPM4Cost):

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:

        class GaussianTransform:
            """
            Transform data into Gaussian by applying psi = Phi^{-1} o F where F is the truncated ECDF.
            :param y: shape (n, dim)
            """

            def __init__(self, y: np.array):
                assert y.ndim == 2
                self.dim = y.shape[1]
                self.sorted = y.copy()
                self.sorted.sort(axis=0)

            @staticmethod
            def z_transform(series, values_sorted=None):
                # applies truncated ECDF then inverse Gaussian CDF.
                if values_sorted is None:
                    values_sorted = sorted(series)

                def winsorized_delta(n):
                    return 1.0 / (4.0 * n ** 0.25 * np.sqrt(np.pi * np.log(n)))

                delta = winsorized_delta(len(series))

                def quantile(values_sorted, values_to_insert, delta):
                    res = np.searchsorted(values_sorted, values_to_insert) / len(values_sorted)
                    return np.clip(res, a_min=delta, a_max=1 - delta)

                quantiles = quantile(
                    values_sorted,
                    series,
                    delta
                )

                quantiles = np.clip(quantiles, a_min=delta, a_max=1 - delta)
                from scipy import stats
                return stats.norm.ppf(quantiles)

            def transform(self, y: np.array):
                """
                :param y: shape (n, dim)
                :return: shape (n, dim), distributed along a normal
                """
                assert y.shape[1] == self.dim
                # compute truncated quantile, apply gaussian inv cdf
                return np.stack([
                    self.z_transform(y[:, i], self.sorted[:, i])
                    for i in range(self.dim)
                ]).T
        psi = GaussianTransform(values)
        return psi.transform(values)


class RunHistory2EPM4GaussianCopulaTurbo(RunHistory2EPM4Cost):
    """TODO"""

    def order_stats(self, X):
        _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
        obs = np.cumsum(cnt)  # Need to do it this way due to ties
        o_stats = obs[idx]
        return o_stats

    def copula_standardize(self, X):
        X = np.nan_to_num(np.asarray(X)).flatten()  # Replace inf by something large
        assert X.ndim == 1 and np.all(np.isfinite(X))
        o_stats = self.order_stats(X)
        quantile = np.true_divide(o_stats, len(X) + 1)
        import scipy.stats as ss
        X_ss = ss.norm.ppf(quantile).reshape((-1, 1))
        return X_ss

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:

        return self.copula_standardize(values)
