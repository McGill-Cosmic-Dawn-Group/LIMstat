"""
utilities for gp regression

copied from gp_errs notebook (Kern+2021) but updated for sklearn >= 0.24.0

July, 2022
"""
from sklearn import gaussian_process as gp
from sklearn.utils import check_X_y, check_array, check_random_state, itemgetter
from scipy.linalg import cholesky, cho_solve, solve_triangular
import numpy as np
from scipy import stats, special, optimize
import functools
import operator
import warnings

try:
    import emcee
except ImportError:
    print("could not import emcee")
try:
    import uvtools as uvt
except ImportError:
    print("could not import uvtools")


def setup_gp(kernels, kdict, optimizer='fmin_l_bfgs_b', n_restarts=10, norm_y=False, alpha=1e-10):
    """setup GP
    
    Args:
    kernels : list of str, kernel names found in kdict to include in model
    kdict : dict, keys kernel names and values GP kernels

    Returns:
    G : GaussianProcess model
    Knames : list of kernel names in G.kernel_.get_params() in the order fed by kernels
    theta_labels : list of latex labels
    theta_bounds : list of hyperparameter theta bounds determined by input kdict
    """
    # setup G
    kerns = [kdict[k] for k in kernels]
    kernel = functools.reduce(operator.add, kerns)
    G = GaussianProcess(kernel=kernel, optimizer=optimizer, n_restarts_optimizer=n_restarts,
                        copy_X_train=False, normalize_y=norm_y, alpha=alpha)
   
    # get Knames
    root = ''
    Knames = []
    Nk = len(kernels)
    for i in range(Nk):
        if i == Nk - 1:
            Knames.append(root[:-2])
        else:
            Knames.append(root + 'k2')
            root += 'k1__'
    Knames = Knames[::-1]

    # get theta bounds
    theta_labels = []
    theta_bounds = []
    for kern in kerns:
        theta_bounds.extend([tuple(b) for b in kern.bounds])
        theta_labels.extend(kern.labels)

    return G, Knames, theta_labels, theta_bounds


def get_kernel(kernel, k):
    """
    Given composite sklearn kernel, isolate and return
    a specific component given by k, conforming to Knames
    output of setup_gp

    Parameters
    ----------
    kernel : sklearn kernel object
    k : str, component of kernel to return
        Ex. 'k1__k1__k2' means kernel.k1.k1.k2
    
    Returns
    -------
    sklearn kernel object
    """
    if '__' in k:
        k_opts = k.split('__')
        return get_kernel(getattr(kernel, k_opts[0]), '__'.join(k_opts[1:])) 
    else:
        return getattr(kernel, k)

def gp_fit_predict(data, G, freqs, Kfg_name=None, Keor_name=None):
    """given data and GP, train and predict for FG term
    
    Args:
    data : ndarray
    G : GaussianProcess object
    freqs : frequency array [Hz]
    Kfg_name : name of FG kernel in G.kernel_.get_params()
    Keor_name : name of EoR kernel in G.kernel_.get_params()

    Returns:

    """
    theta_ml = {}
    fg, eor = {}, {}
    fg_cov, eor_cov = {}, {}

    # stack real and imag
    Ntimes, Nfreqs = data.shape
    ydata = G.prep_ydata(freqs[:, None] / 1e6, data.T)

    if G.optimizer is not None:
        # re-optimize kernel_
        G.fit(freqs[:, None] / 1e6, ydata)
        kernel = G.kernel_
    else:
        kernel = G.kernel

    params = kernel.get_params()

    # get FG conditional distribution given trained kernel
    if Kfg_name in params:
        ypred, ycov = G.predict(freqs / 1e6, kernel=params[Kfg_name], return_cov=True)
        ypred = ypred[:, :Ntimes] + 1j * ypred[:, Ntimes:]
        fg = ypred.T
        fg_cov = ycov
    else:
        fg = np.zeros_like(data[k])
        fg_cov = np.eye(Nfreqs)

    # get EoR conditional distribution given trained kernel
    if Keor_name in params:
        ypred, ycov = G.predict(freqs, kernel=params[Keor_name], return_cov=True)
        ypred = ypred[:, :Ntimes] + 1j * ypred[:, Ntimes:]
        eor = ypred.T
        eor_cov = ycov
    else:
        eor = np.zeros_like(data[k])
        eor_cov = np.eye(Nfreqs) 

    return kernel.theta, fg, fg_cov, eor, eor_cov 


# define custom GP class
class GaussianProcess(gp.GaussianProcessRegressor):
    """
    Adapted from sklearn version 0.20.3
    """
    def fit(self, X, y, init_theta=None):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values
        init_theta : array-like, shape = (n_samples,)
            Initial theta. Default is self.kernel.theta
        Returns
        -------
        self : returns an instance of self.
        """
        if init_theta is not None:
            self.kernel_ = self.kernel.clone_with_theta(init_theta)
        else:
            self.kernel_ = self.kernel.clone_with_theta(self.kernel.theta)

        self._rng = check_random_state(self.random_state)

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            # demean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        if np.iterable(self.alpha) \
           and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel_.theta,
                                                      self.kernel_.bounds))]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        return self

    def predict(self, X, kernel=None, Xydata=None, return_std=False, return_cov=False):
        """Predict using the GP regression model
        
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        kernel : Kernel object, default is self.kernel
            Kernel object to use as K_11 and K_21 in prediction function.

        Xydata : tuple holding (X, y) of new inputs
            where X is shape (n_samples, n_features)
            and y is shape = (n_samples, n_output_dims).
            Target-values to condition on. If None use pre-computed
            self.alpha_ which comes from  self.y_train

        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.

        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")
        if X.ndim == 1:
            X = X[:, None]
        X = check_array(X)

        # check kernel
        if kernel is None:
            # if no kernel provided, look for one
            if self.kernel is None:
                raise ValueError("kernel argument or self.kernel must not be None")
            else:
                # if a fitted kernel exists, take it
                if hasattr(self, 'kernel_'):
                    kernel = self.kernel_
                # otherwise take init kernel
                else:
                    kernel = self.kernel

        # check y
        if Xydata is not None:
            self.prep_ydata(*Xydata)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            K_trans = kernel(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
            y_mean = self._y_train_mean + y_mean  # undo normal.
            if return_cov:
                v = cho_solve((self.L_, True), K_trans.T)  # Line 5
                y_cov = kernel(X) - K_trans.dot(v)  # Line 6
                return y_mean, y_cov
            elif return_std:
                # cache result of K_inv computation
                if self._K_inv is None:
                    # compute inverse K_inv of K based on its Cholesky
                    # decomposition L and its inverse L_inv
                    L_inv = solve_triangular(self.L_.T,
                                             np.eye(self.L_.shape[0]))
                    self._K_inv = L_inv.dot(L_inv.T)

                # Compute variance of predictive distribution
                y_var = kernel.diag(X)
                y_var -= np.einsum("ij,ij->i",
                                   np.dot(K_trans, self._K_inv), K_trans)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    def prep_ydata(self, X, ydata, kernel=None):
        """Take complex waterfall and prepare fit matrices
       
        Note: because real and imag components of ydata are
        separated as distinct features, the fitted covariance
        will be a factor of 2 lower than the covariance of
        the full complex data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        ydata : array-like, shape = (n_samples, n_output_dims)
            Target values to condition on

        kernel : kernel object, optional
            Kernel object to use, default is self.kernel_ or self.kernel

        Returns
        -------
        array-like (n_samples, n_output_dims*2)
        """
        # stack ydata
        y = np.hstack([ydata.real, ydata.imag])

        # Precompute quantities required for predictions which are independent
        # of actual query points
        if X.ndim == 1:
            X = X[:, None]
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            # de-mean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)
        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        # precompute Kernel matrix inversions
        if kernel is None:
            if hasattr(self, 'kernel_'):
                kernel = self.kernel_
            else:
                kernel = self.kernel
        K = kernel(X)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), y)  # Line 3

        return y + self._y_train_mean

    def refactor(self, y):
        """
        Take predictions that have been
        cast to real and stacked and
        re-organize them into complex waterfalls

        Parameters
        ----------
        y : array_like, (Nsamples, Nfeatures, ...)
        """
        N = y.shape[1]
        return y[:, :N//2] + 1j * y[:, N//2:]

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.
        A log prior is added if self.theta_priors exists.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        kernel = self.kernel.clone_with_theta(theta)

        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)

        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        alpha = cho_solve((L, True), y_train)  # Line 3

        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

        # Compute log-prior and add
        log_likelihood += self.log_prior(theta)

        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def log_prior(self, theta):
        """
        Compute log prior if self.theta_priors list exists.

        self.theta_priors is a list of callable of len(theta),
        each of which indexes the corresponding element in theta
        and returns its log prior

        Parameters
        ----------
        theta : array-like, shape (n_kernel_params,)
        """
        log_prior = np.zeros_like(theta, dtype=np.float)
        if hasattr(self, 'theta_priors') and self.theta_priors is not None:
            for i, pr in enumerate(self.theta_priors):
                if pr is not None:
                    log_prior[i] += pr(theta[i])
        return np.sum(log_prior, axis=0)


from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel, Hyperparameter


def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale


class Sinc(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 length_scale_fixed=False):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.length_scale_fixed = length_scale_fixed

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale),
                                  fixed=self.length_scale_fixed)
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds,
            fixed=self.length_scale_fixed
            )

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        # factor of 2 makes this match freq-freq cov of RBF length_scale
        length_scale = _check_length_scale(X, self.length_scale) * 2
        if Y is None:
            dists = pdist(X / length_scale, metric="euclidean")
            K = np.sinc(dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")
            K = np.sinc(dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = 1 / length_scale * (np.sinc(dists) - np.cos(dists * np.pi))
                K_gradient = squareform(K_gradient)[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                raise NotImplementedError
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0]
            )


class GaussSinc(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, gauss_length=1.0, gauss_length_bounds=(1e-1, 1e3),
                 gauss_length_fixed=False, sinc_length=1.0,
                 sinc_length_bounds=(1e-1, 1e3), sinc_length_fixed=False,
                 high_prec=False, dx=1e-5):
        """
        Convolution of a Gaussian and Sinc covariance function
        See appendix A2 of arxiv:1608.05854
        """
        self.gauss_length = gauss_length
        self.gauss_length_bounds = gauss_length_bounds
        self.gauss_length_fixed = gauss_length_fixed
        self.sinc_length = sinc_length
        self.sinc_length_bounds = sinc_length_bounds
        self.sinc_length_fixed = sinc_length_fixed
        self.high_prec = high_prec
        self.dx = dx

    @property
    def hyperparameter_gauss_length(self):
        if self.anisotropic:
            return Hyperparameter("gauss_length", "numeric",
                                  self.gauss_length_bounds,
                                  len(self.gauss_length),
                                  fixed=self.gauss_length_fixed)
        return Hyperparameter(
            "gauss_length", "numeric", self.gauss_length_bounds,
            fixed=self.gauss_length_fixed
            )

    @property
    def hyperparameter_sinc_length(self):
        if self.anisotropic:
            return Hyperparameter("sinc_length", "numeric",
                                  self.sinc_length_bounds,
                                  len(self.sinc_length),
                                  fixed=self.sinc_length_fixed)
        return Hyperparameter(
            "sinc_length", "numeric", self.sinc_length_bounds,
            fixed=self.sinc_length_fixed
            )

    @property
    def anisotropic(self):
        return False

    def __call__(self, X, Y=None, eval_gradient=False,
                 gauss_length=None, sinc_length=None):

        X = np.atleast_2d(X)

        gauss_length = gauss_length if gauss_length is not None else self.gauss_length
        sinc_length = sinc_length if sinc_length is not None else self.sinc_length        
        gauss_length = _check_length_scale(X, gauss_length)
        # factor of 2/pi makes sinc match Sinc kernel length
        sinc_length = _check_length_scale(X, sinc_length) * (2 / np.pi)

        # get prefactors
        arg = gauss_length / np.sqrt(2) / sinc_length
        Xc = X / gauss_length / np.sqrt(2)

        def func(dists, arg, high_prec=self.high_prec):
            """
            evaluate covariance depending on whether you
            need arbitrary precision arithmetic.
            If normalized distances exceeds ~20, you will need this,
            but note it is much slower.
            Appendix A2 of arxiv:1608.05854
            """
            if high_prec:
                import mpmath
                fn = lambda z: mpmath.exp(-z**2) * (mpmath.erf(arg + 1j*z) + mpmath.erf(arg - 1j*z)).real
                K = 0.5 * np.asarray(np.frompyfunc(fn, 1, 1)(dists), dtype=float) / special.erf(arg)

            else:
                K = 0.5 * np.exp(-dists**2) / special.erf(arg) \
                    * (special.erf(arg + 1j*dists) + special.erf(arg - 1j*dists))
                # replace nans with zero: in this limit, you should use high_prec
                # but this is a faster approximation
                K[np.isnan(K)] = 0.0

            return K

        if Y is None:
            # get x - x^prime distances
            dists = pdist(Xc, metric="euclidean")
            # evaluate covariance
            K = func(dists, arg)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            # get distances
            Yc = Y / gauss_length / np.sqrt(2)
            dists = cdist(Xc, Yc, metric="euclidean")
            # evaluate covariance
            K = func(dists, arg)

        K = K.real

        def grad(hparam, X=X, K=K):
            """
            numerical gradient approimation evaluate covariance gradient
            """
            if hparam == 'gauss_length':
                K2 = self.__call__(X, gauss_length=self.gauss_length + self.dx)
            elif hparam == 'sinc_length':
                K2 = self.__call__(X, sinc_length=self.sinc_length + self.dx)
            Kgrad = (K2 - K) / self.dx
            return Kgrad

        if eval_gradient:
            if not self.anisotropic or gauss_length.shape[0] == 1:
                if not self.hyperparameter_gauss_length.fixed:
                    gauss_length_grad = grad('gauss_length')
                else:
                    gauss_length_grad = np.empty((X.shape[0], X.shape[0], 0))
                if not self.hyperparameter_sinc_length.fixed:
                    sinc_length_grad = grad('sinc_length')
                else:
                    sinc_length_grad = np.empty((X.shape[0], X.shape[0], 0))

                return K, np.dstack((gauss_length_grad, sinc_length_grad))

            elif self.anisotropic:
                raise NotImplementedError

        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(gauss_length=[{1}], sinc_length=[{2}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.gauss_length)),
                ", ".join(map("{0:.3g}".format, self.sinc_length)),
            )
        else:  # isotropic
            return "{0}(gauss_length={1:.3g}, sinc_length={2:.3g})".format(
                self.__class__.__name__,
                np.ravel(self.gauss_length)[0],
                np.ravel(self.sinc_length)[0]
            )


class PowerLawKernel(gp.kernels.Kernel):
    """
    A power law (non-staionary) kernel
    wrapper around an existing kernel

    k(x1, x2) = T @ K @ T

    where K is an input kernel (identity if None) and

    T(x1, x2) = kronecker(x1, x2) * (x1 / anchorx)**((1-beta)/2)

    Note exponential is parameterized this way because
    sklearn takes log of hyperparams, thus beta must be non-negative
    and encompass possible values for synchrotron emission (-3 < alpha < 1)
    """
    def __init__(self, K=None, beta=1.0, beta_bounds=(1e-10, 3), anchorx=150.0):
        """
        K : kernel object, optional
            An existing kernel object to wrap around
        beta : float, optional
            Starting beta parameter
        beta_bounds : tuple, optional
            Bounds for beta
        anchorx : float, optional
            Anchor x-value for powerlaw
        """
        self.K = K
        self.beta = beta
        self.beta_bounds = beta_bounds
        self.anchorx = anchorx

    @property
    def hyperparameter_beta(self):
        return gp.kernels.Hyperparameter(
            "beta", "numeric", self.beta_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            Kstart = self.K(X) if self.K is not None else np.eye(X.shape[0])
            T = np.diag((X[:, 0] / self.anchorx)**((1 - self.beta)/2))
            K = T @ Kstart @ T
            if eval_gradient:
                raise NotImplementedError
                ## TODO: compute gradient for PowerLawKernel, not just DiagPLK
                if not self.hyperparameter_beta.fixed:
                    return (K, ((1 - self.beta) * X[:, 0]**(-self.beta)\
                               * np.eye(X.shape[0]))[:, :, None])
                else:
                    return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                return K
        else:
            return np.zeros((X.shape[0], Y.shape[0]))

    def is_stationary(self):
        """
        Returns whether the kernel is stationary.
        """
        return False

    def diag(self, X):
        """
        Returns the diagonal of the kernel k(X, X).

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.diag(np.diag(self.___call__(X, Y=None, eval_gradient=False)))

    def __repr__(self):
        out = "{0}(beta={1:.3g})".format(self.__class__.__name__, self.beta)
        if self.K:
            out = "{0} * {1}".format(out, self.K)
        return out


class DiagPowerLawKernel(gp.kernels.Kernel):
    """
    A diagonal power-law kernel.

    k(x_1, x_2) = (x_1 / anchorx) ** (1 - beta) if x_1 == x_2 else 0

    Parameters
    ----------
    beta : float, default: 1.0
        Parameter controlling power law amplitude w.r.t.
        dependent axis (i.e. x_train)

    beta_bounds : pair of floats, default (1e-10, 3)
        Hard prior bounds on beta parameter

    anchorx : float, default: 100.0
        Anchoring point along X_train for power law
    """

    def __init__(self, beta=1.0, beta_bounds=(1e-10, 3), anchorx=100.0):
        self.beta = beta
        self.beta_bounds = beta_bounds
        self.anchorx = anchorx

    @property
    def hyperparameter_beta(self):
        return gp.kernels.Hyperparameter(
            "beta", "numeric", self.beta_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.eye(X.shape[0]) * (X[:, 0] / self.anchorx)**(1 - self.beta)
            if eval_gradient:
                if not self.hyperparameter_beta.fixed:
                    return (K, ((1 - self.beta) * X[:, 0]**(-self.beta)\
                               * np.eye(X.shape[0]))[:, :, None])
                else:
                    return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                return K
        else:
            return np.zeros((X.shape[0], Y.shape[0]))

    def is_stationary(self):
        """
        Returns whether the kernel is stationary.
        """
        return False

    def diag(self, X):
        """
        Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.___call__(X, Y=None, eval_gradient=False)

    def __repr__(self):
        return "{0}(beta={1:.3g})".format(self.__class__.__name__, self.beta)


# optimized sampling function
def sample_y(GP, X, n_samples=1, random_state=0):
    """
    This is a faster version of
    GaussianProcessRegressor.sample_y

    Parameters
    ----------
    GP : GaussianProcessRegressor object
    X : array_like
    """
    rng = check_random_state(random_state)
    y_mean, y_cov = GP.predict(X, return_cov=True)
    if y_mean.ndim == 1:
        y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T

    else:
        y_samples = rng.multivariate_normal(np.zeros(y_mean.shape[0]), y_cov,
                                            n_samples * y_mean.shape[1])
        y_samples = y_mean[:, :, None] + y_samples.reshape(y_mean.shape + (n_samples,))

    return y_samples


# define priors
def log_gamma(a, b, unlog=True):
    """
    Returns a callable log gamma function hard-coded
    with parameters a, b.

    Parameters
    ----------
    a, b : float, gamma function parameters
    unlog : bool, if True, "unlog" input as np.e**x
    """
    def func(x, a=a, b=b, unlog=unlog):
        if unlog:
            x = np.e**x
        return np.log(b**a*x**(a-1)*np.exp(-b*x)/special.gamma(a))
    return gam

def log_gauss(mean, ell, unlog=True):
    """
    Returns a callable log gaussian function hard-coded
    with parameters mean and ell.

    Parameters
    ---------
    mean, ell : float, mean and lengthscale parameter
    unlog : bool, if True, "unlog" input as np.e**x
    """
    def func(x, mean=mean, ell=ell, unlog=unlog):
        if unlog:
            x = np.e**x
        return np.log(np.exp(-0.5*((x-mean)/ell)**2))
    return func

def log_uniform(a, b, unlog=True):
    """
    Returns a callable log uniform function hard-coded
    with parameters a, b.

    Parameters
    ----------
    a,b : float, upper and lower limit in uniform
    unlog : bool, if True, "unlog" input as np.e**x
    """
    def func(x, a=a, b=b, unlog=unlog):
        if unlog:
            x = np.e**x
        return np.log(((x >= a) & (x <= b)).astype(np.float))
    return func

def flat_log_prior(theta, theta_bounds=None):
    if theta_bounds is None:
        return 0
    in_bounds = [(th >= thb[0]) and (th <= thb[1]) for th, thb in zip(theta, theta_bounds)]
    if not np.all(in_bounds) or not np.all(np.isfinite(theta)):
        return -np.inf
    else:
        return 0

# define posterior probability
def log_prob(theta, GP, theta_bounds=None, unlogged=True, return_grad=False, prepend=None):
    """
    Compute log posterior. A prior on theta is computed if GP.theta_priors exits.
    See GP.log_marginal_likelihood for details

    Parameters
    ----------
    theta : 1d numpy array
        Vector of parameter values
    GP : GaussianProcess object
    theta_bounds : list of len-2 tuples
        Hard bounds on parameter prior, enacted through uniform prior
    unlogged : bool
        If True, theta and theta_bounds are taken to be in real space,
        in which case they are logged before passing to GP.log_marginal_like()
    return_grad : bool
        If True, return gradient
    prepend : float or list of float
        List of parameters to prepend to theta before passing to GP.
        theta bounds for parameters in prepend should already be present
        in theta_bounds input.

    Returns
    -------
    float
        log marginal likelihood: log(like) + log(prior)
    float (if return_grad is True)
        log marginal likelihood gradient
    """
    # prepend a parameter if desired
    if prepend is not None:
        theta = np.append(prepend, theta)
    # enact hard bounds if theta_bounds is fed
    pr = flat_log_prior(theta, theta_bounds=theta_bounds)
    if not np.isfinite(pr):
        return -np.inf
    if unlogged:
        theta = np.log(theta)
    if return_grad:
        lml, grad = GP.log_marginal_likelihood(theta, eval_gradient=True)
        return lml + pr, grad
    else:
        lml = GP.log_marginal_likelihood(theta, eval_gradient=False)
        return lml + pr

# explore for hyperparameters w/ emcee
def emcee_run(start_pos, GP, nstep=100, nwalkers=10, unlogged=False, theta_bounds=None,
              prepend=None, moves=None):
    """
    Run emcee EnsembleSampler

    Parameters
    ----------
    start_pos : ndarray, shape (Nwalkers, Nparams)
        Starting position of walkers across parameter space
    GP : GaussianProcess object
    nstep : int
        Number of steps to take. No burn-in. You can truncate the chains
        yourself if burn-in is a concern.
    nwalkers : int
        Number of walkers
    unlogged : bool
        If True, start_pos and theta_bounds are the not in log form,
        which is to say, the parameter space is explored in the unlogged form.
        The GP object only takes parameters in logged form.
        If False, then the parameter space is assumed to be already log-projected.
    theta_bounds : list of len-2 tuples
        Hard bounds on parameter space prior.
    prepend : float or list of float
        Parameter(s) that are not explored by emcee but are necessary when
        the log-likelihood is evaluated by the GP. These parameter(s) are
        prepended to the theta_step vector during the likelihood call.
        Parameters fed as prepend should already be have their bounds set
        in the input theta_bounds array
    moves : emcee move to use
        See emcee.moves for details.

    Returns
    -------
    emcee.EnsembleSampler object
    """
    nwalkers, ndim = start_pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(GP,), moves=moves,
            kwargs={'unlogged':unlogged, 'theta_bounds':theta_bounds, 'prepend':prepend})
    sampler.run_mcmc(start_pos, nstep)
    
    return sampler

def plot_pspec(x, y, ax, comp='real', yerr=None, **kwargs):
    if comp in ['abs', 'abs-real']:
        if comp == 'abs':
            ax.errorbar(x, np.abs(y), yerr=yerr, **kwargs)
        elif comp == 'abs-real':
            ax.errorbar(x, np.abs(np.real(y)), yerr=yerr, **kwargs)
    elif comp == 'real':
        pos = y >= 0.0
        _yerr = None
        if yerr is not None:
            _yerr = yerr[pos]
        ax.errorbar(x[pos], y[pos], yerr=_yerr, **kwargs)
        neg = y <= 0.0
        if yerr is not None:
            _yerr = yerr[neg]
        if 'label' in kwargs:
            kwargs.pop('label')
        ax.errorbar(x[neg], np.abs(y[neg]), yerr=_yerr, markerfacecolor='None', **kwargs)

def get_cov(data):
    cov = np.cov(data)
    for i in range(cov.shape[0]):
        cov[i] = np.roll(np.real(cov[i]), -i, axis=-1)
    cov = np.mean(cov, axis=0)
    cov /= np.abs(cov[0])
    cov[cov < 0] = 0
    return cov

def cov2pspec(K, scalar, window='bh7'):
    """convert a covariance matrix to a power spectrum
    
    Parameters
    ----------
    K : ndarray, square covariance matrix
    window : str, FFT tapering function
    scalar : float, normalization factor

    Returns
    -------
    ndarray : bandpowers
    """
    # get FFT operator
    nfreqs = len(K)
    q = np.fft.ifft(np.eye(K.shape[0]), axis=1) * uvt.dspec.gen_window(window, nfreqs)[None, :]
    # form pspec
    pspec = np.fft.fftshift(np.array([q[i].T.conj().dot(K).dot(q[i]) for i in range(len(q))])) * scalar
    return pspec

def draw_from_cov(cov, Nsamples=1):
    nfreqs = len(cov)
    real = stats.multivariate_normal.rvs(np.zeros(nfreqs), cov, Nsamples)
    imag = stats.multivariate_normal.rvs(np.zeros(nfreqs), cov, Nsamples)
    return real + 1j * imag


