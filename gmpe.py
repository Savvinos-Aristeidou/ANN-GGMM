import warnings
import re
import json
from pathlib import Path
from typing import Union
import numpy as np
from scipy.interpolate import interp1d
from openquake.hazardlib import const
from openquake.hazardlib.imt import RSD575, RSD595, AvgSA, SA, PGA, PGV, PGD


def get_period_im(name: str):
    pattern = r"\((\d+\.\d+)\)?"

    if '(' in name:
        im_type = name.split('(', 1)[0].strip()
    else:
        im_type = name

    if re.search(pattern, name):
        period = float(re.search(pattern, name).group(1))
    else:
        period = None

    return im_type, period


def read_json(filename: Union[Path, dict]):
    if isinstance(filename, Path) or isinstance(filename, str):
        filename = Path(filename)

        with open(filename) as f:
            filename = json.load(f)

    return filename


def FIV3(period):
    period = float(period)
    return 'FIV3(%s)' % period


class AristeidouEtAl2024:
    DATA = read_json("gmm_ann.json")

    #: Supported tectonic region type is subduction interface
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {
        SA, AvgSA, RSD595, RSD575, PGA, PGV, PGD, FIV3}

    #: Supported intensity measure components
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = {
        const.IMC.RotD50, const.IMC.RotD100, const.IMC.GEOMETRIC_MEAN}

    #: Supported standard deviation types
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL}

    #: Requires sites parameters
    REQUIRES_SITES_PARAMETERS = {'vs30'}

    #: Required rupture parameters
    REQUIRES_RUPTURE_PARAMETERS = {
        'mag', 'ztor', 'z2pt5', 'd_hyp', 'mechanism'}

    #: Required distance measures
    REQUIRES_DISTANCES = {'rrup', 'rjb', 'rx'}

    SUGGESTED_LIMITS = {
        "magnitude": [4.5, 7.9],
        "Rjb": [0., 299.44],
        "Rrup": [0.07, 299.59],
        "D_hyp": [2.3, 18.65],
        "Vs30": [106.83, 1269.78],
        "mechanism": [0, 4],
        "Z2pt5": [0., 7780.],
        "Rx": [-297.13, 292.39],
        "Ztor": [0, 16.23],
    }

    def get_mean_and_stddevs(
        self, imt: str, mag: float, rjb: float, rrup: float, d_hyp: float,
        vs30: float, mechanism: int, z2pt5: float, rx: float, ztor: float,
        component_definition: str = "RotD50"
    ):
        """Provides the ground motion prediction equation for the following
        intensity measures:
        1. Significant duration defined at the time from 5-75% or 5-95% of
        the Arias intensity: Ds575, Ds595
        2. PGA, PGV, PGD
        3. Spectral accelerations (RotD50, RotD100 and geomean components):
        SA_RotD50({T}s), SA_RotD100({T}) and SA_geomean({T})
        4. Filtered incremental velocity (geomean): FIV3({T})
        5. Average spectral accelerations (RotD50, RotD100 and geomean
        components): Sa_avg2_RotD50({T}), Sa_avg2_RotD100({T}),
        Sa_avg2_geomean({T}), Sa_avg3_RotD50({T}), Sa_avg3_RotD100({T}),
        Sa_avg3_geomean({T})

        where T stands for the period value; Sa_avg2 is computed using
        0.2T to 2.0T; Sa_avg3 is computed using 0.2T to 3.0T;

        Parameters
        ----------
        imt : str
            Intensity measure name, e.g. PGA, SA(0.1)
        mag : float
            Moment magnitude
        rjb : float
            Joyner-Boore distance [km]
        rrup : float
            Closest rupture distance [km]
        d_hyp : float
            Hypocentral depth [km]
        vs30 : float
            Time-averaged shear-wave velocity to 30m depth [m/s]
        mechanism : int
            Faulting mechanism, from 0 to 4
        z2pt5 : float
            Depth to 2.5 km/s shear-wave velocity horizon (a.k.a., basin
            or sediment depth) [m]
        rx : float
            Distance measured perpendicular to the fault strike from the
            surfance projection of the up-dip edge of the faul plane [km]
        ztor : float
            Depth to top of the fault rupture [km]
        component_definition: str, optional
            Component defintion: RotD50, RotD100, geomean
            Relevant only for SA, Sa_avg2, and Sa_avg3

        Returns
        -------
        numpy.ndarray and float
            Means and stadard deviations
        """
        imt = str(imt)

        if imt == "RSD575":
            imt = "Ds575"
        if imt == "RSD595":
            imt = "Ds595"

        mag = np.array(mag).reshape(-1, 1)
        rjb = np.array(rjb).reshape(-1, 1)
        rrup = np.array(rrup).reshape(-1, 1)
        d_hyp = np.array(d_hyp).reshape(-1, 1)
        vs30 = np.array(vs30).reshape(-1, 1)
        mechanism = np.array(mechanism).reshape(-1, 1)
        z2pt5 = np.array(z2pt5).reshape(-1, 1)
        rx = np.array(rx).reshape(-1, 1)
        ztor = np.array(ztor).reshape(-1, 1)

        x = np.column_stack([
            rjb, rrup, d_hyp, mag, vs30, mechanism, z2pt5, rx, ztor
        ])

        # Validate the input parameters and raise warnings if not within limits
        self._validate_input(x)

        # Get biases and weights of the ANN model
        biases = self.DATA["biases"]
        weights = self.DATA["weights"]

        # Input layer
        # Transform the input
        x_transformed = self._minmax_scaling(x)

        _data = self._generate_function(
            x_transformed, biases[0], weights[0])
        a1 = self.softmax(_data)

        # Hidden layer
        _data = self._generate_function(
            a1, biases[1], weights[1]
        )
        a2 = self.tanh(_data)

        # Output layer
        _data = self._generate_function(
            a2, biases[2], weights[2]
        )
        output = self.linear(_data)

        # Reverse log10
        output = self.log10_reverse(output)

        # Means (shape=(cases, n_im)) and standard deviations (n_im, )
        means = np.squeeze(np.log(output))

        stddevs = np.asarray((self.DATA["total-stdev"],
                              self.DATA["inter-stdev"],
                              self.DATA["intra-stdev"]))

        # Transform the standard deviations from log10 to natural logarithm
        stddevs = np.log(10**stddevs)

        # Get the means and stddevs at index corresponding to the IM
        return self._get_means_stddevs(imt, means, stddevs,
                                       component_definition)

    def _get_means_stddevs(self, im_name, means, stddevs,
                           component_definition):
        if len(means.shape) == 1:
            means = means.reshape(1, means.shape[0])

        supported_ims = np.asarray(self.DATA["output-ims"])

        if im_name.startswith("SA") or im_name.startswith("Sa"):
            im_name = re.sub(r'\(', f'_{component_definition}(', im_name,
                             count=1)

        if im_name in supported_ims:
            idx = np.where(supported_ims == im_name)[0][0]
            return means[:, idx], stddevs[:, idx]

        im_type, period = get_period_im(im_name)

        if period is None:
            raise ValueError(
                f"IM type {im_type} is not supported, to get list "
                "of supported IMs, run method: get_supported_ims()")

        # Period not supported, perform linear interpolation
        idxs = np.where(np.char.find(supported_ims, im_type) != -1)

        ims = supported_ims[idxs]
        means = means[:, idxs]
        stddevs = stddevs[:, idxs]

        periods = []
        for im in ims:
            _, _t = get_period_im(im)
            periods.append(_t)

        # TODO: make the interpolation on the logarithmic scale (like openquake)
        # Create interpolators
        interp_stddevs = interp1d(np.log(periods), stddevs)
        interp_means = interp1d(np.log(periods), means)

        mean, stddev = np.squeeze(interp_means(np.log(period))), interp_stddevs(np.log(period))

        return mean, stddev

    def _validate_input(self, x):
        pars = self.DATA["parameters"]

        min_max = np.asarray([self.SUGGESTED_LIMITS[par] for par in pars])

        idxs = np.where((x < min_max[:, 0]) | (x > min_max[:, 1]))

        for row, col in zip(*idxs):
            val = x[row, col]
            warnings.warn(
                f"Value of {pars[col]}: {val} is not within the "
                f"suggested limits {self.SUGGESTED_LIMITS[pars[col]]}")

    def log10_reverse(self, x):
        return 10 ** x

    def _generate_function(self, x, biases, weights):
        biases = np.asarray(biases)
        weights = np.asarray(weights).T

        return biases.reshape(1, -1) + np.dot(weights, x.T).T

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def linear(self, x):
        return x

    def tanh(self, x):
        return np.tanh(x)

    def _minmax_scaling(self, data, feature_range=(-3, 3)):
        pars = self.DATA["parameters"]
        min_max = np.asarray([
            self.SUGGESTED_LIMITS[par] for par in pars])

        scaled_data = (data - min_max[:, 0]) / (min_max[:, 1] - min_max[:, 0])
        scaled_data = scaled_data * \
            (feature_range[1] - feature_range[0]) + feature_range[0]

        return scaled_data

    def get_suggested_parameter_limits(self):
        """Returns the suggested limits of the input parameters

        Returns
        -------
        dict
            Suggested input parameter limitations
        """
        return self.SUGGESTED_LIMITS

    def get_supported_ims(self):
        return self.DATA["output-ims"]
