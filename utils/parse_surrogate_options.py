import argparse


class ParseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = []

    def initialize(self):
        self.parser.add_argument('in_files', metavar='filename', nargs='+', help='filename of node status time-series (node x frames)')
        self.parser.add_argument('--rg',  action='store_true', help='output Random Gaussian (RG) surrogate (<filename>_rg_<variate>_<num>.csv)')
        self.parser.add_argument('--rs',  action='store_true', help='output Random Shuffling (RS) surrogate (<filename>_rs_<variate>_<num>.csv)')
        self.parser.add_argument('--ft',  action='store_true', help='output Fourier Transform (FT) surrogate (<filename>_ft_<variate>_<num>.csv)')
        self.parser.add_argument('--aaft',  action='store_true', help='output Amplitude Adjusted FT (AAFT) surrogate (<filename>_aaft_<variate>_<num>.csv)')
        self.parser.add_argument('--iaaft',  action='store_true', help='output Iterated AAFT (IAAFT) surrogate (<filename>_iaaft_<variate>_<num>.csv)')
        self.parser.add_argument('--var',  action='store_true', help='output Vector Auto-Regression (VAR) surrogate (<filename>_var_<variate>_<num>.csv)')
        self.parser.add_argument('--multi',  action='store_true', help='output multivariate surrogate (default:on)')
        self.parser.add_argument('--uni',  action='store_true', help='output univariate surrogate (default:off)')
        self.parser.add_argument('--noise', type=str, nargs=1, default='gaussian', help='noise type for VAR surrogate (default:"gaussian")')
        self.parser.add_argument('--surrnum', type=int, default=1, help='output surrogate sample number <num> (default:1)')
        self.parser.add_argument('--outpath', nargs=1, default='results', help='output files path (default:"results")')
        self.parser.add_argument('--format', type=int, default=2, help='save file format <type> 0:csv, 1:mat(each), 2:mat(all) (default:2)')
        self.parser.add_argument('--transform', type=int, default=0, help='input signal transform  0:raw, 1:sigmoid (default:0)')
        self.parser.add_argument('--transopt', type=float, default=float('NaN'), help='signal transform option (for type 1:centroid value)')
        self.parser.add_argument('--lag', type=int, default=3, help='time lag <num> for VAR (default:3)')
        self.parser.add_argument('--showsig',  action='store_true', help='show input time-series data of <filename>.csv')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

