import argparse


class ParseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = []

    def initialize(self):
        self.parser.add_argument('in_files', metavar='filename', nargs='+', help='filename of node status time-series (node x frames)')
        self.parser.add_argument('--var',  action='store_true', help='output Vector Auto-Regression (VAR) group surrogate model (<filename>_gsm_var.mat)')
        self.parser.add_argument('--lag', type=int, default=3, help='time lag <num> for VAR (default:3)')
        self.parser.add_argument('--noise', type=str, nargs=1, default='gaussian', help='noise type for VAR surrogate model (default:"gaussian" or "residuals")')
        self.parser.add_argument('--outpath', nargs=1, default='results', help='output files path (default:"results")')
        self.parser.add_argument('--transform', type=int, default=0, help='input signal transform  0:raw, 1:sigmoid (default:0)')
        self.parser.add_argument('--transopt', type=float, default=float('NaN'), help='signal transform option (for type 1:centroid value)')
        self.parser.add_argument('--format', type=int, default=2, help='save file format <type> 0:csv, 1:mat(each), 2:mat(all) (default:2)')
        self.parser.add_argument('--surrnum', type=int, default=1, help='output surrogate sample number <num> (default:1)')
        self.parser.add_argument('--siglen', type=int, default=0, help='output time-series length <num> (default:same as input time-series)')
        self.parser.add_argument('--range', type=str, nargs=1, default='auto', help='output surrogate value range (default:"auto", sigma:<num>, full:<num>, <min>:<max> or "none")')
        self.parser.add_argument('--showinsig',  action='store_true', help='show input time-series data of <filename>.csv')
        self.parser.add_argument('--showinras',  action='store_true', help='show raster plot of input time-series data of <filename>.csv')
        self.parser.add_argument('--showsig',  action='store_true', help='show output surrogate time-series data')
        self.parser.add_argument('--showras',  action='store_true', help='show raster plot of output surrogate time-series data')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

