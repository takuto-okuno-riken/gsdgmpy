import argparse


class ParseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = []

    def initialize(self):
        self.parser.add_argument('in_files', metavar='filename', nargs='+', help='filename of node status time-series (node x frames)')
        self.parser.add_argument('--range', type=str, nargs=1, default='auto', help='input group value range (default:"auto", sigma:<num>, full:<num> or <min>:<max>)')
        self.parser.add_argument('--aclag', type=int, default=15, help='time lag <num> for Auto Correlation (default:15)')
        self.parser.add_argument('--cclag', type=int, default=8, help='time lag <num> for Cross Correlation (default:8)')
        self.parser.add_argument('--pcclag', type=int, default=8, help='time lag <num> for Partial Cross Correlation (default:8)')
        self.parser.add_argument('--outpath', nargs=1, default='results', help='output files path (default:"results")')
        self.parser.add_argument('--format', type=int, default=1, help='save file format <type> 0:csv, 1:mat (default:1)')
        self.parser.add_argument('--transform', type=int, default=0, help='input signal transform  0:raw, 1:sigmoid (default:0)')
        self.parser.add_argument('--transopt', type=float, default=float('NaN'), help='signal transform option (for type 1:centroid value)')
        self.parser.add_argument('--showinsig',  action='store_true', help='show input time-series data of <filename>.csv')
        self.parser.add_argument('--showinras',  action='store_true', help='show raster plot of input time-series data of <filename>.csv')
        self.parser.add_argument('--showmat',  action='store_true', help='show result MTESS matrix')
        self.parser.add_argument('--showsig',  action='store_true', help='show 1 vs. others node signals')
        self.parser.add_argument('--showprop',  action='store_true', help='show result polar chart of 1 vs. others MTESS statistical properties')
        self.parser.add_argument('--shownode',  action='store_true', help='show result line plot of 1 vs. others node MTESS')
        self.parser.add_argument('--showdend',  type=str, nargs=1, default='', help='show dendrogram of <algo> hierarchical clustering based on MTESS matrix.')
        self.parser.add_argument('--cache',  action='store_true', help='use cache file for MTESS calculation')
        self.parser.add_argument('--cachepath', nargs=1, default='results/cache', help='cache files <path> (default:"results/cache")')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

