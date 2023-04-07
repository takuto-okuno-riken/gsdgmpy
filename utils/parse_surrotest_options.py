import argparse


class ParseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = []

    def initialize(self):
        self.parser.add_argument('in_files', metavar='filename', nargs='+', help='filename of node status time-series (node x frames)')
        self.parser.add_argument('--gaussian',  action='store_true', help='output Gaussian distribution test (<original>_gauss_test.csv)')
        self.parser.add_argument('--linear',  action='store_true', help='output Linearity test  (<original>_linear_test.csv)')
        self.parser.add_argument('--iid',  action='store_true', help='output I.I.D test (<original>_iid_test.csv)')
        self.parser.add_argument('--side', type=int, default=2, help='bottom-side(1), both-side(2), top-side(3) (default:2)')
        self.parser.add_argument('--outpath', nargs=1, default='results', help='output files path (default:"results")')
        self.parser.add_argument('--format', type=int, default=1, help='save file format <type> 0:csv, 1:mat (default:1)')
        self.parser.add_argument('--showsig',  action='store_true', help='show input time-series data of <filename>.csv')
        self.parser.add_argument('--showrank',  action='store_true', help='show raster plot of input time-series data of <filename>.csv')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

