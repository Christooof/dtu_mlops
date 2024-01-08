"""import pstats
from pstats import SortKey
p = pstats.Stats('profiling_results')
p.strip_dirs().sort_stats(-1).print_stats()"""
import cProfile
import re
cProfile.run('re.compile("profiling_results")')