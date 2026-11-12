"""Academic customer intelligence experiment, independent of the legacy app."""
import os

# Bound worker discovery and numerical thread use on small Windows laptops.
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
