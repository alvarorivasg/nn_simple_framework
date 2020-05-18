import data_loader_two_by_two as dat
import numpy as np

training_set, evaluation_set = dat.get_data_sets()

sample = next(training_set())
n_pixels = sample.size
n_nodes = [n_pixels, n_pixels]