import unittest
import numpy as np
from somoclu import Somoclu


class DeterministicCodebook(unittest.TestCase):

    def test_deterministic_codebook(self):
        n_rows, n_columns = 2, 2
        codebook = np.zeros((2*2, 2), dtype=np.float32)
        data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        som = Somoclu(n_columns, n_rows, data=data, initialcodebook=codebook)
        som.train()
        correct_codebook = np.array([[[ 0.2,  0.29999998],
                                      [ 0.15378828,  0.25378826]],
                                     [[ 0.24621172,  0.34621173],
                                      [ 0.2       ,  0.29999998]]], dtype=np.float32)
        self.assertTrue(sum(codebook.reshape((n_rows*n_columns*2)) -
                            correct_codebook.reshape((n_rows*n_columns*2))) < 10e-8)



if __name__ == '__main__':
    test_main()
