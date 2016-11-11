import unittest
import numpy as np
from somoclu import Somoclu


class DeterministicCodebook(unittest.TestCase):

    def test_deterministic_codebook(self):
        n_rows, n_columns = 2, 2
        codebook = np.zeros((2*2, 2), dtype=np.float32)
        data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        som = Somoclu(n_columns, n_rows, data=data, initialcodebook=codebook,
                      compactsupport=False)
        som.train()
        correct_codebook = np.array([[[ 0.2       ,  0.30000001],
                                      [ 0.10359724,  0.20359723]],
                                     [[ 0.29640275,  0.39640275],
                                      [ 0.2       ,  0.30000001]]], dtype=np.float32)
        self.assertTrue(sum(codebook.reshape((n_rows*n_columns*2)) -
                            correct_codebook.reshape((n_rows*n_columns*2))) < 10e-8)


if __name__ == '__main__':
    test_main()
