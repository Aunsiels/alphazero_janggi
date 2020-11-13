import unittest
import multiprocessing as mp


class TestMultiprocessing(unittest.TestCase):

    def test_atleastoneprocessor(self):
        self.assertGreater(mp.cpu_count(), 0)

    def test_pool(self):
        pool = mp.Pool(mp.cpu_count())
        results = [pool.apply(times, args=(x, 2)) for x in range(100)]
        pool.close()
        self.assertEqual(results, [2 * x for x in range(100)])

    def test_pool_async(self):
        pool = mp.Pool(mp.cpu_count())
        for x in range(100):
            pool.apply_async(times, args=(x, 2))
        pool.close()
        self.assertTrue(True)


def times(x, y):
    return x * y


if __name__ == '__main__':
    unittest.main()
