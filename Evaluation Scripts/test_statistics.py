"""Small deterministic checks; no inference or dataset downloads."""
import unittest
from core import wilson, mcnemar, holm

class StatisticsTests(unittest.TestCase):
    def test_exact_paired_test(self):
        self.assertEqual(mcnemar(0,0),1)
        self.assertEqual(mcnemar(0,12),0.00048828125)
        self.assertAlmostEqual(mcnemar(1,4),0.375)
        self.assertEqual(mcnemar(4,1),mcnemar(1,4))

    def test_holm_preserves_order(self):
        self.assertEqual(holm([.04,.01,.03]),[.06,.03,.06])
        self.assertEqual(holm([]),[])

    def test_wilson_boundaries(self):
        self.assertEqual(wilson(0,0),(None,None))
        lo,hi=wilson(0,100)
        self.assertEqual(lo,0)
        self.assertAlmostEqual(hi,0.03699349820698568)
        self.assertEqual(wilson(100,100)[1],1)

if __name__=='__main__':unittest.main()
