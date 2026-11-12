"""Exercise order-date preprocessing without warehouse access or model training."""
import ast
from pathlib import Path
import unittest

import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / 'churn_prediction.py'


def preprocess_orders(order_numbers):
    tree = ast.parse(SOURCE.read_text())
    targets = {"orderitem_df['OrderDate']", 'orderitem_df_recent'}
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and ast.unparse(node.targets[0]) in targets]
    if len(nodes) != 2:
        raise AssertionError('Expected the date extraction and recent-order filter')
    frame = pd.DataFrame({'OrderNo': order_numbers})
    namespace = {'pd': pd, 'orderitem_df': frame}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return frame, namespace['orderitem_df_recent']


class OrderDateTests(unittest.TestCase):
    def test_string_numeric_and_mixed_ids_keep_the_same_dates(self):
        fixtures = [
            ['202401010001', '202401020002', '202402290003'],
            [202401010001, 202401020002, 202402290003],
            ['202401010001', 202401020002, '202402290003'],
        ]
        for values in fixtures:
            with self.subTest(values=values):
                original = pd.Series(values, name='OrderNo')
                frame, recent = preprocess_orders(original)
                self.assertEqual(frame['OrderDate'].dt.strftime('%Y-%m-%d').tolist(),
                                 ['2024-01-01', '2024-01-02', '2024-02-29'])
                # Keep the existing strict cutoff and the original join keys/dtype.
                self.assertEqual(recent.index.tolist(), [1, 2])
                pd.testing.assert_series_equal(frame['OrderNo'], original)

    def test_missing_ids_are_not_recent_orders(self):
        for values in [pd.Series(['202401020001', None], dtype=object),
                       pd.Series([202401020001, pd.NA], dtype='Int64')]:
            with self.subTest(dtype=str(values.dtype)):
                frame, recent = preprocess_orders(values)
                self.assertTrue(pd.isna(frame.loc[1, 'OrderDate']))
                self.assertEqual(recent.index.tolist(), [0])

    def test_empty_input_keeps_an_empty_result(self):
        frame, recent = preprocess_orders(pd.Series([], dtype='int64'))
        self.assertTrue(frame.empty)
        self.assertTrue(recent.empty)
        self.assertIn('OrderDate', recent.columns)

    def test_invalid_dates_still_raise(self):
        for value in ['not-an-order', '202402300001', 202402300001]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                preprocess_orders([value])


if __name__ == '__main__':
    unittest.main()
