# Reading the churn model output

The model considers customers with a size 3 purchase. Label `1` means a size 4 purchase appears in the available order history. Label `0` means no such purchase was observed in that history. A missing purchase does not establish that the customer permanently left.

The final prediction table contains the held-out test split. The printed precision, recall and F1 scores are weighted averages across both labels. To evaluate how many observed non-transitioners were detected, inspect recall for label `0` separately.

When using a different product catalog, check the size labels first. Size flags recognize labels such as `1step` and `3step`. A leading digit is excluded, so `11step` is not counted as size 1 and `13step` is not counted as size 3.
