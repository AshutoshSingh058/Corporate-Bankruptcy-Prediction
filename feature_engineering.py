import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rename_map = {
            'X1': 'current_assets',
            'X2': 'cost_of_goods_sold',
            'X3': 'depreciation_amortization',
            'X4': 'ebitda',
            'X5': 'inventory',
            'X6': 'net_income',
            'X7': 'total_receivables',
            'X8': 'market_value',
            'X9': 'net_sales',
            'X10': 'total_assets',
            'X11': 'long_term_debt',
            'X12': 'ebit',
            'X13': 'gross_profit',
            'X14': 'current_liabilities',
            'X15': 'retained_earnings',
            'X16': 'total_revenue',
            'X17': 'total_liabilities',
            'X18': 'total_operating_expenses'
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.rename(columns=self.rename_map)

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        eps = 1e-6

        X["net_profit_margin"] = X["net_income"] / (X["total_revenue"] + eps)
        X["gross_profit_margin"] = X["gross_profit"] / (X["net_sales"] + eps)
        X["roa"] = X["net_income"] / (X["total_assets"] + eps)
        X["ros"] = X["net_income"] / (X["net_sales"] + eps)
        X["current_ratio"] = X["current_assets"] / (X["current_liabilities"] + eps)
        X["quick_ratio"] = (X["current_assets"] - X["inventory"]) / (X["current_liabilities"] + eps)
        X["debt_to_asset_ratio"] = X["total_liabilities"] / (X["total_assets"] + eps)

        return X.replace([np.inf, -np.inf], 0).fillna(0)
