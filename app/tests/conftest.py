"""
Shared pytest fixtures for the bedrock-genai test suite.
"""
import pandas as pd
import pytest


# ── Sample DataFrames ────────────────────────────────────────────────────────

@pytest.fixture
def employees_df():
    """Simple employee DataFrame for engine tests."""
    return pd.DataFrame(
        {
            "employee_id": [1001, 1002, 1003, 1004, 1005],
            "name": ["Alice Smith", "Bob Jones", "Carol Lee", "Dave Brown", "Eve White"],
            "department": ["Engineering", "Marketing", "Engineering", "HR", "Marketing"],
            "salary": [90000, 75000, 95000, 60000, 80000],
            "hire_date": ["2020-01-15", "2019-06-01", "2021-03-10", "2018-11-22", "2022-07-05"],
            "notes": [
                "Strong Python skills",
                "Leads social media campaigns",
                "Senior engineer, great mentor",
                "Handles payroll and benefits",
                "Digital marketing specialist",
            ],
        }
    )


@pytest.fixture
def products_df():
    """Product catalog DataFrame."""
    return pd.DataFrame(
        {
            "product_id": [101, 102, 103, 104],
            "product_name": ["Laptop Pro", "Budget Tablet", "Wireless Mouse", "USB Hub"],
            "category": ["Electronics", "Electronics", "Accessories", "Accessories"],
            "price": [1299.99, 299.99, 29.99, 19.99],
            "stock": [50, 200, 500, 300],
        }
    )


@pytest.fixture
def sales_df():
    """Sales DataFrame with date column as YYYYMMDD integer."""
    return pd.DataFrame(
        {
            "sale_id": [1, 2, 3, 4, 5],
            "amount": [1000.0, 2500.0, 750.0, 3200.0, 600.0],
            "sale_date": [20230101, 20230215, 20230310, 20230415, 20230520],
            "region": ["North", "South", "North", "East", "West"],
        }
    )


@pytest.fixture
def text_heavy_df():
    """DataFrame with long text columns for text engine tests."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Report on Q1 performance", "Annual budget review 2023", "Team meeting notes"],
            "content": [
                "This quarter showed strong revenue growth across all business units.",
                "The annual budget proposal includes increases in marketing and R&D spend.",
                "Key decisions: new hire approved, roadmap updated for next sprint.",
            ],
        }
    )
