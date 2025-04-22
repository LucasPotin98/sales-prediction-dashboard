import pandas as pd

def compute_seasonality(df, selected_families):
    df = df.copy()
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)
    seasonality = (
        df[df['family'].isin(selected_families)]
        .groupby(['month', 'family'])['quantity']
        .sum()
        .reset_index()
    )
    return seasonality

def compute_family_distribution(df, selected_families):
    df = df.copy()
    filtered = df[df['family'].isin(selected_families)]
    grouped = filtered.groupby(['family', 'product_label'])['quantity'].sum().reset_index()
    return grouped
