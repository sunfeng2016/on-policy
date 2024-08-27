import pandas as pd
import numpy as np
import os

def append_to_csv(data, filename='output.csv'):

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Replace numeric values in '打击' with corresponding text
    df['打击'] = df['打击'].replace({np.nan: np.nan, 0: '不打击', 1: '软毁伤', 2: '硬毁伤'})
    
    # Replace numeric values in '航向' with corresponding text
    df['航向'] = df['航向'].apply(lambda x: np.nan if pd.isnull(x) else (f'顺时针转{-x}度' if x < 0 else (f'逆时针转{x}度' if x > 0 else f'保持航向不变')))
    
    # Replace numeric values in '加速度' with corresponding text
    df['加速度'] = df['加速度'].apply(lambda x: np.nan if pd.isnull(x) else (f'加速{x}' if x > 0 else (f'减速{-x}' if x < 0 else '保持' )))

    # Save DataFrame to CSV
    df.to_csv(filename, index=False, encoding='utf-8', mode='w')
