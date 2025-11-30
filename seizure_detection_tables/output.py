import pandas as pd; 
df = pd.read_excel('seizure_detection_tables_MP/window/Sensitivity/seizure_analysis_Sensitivity.xlsx')
print(df.head(20)); print('\nColumns:', df.columns.tolist())