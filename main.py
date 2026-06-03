# impoort necessary libraries
# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

#  Global style 
plt.rcParams.update({
    'figure.facecolor' : '#F8FAFC',
    'axes.facecolor'   : '#F8FAFC',
    'axes.grid'        : True,
    'grid.alpha'       : 0.35,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.titlesize'   : 13,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
})
BLUE   = '#2563EB'
AMBER  = '#F59E0B'
GREEN  = '#10B981'
RED    = '#EF4444'
PURPLE = '#8B5CF6'
PAL    = [BLUE, AMBER, GREEN, RED, PURPLE, '#F97316']
sns.set_palette(PAL)
print('✅ Libraries loaded')


# Load dataset
df = pd.read_csv('/content/ola_driver_scaler.csv')

print(f'  Dataset shape : {df.shape[0]:,} rows x {df.shape[1]} columns')
print(f'  Unique Drivers: {df['Driver_ID'].nunique():,}')
print(f'  Avg records per driver: {df.shape[0]/df['Driver_ID'].nunique():.1f} months')

# checking raw dtypes 
print(df.dtypes)

# Convert date columns to datetime:
date_cols = ['MMM-YY', 'Dateofjoining', 'LastWorkingDate']
for col in date_cols:
  df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=True, errors='coerce')

# convert categorical columns to category dtype.
cat_cols = ['Gender', 'City', 'Education_Level', 'Grade', 'Joining Designation']
for col in cat_cols:
  df[col] = df[col].astype('category')

# check dtypes
print(df.dtypes)

# missing values detections.
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0 ].sort_values('Missing Count',ascending=False))

# statistical summary 
df.describe()

# statitical summary for categorical columns.
df.describe(include='category')

# Continous variables: Histogram + box plots for each
cont_cols = ['Age', 'Income', 'Total Business Value']

fig, axes = plt.subplots(2,3, figsize=(18,10))
fig.suptitle('Univariate Analysis - Continous Variables\n(Row 1: Histograms | Row 2: Boxplots)', fontsize=14,fontweight ='bold',y=1.01)

for i, col in enumerate(cont_cols):
  data = df[col].dropna().astype(float)
  mean_val = data.mean()
  median_val = data.median()
  skew_val = data.skew()

  # Row 0: Histogram
  ax = axes[0, i]
  ax.hist(data, bins=40, color=PAL[i], edgecolor='white', alpha=0.85)
  ax.axvline(mean_val, color=RED, lw=2, ls='--',label=f'Mean = {mean_val:,.0f}')
  ax.axvline(median_val, color='black',lw=2,ls=':',label=f'Median = {median_val:,.0f}')
  ax.set_title(f'{col} - Histogram')
  ax.set_xlabel(col); ax.set_ylabel('Frequency')
  ax.legend(fontsize=8)
  ax.annotate(f'Skew = {skew_val:.2f}',xy=(0.97,0.95), xycoords='axes fraction', ha='right', va='top',fontsize=9,
              bbox=dict(
                  boxstyle='round,pad=0.3',
                  facecolor = 'white',
                  edgecolor = 'gray',
                  alpha = 0.7))
  # Row 1: Box plot
  ax = axes[1, i]
  bp = ax.boxplot(data, patch_artist=True, vert=True, widths=0.5,
                  boxprops    = dict(facecolor=PAL[i], alpha=0.55),
                  medianprops = dict(color='black', lw=2.5),
                  whiskerprops= dict(lw=1.5),
                  capprops    = dict(lw=1.5),
                  flierprops  = dict(marker='o', markerfacecolor=RED,
                                      markersize=3, alpha=0.4, lw=0))
  # IQR outliner count
  Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
  n_out = ((data < Q1-1.5*(Q3-Q1)) | (data > Q3+1.5*(Q3-Q1))).sum()
  ax.set_title(f'{col} - Boxplot (Outliers: {n_out:,})')
  ax.set_ylabel(col)

plt.tight_layout()
plt.show()