import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

from sklearn.impute          import KNNImputer
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     ConfusionMatrixDisplay, roc_auc_score,
                                     roc_curve, accuracy_score,
                                     precision_score, recall_score, f1_score)
from imblearn.over_sampling  import SMOTE

plt.rcParams.update({
    'figure.facecolor' : '#F8FAFC', 'axes.facecolor'   : '#F8FAFC',
    'axes.grid': True, 'grid.alpha': 0.35,
    'axes.spines.top'  : False,     'axes.spines.right' : False,
    'axes.titlesize'   : 13,        'axes.titleweight'  : 'bold',
    'axes.labelsize'   : 11,        'xtick.labelsize'   : 9,
    'ytick.labelsize'  : 9,
})
BLUE='#2563EB'; AMBER='#F59E0B'; GREEN='#10B981'
RED='#EF4444';  PURPLE='#8B5CF6'; ORANGE='#F97316'
PAL=[BLUE,AMBER,GREEN,RED,PURPLE,ORANGE]
sns.set_palette(PAL)

# Load dataset
df = pd.read_csv('/content/ola_driver_scaler.csv')

print(f'  Dataset shape : {df.shape[0]:,} rows x {df.shape[1]} columns')
print(f'  Unique Drivers: {df['Driver_ID'].nunique():,}')
print(f'  Avg records per driver: {df.shape[0]/df['Driver_ID'].nunique():.1f} months')


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
