# impoort necessary libraries
# 
# impoort necessary libraries
#
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

from sklearn.impute        import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedKFold)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    BaggingClassifier, AdaBoostClassifier)
from sklearn.tree    import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score, f1_score)
from imblearn.over_sampling import SMOTE

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


# categorical variables: bar charts wtih count labels:
fig, axes = plt.subplots(2,3, figsize=(18,11))
fig.suptitle('Univariate Analysis - Cateogrical Variables', fontsize=14, fontweight='bold', y = 1.01)
axes = axes.flatten()

# Gender
ax = axes[0]
vc = df['Gender'].value_counts().sort_index()
bars = ax.bar(['Male (0)', 'Female (1)'], vc.values, color=[BLUE, PURPLE], edgecolor='white', width=0.5)
ax.bar_label(bars, labels=[f'{v:,}\n({v/len(df)*100:.1f}%)' for v in vc.values],padding=4,fontsize=10,fontweight='bold')
ax.set_title('Gender Distribution')
ax.set_ylabel('Count'); ax.set_ylim(0,max(vc.values)*1.2)

# Education level
ax = axes[1]
vc = df['Education_Level'].value_counts().sort_index()
bars = ax.bar(['10+  (0)', '12+  (1)', 'Graduate (2)'], vc.values,
              color=[AMBER, GREEN, BLUE], edgecolor='white', width=0.5)
ax.bar_label(bars, labels=[f'{v:,}\n({v/len(df)*100:.1f}%)' for v in vc.values],
             padding=4, fontsize=10, fontweight='bold')
ax.set_title('Education Level Distribution')
ax.set_ylabel('Count'); ax.set_ylim(0, max(vc.values)*1.2)

# Joining Designation
ax = axes[2]
vc = df['Joining Designation'].value_counts().sort_index()
bars = ax.bar([str(k) for k in vc.index], vc.values,
              color=PAL[:len(vc)], edgecolor='white', width=0.6)
ax.bar_label(bars, labels=[f'{v:,}' for v in vc.values],
             padding=4, fontsize=9, fontweight='bold')
ax.set_title('Joining Designation Distribution')
ax.set_xlabel('Designation Level'); ax.set_ylabel('Count')

# Grade 
ax = axes[3]
vc = df['Grade'].value_counts().sort_index()
bars = ax.bar([str(k) for k in vc.index], vc.values,
              color=PAL[:len(vc)], edgecolor='white', width=0.6)
ax.bar_label(bars, labels=[f'{v:,}' for v in vc.values],
             padding=4, fontsize=9, fontweight='bold')
ax.set_title('Grade Distribution')
ax.set_xlabel('Grade Level'); ax.set_ylabel('Count')

# Quarterly Rating 
ax = axes[4]
vc = df['Quarterly Rating'].value_counts().sort_index()
bars = ax.bar([str(k) for k in vc.index], vc.values,
              color=[RED, AMBER, GREEN, BLUE], edgecolor='white', width=0.5)
ax.bar_label(bars, labels=[f'{v:,}\n({v/len(df)*100:.1f}%)' for v in vc.values],
             padding=4, fontsize=7, fontweight='bold')
ax.set_title('Quarterly Rating Distribution (1=Low, 4=High)')
ax.set_xlabel('Quarterly Rating'); ax.set_ylabel('Count')

# Top 10 Cities 
ax = axes[5]
top_cities = df['City'].value_counts().head(10)
bars = ax.bar(top_cities.index.astype(str), top_cities.values,
              color=PURPLE, edgecolor='white', alpha=0.85)
ax.bar_label(bars, labels=[f'{v:,}' for v in top_cities.values],
             padding=4, fontsize=8, fontweight='bold')
ax.set_title(f'Top 10 Cities by Driver Count\n(Total: {df["City"].nunique()} unique cities)')
ax.set_xlabel('City Code'); ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout(); plt.show()


# Bivariate Plot: Quarterly Rating vs Churn Rate
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Bivariate: Quarterly Rating vs Churn', fontsize=13, fontweight='bold')

# churn rate % per rating
ax = axes[0]
cr_qr = df.groupby('Quarterly Rating')['churned'].mean() * 100
colors_qr = [RED, AMBER, GREEN, BLUE]
bars = ax.bar(cr_qr.index.astype(str), cr_qr.values, color=colors_qr, edgecolor='white', width=0.6)
ax.bar_label(bars, labels=[f'{v:.1f}%' for v in cr_qr.values],
             padding=4, fontsize=11, fontweight='bold')
ax.set_title('Churn Rate % by Quarterly Rating')
ax.set_xlabel('Quarterly Rating (1=Low, 4=High)')
ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, max(cr_qr.values) * 1.25)


# Count of churned vs retained per rating (stacked bar)
ax = axes[1]
ct = pd.crosstab(df['Quarterly Rating'], df['churned'])
ct.columns = ['Retained', 'Churned']
ct.plot(kind='bar', ax=ax, color=[BLUE, RED], edgecolor='white', width=0.65)
ax.set_title('Count: Retained vs Churned by Rating')
ax.set_xlabel('Quarterly Rating'); ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)
ax.legend(loc='upper right')

plt.tight_layout(); plt.show()


# Bivariate Plot: Grade Vs Chrun Rate
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Bivariate: Grade vs Churn', fontsize=13, fontweight='bold')

ax = axes[0]
cr_g = df.groupby('Grade')['churned'].mean() * 100
bars = ax.bar(cr_g.index.astype(str), cr_g.values, color=PAL[:len(cr_g)], edgecolor='white', width=0.6)
ax.bar_label(bars, labels=[f'{v:.1f}%' for v in cr_g.values],
             padding=4, fontsize=11, fontweight='bold')
ax.set_title('Churn Rate % by Grade')
ax.set_xlabel('Grade Level'); ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, max(cr_g.values) * 1.3)

# TBV by Grade
ax = axes[1]
tbv_grade = df.groupby('Grade')['Total Business Value'].mean() / 1000   # in thousands
bars = ax.bar(tbv_grade.index.astype(str), tbv_grade.values, color=GREEN, edgecolor='white', width=0.6, alpha=0.85)
ax.bar_label(bars, labels=[f'₹{v:.0f}K' for v in tbv_grade.values],
             padding=4, fontsize=10, fontweight='bold')
ax.set_title('Average Business Value by Grade')
ax.set_xlabel('Grade Level'); ax.set_ylabel('Avg TBV (₹ Thousands)')

plt.tight_layout(); plt.show()


# Bivariate plot: Education level vs Churn Rate.
fig, ax = plt.subplots(figsize=(8, 5))
cr_edu = df.groupby('Education_Level')['churned'].mean() * 100
bars = ax.bar(['10+  (0)', '12+  (1)', 'Graduate (2)'], cr_edu.values,
              color=[AMBER, GREEN, BLUE], edgecolor='white', width=0.5)
ax.bar_label(bars, labels=[f'{v:.1f}%' for v in cr_edu.values],
             padding=4, fontsize=12, fontweight='bold')
ax.set_title('Bivariate: Churn Rate % by Education Level', fontsize=13, fontweight='bold')
ax.set_xlabel('Education Level'); ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, max(cr_edu.values) * 1.3)
plt.tight_layout(); plt.show()


# Bivariate plot: Income and Age vs Churn Status.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Bivariate: Income & Age vs Churn Status', fontsize=13, fontweight='bold')

# Income
ax = axes[0]
for v, label, c, a in [(0,'Retained',BLUE,0.65),(1,'Churned',RED,0.65)]:
    subset = df[df['churned']==v]['Income']
    ax.hist(subset, bins=35, alpha=a, color=c, label=f'{label} (mean=₹{subset.mean():,.0f})',
            edgecolor='white')
ax.set_title('Income Distribution by Churn Status')
ax.set_xlabel('Monthly Income (₹)'); ax.set_ylabel('Count')
ax.legend(fontsize=9)

# Age
ax = axes[1]
for v, label, c in [(0,'Retained',BLUE),(1,'Churned',RED)]:
    subset = df[df['churned']==v]['Age'].dropna().astype(float)
    ax.hist(subset, bins=28, alpha=0.65, color=c, label=f'{label} (mean={subset.mean():.1f} yrs)',
            edgecolor='white')
ax.set_title('Age Distribution by Churn Status')
ax.set_xlabel('Age'); ax.set_ylabel('Count')
ax.legend(fontsize=9)

plt.tight_layout(); plt.show()

# Bivariate Plot: Total Business Value vs Churn Status:

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Bivariate: Total Business Value vs Churn Status', fontsize=13, fontweight='bold')

# Boxplot
ax = axes[0]
clip_val = 1_500_000
d0 = df[df['churned']==0]['Total Business Value'].clip(-clip_val, clip_val)
d1 = df[df['churned']==1]['Total Business Value'].clip(-clip_val, clip_val)
bp = ax.boxplot([d0, d1], patch_artist=True, labels=['Retained','Churned'],
                medianprops=dict(color='black', lw=2.5),
                whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5),
                flierprops=dict(marker='o', markerfacecolor=AMBER, markersize=3, alpha=0.4))
for patch, c in zip(bp['boxes'], [BLUE, RED]):
    patch.set_facecolor(c); patch.set_alpha(0.55)
ax.set_title(f'Boxplot — TBV by Churn Status (clipped to ±₹1.5M)')
ax.set_ylabel('Total Business Value (₹)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))

# Mean TBV bar
ax = axes[1]
tbv_churn = df.groupby('churned')['Total Business Value'].mean() / 1000
bars = ax.bar(['Retained','Churned'], tbv_churn.values, color=[BLUE, RED], edgecolor='white', width=0.5)
ax.bar_label(bars, labels=[f'₹{v:.0f}K' for v in tbv_churn.values],
             padding=5, fontsize=12, fontweight='bold')
ax.set_title('Mean Business Value: Retained vs Churned')
ax.set_ylabel('Avg Total Business Value (₹ Thousands)')
ax.set_ylim(0, max(tbv_churn.values) * 1.25)

plt.tight_layout(); plt.show()

# Bivariate Plot: Correlation Heatmap 
num_df = df[['Age','Income','Total Business Value',
             'Quarterly Rating','Grade','churned']].copy()
for c in num_df.columns:
    num_df[c] = pd.to_numeric(num_df[c], errors='coerce')

corr = num_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))    # show lower triangle only
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, square=True,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 11, 'weight': 'bold'}, ax=ax,
            vmin=-1, vmax=1)
ax.set_title('Correlation Heatmap — Numerical Features\n(Negative correlation with churned = good retention predictor)',
             fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

# Complete outlier analysis using IQR 
print('=== Outlier Analysis — IQR Method ===')
print(f'{"Column":<25} {"Q1":>10} {"Q3":>10} {"IQR":>10} {"Lower":>12} {"Upper":>12} {"Outliers":>10} {"Action"}')
print('─'*110)

outlier_info = {
    'Age'                 : 'Retain — business-plausible senior drivers',
    'Income'              : 'Retain — genuine high-performing drivers',
    'Total Business Value': 'Cap at 1st–99th percentile (Winsorize)'
}

for col in ['Age', 'Income', 'Total Business Value']:
    data = df[col].dropna().astype(float)
    Q1 = data.quantile(0.25); Q3 = data.quantile(0.75); IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR; upper = Q3 + 1.5 * IQR
    n_out = ((data < lower) | (data > upper)).sum()
    pct   = n_out / len(data) * 100
    print(f'{col:<25} {Q1:>10.0f} {Q3:>10.0f} {IQR:>10.0f} {lower:>12.0f} {upper:>12.0f} '
          f'{n_out:>5} ({pct:.1f}%)  {outlier_info[col]}')
    
    # Skewness summary 
print('=== Skewness Analysis ===')
skew_data = {
    'Age'                 : (df['Age'].skew(),       'Mild right skew',   'Slight excess of older drivers'),
    'Income'              : (df['Income'].skew(),    'Moderate right skew', 'Most earn moderate; few earn very high'),
    'Total Business Value': (df['Total Business Value'].skew(), 'Severe right skew (+6.97)', 'Extreme outliers from top performers'),
}
print(f'{"Column":<25} {"Skew Value":>12} {"Type":<25} {"Business Reason"}')
print('─'*95)
for col, (skew, stype, reason) in skew_data.items():
    print(f'{col:<25} {skew:>12.2f} {stype:<25} {reason}')


# EDA summary dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('EDA Summary — Key Insights Dashboard', fontsize=15, fontweight='bold', y=1.01)

# Churn rate by Rating (most important)
ax = axes[0,0]
cr_qr = df.groupby('Quarterly Rating')['churned'].mean() * 100
ax.bar(cr_qr.index.astype(str), cr_qr.values, color=[RED,AMBER,GREEN,BLUE], edgecolor='white')
for j, v in enumerate(cr_qr.values):
    ax.text(j, v+0.3, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Churn Rate % by Quarterly Rating\n⭐ Strongest predictor')
ax.set_xlabel('Rating'); ax.set_ylabel('Churn Rate (%)')

# Income by churn (KDE)
ax = axes[0,1]
for v, label, c in [(0,'Retained',BLUE),(1,'Churned',RED)]:
    df[df['churned']==v]['Income'].plot(kind='density', ax=ax, color=c, lw=2, label=label)
ax.set_title('Income Density by Churn Status')
ax.set_xlabel('Income (₹)'); ax.legend()

# Grade vs Churn
ax = axes[0,2]
cr_g = df.groupby('Grade')['churned'].mean() * 100
ax.bar(cr_g.index.astype(str), cr_g.values, color=PAL[:len(cr_g)], edgecolor='white')
for j, v in enumerate(cr_g.values):
    ax.text(j, v+0.2, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Churn Rate % by Grade')
ax.set_xlabel('Grade'); ax.set_ylabel('Churn Rate (%)')

# TBV by churn status (violin)
ax = axes[1,0]
tbv_clip = df.copy()
tbv_clip['TBV_clipped'] = tbv_clip['Total Business Value'].clip(-500_000, 3_000_000)
retained_tbv = tbv_clip[tbv_clip['churned']==0]['TBV_clipped']
churned_tbv  = tbv_clip[tbv_clip['churned']==1]['TBV_clipped']
ax.violinplot([retained_tbv, churned_tbv], positions=[1,2], showmedians=True)
ax.set_xticks([1,2]); ax.set_xticklabels(['Retained','Churned'])
ax.set_title('Business Value Distribution by Churn\n(Clipped for readability)')
ax.set_ylabel('Total Business Value (₹)')

# Correlation bar chart (with churned)
ax = axes[1,1]
corr_with_target = num_df.corr()['churned'].drop('churned').sort_values()
colors_corr = [RED if v < 0 else GREEN for v in corr_with_target.values]
bars = ax.barh(corr_with_target.index, corr_with_target.values, color=colors_corr, edgecolor='white')
ax.axvline(0, color='black', lw=0.8)
ax.bar_label(bars, labels=[f'{v:.2f}' for v in corr_with_target.values],
             padding=3, fontsize=6, fontweight='bold')
ax.set_title('Correlation with Churn Target\n(Red=negative = protective factor)')
ax.set_xlabel('Pearson Correlation Coefficient')

# Monthly reporting count (temporal)
ax = axes[1,2]
monthly = df.groupby('MMM-YY').size()
ax.plot(monthly.index, monthly.values, color=BLUE, lw=2.5, marker='o', markersize=4)
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color=BLUE)
ax.set_title('Monthly Record Volume\n(Temporal distribution of data)')
ax.set_xlabel('Month'); ax.set_ylabel('Record Count')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout(); plt.show()


# Checking identical rows
exact_dups = df.duplicated().sum()
print(f'Exact duplicate rows: {exact_dups}')

# Same driver appearing more than once in same month
subset_dups = df.duplicated(subset=['Driver_ID','MMM-YY']).sum()
print(f'Driver + MOnth duplicates : {subset_dups}')

# Driver Id appearing in multiple months
records_per_driver = df.groupby('Driver_ID').size()
print(f'\nRecords per driver (expected multiple months):')
print(f'  Min : {records_per_driver.min()}')
print(f'  Max : {records_per_driver.max()}')
print(f'  Mean: {records_per_driver.mean():.1f}')
print(f'  Drivers with only 1 record: {(records_per_driver == 1).sum()}')

# Missing value Treatment -- KNN imputation
missing = df.isnull().sum()
print(missing[missing > 0])
print(f'\nNote: LastWorkingDate ({missing["LastWorkingDate"]:,} missing) is NOT imputed —')
print('      it is missing because the driver is still working (retained). Used to create target.')

age_missing_idx    = df['Age'].isna()
gender_missing_idx = df['Gender'].isna()

print(f'Age   missing: {age_missing_idx.sum()} rows → indices: {df[age_missing_idx].index[:5].tolist()} ...')
print(f'Gender missing: {gender_missing_idx.sum()} rows')

num_impute_cols = [
    'Age', 'Gender', 'Income', 'Education_Level',
    'Joining Designation', 'Grade',
    'Total Business Value', 'Quarterly Rating'
]

imputer = KNNImputer(n_neighbors=5)
df[num_impute_cols] = imputer.fit_transform(df[num_impute_cols])
df[num_impute_cols]
# Round Gender back to 0 or 1 (binary variable — KNN may give 0.4, 0.8 etc.)
df['Gender'] = df['Gender'].round().astype(int)

print('=== Missing Values AFTER KNN Imputation ===')
print(df[num_impute_cols].isnull().sum())
print('\n All missing values filled — zero nulls in numerical columns')

# Visualise before vs after for age.
# We compare imputed age vs overalll distribution to verify quality.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('KNN Imputation Quality Check — Age', fontsize=13, fontweight='bold')

# Left: full age distribution after imputation
ax = axes[0]
ax.hist(df['Age'], bins=35, color=BLUE, edgecolor='white', alpha=0.8, label='All ages (post-imputation)')
# Mark imputed values
imputed_ages = df.loc[age_missing_idx, 'Age']
ax.hist(imputed_ages, bins=20, color=RED, edgecolor='white', alpha=0.7, label=f'Imputed values (n={len(imputed_ages)})')
ax.axvline(df['Age'].mean(), color='black', lw=2, ls='--', label=f'Mean={df["Age"].mean():.1f}')
ax.set_title('Age Distribution — Imputed values highlighted')
ax.set_xlabel('Age'); ax.set_ylabel('Count')
ax.legend(fontsize=9)

# Right: gender count before vs after
ax = axes[1]
vc_gender = df['Gender'].value_counts().sort_index()
bars = ax.bar(['Male (0)', 'Female (1)'], vc_gender.values, color=[BLUE, PURPLE], edgecolor='white', width=0.5)
ax.bar_label(bars, labels=[f'{v:,}\n({v/len(df)*100:.1f}%)' for v in vc_gender.values],
             padding=4, fontsize=11, fontweight='bold')
ax.set_title(f'Gender Distribution After Imputation\n(52 missing values filled)')
ax.set_ylabel('Count'); ax.set_ylim(0, max(vc_gender.values)*1.2)

plt.tight_layout(); plt.show()

#  IQR-based outlier detection for all 3 numerical columns
print('=== Outlier Analysis — IQR Method (1.5×IQR Rule) ===')
print(f'{"Column":<25} {"Min":>12} {"Max":>12} {"Q1":>10} {"Q3":>10} {"Outliers":>10} {"Action"}')
print('─'*100)

outlier_actions = {
    'Age'                 : 'RETAIN — ages 21–58 are all realistic driver ages',
    'Income'              : 'RETAIN — high earners are genuine top performers',
    'Total Business Value': 'CAP — extreme values distort model training'
}

for col in ['Age', 'Income', 'Total Business Value']:
    data = df[col].astype(float)
    Q1 = data.quantile(0.25); Q3 = data.quantile(0.75); IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR;   upper = Q3 + 1.5 * IQR
    n_out = ((data < lower) | (data > upper)).sum()
    pct   = n_out / len(data) * 100
    print(f'{col:<25} {data.min():>12.0f} {data.max():>12.0f} {Q1:>10.0f} {Q3:>10.0f} '
          f'{n_out:>5} ({pct:.1f}%)  {outlier_actions[col]}')

# ── Visualise outliers: before vs after capping ───────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Outlier Treatment — Total Business Value', fontsize=13, fontweight='bold')

# Before: histogram
ax = axes[0, 0]
ax.hist(df['Total Business Value'], bins=50, color=AMBER, edgecolor='white', alpha=0.8)
ax.set_title('TBV Histogram — BEFORE Capping')
ax.set_xlabel('Total Business Value (₹)'); ax.set_ylabel('Frequency')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))

# Before: boxplot
ax = axes[0, 1]
bp = ax.boxplot(df['Total Business Value'], patch_artist=True,
                boxprops=dict(facecolor=AMBER, alpha=0.6),
                medianprops=dict(color='black', lw=2),
                flierprops=dict(marker='o', markerfacecolor=RED, markersize=3, alpha=0.4))
ax.set_title('TBV Boxplot — BEFORE Capping')
ax.set_ylabel('Total Business Value (₹)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))

# ── Apply Winsorization: cap at 1st–99th percentile ──────────────────────────
p1  = df['Total Business Value'].quantile(0.01)
p99 = df['Total Business Value'].quantile(0.99)
tbv_before = df['Total Business Value'].copy()   # keep original for plot
df['Total Business Value'] = df['Total Business Value'].clip(lower=p1, upper=p99)

print(f'Capped at: lower = ₹{p1:,.0f} (1st percentile)')
print(f'           upper = ₹{p99:,.0f} (99th percentile)')
print(f'Range before: ₹{tbv_before.min():,.0f} → ₹{tbv_before.max():,.0f}')
print(f'Range after : ₹{df["Total Business Value"].min():,.0f} → ₹{df["Total Business Value"].max():,.0f}')

# After: histogram
ax = axes[1, 0]
ax.hist(df['Total Business Value'], bins=50, color=GREEN, edgecolor='white', alpha=0.8)
ax.set_title('TBV Histogram — AFTER Capping (1st–99th pct)')
ax.set_xlabel('Total Business Value (₹)'); ax.set_ylabel('Frequency')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))

# After: boxplot
ax = axes[1, 1]
bp = ax.boxplot(df['Total Business Value'], patch_artist=True,
                boxprops=dict(facecolor=GREEN, alpha=0.6),
                medianprops=dict(color='black', lw=2),
                flierprops=dict(marker='o', markerfacecolor=RED, markersize=3, alpha=0.4))
ax.set_title('TBV Boxplot — AFTER Capping')
ax.set_ylabel('Total Business Value (₹)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))

plt.tight_layout(); plt.show()

# Feature Engineering
# ── Sort by Driver_ID and date BEFORE aggregating ─────────────────────────────
# Critical: first/last values only make sense when data is chronologically sorted
df.sort_values(['Driver_ID', 'MMM-YY'], inplace=True)
g = df.groupby('Driver_ID')

# Build aggregated driver-level dataframe
agg = pd.DataFrame({

    # ── Static features: take FIRST known value (don't change over time) ──────
    'Driver_ID'      : g['Driver_ID'].first().values,
    'Age'            : g['Age'].first().values,
    'Gender'         : g['Gender'].first().values,
    'City'           : g['City'].first().values,
    'Education_Level': g['Education_Level'].first().values,
    'Joining_Desig'  : g['Joining Designation'].first().values,

    # ── Grade: take LAST (most recent grade is most relevant) ─────────────────
    'Grade_last'     : g['Grade'].last().values,

    # ── Income: capture both endpoints + average ──────────────────────────────
    'Income_first'   : g['Income'].first().values,   # will be used for trend flag
    'Income_last'    : g['Income'].last().values,    # will be used for trend flag
    'Income_mean'    : g['Income'].mean().values,    # kept as a model feature

    # ── Total Business Value: mean and sum as productivity metrics ────────────
    'TBV_mean'       : g['Total Business Value'].mean().values,
    'TBV_sum'        : g['Total Business Value'].sum().values,

    # ── Quarterly Rating: capture both endpoints for trend detection ──────────
    'QR_first'       : g['Quarterly Rating'].first().values,
    'QR_last'        : g['Quarterly Rating'].last().values,

    # ── Dates: needed for feature engineering below ───────────────────────────
    'Dateofjoining'  : g['Dateofjoining'].first().values,
    'LastWorkingDate': g['LastWorkingDate'].first().values,
})

print(f'✅ Aggregated: {len(df):,} monthly rows → {len(agg):,} unique drivers')
print(f'   Shape: {agg.shape}')
agg.head(4)

