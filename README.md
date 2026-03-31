## Coordinated Relational Restructuring of the Gut Microbiome and Serum Metabolome Encodes Lung Adenocarcinoma Progression


<img src="figures/flurico_banner.png" align="right"  width="300">

<div style="display: flex; align-items: center;">
  <img src="figures/flurico_icon.png"  width="300" style="margin-right: 10px;">
  
</div>

### *Fluctuation, Remote Interaction and Coordination*







## Usage
1. *statsmodels* and *scipy* packages are essential for FluRICo. You should at least install these 2 packages in your python environment.
2. Prepare your own data. Co-abundance for microbiome should follow https://github.com/scwatts/fastspar guide.
2. Check the example in FluRICo_example.ipynb.
```python
# Quick Start
from core.flurico import FluRiCoAnalysis

# Load your data
df_mb = pd.read_csv('xxx/mb.csv',index_col=0) 
df_mt = pd.read_csv('xxx/mt.csv',index_col=0)

# Load co-coabundance data
# Fastspar co-abundance should be calculated within each group
df_mb_corr = {}
for i in ['Co_1','Co_2','Co_3']:
    df_mb_corr[i] = pd.read_csv(f'xxx/{i}_coabundance.tsv',sep='\t',index_col=0)

# Define your analyzer
analyzer = FluRiCoAnalysis(microbe_data=df_mb, metabolite_data=df_mt, 
                           group_labels=['Co_1','Co_2','Co_3'], microbe_coab_data = df_mb_corr)

# start flurico scoring
all_scores = analyzer.calculate_all_scores()

# filter top 20% feature for microbiome and metabolome in each cohort
analyzer.save_results(output_dir='your_dir')

```
