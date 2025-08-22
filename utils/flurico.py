import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr, levene,pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

class FluRiCoAnalysis:
    """
    FluRiCo Analysis Class: Calculate Fluctuation, Remote Interaction, Coordination
    """
  
    def __init__(self, microbe_data=None, metabolite_data=None, 
                 group_labels=None, microbe_coab_data = None):
        """
        Initialization function
      
        Parameters:
        -----------
        microbe_data : DataFrame
            Gut microbe data, rows as samples, columns as microbes, index contains group information
        metabolite_data : DataFrame
            Metabolite data, rows as samples, columns as metabolites, index contains group information
        group_labels : list
            Group labels, e.g. ["HC", "MIA", "IA"]
        microbe_coab_data : dict
            Microbe coabundance data
        """
        self.microbe_data = microbe_data
        self.microbe_coab_data = microbe_coab_data
        self.metabolite_data = metabolite_data
        self.group_labels = group_labels or ["HC", "MIA", "IA"]
      
        self._validate_data()
      
        self.results = {
            'microbe': {
                'Fluctuation': None,
                'Remote_Interaction': {},
                'Coordination': {}
            },
            'metabolite': {
                'Fluctuation': None,
                'Remote_Interaction': {},
                'Coordination': {}
            }
        }
  
    def _validate_data(self):
        """Validate data format"""
        if self.microbe_data is None or self.metabolite_data is None:
            print("Warning: Data not fully loaded, please load data before running analysis")
            return
      
        for label in self.group_labels:
            if not any(self.microbe_data.index.str.contains(label)):
                print(f"Warning: Group label '{label}' not found in microbe data")
            if not any(self.metabolite_data.index.str.contains(label)):
                print(f"Warning: Group label '{label}' not found in metabolite data")
      
        try:
            if len(self.microbe_coab_data.keys()) != len(set(self.microbe_data.index)):
                print('Input microbe coabundance data does not match original data')
            
        except:
            print('Error: Microbe coabundance matrix input error, should be dictionary')
  
    def load_data(self, microbe_data, metabolite_data,microbe_coab_data, group_labels=None):
        """Load data"""
        self.microbe_data = microbe_data
        self.metabolite_data = metabolite_data
        self.microbe_coab_data = microbe_coab_data
        if group_labels:
            self.group_labels = group_labels
        self._validate_data()
  
    def calculate_fluctuation(self, data_type='both'):
        """
        Calculate Fluctuation

        Parameters:
        -----------
        data_type : str
            'microbe', 'metabolite' or 'both'

        Returns:
        --------
        DataFrame or dict: Contains fluctuation scores
        """
        print("Calculating Fluctuation...")

        if data_type in ['microbe', 'both'] and self.microbe_data is not None:
            print("Calculating microbe fluctuation...")
            microbe_dv, microbe_p_values = self._calculate_dv_score(self.microbe_data, return_p_values=True)
            self.results['microbe']['Fluctuation'] = microbe_dv
            self.results['microbe']['Fluctuation_P_Values'] = microbe_p_values

        if data_type in ['metabolite', 'both'] and self.metabolite_data is not None:
            print("Calculating metabolite fluctuation...")
            metabolite_dv, metabolite_p_values = self._calculate_dv_score(self.metabolite_data, return_p_values=True)
            self.results['metabolite']['Fluctuation'] = metabolite_dv
            self.results['metabolite']['Fluctuation_P_Values'] = metabolite_p_values

        if data_type == 'microbe':
            return self.results['microbe']['Fluctuation']
        elif data_type == 'metabolite':
            return self.results['metabolite']['Fluctuation']
        else:
            return {
                'microbe': self.results['microbe']['Fluctuation'],
                'metabolite': self.results['metabolite']['Fluctuation']
            }

    def _calculate_dv_score(self, df, return_p_values=True):
        """
        Calculate variation difference of each feature between different groups, with FDR correction

        Parameters:
        -----------
        df : DataFrame
            Rows are samples, columns are features, index contains group information
        return_p_values : bool, optional
            Whether to return original p-values and corrected p-values, default False

        Returns:
        --------
        Series: Feature fluctuation scores (based on FDR corrected p-values)
        DataFrame (if return_p_values=True): Contains original p-values and FDR corrected p-values
        """
        dv_score = {}
        p_values = []
        feature_names = []
        error_list = []
        for feature in tqdm(df.columns):
            group_vals = []
            valid_groups = []

            for g in self.group_labels:
                group_samples = [idx for idx in df.index if g in idx]
                if len(group_samples) > 1:
                    group_data = df.loc[group_samples, feature]
                    if group_data.nunique() > 1:
                        group_vals.append(group_data.values)
                        valid_groups.append(g)

            if len(group_vals) >= 2:
                try:
                    stat, p = levene(*group_vals, center='median')
                    dv_score[feature] = -np.log10(p) if p > 0 else 50
                    p_values.append(p)
                    feature_names.append(feature)
                except Exception as e:
                    print(f"Levene test failed, feature: {feature}, error: {e}")
                    dv_score[feature] = np.nan
                    p_values.append(np.nan)
                    feature_names.append(feature)
                    error_list.append(feature)
            else:
                dv_score[feature] = np.nan
                p_values.append(np.nan)
                feature_names.append(feature)
                error_list.append(feature)
        print(f'Total {len(error_list)} features do not have sufficient variability data for Levene test')

        non_nan_p_values = np.array([p for p in p_values if not np.isnan(p)])
        corrected_p_values = np.full(len(p_values), np.nan)

        if len(non_nan_p_values) > 0:
            reject, p_corrected, _, _ = multipletests(non_nan_p_values, method='fdr_bh')

            non_nan_idx = 0
            for i in range(len(p_values)):
                if not np.isnan(p_values[i]):
                    corrected_p_values[i] = p_corrected[non_nan_idx]
                    non_nan_idx += 1

        corrected_dv_score = {}
        fdr_results = {}
        for i, feature in enumerate(feature_names):
            if np.isnan(corrected_p_values[i]):
                corrected_dv_score[feature] = 0
                fdr_results[feature] = {'original_p': np.nan, 'fdr_p': np.nan}
            else:
                corrected_dv_score[feature] = -np.log10(corrected_p_values[i]) if corrected_p_values[i] > 0 else 50
                fdr_results[feature] = {'original_p': p_values[i], 'fdr_p': corrected_p_values[i]}

        if return_p_values:
            return pd.Series(corrected_dv_score).fillna(0), pd.DataFrame(fdr_results).T
        else:
            return pd.Series(corrected_dv_score).fillna(0)

  
    def calculate_coordination(self, data_type='both'):
        """
        Calculate Coordination
        - For microbes: use coabundance
        - For metabolites: calculate average correlation with all metabolites
      
        Parameters:
        -----------
        data_type : str
            'microbe', 'metabolite' or 'both'
          
        Returns:
        --------
        dict: Contains coordination scores for each group
        """
        
        for label in self.group_labels:
          
            if data_type in ['microbe', 'both'] and self.microbe_coab_data is not None:
                print(f"Applying {label} group microbe coabundance...")
                mb_data = self.microbe_coab_data[label]
                mb_corr = mb_data.abs()
                np.fill_diagonal(mb_corr.values, np.nan)
                mb_coord = mb_corr.mean(axis=1, skipna=True)
              
                self.results['microbe']['Coordination'][label] = mb_coord
          
            if data_type in ['metabolite', 'both'] and self.metabolite_data is not None:
                print(f"Calculating {label} group metabolite average correlation...")
                met_data = self.metabolite_data.loc[label]
                met_corr = met_data.corr(method='spearman').abs()
                np.fill_diagonal(met_corr.values, np.nan)
                met_coord = met_corr.mean(axis=1, skipna=True)
              
                self.results['metabolite']['Coordination'][label] = met_coord
      
        if data_type == 'microbe':
            return self.results['microbe']['Coordination']
        elif data_type == 'metabolite':
            return self.results['metabolite']['Coordination']
        else:
            return {
                'microbe': self.results['microbe']['Coordination'],
                'metabolite': self.results['metabolite']['Coordination']
            }
  
    def calculate_remote_interaction(self, data_type='both', n_jobs=-1):
        """
        Calculate Remote Interaction
        - For microbes: weighted average correlation with metabolites
        - For metabolites: weighted average correlation with microbes
        
        Parameters:
        -----------
        data_type : str
            'microbe', 'metabolite' or 'both'
        n_jobs : int
            Number of cores for parallel computing, default uses all available cores
        
        Returns:
        --------
        dict: Contains remote interaction scores for each group
        """
        print("Calculating remote interaction...")
    
        for label in self.group_labels:
            mb_data = self.microbe_data.loc[label]
            met_data = self.metabolite_data.loc[label]
    
            if data_type in ['microbe', 'both']:
                print(f"Calculating {label} group microbe-metabolite remote interaction...")
                mb_remote = Parallel(n_jobs=n_jobs)(
                    delayed(self._calculate_wmc)(mb_data[microbe], met_data)
                    for microbe in tqdm(mb_data.columns, desc=f"{label} - microbe")
                )
                self.results['microbe']['Remote_Interaction'][label] = pd.Series(mb_remote, index=mb_data.columns)
    
            if data_type in ['metabolite', 'both']:
                print(f"Calculating {label} group metabolite-microbe remote interaction...")
                met_remote = Parallel(n_jobs=n_jobs)(
                    delayed(self._calculate_wmc)(met_data[metabolite], mb_data)
                    for metabolite in tqdm(met_data.columns, desc=f"{label} - metabolite")
                )
                self.results['metabolite']['Remote_Interaction'][label] = pd.Series(met_remote, index=met_data.columns)
    
        if data_type == 'microbe':
            return self.results['microbe']['Remote_Interaction']
        elif data_type == 'metabolite':
            return self.results['metabolite']['Remote_Interaction']
        else:
            return {
                'microbe': self.results['microbe']['Remote_Interaction'],
                'metabolite': self.results['metabolite']['Remote_Interaction']
            }

    
    def _calculate_wmc(self, feature_series=None, df_target=None):
        """
        Calculate weighted average correlation
      
        Parameters:
        -----------
        feature_series : Series
            Feature for correlation calculation
        df_target : DataFrame
            Target dataset
        Returns:
        --------
        float: Weighted average correlation
        """
        weighted_sum, weight_total = 0, 0
        for target in df_target.columns:
            rho, pval = pearsonr(feature_series, df_target[target])
            abs_rho = abs(rho)
            if np.isnan(rho) or np.isnan(pval):
                continue
          
            if pval < 0.01 and abs_rho > 0.4:
                weight = 2
            elif pval < 0.05 and abs_rho > 0.3:
                weight = 1
            else:
                weight = 0
            weighted_sum += abs_rho * weight
            weight_total += weight
      
        return weighted_sum / weight_total if weight_total != 0 else 0
  
    def calculate_all_scores(self, data_type='both',rule={'flu': 1, 'ri': 1, 'co': 1}):
        """Calculate all scores"""
        self.calculate_fluctuation(data_type)
        self.calculate_coordination(data_type)
        self.calculate_remote_interaction(data_type)
        return self.integrate_scores(data_type,rule)
  
    def integrate_scores(self, data_type='both', rule={'flu': 1, 'ri': 1, 'co': 1}):
        """
        Integrate scores from three dimensions
    
        Parameters:
        -----------
        data_type : str
            'microbe', 'metabolite' or 'both'
        rule : dict
            Weights for three dimensions, e.g. {'flu':1, 'ri':1, 'co':1} or {'flu':0.4, 'ri':0.3, 'co':0.3}
    
        Returns:
        --------
        dict: Integrated scores DataFrame for each group (includes three dimensions + Integrated_Score)
        """
        w_flu = rule.get('flu', 1)
        w_ri = rule.get('ri', 1)
        w_co = rule.get('co', 1)
    
        weights = [w_flu, w_ri, w_co]
        if len(set(weights)) != 1:
            total = sum(weights)
            if not np.isclose(total, 1.0):
                raise ValueError("Rule weights must be either all equal or sum to 1")
    
        integrated_scores = {
            'microbe': {},
            'metabolite': {}
        }
    
        def _process_block(fluct, coord, remote, label):
            all_features = set(fluct.index) | set(coord.index) | set(remote.index)
            data = { 'Fluctuation': [fluct.get(f, 0) for f in all_features],
                'Coordination': [coord.get(f, 0) for f in all_features],
                'Remote_Interaction': [remote.get(f, 0) for f in all_features]
            }
            df = pd.DataFrame(data, index=list(all_features))
    
            for col in df.columns:
                if df[col].max() > df[col].min():
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
            df['Integrated_Score'] = (
                df['Fluctuation'] * w_flu +
                df['Coordination'] * w_co +
                df['Remote_Interaction'] * w_ri
            )
            return df
    
        if data_type in ['microbe', 'both']:
            for label in self.group_labels:
                if (label in self.results['microbe']['Coordination'] and
                    label in self.results['microbe']['Remote_Interaction']):
                    fluct = self.results['microbe']['Fluctuation']
                    coord = self.results['microbe']['Coordination'][label]
                    remote = self.results['microbe']['Remote_Interaction'][label]
                    integrated_scores['microbe'][label] = _process_block(fluct, coord, remote, label)
    
        if data_type in ['metabolite', 'both']:
            for label in self.group_labels:
                if (label in self.results['metabolite']['Coordination'] and
                    label in self.results['metabolite']['Remote_Interaction']):
                    fluct = self.results['metabolite']['Fluctuation']
                    coord = self.results['metabolite']['Coordination'][label]
                    remote = self.results['metabolite']['Remote_Interaction'][label]
                    integrated_scores['metabolite'][label] = _process_block(fluct, coord, remote, label)
    
        if data_type == 'microbe':
            return integrated_scores['microbe']
        elif data_type == 'metabolite':
            return integrated_scores['metabolite']
        else:
            return integrated_scores


    def save_results(self, output_dir="results", rule={'flu':1, 'ri':1, 'co':1}):
        """
        Save analysis results (one file per category, includes four score columns, sorted by integrated score) and p-values from fluctuation analysis.

        Parameters:
        -----------
        output_dir : str
            Output directory path
        rule : dict
            Weight configuration for integrated scores, used to call integrate_scores
        """
        os.makedirs(output_dir, exist_ok=True)

        for data_type in ['microbe', 'metabolite']:
            integrated = self.integrate_scores(data_type=data_type, rule=rule)
            if integrated:
                for label, df in integrated.items():
                    ordered_cols = ['Fluctuation', 'Coordination', 'Remote_Interaction', 'Integrated_Score']
                    df = df[[col for col in ordered_cols if col in df.columns]]
                    df = df.sort_values('Integrated_Score', ascending=False)

                    file_path = os.path.join(output_dir, f"{data_type}_summary_{label}.csv")
                    df.to_csv(file_path)
                    print(f"Integrated scores saved to {file_path}")

            if self.results[data_type]['Fluctuation_P_Values'] is not None:
                p_value_df = self.results[data_type]['Fluctuation_P_Values']
                file_path_p_values = os.path.join(output_dir, f"{data_type}_fluctuation_p_values.csv")
                p_value_df.to_csv(file_path_p_values)
                print(f"Fluctuation analysis p-values saved to {file_path_p_values}")

        print(f"All results saved to {output_dir} directory")

    
    def query_common_features(self, data_type='both', top_n=None, top_percent=None):
        """
        Query common important features across different labels
        
        Parameters:
        -----------
        data_type : str
            'microbe', 'metabolite' or 'both'
        top_n : int, optional
            Top n features
        top_percent : float, optional
            Top percentage of features, between 0-1
            
        Returns:
        --------
        dict: Contains common features and unique features
        """
        if top_n is None and top_percent is None:
            top_percent = 0.1
        
        result = {}
        
        if data_type in ['microbe', 'both']:
            integrated = self.integrate_scores('microbe')
            if integrated:
                result['microbe'] = self._find_common_features(integrated, top_n, top_percent)
        
        if data_type in ['metabolite', 'both']:
            integrated = self.integrate_scores('metabolite')
            if integrated:
                result['metabolite'] = self._find_common_features(integrated, top_n, top_percent)
        
        if data_type == 'microbe':
            return result['microbe']
        elif data_type == 'metabolite':
            return result['metabolite']
        else:
            return result
    
    def _find_common_features(self, integrated_scores, top_n=None, top_percent=None):
        """
        Find common features
        
        Parameters:
        -----------
        integrated_scores : dict
            Integrated scores for each label
        top_n : int, optional
            Top n features
        top_percent : float, optional
            Top percentage of features, between 0-1
            
        Returns:
        --------
        dict: Contains common features and unique features
        """
        top_features = {}
        
        for label, df in integrated_scores.items():
            sorted_features = df.sort_values('Integrated_Score', ascending=False)
            
            if top_n is not None:
                cutoff = min(top_n, len(sorted_features))
                top_features[label] = set(sorted_features.index[:cutoff])
            elif top_percent is not None:
                cutoff = int(len(sorted_features) * top_percent)
                top_features[label] = set(sorted_features.index[:cutoff])
        
        all_labels = list(top_features.keys())
        common_features = set.intersection(*[top_features[l] for l in all_labels])
        
        unique_features = {}
        for label in all_labels:
            unique_to_label = top_features[label] - set.union(*[top_features[l] for l in all_labels if l != label])
            unique_features[label] = unique_to_label
        
        pairwise_common = {}
        if len(all_labels) >= 3:
            for i in range(len(all_labels)):
                for j in range(i+1, len(all_labels)):
                    label_pair = (all_labels[i], all_labels[j])
                    common_pair = top_features[label_pair[0]] & top_features[label_pair[1]]
                    other_labels = [l for l in all_labels if l not in label_pair]
                    exclusive_to_pair = common_pair - set.union(*[top_features[l] for l in other_labels])
                    if exclusive_to_pair:
                        pairwise_common[f"{label_pair[0]}_and_{label_pair[1]}"] = exclusive_to_pair
        
        return {
            'common_to_all': common_features,
            'unique_to_each': unique_features,
            'pairwise_common': pairwise_common
        }

