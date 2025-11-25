"""
Bike Demand Prediction - Exploratory Data Analysis


- Linear Discriminant Analysis (LDA)
- Tree-based Methods (Decision Trees, Random Forests)
- Logistic Regression
- K-Nearest Neighbors (KNN)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, shapiro
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
import os
OUTPUT_DIR = os.path.join(os.getcwd(), 'bike_demand_eda_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class BikeDataAnalyzer:
    """
    Handles comprehensive EDA with model-specific recommendations.
    Designed for scalability and production deployment.
    """
    
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.target_col = 'increase_stock'
        self.data['target_binary'] = (self.data[self.target_col] == 'high_bike_demand').astype(int)
        
        # Feature categorization based on data dictionary
        self.categorical_features = ['hour_of_day', 'day_of_week', 'month', 
                                     'holiday', 'weekday', 'summertime']
        self.numerical_features = ['temp', 'dew', 'humidity', 'precip', 'snow', 
                                   'snowdepth', 'windspeed', 'cloudcover', 'visibility']
        
        self.reports = {}
        
    def generate_summary_stats(self):
        """Basic dataset statistics - foundation for all models"""
        report = []
        report.append("="*80)
        report.append("DATASET OVERVIEW")
        report.append("="*80)
        report.append(f"\nShape: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        report.append(f"Target distribution:")
        
        target_dist = self.data[self.target_col].value_counts()
        for label, count in target_dist.items():
            pct = (count / len(self.data)) * 100
            report.append(f"  {label}: {count} ({pct:.2f}%)")
        
        # Class imbalance critical for all models
        imbalance_ratio = target_dist.max() / target_dist.min()
        report.append(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 1.5:
            report.append("WARNING: Imbalanced classes detected")
            report.append("  - Use stratified sampling for train/test splits")
            report.append("  - Consider class weights or SMOTE")
            report.append("  - Evaluate using F1-score, not just accuracy")
        
        report.append("\n" + "-"*80)
        report.append("FEATURE TYPES")
        report.append("-"*80)
        report.append(f"\nCategorical features ({len(self.categorical_features)}):")
        for feat in self.categorical_features:
            unique = self.data[feat].nunique()
            report.append(f"  {feat}: {unique} unique values")
        
        report.append(f"\nNumerical features ({len(self.numerical_features)}):")
        for feat in self.numerical_features:
            report.append(f"  {feat}: [{self.data[feat].min():.2f}, {self.data[feat].max():.2f}]")
        
        return "\n".join(report)
    
    def analyze_temporal_patterns(self):
        """Temporal analysis - critical for understanding cyclical patterns"""
        report = []
        report.append("\n" + "="*80)
        report.append("TEMPORAL PATTERN ANALYSIS")
        report.append("="*80)
        
        # Hour of day - captures commute patterns
        report.append("\n1. HOURLY PATTERNS")
        report.append("-"*80)
        hourly = self.data.groupby('hour_of_day')['target_binary'].agg(['mean', 'count'])
        hourly.columns = ['demand_rate', 'n_samples']
        hourly['demand_rate'] *= 100
        
        peak_hours = hourly.nlargest(5, 'demand_rate')
        report.append(f"Peak demand hours:")
        for hour, row in peak_hours.iterrows():
            report.append(f"  {hour:02d}:00 - {row['demand_rate']:.1f}% high demand ({row['n_samples']} samples)")
        
        low_hours = hourly.nsmallest(5, 'demand_rate')
        report.append(f"\nLowest demand hours:")
        for hour, row in low_hours.iterrows():
            report.append(f"  {hour:02d}:00 - {row['demand_rate']:.1f}% high demand ({row['n_samples']} samples)")
        
        # Day of week - weekday vs weekend behavior
        report.append("\n2. WEEKLY PATTERNS")
        report.append("-"*80)
        weekday_stats = self.data.groupby('weekday')['target_binary'].agg(['mean', 'count'])
        weekday_stats.index = ['Weekend', 'Weekday']
        weekday_stats.columns = ['demand_rate', 'n_samples']
        weekday_stats['demand_rate'] *= 100
        
        for idx, row in weekday_stats.iterrows():
            report.append(f"{idx}: {row['demand_rate']:.1f}% high demand ({row['n_samples']} samples)")
        
        # Statistical significance
        weekday_mask = self.data['weekday'] == 1
        t_stat, p_val = stats.ttest_ind(
            self.data[weekday_mask]['target_binary'],
            self.data[~weekday_mask]['target_binary']
        )
        report.append(f"\nWeekday vs Weekend significance: p={p_val:.4f}")
        
        # Monthly patterns - seasonal variation
        report.append("\n3. SEASONAL PATTERNS")
        report.append("-"*80)
        monthly = self.data.groupby('month')['target_binary'].agg(['mean', 'count'])
        monthly.columns = ['demand_rate', 'n_samples']
        monthly['demand_rate'] *= 100
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month_num, row in monthly.iterrows():
            month_name = months[month_num-1] if 1 <= month_num <= 12 else str(month_num)
            report.append(f"{month_name}: {row['demand_rate']:.1f}% high demand ({row['n_samples']} samples)")
        
        return "\n".join(report)
    
    def analyze_weather_impact(self):
        """Weather analysis - environmental factors affecting bike usage"""
        report = []
        report.append("\n" + "="*80)
        report.append("WEATHER IMPACT ANALYSIS")
        report.append("="*80)
        
        # Temperature - primary weather factor
        report.append("\n1. TEMPERATURE")
        report.append("-"*80)
        temp_corr = self.data['temp'].corr(self.data['target_binary'])
        report.append(f"Correlation with demand: {temp_corr:+.4f}")
        
        temp_bins = [-20, 0, 10, 20, 30, 40]
        temp_labels = ['<0°C', '0-10°C', '10-20°C', '20-30°C', '>30°C']
        temp_cat = pd.cut(self.data['temp'], bins=temp_bins, labels=temp_labels)
        temp_analysis = self.data.groupby(temp_cat, observed=True)['target_binary'].agg(['mean', 'count'])
        temp_analysis['mean'] *= 100
        
        for cat, row in temp_analysis.iterrows():
            report.append(f"  {cat}: {row['mean']:.1f}% demand ({row['count']} samples)")
        
        # Precipitation - inhibits cycling
        report.append("\n2. PRECIPITATION")
        report.append("-"*80)
        has_precip = self.data['precip'] > 0
        precip_analysis = self.data.groupby(has_precip)['target_binary'].agg(['mean', 'count'])
        precip_labels = {False: 'No rain', True: 'Rain'}
        precip_analysis.index = [precip_labels.get(x, str(x)) for x in precip_analysis.index]
        precip_analysis['mean'] *= 100
        
        for cond, row in precip_analysis.iterrows():
            report.append(f"  {cond}: {row['mean']:.1f}% demand ({row['count']} samples)")
        
        # Snow - severe weather condition
        report.append("\n3. SNOW CONDITIONS")
        report.append("-"*80)
        has_snow = self.data['snow'] > 0
        snow_analysis = self.data.groupby(has_snow)['target_binary'].agg(['mean', 'count'])
        snow_labels = {False: 'No snow', True: 'Snow'}
        snow_analysis.index = [snow_labels.get(x, str(x)) for x in snow_analysis.index]
        snow_analysis['mean'] *= 100
        
        if len(snow_analysis) == 1:
            report.append(f"  Only one category present in dataset")
        for cond, row in snow_analysis.iterrows():
            report.append(f"  {cond}: {row['mean']:.1f}% demand ({row['count']} samples)")
        
        # Wind speed - impacts cycling comfort
        report.append("\n4. WIND SPEED")
        report.append("-"*80)
        wind_corr = self.data['windspeed'].corr(self.data['target_binary'])
        report.append(f"Correlation with demand: {wind_corr:+.4f}")
        
        return "\n".join(report)
    
    def analyze_for_lda(self):
        """
        LDA-specific analysis
        Key requirements: normality, equal covariance, class separation
        """
        report = []
        report.append("\n" + "="*80)
        report.append("LINEAR DISCRIMINANT ANALYSIS (LDA) - SPECIFIC INSIGHTS")
        report.append("="*80)
        
        report.append("\nLDA ASSUMPTIONS CHECK:")
        report.append("-"*80)
        
        # Normality testing - LDA assumes Gaussian distributions
        report.append("\n1. NORMALITY ASSESSMENT")
        report.append("   LDA performs best when features follow normal distributions")
        report.append("   Testing each numerical feature (Shapiro-Wilk test):\n")
        
        normality_results = []
        for feat in self.numerical_features:
            # Use sample for computational efficiency
            sample = self.data[feat].dropna().sample(min(5000, len(self.data)))
            stat, p_value = shapiro(sample)
            is_normal = p_value > 0.05
            normality_results.append({
                'feature': feat,
                'p_value': p_value,
                'is_normal': is_normal
            })
            status = "✓ Normal" if is_normal else "✗ Non-normal"
            report.append(f"   {feat:15s}: p={p_value:.4f} {status}")
        
        non_normal_count = sum(1 for r in normality_results if not r['is_normal'])
        if non_normal_count > 0:
            report.append(f"\n   WARNING: {non_normal_count}/{len(self.numerical_features)} features are non-normal")
            report.append("   RECOMMENDATION: Apply transformations (log, Box-Cox) before LDA")
        
        # Class separation analysis
        report.append("\n2. CLASS SEPARATION ANALYSIS")
        report.append("   Measuring how well classes separate in feature space:\n")
        
        for feat in self.numerical_features[:5]:  # Top 5 for brevity
            class0 = self.data[self.data['target_binary'] == 0][feat]
            class1 = self.data[self.data['target_binary'] == 1][feat]
            
            # Cohen's d effect size
            pooled_std = np.sqrt((class0.std()**2 + class1.std()**2) / 2)
            cohens_d = (class1.mean() - class0.mean()) / pooled_std
            
            interpretation = "Negligible"
            if abs(cohens_d) > 0.8: interpretation = "Large"
            elif abs(cohens_d) > 0.5: interpretation = "Medium"
            elif abs(cohens_d) > 0.2: interpretation = "Small"
            
            report.append(f"   {feat:15s}: Cohen's d = {cohens_d:+.3f} ({interpretation})")
        
        # Correlation analysis - multicollinearity issues
        report.append("\n3. MULTICOLLINEARITY CHECK")
        report.append("   High correlation between features can destabilize LDA:\n")
        
        corr_matrix = self.data[self.numerical_features].corr()
        high_corr = []
        for i in range(len(self.numerical_features)):
            for j in range(i+1, len(self.numerical_features)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr.append((self.numerical_features[i], 
                                     self.numerical_features[j], corr_val))
        
        if high_corr:
            report.append("   HIGH CORRELATIONS DETECTED:")
            for feat1, feat2, corr in high_corr:
                report.append(f"   {feat1} <-> {feat2}: {corr:.3f}")
            report.append("\n   RECOMMENDATION: Consider removing one from each pair")
        else:
            report.append("   ✓ No severe multicollinearity detected")
        
        report.append("\n4. LDA PREPROCESSING RECOMMENDATIONS")
        report.append("-"*80)
        report.append("   • Feature scaling: NOT required (LDA handles different scales)")
        report.append("   • Transformation: Apply Box-Cox to non-normal features")
        report.append("   • Encoding: One-hot encode categorical features")
        report.append("   • Outliers: Consider removal (can skew covariance estimates)")
        
        return "\n".join(report)
    
    def analyze_for_trees(self):
        """
        Tree-based methods analysis
        Advantages: handle non-linearity, no scaling needed, automatic feature selection
        """
        report = []
        report.append("\n" + "="*80)
        report.append("TREE-BASED METHODS - SPECIFIC INSIGHTS")
        report.append("="*80)
        
        report.append("\nTREE ADVANTAGES FOR THIS DATASET:")
        report.append("-"*80)
        
        # Non-linear relationships
        report.append("\n1. NON-LINEAR PATTERNS DETECTION")
        report.append("   Trees excel with non-monotonic relationships:\n")
        
        # Check for non-monotonic patterns
        for feat in ['hour_of_day', 'temp', 'windspeed']:
            if feat == 'hour_of_day':
                # Hourly pattern is typically U-shaped
                hourly_demand = self.data.groupby(feat)['target_binary'].mean()
                variance = hourly_demand.var()
                report.append(f"   {feat}: High variance ({variance:.3f}) - likely non-linear")
            else:
                corr_spearman = self.data[feat].corr(self.data['target_binary'], method='spearman')
                corr_pearson = self.data[feat].corr(self.data['target_binary'], method='pearson')
                diff = abs(corr_spearman - corr_pearson)
                if diff > 0.1:
                    report.append(f"   {feat}: Spearman≠Pearson (Δ={diff:.3f}) - suggests non-linearity")
        
        # Feature importance indicators
        report.append("\n2. EXPECTED FEATURE IMPORTANCE RANKING")
        report.append("   Based on correlation and variance analysis:\n")
        
        importances = []
        for feat in self.numerical_features:
            corr = abs(self.data[feat].corr(self.data['target_binary']))
            importances.append((feat, corr))
        
        importances.sort(key=lambda x: x[1], reverse=True)
        for i, (feat, importance) in enumerate(importances[:8], 1):
            report.append(f"   {i}. {feat:15s}: |correlation| = {importance:.4f}")
        
        # Categorical features handling
        report.append("\n3. CATEGORICAL FEATURE HANDLING")
        report.append("-"*80)
        report.append("   Trees handle categorical features natively")
        report.append("   Categorical features in dataset:")
        for feat in self.categorical_features:
            cardinality = self.data[feat].nunique()
            report.append(f"   • {feat}: {cardinality} categories")
        
        # Interaction effects
        report.append("\n4. POTENTIAL FEATURE INTERACTIONS")
        report.append("   Trees automatically capture interactions:\n")
        report.append("   Expected strong interactions:")
        report.append("   • hour_of_day × weekday (commute patterns)")
        report.append("   • temp × month (seasonal temperature effects)")
        report.append("   • precip × windspeed (severe weather conditions)")
        
        report.append("\n5. TREE PREPROCESSING RECOMMENDATIONS")
        report.append("-"*80)
        report.append("   • Feature scaling: NOT needed")
        report.append("   • Encoding: Leave categorical as-is or use label encoding")
        report.append("   • Missing values: Trees handle naturally (surrogate splits)")
        report.append("   • Outliers: Generally not a concern for trees")
        report.append("\n   HYPERPARAMETERS TO TUNE:")
        report.append("   • max_depth: Start with 5-15")
        report.append("   • min_samples_split: Try 10-50")
        report.append("   • min_samples_leaf: Try 5-20")
        report.append("   • For Random Forest: n_estimators=100-500, max_features='sqrt'")
        
        return "\n".join(report)
    
    def analyze_for_logistic_regression(self):
        """
        Logistic Regression analysis
        Key: linear decision boundaries, feature scaling, multicollinearity
        """
        report = []
        report.append("\n" + "="*80)
        report.append("LOGISTIC REGRESSION - SPECIFIC INSIGHTS")
        report.append("="*80)
        
        report.append("\nLINEARITY ASSESSMENT:")
        report.append("-"*80)
        
        # Linear relationship with log-odds
        report.append("\n1. FEATURE-TARGET RELATIONSHIPS")
        report.append("   Pearson correlation (linear relationship indicator):\n")
        
        correlations = []
        for feat in self.numerical_features:
            corr = self.data[feat].corr(self.data['target_binary'])
            correlations.append((feat, corr))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat, corr in correlations:
            strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.1 else "Weak"
            report.append(f"   {feat:15s}: {corr:+.4f} ({strength})")
        
        # Scaling requirement
        report.append("\n2. FEATURE SCALE ANALYSIS")
        report.append("   Logistic Regression sensitive to feature scales:\n")
        
        scales = []
        for feat in self.numerical_features:
            scale = self.data[feat].max() - self.data[feat].min()
            scales.append((feat, scale, self.data[feat].std()))
        
        scales.sort(key=lambda x: x[1], reverse=True)
        max_scale = scales[0][1]
        
        report.append(f"   Feature ranges:")
        for feat, scale, std in scales[:5]:
            report.append(f"   {feat:15s}: range={scale:8.2f}, std={std:7.2f}")
        
        report.append(f"\n   Range ratio (max/min): {scales[0][1]/scales[-1][1]:.1f}:1")
        report.append("   RECOMMENDATION: Standardization (StandardScaler) is REQUIRED")
        
        # Multicollinearity
        report.append("\n3. MULTICOLLINEARITY ANALYSIS")
        report.append("   VIF-style correlation check (corr > 0.7 indicates issues):\n")
        
        corr_matrix = self.data[self.numerical_features].corr()
        high_corr_pairs = []
        for i in range(len(self.numerical_features)):
            for j in range(i+1, len(self.numerical_features)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((self.numerical_features[i], 
                                          self.numerical_features[j], corr_val))
        
        if high_corr_pairs:
            report.append("   MULTICOLLINEARITY DETECTED:")
            for feat1, feat2, corr in high_corr_pairs:
                report.append(f"   {feat1} <-> {feat2}: {corr:.3f}")
            report.append("\n   RECOMMENDATION: Remove one feature from each highly correlated pair")
        else:
            report.append("   ✓ No severe multicollinearity detected")
        
        # Class separation
        report.append("\n4. CLASS SEPARABILITY")
        report.append("   Logistic regression works best with linearly separable classes:\n")
        
        # Calculate separation for top features
        for feat in [correlations[0][0], correlations[1][0]]:
            class0_mean = self.data[self.data['target_binary'] == 0][feat].mean()
            class1_mean = self.data[self.data['target_binary'] == 1][feat].mean()
            class0_std = self.data[self.data['target_binary'] == 0][feat].std()
            class1_std = self.data[self.data['target_binary'] == 1][feat].std()
            
            separation = abs(class1_mean - class0_mean) / ((class0_std + class1_std) / 2)
            report.append(f"   {feat}: separation score = {separation:.3f}")
        
        report.append("\n5. LOGISTIC REGRESSION PREPROCESSING RECOMMENDATIONS")
        report.append("-"*80)
        report.append("   • Feature scaling: REQUIRED (use StandardScaler)")
        report.append("   • Encoding: One-hot encode categoricals (avoid label encoding)")
        report.append("   • Polynomial features: Consider for temp, hour_of_day")
        report.append("   • Regularization: Start with L2 (Ridge), tune C parameter")
        report.append("\n   HYPERPARAMETERS TO TUNE:")
        report.append("   • C (inverse regularization): Try [0.001, 0.01, 0.1, 1, 10, 100]")
        report.append("   • penalty: Try both 'l1' and 'l2'")
        report.append("   • class_weight: Use 'balanced' for imbalanced data")
        
        return "\n".join(report)
    
    def analyze_for_knn(self):
        """
        KNN analysis
        Critical: feature scaling, curse of dimensionality, local class distributions
        """
        report = []
        report.append("\n" + "="*80)
        report.append("K-NEAREST NEIGHBORS (KNN) - SPECIFIC INSIGHTS")
        report.append("="*80)
        
        report.append("\nKNN CRITICAL REQUIREMENTS:")
        report.append("-"*80)
        
        # Distance metric considerations
        report.append("\n1. FEATURE SCALING ANALYSIS")
        report.append("   KNN is EXTREMELY sensitive to feature scales (uses distance):\n")
        
        scales = []
        for feat in self.numerical_features:
            feat_range = self.data[feat].max() - self.data[feat].min()
            feat_std = self.data[feat].std()
            scales.append((feat, feat_range, feat_std))
        
        scales.sort(key=lambda x: x[1], reverse=True)
        
        report.append("   Raw feature scales:")
        for feat, feat_range, std in scales:
            report.append(f"   {feat:15s}: range={feat_range:8.2f}, std={std:7.2f}")
        
        # Demonstrate scaling impact
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data[self.numerical_features])
        
        report.append(f"\n   Scale ratio (max/min): {scales[0][1]/scales[-1][1]:.0f}:1")
        report.append("   ⚠️  CRITICAL: Without scaling, high-range features will dominate distance")
        report.append("   RECOMMENDATION: StandardScaler or MinMaxScaler is MANDATORY")
        
        # Dimensionality assessment
        report.append("\n2. CURSE OF DIMENSIONALITY")
        report.append(f"   Total features: {len(self.numerical_features) + len(self.categorical_features)}")
        report.append(f"   Sample size: {len(self.data)}")
        report.append(f"   Samples per dimension: {len(self.data) / (len(self.numerical_features) + len(self.categorical_features)):.0f}")
        
        if len(self.data) / (len(self.numerical_features) + len(self.categorical_features)) < 50:
            report.append("   WARNING: Low samples-to-dimensions ratio")
            report.append("   RECOMMENDATION: Feature selection or dimensionality reduction")
        else:
            report.append("   ✓ Adequate samples-to-dimensions ratio")
        
        # Local class distribution
        report.append("\n3. LOCAL CLASS BALANCE ANALYSIS")
        report.append("   KNN predictions depend on local neighborhood composition:\n")
        
        # Analyze class balance in different regions
        for feat in ['hour_of_day', 'temp']:
            if feat == 'hour_of_day':
                bins = range(0, 25, 4)
                labels = [f"{i}-{i+3}h" for i in range(0, 24, 4)]
            else:
                bins = [-20, 0, 10, 20, 30, 40]
                labels = ['<0°C', '0-10°C', '10-20°C', '20-30°C', '>30°C']
            
            feat_binned = pd.cut(self.data[feat], bins=bins, labels=labels)
            local_balance = self.data.groupby(feat_binned, observed=True)['target_binary'].agg(['mean', 'count'])
            local_balance['mean'] *= 100
            
            report.append(f"\n   Local class balance by {feat}:")
            for region, row in local_balance.iterrows():
                report.append(f"   {str(region):12s}: {row['mean']:5.1f}% high demand ({row['count']:3.0f} samples)")
        
        # Distance metric recommendations
        report.append("\n4. DISTANCE METRIC SELECTION")
        report.append("-"*80)
        report.append("   • Euclidean: Default, works well after standardization")
        report.append("   • Manhattan: More robust to outliers")
        report.append("   • Minkowski (p=1.5): Compromise between Euclidean and Manhattan")
        
        # Optimal K estimation
        report.append("\n5. INITIAL K VALUE ESTIMATION")
        report.append("-"*80)
        n_samples = len(self.data)
        sqrt_n = int(np.sqrt(n_samples))
        
        report.append(f"   Dataset size: {n_samples}")
        report.append(f"   sqrt(n) = {sqrt_n}")
        report.append(f"   Rule of thumb: Start with K around {sqrt_n}")
        report.append(f"   RECOMMENDATION: Grid search K in [{max(3, sqrt_n-20)}, {sqrt_n}, {sqrt_n+20}]")
        report.append("   Use odd K values to avoid ties")
        
        report.append("\n6. KNN PREPROCESSING RECOMMENDATIONS")
        report.append("-"*80)
        report.append("   • Feature scaling: MANDATORY (StandardScaler or MinMaxScaler)")
        report.append("   • Encoding: One-hot encode categoricals (increases dimensionality)")
        report.append("   • Outliers: Can significantly impact predictions - consider removal")
        report.append("   • Feature selection: Reduce dimensionality if possible")
        report.append("\n   HYPERPARAMETERS TO TUNE:")
        report.append(f"   • n_neighbors: Try odd values in [3, 5, 7, ..., {sqrt_n+20}]")
        report.append("   • weights: Try both 'uniform' and 'distance'")
        report.append("   • metric: Try 'euclidean', 'manhattan', 'minkowski'")
        report.append("   • p (for Minkowski): Try [1, 1.5, 2]")
        
        return "\n".join(report)
    
    def create_visualizations(self):
        """Generate model-specific visualization suites"""
        
        # Create figure directory structure
        model_dirs = ['LDA', 'Tree', 'LogisticRegression', 'KNN']
        for model in model_dirs:
            model_path = os.path.join(OUTPUT_DIR, f'{model}_EDA_results')
            os.makedirs(model_path, exist_ok=True)
            print(f"Created directory: {model_path}")
        
        # 1. Distribution plots for LDA
        self._plot_distributions_lda()
        
        # 2. Feature importance indicators for Trees
        self._plot_feature_importance_trees()
        
        # 3. Linear relationships for Logistic Regression
        self._plot_linear_relationships_logreg()
        
        # 4. Distance and scaling analysis for KNN
        self._plot_distance_analysis_knn()
        
        # 5. Common plots for all models
        self._plot_common_analysis()
    
    def _plot_distributions_lda(self):
        """LDA-specific: Check normality assumptions"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('LDA: Feature Distribution Analysis (Normality Check)', 
                     fontsize=16, fontweight='bold')
        
        for idx, feat in enumerate(self.numerical_features):
            ax = axes[idx // 3, idx % 3]
            
            # Plot distribution for each class
            for target_val, label in [(0, 'Low Demand'), (1, 'High Demand')]:
                data_subset = self.data[self.data['target_binary'] == target_val][feat]
                ax.hist(data_subset, bins=30, alpha=0.5, label=label, density=True)
            
            ax.set_title(f'{feat}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'LDA_EDA_results', '01_feature_distributions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Q-Q plots for normality
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('LDA: Q-Q Plots for Normality Assessment', 
                     fontsize=16, fontweight='bold')
        
        for idx, feat in enumerate(self.numerical_features):
            ax = axes[idx // 3, idx % 3]
            stats.probplot(self.data[feat].dropna(), dist="norm", plot=ax)
            ax.set_title(f'{feat}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'LDA_EDA_results', '02_qq_plots.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation heatmap for multicollinearity
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = self.data[self.numerical_features].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('LDA: Feature Correlation Matrix (Multicollinearity Check)', 
                     fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'LDA_EDA_results', '03_correlation_matrix.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance_trees(self):
        """Tree-specific: Potential feature importance and interactions"""
        
        # Correlation-based importance proxy
        fig, ax = plt.subplots(figsize=(10, 8))
        correlations = []
        for feat in self.numerical_features:
            corr = abs(self.data[feat].corr(self.data['target_binary']))
            correlations.append((feat, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        features, importance = zip(*correlations)
        
        ax.barh(features, importance, color='forestgreen', alpha=0.7)
        ax.set_xlabel('|Correlation with Target|', fontsize=12, fontweight='bold')
        ax.set_title('Trees: Expected Feature Importance\n(Based on Absolute Correlation)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Tree_EDA_results', '01_expected_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interaction plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Trees: Potential Feature Interactions', 
                     fontsize=16, fontweight='bold')
        
        # hour_of_day x weekday
        ax = axes[0, 0]
        interaction_data = self.data.groupby(['hour_of_day', 'weekday'])['target_binary'].mean().unstack()
        interaction_data.columns = ['Weekend', 'Weekday']
        interaction_data.plot(ax=ax, marker='o')
        ax.set_xlabel('Hour of Day', fontweight='bold')
        ax.set_ylabel('High Demand Rate', fontweight='bold')
        ax.set_title('Hour × Weekday Interaction')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Day Type')
        
        # temp x month (seasonal)
        ax = axes[0, 1]
        interaction_data = self.data.groupby(['month', pd.cut(self.data['temp'], bins=5)])['target_binary'].mean().unstack()
        interaction_data.plot(ax=ax, marker='o', legend=False)
        ax.set_xlabel('Month', fontweight='bold')
        ax.set_ylabel('High Demand Rate', fontweight='bold')
        ax.set_title('Temperature × Month Interaction')
        ax.grid(True, alpha=0.3)
        
        # precip x windspeed
        ax = axes[1, 0]
        self.data['weather_severity'] = (self.data['precip'] > 0).astype(int) + \
                                         (self.data['windspeed'] > 20).astype(int)
        weather_mapping = {0: 'Good', 1: 'Moderate', 2: 'Severe'}
        interaction_data = self.data.groupby(self.data['weather_severity'].map(weather_mapping))['target_binary'].mean()
        interaction_data.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'], alpha=0.7)
        ax.set_xlabel('Weather Severity', fontweight='bold')
        ax.set_ylabel('High Demand Rate', fontweight='bold')
        ax.set_title('Precipitation × Wind Interaction')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Nonlinear pattern: hour of day
        ax = axes[1, 1]
        hourly_pattern = self.data.groupby('hour_of_day')['target_binary'].mean()
        ax.plot(hourly_pattern.index, hourly_pattern.values, marker='o', 
                linewidth=2, markersize=6, color='darkgreen')
        ax.fill_between(hourly_pattern.index, 0, hourly_pattern.values, alpha=0.3, color='lightgreen')
        ax.set_xlabel('Hour of Day', fontweight='bold')
        ax.set_ylabel('High Demand Rate', fontweight='bold')
        ax.set_title('Non-linear Hourly Pattern\n(Trees capture this naturally)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Tree_EDA_results', '02_interaction_effects.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_linear_relationships_logreg(self):
        """Logistic Regression: Linear relationship checks"""
        
        # Scatter plots with regression lines
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Logistic Regression: Feature-Target Relationships', 
                     fontsize=16, fontweight='bold')
        
        for idx, feat in enumerate(self.numerical_features):
            ax = axes[idx // 3, idx % 3]
            
            # Skip features with zero variance
            if self.data[feat].std() == 0:
                ax.text(0.5, 0.5, f'{feat}\n(constant feature)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue
            
            # Scatter plot with jitter for binary target
            jittered_target = self.data['target_binary'] + np.random.normal(0, 0.02, len(self.data))
            ax.scatter(self.data[feat], jittered_target, alpha=0.3, s=10)
            
            # Add trend line with error handling
            try:
                z = np.polyfit(self.data[feat].dropna(), 
                              self.data.loc[self.data[feat].notna(), 'target_binary'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(self.data[feat].min(), self.data[feat].max(), 100)
                ax.plot(x_line, p(x_line), "r-", linewidth=2, label='Linear trend')
            except:
                pass  # Skip trend line if polyfit fails
            
            corr = self.data[feat].corr(self.data['target_binary'])
            ax.set_title(f'{feat}\n(r={corr:+.3f})', fontsize=10, fontweight='bold')
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Target (jittered)')
            ax.grid(True, alpha=0.3)
            if 'Linear trend' in [t.get_label() for t in ax.get_lines()]:
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'LogisticRegression_EDA_results', '01_linear_relationships.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature scaling comparison
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Logistic Regression: Importance of Feature Scaling', 
                     fontsize=16, fontweight='bold')
        
        # Before scaling
        ax = axes[0]
        data_subset = self.data[self.numerical_features[:5]].iloc[::10]  # Sample for visibility
        ax.boxplot([data_subset[feat] for feat in self.numerical_features[:5]], 
                   labels=self.numerical_features[:5])
        ax.set_ylabel('Raw Values', fontweight='bold', fontsize=12)
        ax.set_title('Before Scaling (Different scales affect convergence)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        # After scaling
        ax = axes[1]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_subset)
        ax.boxplot([scaled_data[:, i] for i in range(5)], 
                   labels=self.numerical_features[:5])
        ax.set_ylabel('Standardized Values', fontweight='bold', fontsize=12)
        ax.set_title('After StandardScaler (Equal scale for fair comparison)', fontsize=12)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.5, label='Mean=0')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'LogisticRegression_EDA_results', '02_scaling_importance.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distance_analysis_knn(self):
        """KNN: Distance-based analysis"""
        
        # Feature scale impact on distance
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('KNN: Feature Scaling Impact on Distance Calculations', 
                     fontsize=16, fontweight='bold')
        
        # Take two features with very different scales
        feat1, feat2 = 'temp', 'cloudcover'
        sample_data = self.data[[feat1, feat2, 'target_binary']].sample(200, random_state=42)
        
        # Unscaled
        ax = axes[0, 0]
        colors = ['red' if x == 1 else 'blue' for x in sample_data['target_binary']]
        ax.scatter(sample_data[feat1], sample_data[feat2], c=colors, alpha=0.6, s=50)
        ax.set_xlabel(f'{feat1} (unscaled)', fontweight='bold')
        ax.set_ylabel(f'{feat2} (unscaled)', fontweight='bold')
        ax.set_title('Unscaled Features\n(cloudcover dominates distance)')
        ax.grid(True, alpha=0.3)
        ax.legend(['Low Demand', 'High Demand'])
        
        # Scaled
        ax = axes[0, 1]
        scaler = StandardScaler()
        scaled_sample = scaler.fit_transform(sample_data[[feat1, feat2]])
        ax.scatter(scaled_sample[:, 0], scaled_sample[:, 1], c=colors, alpha=0.6, s=50)
        ax.set_xlabel(f'{feat1} (scaled)', fontweight='bold')
        ax.set_ylabel(f'{feat2} (scaled)', fontweight='bold')
        ax.set_title('Scaled Features\n(Equal contribution to distance)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Distance comparison
        ax = axes[1, 0]
        # Calculate distances before and after scaling
        from scipy.spatial.distance import pdist
        dist_unscaled = pdist(sample_data[[feat1, feat2]].values[:50])
        dist_scaled = pdist(scaled_sample[:50])
        
        ax.hist(dist_unscaled, bins=30, alpha=0.5, label='Unscaled', color='red')
        ax.hist(dist_scaled, bins=30, alpha=0.5, label='Scaled', color='blue')
        ax.set_xlabel('Pairwise Distance', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distance Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # K value sensitivity
        ax = axes[1, 1]
        k_values = range(1, 51, 2)
        sqrt_n = int(np.sqrt(len(self.data)))
        ax.axvline(x=sqrt_n, color='r', linestyle='--', linewidth=2, 
                   label=f'sqrt(n)={sqrt_n}')
        ax.fill_betweenx([0, 1], sqrt_n-10, sqrt_n+10, alpha=0.2, color='green', 
                         label='Recommended range')
        ax.set_xlabel('K Value', fontweight='bold')
        ax.set_ylabel('', fontweight='bold')
        ax.set_title('K Value Selection Guide\n(Start with sqrt(n))')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'KNN_EDA_results', '01_distance_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Local neighborhood analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('KNN: Local Class Distribution Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Hour of day neighborhoods
        ax = axes[0, 0]
        hourly_balance = self.data.groupby('hour_of_day')['target_binary'].agg(['mean', 'count'])
        hourly_balance['mean'] *= 100
        ax.bar(hourly_balance.index, hourly_balance['mean'], alpha=0.7, color='steelblue')
        ax.set_xlabel('Hour of Day', fontweight='bold')
        ax.set_ylabel('% High Demand', fontweight='bold')
        ax.set_title('Local Class Balance by Hour\n(KNN predictions vary by region)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend()
        
        # Temperature neighborhoods
        ax = axes[0, 1]
        temp_bins = pd.cut(self.data['temp'], bins=10)
        temp_balance = self.data.groupby(temp_bins, observed=True)['target_binary'].agg(['mean', 'count'])
        temp_balance['mean'] *= 100
        temp_balance.plot(y='mean', kind='bar', ax=ax, color='coral', alpha=0.7, legend=False)
        ax.set_xlabel('Temperature Range', fontweight='bold')
        ax.set_ylabel('% High Demand', fontweight='bold')
        ax.set_title('Local Class Balance by Temperature')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        # Sample density
        ax = axes[1, 0]
        ax.hist2d(self.data['temp'], self.data['humidity'], bins=30, cmap='YlOrRd')
        ax.set_xlabel('Temperature', fontweight='bold')
        ax.set_ylabel('Humidity', fontweight='bold')
        ax.set_title('Sample Density Heatmap\n(Sparse regions may have unreliable predictions)')
        plt.colorbar(ax.collections[0], ax=ax, label='Sample Count')
        
        # Dimensionality impact
        ax = axes[1, 1]
        n_features = range(1, len(self.numerical_features) + 1)
        samples_per_dim = [len(self.data) / n for n in n_features]
        ax.plot(n_features, samples_per_dim, marker='o', linewidth=2, markersize=8, color='darkgreen')
        ax.axhline(y=50, color='r', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Minimum threshold (50 samples/dim)')
        ax.set_xlabel('Number of Features', fontweight='bold')
        ax.set_ylabel('Samples per Dimension', fontweight='bold')
        ax.set_title('Curse of Dimensionality Impact')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'KNN_EDA_results', '02_local_neighborhoods.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_common_analysis(self):
        """Common analysis plots for all models"""
        
        # Temporal patterns
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Common Analysis: Temporal Patterns', 
                     fontsize=16, fontweight='bold')
        
        # Hourly
        ax = axes[0, 0]
        hourly = self.data.groupby('hour_of_day')['target_binary'].mean() * 100
        ax.plot(hourly.index, hourly.values, marker='o', linewidth=2, markersize=6, color='navy')
        ax.fill_between(hourly.index, 0, hourly.values, alpha=0.3)
        ax.set_xlabel('Hour of Day', fontweight='bold')
        ax.set_ylabel('% High Demand', fontweight='bold')
        ax.set_title('Demand by Hour of Day')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))
        
        # Weekly
        ax = axes[0, 1]
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly = self.data.groupby('day_of_week')['target_binary'].mean() * 100
        ax.bar(range(7), weekly.values, color=['steelblue']*5 + ['coral']*2, alpha=0.7)
        ax.set_xticks(range(7))
        ax.set_xticklabels(days)
        ax.set_xlabel('Day of Week', fontweight='bold')
        ax.set_ylabel('% High Demand', fontweight='bold')
        ax.set_title('Demand by Day of Week')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Monthly
        ax = axes[1, 0]
        monthly = self.data.groupby('month')['target_binary'].mean() * 100
        months_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.plot(monthly.index, monthly.values, marker='o', linewidth=2, markersize=8, color='darkgreen')
        ax.fill_between(monthly.index, 0, monthly.values, alpha=0.3, color='lightgreen')
        ax.set_xlabel('Month', fontweight='bold')
        ax.set_ylabel('% High Demand', fontweight='bold')
        ax.set_title('Seasonal Pattern')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(monthly.index)
        ax.set_xticklabels([months_short[i-1] for i in monthly.index], rotation=45)
        
        # Weekday vs Weekend
        ax = axes[1, 1]
        weekday_data = self.data.groupby('weekday')['target_binary'].mean() * 100
        ax.bar(['Weekend', 'Weekday'], weekday_data.values, color=['coral', 'steelblue'], alpha=0.7)
        ax.set_ylabel('% High Demand', fontweight='bold')
        ax.set_title('Weekday vs Weekend Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add sample sizes
        counts = self.data.groupby('weekday')['target_binary'].count()
        for i, (label, val, count) in enumerate(zip(['Weekend', 'Weekday'], weekday_data.values, counts.values)):
            ax.text(i, val + 1, f'n={count}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '01_common_temporal_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Weather analysis
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Common Analysis: Weather Conditions Impact', 
                     fontsize=16, fontweight='bold')
        
        # Temperature
        ax = axes[0, 0]
        temp_bins = pd.cut(self.data['temp'], bins=8)
        temp_analysis = self.data.groupby(temp_bins, observed=True)['target_binary'].agg(['mean', 'count'])
        temp_analysis['mean'] *= 100
        temp_analysis.plot(y='mean', kind='bar', ax=ax, color='orangered', alpha=0.7, legend=False)
        ax.set_xlabel('Temperature Range (°C)', fontweight='bold')
        ax.set_ylabel('% High Demand', fontweight='bold')
        ax.set_title('Temperature Impact')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Precipitation
        ax = axes[0, 1]
        precip_grouped = self.data.groupby(self.data['precip'] > 0)['target_binary'].mean() * 100
        precip_labels = {False: 'No Rain', True: 'Rain'}
        precip_data = pd.Series({precip_labels.get(k, str(k)): v for k, v in precip_grouped.items()})
        colors_precip = ['skyblue' if 'No' in str(idx) else 'navy' for idx in precip_data.index]
        ax.bar(range(len(precip_data)), precip_data.values, color=colors_precip, alpha=0.7)
        ax.set_xticks(range(len(precip_data)))
        ax.set_xticklabels(precip_data.index)
        ax.set_ylabel('% High Demand', fontweight='bold')
        ax.set_title('Precipitation Effect')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Snow
        ax = axes[0, 2]
        snow_grouped = self.data.groupby(self.data['snow'] > 0)['target_binary'].mean() * 100
        snow_labels = {False: 'No Snow', True: 'Snow'}
        snow_data = pd.Series({snow_labels.get(k, str(k)): v for k, v in snow_grouped.items()})
        colors_snow = ['lightblue' if 'No' in str(idx) else 'white' for idx in snow_data.index]
        ax.bar(range(len(snow_data)), snow_data.values, color=colors_snow, 
               alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(snow_data)))
        ax.set_xticklabels(snow_data.index)
        ax.set_ylabel('% High Demand', fontweight='bold')
        ax.set_title('Snow Effect')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Humidity
        ax = axes[1, 0]
        ax.scatter(self.data['humidity'], self.data['target_binary'], alpha=0.1, s=10)
        z = np.polyfit(self.data['humidity'], self.data['target_binary'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.data['humidity'].min(), self.data['humidity'].max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2)
        ax.set_xlabel('Humidity (%)', fontweight='bold')
        ax.set_ylabel('Target (jittered)', fontweight='bold')
        ax.set_title(f'Humidity Effect (r={self.data["humidity"].corr(self.data["target_binary"]):+.3f})')
        ax.grid(True, alpha=0.3)
        
        # Wind speed
        ax = axes[1, 1]
        wind_bins = pd.cut(self.data['windspeed'], bins=6)
        wind_analysis = self.data.groupby(wind_bins, observed=True)['target_binary'].mean() * 100
        wind_analysis.plot(kind='bar', ax=ax, color='lightseagreen', alpha=0.7)
        ax.set_xlabel('Wind Speed Range (km/h)', fontweight='bold')
        ax.set_ylabel('% High Demand', fontweight='bold')
        ax.set_title('Wind Speed Impact')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Cloud cover
        ax = axes[1, 2]
        cloud_bins = pd.cut(self.data['cloudcover'], bins=5)
        cloud_analysis = self.data.groupby(cloud_bins, observed=True)['target_binary'].mean() * 100
        cloud_analysis.plot(kind='bar', ax=ax, color='slategray', alpha=0.7)
        ax.set_xlabel('Cloud Cover Range (%)', fontweight='bold')
        ax.set_ylabel('% High Demand', fontweight='bold')
        ax.set_title('Cloud Cover Effect')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '02_common_weather_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Target distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Common Analysis: Target Variable Distribution', 
                     fontsize=16, fontweight='bold')
        
        # Count plot
        ax = axes[0]
        target_counts = self.data[self.target_col].value_counts()
        colors_map = {'low_bike_demand': 'steelblue', 'high_bike_demand': 'coral'}
        ax.bar(range(len(target_counts)), target_counts.values, 
               color=[colors_map[x] for x in target_counts.index], alpha=0.7)
        ax.set_xticks(range(len(target_counts)))
        ax.set_xticklabels(['Low Demand', 'High Demand'])
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Class Distribution (Absolute)')
        for i, (label, count) in enumerate(target_counts.items()):
            ax.text(i, count + 20, f'{count}\n({count/len(self.data)*100:.1f}%)', 
                   ha='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Pie chart
        ax = axes[1]
        ax.pie(target_counts.values, labels=['Low Demand', 'High Demand'], 
               autopct='%1.1f%%', colors=['steelblue', 'coral'], startangle=90, 
               explode=(0.05, 0.05))
        ax.set_title('Class Distribution (Proportion)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '03_common_target_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Visualizations saved to {OUTPUT_DIR}/")
    
    def save_reports(self):
        """Save all analysis reports to text files"""
        
        # Generate all reports
        reports = {
            'summary': self.generate_summary_stats(),
            'temporal': self.analyze_temporal_patterns(),
            'weather': self.analyze_weather_impact(),
            'lda': self.analyze_for_lda(),
            'trees': self.analyze_for_trees(),
            'logreg': self.analyze_for_logistic_regression(),
            'knn': self.analyze_for_knn()
        }
        
        # Save comprehensive report
        comprehensive_path = f'{OUTPUT_DIR}/COMPREHENSIVE_EDA_REPORT.txt'
        with open(comprehensive_path, 'w') as f:
            f.write("COMPREHENSIVE EXPLORATORY DATA ANALYSIS\n")
            f.write("Bike Demand Prediction Project\n")
            f.write("Statistical Machine Learning HT2025\n")
            f.write("Uppsala University\n")
            f.write("="*80 + "\n\n")
            
            for section, content in reports.items():
                f.write(content)
                f.write("\n\n")
        
        # Save model-specific reports
        model_reports = {
            'LDA_EDA_results': reports['lda'],
            'Tree_EDA_results': reports['trees'],
            'LogisticRegression_EDA_results': reports['logreg'],
            'KNN_EDA_results': reports['knn']
        }
        
        for model_dir, content in model_reports.items():
            report_path = f'{OUTPUT_DIR}/{model_dir}/ANALYSIS_REPORT.txt'
            with open(report_path, 'w') as f:
                f.write(f"{model_dir.replace('_EDA_results', '')} - SPECIFIC ANALYSIS\n")
                f.write("="*80 + "\n\n")
                f.write(reports['summary'])
                f.write("\n\n")
                f.write(reports['temporal'])
                f.write("\n\n")
                f.write(reports['weather'])
                f.write("\n\n")
                f.write(content)
        
        print(f"✓ Reports saved successfully")
        print(f"  - Comprehensive report: {comprehensive_path}")
        for model_dir in model_reports.keys():
            print(f"  - {model_dir}: {OUTPUT_DIR}/{model_dir}/ANALYSIS_REPORT.txt")


def main():
    """Execute complete EDA pipeline"""
    
    print("="*80)
    print("BIKE DEMAND PREDICTION - PRODUCTION EDA")
    print("="*80)
    print()
    
    # Initialize analyzer
    analyzer = BikeDataAnalyzer('./training_data_ht2025.csv')
    
    # Run analysis
    print("Running comprehensive analysis...")
    print()
    
    # Console output
    print(analyzer.generate_summary_stats())
    print(analyzer.analyze_temporal_patterns())
    print(analyzer.analyze_weather_impact())
    print(analyzer.analyze_for_lda())
    print(analyzer.analyze_for_trees())
    print(analyzer.analyze_for_logistic_regression())
    print(analyzer.analyze_for_knn())
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.create_visualizations()
    
    # Save reports
    print("\nSaving analysis reports...")
    analyzer.save_reports()
    
    print("\n" + "="*80)
    print("EDA COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nModel-specific folders created:")
    print("  - LDA_EDA_results/")
    print("  - Tree_EDA_results/")
    print("  - LogisticRegression_EDA_results/")
    print("  - KNN_EDA_results/")
    print("\nReview the reports before starting model development.")
    print("="*80)


if __name__ == "__main__":
    main()