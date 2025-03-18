import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import random
import re
import openai

# IMPORTANT: This must be the first Streamlit command
st.set_page_config(
    page_title="Property Clustering Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# PropertyInsightGenerator class for template-based GenAI features
class PropertyInsightGenerator:
    def __init__(self, clustered_data=None):
        self.data = clustered_data
        
    def update_data(self, clustered_data):
        """Update the property data"""
        self.data = clustered_data
        
    def generate_insight(self, query, use_templates=True):
        """Generate insights based on user queries"""
        if self.data is None:
            return "Please run clustering first before generating insights."
        
        # Calculate cluster statistics for reference
        cluster_stats = self.data.groupby('Cluster').agg({
            'Lease_Up_Time': 'mean',
            'Average_Rent_During_LeaseUp': 'mean',
            'Effective_Age': 'mean',
            'Quantity': 'mean',
            'AreaPerUnit': 'mean',
            'Average_Concession_During_LeaseUp': 'mean'
        }).round(2)
        
        # Simple query classification
        if use_templates:
            return self._template_based_response(query, cluster_stats)
        else:
            # In a real implementation, this would call an LLM API
            return "AI insight generation is not available in the demo. Please use the template-based insights."
    
    def _template_based_response(self, query, stats):
        """Use templates to generate insights based on pattern matching"""
        query = query.lower()
        
        # General statistics overview
        if "overview" in query or "summary" in query or "general" in query:
            return self._generate_overview(stats)
            
        # Lease-up time specific queries
        elif any(term in query for term in ["fast", "quick", "speed", "fastest"]):
            return self._generate_lease_up_speed_insight(stats)
            
        # Rent-related queries
        elif any(term in query for term in ["rent", "price", "cost", "expensive"]):
            return self._generate_rent_insight(stats)
            
        # Concession-related queries
        elif any(term in query for term in ["concession", "discount", "incentive"]):
            return self._generate_concession_insight(stats)
            
        # Size-related queries
        elif any(term in query for term in ["size", "area", "square", "footage", "unit size"]):
            return self._generate_size_insight(stats)
            
        # Age-related queries
        elif any(term in query for term in ["age", "old", "new", "year", "built"]):
            return self._generate_age_insight(stats)
            
        # Quantity-related queries
        elif any(term in query for term in ["quantity", "units", "number", "count"]):
            return self._generate_quantity_insight(stats)
            
        # Relationship queries
        elif "relationship" in query or "correlation" in query:
            return self._generate_relationship_insight(stats)
            
        # Anomaly detection
        elif "anomaly" in query or "outlier" in query or "unusual" in query:
            return self._generate_anomaly_insight(stats)
            
        # Recommendations
        elif "recommend" in query or "suggest" in query or "advice" in query:
            return self._generate_recommendation(stats)
            
        # Fallback general analysis when query isn't matched
        else:
            return self._generate_general_insight(query, stats)
    
    def _generate_overview(self, stats):
        """Generate an overview of the clusters"""
        n_clusters = len(stats)
        fastest_cluster = stats['Lease_Up_Time'].idxmin()
        slowest_cluster = stats['Lease_Up_Time'].idxmax()
        highest_rent = stats['Average_Rent_During_LeaseUp'].idxmax()
        lowest_rent = stats['Average_Rent_During_LeaseUp'].idxmin()
        
        response = f"# Cluster Analysis Overview\n\n"
        response += f"The analysis identified **{n_clusters}** distinct property clusters with the following characteristics:\n\n"
        
        response += "## Lease-Up Performance\n"
        response += f"- **Fastest lease-up**: Cluster {fastest_cluster} ({stats.loc[fastest_cluster, 'Lease_Up_Time']:.1f} months)\n"
        response += f"- **Slowest lease-up**: Cluster {slowest_cluster} ({stats.loc[slowest_cluster, 'Lease_Up_Time']:.1f} months)\n\n"
        
        response += "## Rental Rates\n"
        response += f"- **Highest rent**: Cluster {highest_rent} (${stats.loc[highest_rent, 'Average_Rent_During_LeaseUp']:.2f})\n"
        response += f"- **Lowest rent**: Cluster {lowest_rent} (${stats.loc[lowest_rent, 'Average_Rent_During_LeaseUp']:.2f})\n\n"
        
        response += "## Key Insight\n"
        if stats.loc[fastest_cluster, 'Average_Rent_During_LeaseUp'] > stats.loc[slowest_cluster, 'Average_Rent_During_LeaseUp']:
            response += "Properties with higher rents are leasing up faster, suggesting strong demand in premium segments."
        else:
            response += "Properties with lower rents are leasing up faster, suggesting price sensitivity in the market."
            
        return response
    
    def _generate_lease_up_speed_insight(self, stats):
        """Generate insights about lease-up speed"""
        fastest_cluster = stats['Lease_Up_Time'].idxmin()
        fastest_stats = stats.loc[fastest_cluster]
        slowest_cluster = stats['Lease_Up_Time'].idxmax()
        slowest_stats = stats.loc[slowest_cluster]
        
        response = f"# Lease-Up Speed Analysis\n\n"
        response += f"## Fastest Leasing Properties (Cluster {fastest_cluster})\n"
        response += f"- **Average lease-up time**: {fastest_stats['Lease_Up_Time']:.1f} months\n"
        response += f"- **Average rent**: ${fastest_stats['Average_Rent_During_LeaseUp']:.2f}\n"
        response += f"- **Average unit size**: {fastest_stats['AreaPerUnit']:.0f} sq ft\n"
        response += f"- **Average concession**: {fastest_stats['Average_Concession_During_LeaseUp']*100:.1f}%\n"
        response += f"- **Average effective age**: {fastest_stats['Effective_Age']:.1f} years\n\n"
        
        response += f"## Slowest Leasing Properties (Cluster {slowest_cluster})\n"
        response += f"- **Average lease-up time**: {slowest_stats['Lease_Up_Time']:.1f} months\n"
        response += f"- **Average rent**: ${slowest_stats['Average_Rent_During_LeaseUp']:.2f}\n"
        response += f"- **Average unit size**: {slowest_stats['AreaPerUnit']:.0f} sq ft\n"
        response += f"- **Average concession**: {slowest_stats['Average_Concession_During_LeaseUp']*100:.1f}%\n"
        response += f"- **Average effective age**: {slowest_stats['Effective_Age']:.1f} years\n\n"
        
        response += "## Key Differentiators\n"
        
        # Analyze what makes the fastest cluster different
        differentiators = []
        
        # Check rent difference
        rent_diff = fastest_stats['Average_Rent_During_LeaseUp'] - slowest_stats['Average_Rent_During_LeaseUp']
        if abs(rent_diff) > 100:  # Significant difference
            if rent_diff > 0:
                differentiators.append(f"Higher rent (${rent_diff:.0f} more)")
            else:
                differentiators.append(f"Lower rent (${-rent_diff:.0f} less)")
        
        # Check size difference
        size_diff = fastest_stats['AreaPerUnit'] - slowest_stats['AreaPerUnit']
        if abs(size_diff) > 50:  # Significant difference
            if size_diff > 0:
                differentiators.append(f"Larger units ({size_diff:.0f} sq ft larger)")
            else:
                differentiators.append(f"Smaller units ({-size_diff:.0f} sq ft smaller)")
        
        # Check concession difference
        conc_diff = (fastest_stats['Average_Concession_During_LeaseUp'] - slowest_stats['Average_Concession_During_LeaseUp']) * 100
        if abs(conc_diff) > 1:  # More than 1% difference
            if conc_diff > 0:
                differentiators.append(f"Higher concessions ({conc_diff:.1f}% more)")
            else:
                differentiators.append(f"Lower concessions ({-conc_diff:.1f}% less)")
        
        # Check age difference
        age_diff = fastest_stats['Effective_Age'] - slowest_stats['Effective_Age']
        if abs(age_diff) > 3:  # More than 3 years difference
            if age_diff > 0:
                differentiators.append(f"Older properties ({age_diff:.1f} years older)")
            else:
                differentiators.append(f"Newer properties ({-age_diff:.1f} years newer)")
        
        if differentiators:
            response += "The fastest leasing properties are characterized by:\n"
            for diff in differentiators:
                response += f"- {diff}\n"
        else:
            response += "The fastest and slowest leasing properties have similar characteristics, suggesting that lease-up time may be influenced by factors not captured in the basic metrics (such as location quality, amenities, or management)."
        
        return response
    
    def _generate_rent_insight(self, stats):
        """Generate insights about rent and its relationship to lease-up time"""
        # Sort clusters by rent
        sorted_clusters = stats.sort_values('Average_Rent_During_LeaseUp').index
        highest_rent = sorted_clusters[-1]
        lowest_rent = sorted_clusters[0]
        
        # Check if there's a correlation between rent and lease-up time
        rent_values = stats['Average_Rent_During_LeaseUp'].values
        leaseup_values = stats['Lease_Up_Time'].values
        correlation = np.corrcoef(rent_values, leaseup_values)[0, 1]
        
        response = f"# Rent Analysis and Impact on Lease-Up\n\n"
        response += "## Rent by Cluster\n"
        
        for cluster in sorted_clusters:
            response += f"- **Cluster {cluster}**: ${stats.loc[cluster, 'Average_Rent_During_LeaseUp']:.2f}, "
            response += f"Lease-up: {stats.loc[cluster, 'Lease_Up_Time']:.1f} months\n"
        
        response += f"\n## Rent Range\n"
        response += f"- **Highest rent cluster ({highest_rent})**: ${stats.loc[highest_rent, 'Average_Rent_During_LeaseUp']:.2f}\n"
        response += f"- **Lowest rent cluster ({lowest_rent})**: ${stats.loc[lowest_rent, 'Average_Rent_During_LeaseUp']:.2f}\n"
        response += f"- **Rent spread**: ${stats.loc[highest_rent, 'Average_Rent_During_LeaseUp'] - stats.loc[lowest_rent, 'Average_Rent_During_LeaseUp']:.2f}\n\n"
        
        response += "## Relationship with Lease-Up Time\n"
        
        if abs(correlation) < 0.3:
            response += "There is **no strong correlation** between rent and lease-up time across clusters, suggesting that other factors are more important in determining how quickly properties lease up."
        elif correlation > 0:
            response += f"There is a **positive correlation** ({correlation:.2f}) between rent and lease-up time, meaning higher-priced properties generally take longer to lease up."
        else:
            response += f"There is a **negative correlation** ({correlation:.2f}) between rent and lease-up time, meaning higher-priced properties actually lease up faster, possibly due to stronger demand in premium segments."
        
        return response
    
    def _generate_concession_insight(self, stats):
        """Generate insights about concessions"""
        # Sort clusters by concession level
        sorted_clusters = stats.sort_values('Average_Concession_During_LeaseUp').index
        highest_conc = sorted_clusters[-1]
        lowest_conc = sorted_clusters[0]
        
        # Calculate correlation with lease-up time
        conc_values = stats['Average_Concession_During_LeaseUp'].values
        leaseup_values = stats['Lease_Up_Time'].values
        correlation = np.corrcoef(conc_values, leaseup_values)[0, 1]
        
        response = f"# Concession Analysis\n\n"
        response += "## Concessions by Cluster\n"
        
        for cluster in sorted_clusters:
            response += f"- **Cluster {cluster}**: {stats.loc[cluster, 'Average_Concession_During_LeaseUp']*100:.1f}%, "
            response += f"Lease-up: {stats.loc[cluster, 'Lease_Up_Time']:.1f} months\n"
        
        response += f"\n## Concession Range\n"
        response += f"- **Highest concession cluster ({highest_conc})**: {stats.loc[highest_conc, 'Average_Concession_During_LeaseUp']*100:.1f}%\n"
        response += f"- **Lowest concession cluster ({lowest_conc})**: {stats.loc[lowest_conc, 'Average_Concession_During_LeaseUp']*100:.1f}%\n\n"
        
        response += "## Impact on Lease-Up Time\n"
        
        if abs(correlation) < 0.3:
            response += "There is **no strong correlation** between concession levels and lease-up time, suggesting that concessions alone don't significantly impact lease-up speed."
        elif correlation > 0:
            response += f"There is a **positive correlation** ({correlation:.2f}) between concessions and lease-up time. Properties offering higher concessions are taking longer to lease up, which might indicate that concessions are being used reactively to address slow absorption rather than proactively accelerating it."
        else:
            response += f"There is a **negative correlation** ({correlation:.2f}) between concessions and lease-up time. Properties offering higher concessions are leasing up faster, suggesting that concessions may be an effective tool for accelerating absorption."
        
        return response
    
    def _generate_size_insight(self, stats):
        """Generate insights about unit size"""
        # Sort clusters by unit size
        sorted_clusters = stats.sort_values('AreaPerUnit').index
        largest_size = sorted_clusters[-1]
        smallest_size = sorted_clusters[0]
        
        # Calculate correlation with lease-up time
        size_values = stats['AreaPerUnit'].values
        leaseup_values = stats['Lease_Up_Time'].values
        correlation = np.corrcoef(size_values, leaseup_values)[0, 1]
        
        response = f"# Unit Size Analysis\n\n"
        response += "## Unit Size by Cluster\n"
        
        for cluster in sorted_clusters:
            response += f"- **Cluster {cluster}**: {stats.loc[cluster, 'AreaPerUnit']:.0f} sq ft, "
            response += f"Lease-up: {stats.loc[cluster, 'Lease_Up_Time']:.1f} months\n"
        
        response += f"\n## Size Range\n"
        response += f"- **Largest units cluster ({largest_size})**: {stats.loc[largest_size, 'AreaPerUnit']:.0f} sq ft\n"
        response += f"- **Smallest units cluster ({smallest_size})**: {stats.loc[smallest_size, 'AreaPerUnit']:.0f} sq ft\n"
        response += f"- **Size difference**: {stats.loc[largest_size, 'AreaPerUnit'] - stats.loc[smallest_size, 'AreaPerUnit']:.0f} sq ft\n\n"
        
        response += "## Impact on Lease-Up Time\n"
        
        if abs(correlation) < 0.3:
            response += "There is **no strong correlation** between unit size and lease-up time across clusters, suggesting that unit size alone doesn't significantly impact lease-up speed."
        elif correlation > 0:
            response += f"There is a **positive correlation** ({correlation:.2f}) between unit size and lease-up time, meaning properties with larger units generally take longer to lease up."
        else:
            response += f"There is a **negative correlation** ({correlation:.2f}) between unit size and lease-up time, meaning properties with larger units actually lease up faster, possibly indicating stronger demand for larger spaces."
        
        return response
    
    def _generate_age_insight(self, stats):
        """Generate insights about property age"""
        # Sort clusters by age
        sorted_clusters = stats.sort_values('Effective_Age').index
        oldest = sorted_clusters[-1]
        newest = sorted_clusters[0]
        
        # Calculate correlation with lease-up time
        age_values = stats['Effective_Age'].values
        leaseup_values = stats['Lease_Up_Time'].values
        correlation = np.corrcoef(age_values, leaseup_values)[0, 1]
        
        response = f"# Property Age Analysis\n\n"
        response += "## Effective Age by Cluster\n"
        
        for cluster in sorted_clusters:
            response += f"- **Cluster {cluster}**: {stats.loc[cluster, 'Effective_Age']:.1f} years, "
            response += f"Lease-up: {stats.loc[cluster, 'Lease_Up_Time']:.1f} months\n"
        
        response += f"\n## Age Range\n"
        response += f"- **Oldest cluster ({oldest})**: {stats.loc[oldest, 'Effective_Age']:.1f} years\n"
        response += f"- **Newest cluster ({newest})**: {stats.loc[newest, 'Effective_Age']:.1f} years\n"
        response += f"- **Age difference**: {stats.loc[oldest, 'Effective_Age'] - stats.loc[newest, 'Effective_Age']:.1f} years\n\n"
        
        response += "## Impact on Lease-Up Time\n"
        
        if abs(correlation) < 0.3:
            response += "There is **no strong correlation** between property age and lease-up time across clusters, suggesting that age alone doesn't significantly impact lease-up speed."
        elif correlation > 0:
            response += f"There is a **positive correlation** ({correlation:.2f}) between property age and lease-up time, meaning older properties generally take longer to lease up."
        else:
            response += f"There is a **negative correlation** ({correlation:.2f}) between property age and lease-up time, meaning older properties actually lease up faster, which is somewhat counter-intuitive and may indicate other factors at play."
        
        return response
    
    def _generate_quantity_insight(self, stats):
        """Generate insights about property quantity/number of units"""
        # Sort clusters by quantity
        sorted_clusters = stats.sort_values('Quantity').index
        largest = sorted_clusters[-1]
        smallest = sorted_clusters[0]
        
        # Calculate correlation with lease-up time
        size_values = stats['Quantity'].values
        leaseup_values = stats['Lease_Up_Time'].values
        correlation = np.corrcoef(size_values, leaseup_values)[0, 1]
        
        response = f"# Property Size Analysis\n\n"
        response += "## Number of Units by Cluster\n"
        
        for cluster in sorted_clusters:
            response += f"- **Cluster {cluster}**: {stats.loc[cluster, 'Quantity']:.0f} units, "
            response += f"Lease-up: {stats.loc[cluster, 'Lease_Up_Time']:.1f} months\n"
        
        response += f"\n## Size Range\n"
        response += f"- **Largest property cluster ({largest})**: {stats.loc[largest, 'Quantity']:.0f} units\n"
        response += f"- **Smallest property cluster ({smallest})**: {stats.loc[smallest, 'Quantity']:.0f} units\n"
        response += f"- **Size difference**: {stats.loc[largest, 'Quantity'] - stats.loc[smallest, 'Quantity']:.0f} units\n\n"
        
        response += "## Impact on Lease-Up Time\n"
        
        if abs(correlation) < 0.3:
            response += "There is **no strong correlation** between property size and lease-up time across clusters, suggesting that the number of units alone doesn't significantly impact lease-up speed."
        elif correlation > 0:
            response += f"There is a **positive correlation** ({correlation:.2f}) between property size and lease-up time, meaning larger properties generally take longer to lease up, which is expected given the greater number of units to fill."
        else:
            response += f"There is a **negative correlation** ({correlation:.2f}) between property size and lease-up time, meaning larger properties actually lease up faster, which might indicate economies of scale in marketing or stronger demand for larger communities."
        
        return response
    
    def _generate_relationship_insight(self, stats):
        """Generate insights about relationships between variables"""
        response = f"# Relationship Analysis\n\n"
        
        # Calculate correlations between all variables
        correlations = stats.corr()
        
        response += "## Correlation with Lease-Up Time\n"
        lease_up_corr = correlations['Lease_Up_Time'].drop('Lease_Up_Time').sort_values(ascending=False)
        
        for var, corr in lease_up_corr.items():
            var_name = var
            if var == 'Average_Rent_During_LeaseUp':
                var_name = 'Rent'
            elif var == 'Average_Concession_During_LeaseUp':
                var_name = 'Concessions'
            elif var == 'AreaPerUnit':
                var_name = 'Unit Size'
            elif var == 'Effective_Age':
                var_name = 'Effective Age'
            
            if abs(corr) < 0.3:
                strength = "weak"
            elif abs(corr) < 0.7:
                strength = "moderate"
            else:
                strength = "strong"
            
            direction = "positive" if corr > 0 else "negative"
            
            response += f"- **{var_name}**: {corr:.2f} ({strength} {direction} correlation)\n"
        
        response += "\n## Key Relationships\n"
        
        # Find the strongest relationship
        strongest_pair = None
        strongest_corr = 0
        
        for col1 in stats.columns:
            for col2 in stats.columns:
                if col1 != col2:
                    corr = abs(correlations.loc[col1, col2])
                    if corr > strongest_corr:
                        strongest_corr = corr
                        strongest_pair = (col1, col2)
        
        if strongest_pair:
            col1, col2 = strongest_pair
            actual_corr = correlations.loc[col1, col2]
            
            # Clean up variable names
            col1_name = col1
            col2_name = col2
            
            if col1 == 'Average_Rent_During_LeaseUp':
                col1_name = 'Rent'
            elif col1 == 'Average_Concession_During_LeaseUp':
                col1_name = 'Concessions'
            elif col1 == 'AreaPerUnit':
                col1_name = 'Unit Size'
            elif col1 == 'Effective_Age':
                col1_name = 'Effective Age'
            elif col1 == 'Lease_Up_Time':
                col1_name = 'Lease-Up Time'
            
            if col2 == 'Average_Rent_During_LeaseUp':
                col2_name = 'Rent'
            elif col2 == 'Average_Concession_During_LeaseUp':
                col2_name = 'Concessions'
            elif col2 == 'AreaPerUnit':
                col2_name = 'Unit Size'
            elif col2 == 'Effective_Age':
                col2_name = 'Effective Age'
            elif col2 == 'Lease_Up_Time':
                col2_name = 'Lease-Up Time'
            
            relationship = "increases" if actual_corr > 0 else "decreases"
            
            response += f"The strongest relationship is between **{col1_name}** and **{col2_name}** with a correlation of {actual_corr:.2f}.\n\n"
            response += f"This suggests that as {col1_name.lower()} {relationship}, {col2_name.lower()} tends to {'increase' if actual_corr > 0 else 'decrease'} across the different property clusters."
        
        return response
    
    def _generate_anomaly_insight(self, stats):
        """Generate insights about anomalies in the data"""
        response = f"# Anomaly Detection\n\n"
        
        # Look for outlier clusters
        outliers = []
        
        for col in stats.columns:
            col_mean = stats[col].mean()
            col_std = stats[col].std()
            
            # Check for values more than 2 standard deviations from the mean
            for cluster in stats.index:
                z_score = (stats.loc[cluster, col] - col_mean) / col_std if col_std > 0 else 0
                
                if abs(z_score) > 2:
                    # Clean up variable names
                    col_name = col
                    if col == 'Average_Rent_During_LeaseUp':
                        col_name = 'Rent'
                    elif col == 'Average_Concession_During_LeaseUp':
                        col_name = 'Concessions'
                    elif col == 'AreaPerUnit':
                        col_name = 'Unit Size'
                    elif col == 'Effective_Age':
                        col_name = 'Effective Age'
                    elif col == 'Lease_Up_Time':
                        col_name = 'Lease-Up Time'
                    
                    direction = "high" if z_score > 0 else "low"
                    outliers.append((cluster, col_name, direction, abs(z_score), stats.loc[cluster, col]))
        
        if outliers:
            # Sort by z-score (most extreme first)
            outliers.sort(key=lambda x: x[3], reverse=True)
            
            response += "The following anomalies were detected in the clusters:\n\n"
            
            for cluster, metric, direction, z_score, value in outliers:
                if metric == 'Rent':
                    value_str = f"${value:.2f}"
                elif metric == 'Concessions':
                    value_str = f"{value*100:.1f}%"
                elif metric == 'Unit Size':
                    value_str = f"{value:.0f} sq ft"
                elif metric == 'Lease-Up Time':
                    value_str = f"{value:.1f} months"
                else:
                    value_str = f"{value:.1f}"
                
                response += f"- **Cluster {cluster}** has unusually {direction} {metric.lower()} ({value_str}), {z_score:.1f} standard deviations from the mean\n"
        else:
            response += "No significant anomalies were detected in the clusters. All metrics are within expected ranges."
        
        return response
    
    def _generate_recommendation(self, stats):
        """Generate recommendations based on the data"""
        # Placeholder for recommendations
        fastest_cluster = stats['Lease_Up_Time'].idxmin()
        fastest_stats = stats.loc[fastest_cluster]
        
        response = f"# Recommendations for Optimizing Lease-Up Time\n\n"
        response += "Based on the cluster analysis, here are recommendations to potentially improve lease-up performance:\n\n"
        
        response += f"## Target Profile for Fastest Lease-Up\n"
        response += f"Properties with the following characteristics tend to lease up fastest (based on Cluster {fastest_cluster}):\n\n"
        response += f"- **Rent level**: ${fastest_stats['Average_Rent_During_LeaseUp']:.0f}\n"
        response += f"- **Unit size**: {fastest_stats['AreaPerUnit']:.0f} sq ft\n"
        response += f"- **Concession level**: {fastest_stats['Average_Concession_During_LeaseUp']*100:.1f}%\n"
        response += f"- **Property size**: {fastest_stats['Quantity']:.0f} units\n"
        
        response += f"\n## Strategic Recommendations\n"
        
        # Random recommendations based on hypothetical patterns in the data
        recommendations = [
            "Consider adjusting concession levels early in lease-up to accelerate initial absorption.",
            "Focus on properties with unit sizes that align with market demand based on cluster analysis.",
            "Target rent levels that balance revenue goals with lease-up speed objectives.",
            "For slower-leasing property types, consider phased delivery to avoid extended exposure.",
            "Evaluate the relationship between property age and lease-up performance in your portfolio."
        ]
        
        # Add 3 random recommendations
        selected_recommendations = random.sample(recommendations, 3)
        for i, rec in enumerate(selected_recommendations, 1):
            response += f"{i}. {rec}\n"
        
        return response
    
    def _generate_general_insight(self, query, stats):
        """Generate general insights when the query doesn't match specific patterns"""
        response = f"# Analysis Based on Your Query: '{query}'\n\n"
        
        response += "## Cluster Overview\n"
        for cluster in stats.index:
            response += f"**Cluster {cluster}**:\n"
            response += f"- Lease-up time: {stats.loc[cluster, 'Lease_Up_Time']:.1f} months\n"
            response += f"- Average rent: ${stats.loc[cluster, 'Average_Rent_During_LeaseUp']:.2f}\n"
            response += f"- Unit size: {stats.loc[cluster, 'AreaPerUnit']:.0f} sq ft\n"
            response += f"- Concessions: {stats.loc[cluster, 'Average_Concession_During_LeaseUp']*100:.1f}%\n\n"
        
        response += "## Key Observations\n"
        
        # Find the fastest and slowest lease-up clusters
        fastest = stats['Lease_Up_Time'].idxmin()
        slowest = stats['Lease_Up_Time'].idxmax()
        
        response += f"- The fastest leasing cluster ({fastest}) takes {stats.loc[fastest, 'Lease_Up_Time']:.1f} months to stabilize.\n"
        response += f"- The slowest leasing cluster ({slowest}) takes {stats.loc[slowest, 'Lease_Up_Time']:.1f} months to stabilize.\n"
        
        # Find the highest and lowest rent clusters
        highest_rent = stats['Average_Rent_During_LeaseUp'].idxmax()
        lowest_rent = stats['Average_Rent_During_LeaseUp'].idxmin()
        
        response += f"- Rent levels range from ${stats.loc[lowest_rent, 'Average_Rent_During_LeaseUp']:.2f} (Cluster {lowest_rent}) to"
        response += f" ${stats.loc[highest_rent, 'Average_Rent_During_LeaseUp']:.2f} (Cluster {highest_rent}).\n"
        
        response += "\nFor more specific insights, try asking about lease-up speed, rent levels, concessions, or relationships between variables."
        
        return response

# Function to generate AI insights using OpenAI API
from openai import OpenAI  # Add this import at the top of your file

def generate_ai_insights(query, clustered_df, api_key=None):
    """Generate AI insights using OpenAI API"""
    if not api_key:
        return "Please provide an OpenAI API key to use advanced AI insights."
    
    if clustered_df is None:
        return "Please run clustering first before generating insights."
    
    # Calculate cluster statistics
    cluster_stats = clustered_df.groupby('Cluster').agg({
        'Lease_Up_Time': 'mean',
        'Average_Rent_During_LeaseUp': 'mean',
        'Effective_Age': 'mean',
        'Quantity': 'mean',
        'AreaPerUnit': 'mean',
        'Average_Concession_During_LeaseUp': 'mean'
    }).round(2).to_string()
    
    # Create a prompt for the AI
    prompt = f"""
    You are a real estate analytics expert analyzing property clustering results.
    
    Cluster statistics:
    {cluster_stats}
    
    Based on this data, please provide insights for the query: "{query}"
    
    Focus on lease-up time patterns, relationships between variables, and actionable insights.
    Format your response with markdown headings and bullet points for clarity.
    """
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a real estate analytics expert specializing in property clustering and lease-up analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"
# Initialize the insight generator
if 'insight_generator' not in st.session_state:
    st.session_state.insight_generator = PropertyInsightGenerator()

# Page title
st.title("Property Clustering & Lease-Up Analysis Dashboard")

# Sidebar for data upload and controls
with st.sidebar:
    st.header("Controls")
    
    uploaded_file = st.file_uploader("Upload your property dataset (CSV)", type=["csv"])
    
    st.subheader("Clustering Settings")
    clustering_method = st.radio(
        "Clustering Method",
        ["KMeans", "DBSCAN"]
    )
    
    if clustering_method == "KMeans":
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
    else:  # DBSCAN
        eps = st.slider("DBSCAN eps", 0.5, 3.0, 1.2, 0.1)
        min_samples = st.slider("DBSCAN min_samples", 2, 15, 5)
    
    run_clustering = st.button("Run Clustering")
    
    st.divider()
    
    # OpenAI API key input
    with st.expander("🔑 API Settings"):
        api_key = st.text_input("OpenAI API Key", type="password",
                               help="Enter your OpenAI API key to enable advanced AI insights")
        st.caption("Your API key is not stored and is only used for the current session.")
    
    st.subheader("GenAI Property Insights")
    ai_query = st.text_area(
        "Ask a question about the clusters or properties",
        placeholder="Example: Which cluster has the fastest lease-up time?",
        height=100
    )
    generate_insights = st.button("Generate Insights")

# Function to load sample data if no file is uploaded
def load_sample_data():
    # Create a synthetic dataset for demonstration
    np.random.seed(42)
    n_samples = 100
    
    # Create basic properties
    data = {
        'ProjID': [f'P{i+1}' for i in range(n_samples)],
        'Lease_Up_Time': np.random.gamma(shape=3, scale=5, size=n_samples),
        'YearBuilt': np.random.choice(range(1970, 2021), size=n_samples),
        'Quantity': np.random.choice(range(50, 500, 10), size=n_samples),
        'AreaPerUnit': np.random.normal(900, 150, size=n_samples),
        'Average_Rent_During_LeaseUp': np.random.normal(1350, 350, size=n_samples),
        'Average_Concession_During_LeaseUp': np.random.beta(2, 8, size=n_samples),
        'Effective_Age': 2023 - np.random.choice(range(1970, 2021), size=n_samples),
        'Latitude': np.random.uniform(30, 45, size=n_samples),
        'Longitude': np.random.uniform(-100, -80, size=n_samples),
        'YearBuilt_vs_Overall': np.random.normal(0, 5, size=n_samples),
        'AreaPerUnit_vs_Overall': np.random.normal(0, 0.2, size=n_samples),
        'Quantity_vs_Overall': np.random.normal(0, 0.3, size=n_samples),
        'Individual_Competitive_Score': np.random.normal(0, 1, size=n_samples),
    }
    
    # Create property grade one-hot encoded columns
    grades = ['A', 'A+', 'A-', 'B', 'B+', 'B-', 'C', 'C+', 'C-', 'D']
    for grade in grades:
        data[f'Initial_Property_Grade_{grade}'] = [0] * n_samples
    
    # Assign one grade to each property
    for i in range(n_samples):
        grade = np.random.choice(grades)
        data[f'Initial_Property_Grade_{grade}'][i] = 1
    
    # Create market name one-hot encoded columns
    markets = ['Akron, OH', 'Austin-Round Rock, TX']
    for market in markets:
        data[f'MarketName_{market}'] = [0] * n_samples
    
    # Assign one market to each property
    for i in range(n_samples):
        market = np.random.choice(markets)
        data[f'MarketName_{market}'][i] = 1
        
        # Set corresponding state codes
        if 'OH' in market:
            if 'State_OH' not in data:
                data['State_OH'] = [0] * n_samples
            if 'State_TX' not in data:
                data['State_TX'] = [0] * n_samples
            data['State_OH'][i] = 1
            data['State_TX'][i] = 0
        else:
            if 'State_OH' not in data:
                data['State_OH'] = [0] * n_samples
            if 'State_TX' not in data:
                data['State_TX'] = [0] * n_samples
            data['State_OH'][i] = 0
            data['State_TX'][i] = 1
    
    # Create First Status one-hot encoded columns
    statuses = ['LU', 'UC/LU']
    for status in statuses:
        data[f'First_Status_{status}'] = [0] * n_samples
    
    # Assign one status to each property
    for i in range(n_samples):
        status = np.random.choice(statuses)
        data[f'First_Status_{status}'][i] = 1
    
    df = pd.DataFrame(data)
    return df

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded dataset with {df.shape[0]} properties and {df.shape[1]} columns")
else:
    df = load_sample_data()
    st.info("Using sample data. Upload your own CSV file for actual analysis.")

# Function to perform clustering
def perform_clustering(df, method, **kwargs):
    # Define features
    numerical_features = [
        'Lease_Up_Time', 'YearBuilt', 'Quantity', 'AreaPerUnit',
        'Average_Rent_During_LeaseUp', 'Average_Concession_During_LeaseUp',
        'Effective_Age', 'Individual_Competitive_Score'
    ]
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        numerical_features.extend(['Latitude', 'Longitude'])
    if 'YearBuilt_vs_Overall' in df.columns:
        numerical_features.extend(['YearBuilt_vs_Overall', 'AreaPerUnit_vs_Overall', 'Quantity_vs_Overall'])
    
    # Filter to columns that exist in the dataset
    numerical_features = [col for col in numerical_features if col in df.columns]
    
    # Get other feature types
    grade_features = [col for col in df.columns if col.startswith('Initial_Property_Grade_')]
    market_features = [col for col in df.columns if col.startswith('MarketName_')]
    state_features = [col for col in df.columns if col.startswith('State_')]
    status_features = [col for col in df.columns if col.startswith('First_Status_')]
    
    # Combine all features
    embedding_features = numerical_features + grade_features + market_features + state_features + status_features
    
    # Prepare data
    X = df[embedding_features].fillna(df[embedding_features].mean())
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(10, len(embedding_features)))
    X_pca = pca.fit_transform(X_scaled)
    
    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    # Perform clustering
    if method == "KMeans":
        n_clusters = kwargs.get('n_clusters', 5)
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clustering.fit_predict(X_pca)
        silhouette = silhouette_score(X_pca, labels)
        
    elif method == "DBSCAN":
        eps = kwargs.get('eps', 1.2)
        min_samples = kwargs.get('min_samples', 5)
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(X_pca)
        
        # Calculate silhouette score (excluding noise points)
        if len(set(labels)) - (1 if -1 in labels else 0) >= 2:
            non_noise = labels != -1
            if sum(non_noise) > 1:
                silhouette = silhouette_score(X_pca[non_noise], labels[non_noise])
            else:
                silhouette = 0
        else:
            silhouette = 0
    
    # Add results to dataframe
    result_df = df.copy()
    result_df['Cluster'] = labels
    result_df['TSNE_1'] = X_tsne[:, 0]
    result_df['TSNE_2'] = X_tsne[:, 1]
    
    return result_df, silhouette

# Process data when button is clicked
if run_clustering:
    with st.spinner('Running clustering algorithm...'):
        if clustering_method == "KMeans":
            clustered_df, silhouette = perform_clustering(df, "KMeans", n_clusters=n_clusters)
            cluster_method_desc = f"KMeans ({n_clusters} clusters)"
        else:  # DBSCAN
            clustered_df, silhouette = perform_clustering(df, "DBSCAN", eps=eps, min_samples=min_samples)
            cluster_method_desc = f"DBSCAN (eps={eps}, min_samples={min_samples})"
        
        # Store the results in session state
        st.session_state.clustered_df = clustered_df
        st.session_state.cluster_method = clustering_method
        st.session_state.cluster_method_desc = cluster_method_desc
        st.session_state.silhouette = silhouette
        
        # Update the insight generator with the new data
        st.session_state.insight_generator.update_data(clustered_df)
        
        # Count clusters
        n_clusters = len(set(clustered_df['Cluster']))
        if -1 in set(clustered_df['Cluster']):
            n_clusters -= 1
            n_noise = (clustered_df['Cluster'] == -1).sum()
            st.success(f"Clustering complete: Found {n_clusters} clusters and {n_noise} noise points. Silhouette Score: {silhouette:.3f}")
        else:
            st.success(f"Clustering complete: Found {n_clusters} clusters. Silhouette Score: {silhouette:.3f}")

# Generate AI insights when button is clicked
if generate_insights and ai_query:
    if 'clustered_df' in st.session_state:
        with st.spinner('Generating AI insights...'):
            if api_key:
                # Use OpenAI for more advanced insights
                insights = generate_ai_insights(ai_query, st.session_state.clustered_df, api_key)
            else:
                # Fall back to template-based insights if no API key
                insights = st.session_state.insight_generator.generate_insight(ai_query)
            
            st.session_state.insights = insights
    else:
        st.warning("Please run clustering first before generating insights.")

# Main content section with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Cluster Visualization", "Lease-Up Analysis", "Market Comparison", "Property Grades"])

# Tab 1: Cluster Visualization
with tab1:
    if 'clustered_df' in st.session_state:
        st.subheader(f"Property Clusters ({st.session_state.cluster_method_desc})")
        
        # Create cluster visualization with Plotly
        fig = px.scatter(
            st.session_state.clustered_df,
            x='TSNE_1',
            y='TSNE_2',
            color='Cluster',
            hover_data=['Lease_Up_Time', 'Average_Rent_During_LeaseUp', 'AreaPerUnit'],
            title=f"Property Clusters - t-SNE Visualization (Silhouette Score: {st.session_state.silhouette:.3f})"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster statistics
        st.subheader("Cluster Statistics")
        cluster_stats = st.session_state.clustered_df.groupby('Cluster').agg({
            'Lease_Up_Time': 'mean',
            'Average_Rent_During_LeaseUp': 'mean',
            'Effective_Age': 'mean',
            'Quantity': 'mean',
            'AreaPerUnit': 'mean',
            'Average_Concession_During_LeaseUp': lambda x: x.mean() * 100  # Convert to percentage
        }).round(2)
        
        # Rename columns for better readability
        cluster_stats.columns = ['Avg Lease-Up Time (months)', 'Avg Rent ($)', 'Avg Effective Age (years)',
                              'Avg Units', 'Avg Unit Size (sq ft)', 'Avg Concession (%)']
        
        st.dataframe(cluster_stats, use_container_width=True)
    else:
        st.info("Run clustering to visualize property clusters.")

# Tab 2: Lease-Up Analysis
with tab2:
    if 'clustered_df' in st.session_state:
        st.subheader("Lease-Up Time Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot of lease-up time by cluster
            fig = px.box(
                st.session_state.clustered_df,
                x='Cluster',
                y='Lease_Up_Time',
                title='Lease-Up Time Distribution by Cluster'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot of rent vs lease-up time
            fig = px.scatter(
                st.session_state.clustered_df,
                x='Average_Rent_During_LeaseUp',
                y='Lease_Up_Time',
                color='Cluster',
                size='Quantity',
                title='Lease-Up Time vs. Rent',
                labels={'Average_Rent_During_LeaseUp': 'Average Rent ($)',
                       'Lease_Up_Time': 'Lease-Up Time (months)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Factors Influencing Lease-Up Time")
        
        # Select numerical columns for correlation
        numerical_cols = ['Lease_Up_Time', 'Average_Rent_During_LeaseUp', 'AreaPerUnit',
                         'Average_Concession_During_LeaseUp', 'Effective_Age', 'Quantity']
        numerical_cols = [col for col in numerical_cols if col in st.session_state.clustered_df.columns]
        
        corr = st.session_state.clustered_df[numerical_cols].corr()['Lease_Up_Time'].sort_values(ascending=False)
        corr = corr.drop('Lease_Up_Time')  # Remove self-correlation
        
        # Plot correlation values
        fig = px.bar(
            x=corr.values,
            y=corr.index,
            orientation='h',
            title='Correlation with Lease-Up Time',
            labels={'x': 'Correlation Coefficient', 'y': 'Feature'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run clustering to analyze lease-up time patterns.")

# Tab 3: Market Comparison
with tab3:
    if 'clustered_df' in st.session_state:
        st.subheader("Market Analysis")
        
        # Create Market column if it doesn't exist
        if 'Market' not in st.session_state.clustered_df.columns:
            market_cols = [col for col in st.session_state.clustered_df.columns if col.startswith('MarketName_')]
            
            if market_cols:
                # Create a single Market column
                st.session_state.clustered_df['Market'] = None
                for col in market_cols:
                    market_name = col.replace('MarketName_', '')
                    mask = st.session_state.clustered_df[col] == 1
                    st.session_state.clustered_df.loc[mask, 'Market'] = market_name
        
        if 'Market' in st.session_state.clustered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Lease-up time by market
                fig = px.box(
                    st.session_state.clustered_df,
                    x='Market',
                    y='Lease_Up_Time',
                    color='Cluster',
                    title='Lease-Up Time by Market and Cluster'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cluster distribution by market
                cluster_market = pd.crosstab(
                    st.session_state.clustered_df['Cluster'],
                    st.session_state.clustered_df['Market'],
                    normalize='columns'
                ) * 100
                
                fig = px.bar(
                    cluster_market,
                    title='Cluster Distribution by Market (%)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Market metrics comparison
            st.subheader("Market Metrics Comparison")
            market_metrics = st.session_state.clustered_df.groupby('Market').agg({
                'Lease_Up_Time': 'mean',
                'Average_Rent_During_LeaseUp': 'mean',
                'AreaPerUnit': 'mean',
                'Average_Concession_During_LeaseUp': lambda x: x.mean() * 100,
                'Effective_Age': 'mean',
                'Quantity': 'mean'
            }).round(2)
            
            # Rename columns for better readability
            market_metrics.columns = ['Avg Lease-Up Time (months)', 'Avg Rent ($)', 'Avg Unit Size (sq ft)',
                                   'Avg Concession (%)', 'Avg Effective Age (years)', 'Avg Units']
            
            st.dataframe(market_metrics, use_container_width=True)
        else:
            st.info("No market information available in the dataset.")
    else:
        st.info("Run clustering to analyze market patterns.")

# Tab 4: Property Grades
with tab4:
    if 'clustered_df' in st.session_state:
        st.subheader("Property Grade Analysis")
        
        # Create Property_Grade column if it doesn't exist
        if 'Property_Grade' not in st.session_state.clustered_df.columns:
            grade_cols = [col for col in st.session_state.clustered_df.columns if col.startswith('Initial_Property_Grade_')]
            
            if grade_cols:
                # Create a single Property_Grade column
                st.session_state.clustered_df['Property_Grade'] = None
                for col in grade_cols:
                    grade = col.replace('Initial_Property_Grade_', '')
                    mask = st.session_state.clustered_df[col] == 1
                    st.session_state.clustered_df.loc[mask, 'Property_Grade'] = grade
        
        if 'Property_Grade' in st.session_state.clustered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Lease-up time by property grade
                # Define grade order
                grade_order = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D']
                
                # Filter to grades that exist in the data
                existing_grades = st.session_state.clustered_df['Property_Grade'].unique()
                ordered_grades = [g for g in grade_order if g in existing_grades]
                
                # Create visualization
                fig = px.box(
                    st.session_state.clustered_df,
                    x='Property_Grade',
                    y='Lease_Up_Time',
                    color='Cluster',
                    category_orders={'Property_Grade': ordered_grades},
                    title='Lease-Up Time by Property Grade and Cluster'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cluster distribution by property grade
                grade_cluster = pd.crosstab(
                    st.session_state.clustered_df['Property_Grade'],
                    st.session_state.clustered_df['Cluster'],
                    normalize='columns'
                ) * 100
                
                # Reindex to maintain grade order
                if not grade_cluster.empty:
                    grade_cluster = grade_cluster.reindex(ordered_grades, fill_value=0)
                
                fig = px.bar(
                    grade_cluster,
                    title='Property Grade Distribution by Cluster (%)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Grade metrics comparison
            st.subheader("Property Grade Metrics")
            grade_metrics = st.session_state.clustered_df.groupby('Property_Grade').agg({
                'Lease_Up_Time': 'mean',
                'Average_Rent_During_LeaseUp': 'mean',
                'AreaPerUnit': 'mean',
                'Average_Concession_During_LeaseUp': lambda x: x.mean() * 100,
                'Effective_Age': 'mean',
                'Quantity': 'mean'
            }).round(2)
            
            # Sort by grade order
            if not grade_metrics.empty and len(ordered_grades) > 0:
                grade_metrics = grade_metrics.reindex(ordered_grades)
            
            # Rename columns for better readability
            grade_metrics.columns = ['Avg Lease-Up Time (months)', 'Avg Rent ($)', 'Avg Unit Size (sq ft)',
                                    'Avg Concession (%)', 'Avg Effective Age (years)', 'Avg Units']
            
            st.dataframe(grade_metrics, use_container_width=True)
        else:
            st.info("No property grade information available in the dataset.")
    else:
        st.info("Run clustering to analyze property grade patterns.")

# Add a section for the GenAI insights (if available)
if 'insights' in st.session_state:
    st.divider()
    st.header("GenAI Property Insights")
    st.markdown(st.session_state.insights)

# Footer
st.divider()
st.caption("Property Clustering & Lease-Up Analysis Dashboard | © 2025")
