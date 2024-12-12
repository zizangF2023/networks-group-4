import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data():
    """Load and prepare network data"""
    coauthorship = pd.read_csv('dataset/coauthorship.csv')
    features = pd.read_csv('dataset/authorsFeatures.csv')
    fields = pd.read_csv('dataset/authorsFields.csv')
    
    # Build the graph
    G = nx.from_pandas_edgelist(coauthorship, 'Author ID', 'Co-author ID')
    
    # Extract the largest connected component
    giant = max(nx.connected_components(G), key=len)
    G = G.subgraph(giant).copy()
    
    # Compute degrees and degree centrality
    degree_dict = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)
    
    # Convert centrality measures to DataFrame
    centrality_df = pd.DataFrame({
        'Author ID': list(degree_centrality.keys()),
        'degree': list(degree_dict.values()),
        'degree_centrality': list(degree_centrality.values()),
    })
    
    # Merge all data
    df = centrality_df.merge(features, on='Author ID').merge(fields, on='Author ID')
    
    return df, G

def plot_impact_analysis(df):
    """Plot collaboration impact analysis"""
    plt.figure(figsize=(10, 6))
    
    # Create collaboration groups based on degree
    df['collab_group'] = pd.cut(df['degree'], 
                                bins=[0, 1, 3, 7, 15, np.inf],
                                labels=['1', '2-3', '4-7', '8-15', '15+'])
    
    # Calculate statistics
    impact = df.groupby('collab_group', observed=True).agg({
        'Citation Count': 'mean',
        'h-index': 'mean',
        'Author ID': 'count'
    }).reset_index()
    
    # Bar plot for citations
    ax = plt.gca()
    bars = ax.bar(impact['collab_group'], 
                  impact['Citation Count'],
                  color='lightblue',
                  alpha=0.8)
    
    # Line plot for h-index
    ax2 = ax.twinx()
    ax2.plot(range(len(impact)), 
             impact['h-index'],
             'r-o',
             linewidth=2)
    
    # Add value labels
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(idx, height, f'{int(height):,}',
                ha='center', va='bottom')
        ax2.text(idx, impact['h-index'].iloc[idx], 
                 f'{impact["h-index"].iloc[idx]:.1f}',
                 ha='center', va='bottom', color='red')
        ax.text(idx, height/2, 
                f'n={impact["Author ID"].iloc[idx]:,}',
                ha='center', va='center',
                rotation=90)
    
    ax.set_xlabel('Number of Collaborators')
    ax.set_ylabel('Average Citations', color='blue')
    ax2.set_ylabel('Average h-index', color='red')
    plt.title('Impact by Collaboration Level')
    
    plt.tight_layout()
    plt.show()

def plot_gender_analysis(df):
    """Plot gender-based analysis"""
    # Create collaboration groups based on degree
    df['collab_group'] = pd.cut(df['degree'], 
                                bins=[0, 1, 3, 7, 15, np.inf],
                                labels=['1', '2-3', '4-7', '8-15', '15+'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Gender distribution by collaboration level
    gender_dist = pd.crosstab(df['collab_group'], 
                              df['Gender'], 
                              normalize='index')
    gender_dist.plot(kind='bar', 
                     stacked=True, 
                     ax=ax1,
                     color=['#3498db', '#e74c3c'])  # Consistent blue and red
    ax1.set_title('Gender Distribution by\nCollaboration Level')
    ax1.set_xlabel('Number of Collaborators')
    ax1.set_ylabel('Proportion')
    ax1.legend(title='Gender')
    ax1.tick_params(axis='x', rotation=0)
    
    # Plot 2: Citation distribution by gender and collaboration
    sns.boxplot(x='collab_group', 
                y='Citation Count', 
                hue='Gender', 
                data=df,
                ax=ax2,
                palette=['#3498db', '#e74c3c'])  # Matching colors
    ax2.set_yscale('log')
    ax2.set_title('Citation Distribution by\nGender and Collaboration Level')
    ax2.set_xlabel('Number of Collaborators')
    ax2.set_ylabel('Citations (log scale)')
    ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()

def plot_citation_analysis(df):
    """Analyze the relationship between h-index/citation and degree centrality."""
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot for h-index vs Degree Centrality
    sns.scatterplot(data=df, x='degree_centrality', y='h-index', ax=ax[0])
    sns.regplot(data=df, x='degree_centrality', y='h-index', scatter=False, ax=ax[0], color='red')
    ax[0].set_title('h-index vs Degree Centrality')
    ax[0].set_xlabel('Degree Centrality')
    ax[0].set_ylabel('h-index')

    # Scatter plot for Citation Count vs Degree Centrality with log scale
    sns.scatterplot(data=df, x='degree_centrality', y='Citation Count', ax=ax[1])
    sns.regplot(data=df, x='degree_centrality', y='Citation Count', scatter=False, ax=ax[1], color='red')
    ax[1].set_title('Citation Count vs Degree Centrality')
    ax[1].set_xlabel('Degree Centrality')
    ax[1].set_ylabel('Citation Count')
    ax[1].set_yscale('log')  # Apply log scale to y-axis

    plt.tight_layout()
    plt.show()

    # Compute and print correlation coefficients
    print('Correlation between degree centrality and h-index/Citation Count:')
    correlation_matrix = df[['degree_centrality', 'h-index', 'Citation Count']].corr()
    print(correlation_matrix[['h-index', 'Citation Count']])

def plot_ccdf_analysis(df):
    """Plot CCDF analysis with natural logarithm for power-law fitting"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    def compute_ccdf(data):
        # Remove zeros and sort data
        data_clean = data[data > 0]
        sorted_data = np.sort(data_clean)
        # Calculate CCDF
        ccdf = 1 - np.arange(len(sorted_data)) / float(len(sorted_data))
        return sorted_data, ccdf
    
    def fit_power_law(data, ccdf):
        # Log transform using natural logarithm
        log_x = np.log(data)  # Changed to natural log
        log_y = np.log(ccdf)  # Changed to natural log
        
        # Linear regression on log-log data
        coeffs = np.polyfit(log_x, log_y, 1)
        alpha = -coeffs[0]  # The power-law exponent
        
        # Calculate R-squared
        y_fit = coeffs[1] + coeffs[0] * log_x
        ss_res = np.sum((log_y - y_fit) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return alpha, coeffs, r_squared
    
    # Plot settings
    metrics = {
        'degree_centrality': 'Degree Centrality',
        'h-index': 'h-index',
        'Citation Count': 'Citation Count'
    }
    
    for idx, (metric, label) in enumerate(metrics.items()):
        # Compute CCDF
        x, y = compute_ccdf(df[metric])
        
        # Fit power law
        alpha, coeffs, r_squared = fit_power_law(x, y)
        
        # Plot observed data
        axes[idx].loglog(x, y, 'b.', markersize=2, label='Observed')
        
        # Plot fitted line using natural log
        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
        axes[idx].plot(x_fit, y_fit, 'r-', 
                      label=f'Power law (α ≈ {abs(alpha):.2f})')
        
        # Customize plot
        axes[idx].set_title(f'CCDF of {label}')
        axes[idx].set_xlabel(label)
        axes[idx].set_ylabel('P(X ≥ x)')
        axes[idx].grid(True, which="both", ls="-", alpha=0.2)
        axes[idx].legend()
        
        # Display goodness of fit
        axes[idx].text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                      transform=axes[idx].transAxes)
        
        # Print the scaling parameters
        print(f"\n{label}:")
        print(f"Power law exponent (α) = {abs(alpha):.3f}")
        print(f"R-squared = {r_squared:.3f}")
    
    plt.tight_layout()
    plt.show()
    
    return axes

def main():
    df, G = load_data()
    
    print("Plotting impact analysis...")
    plot_impact_analysis(df)
    
    print("Plotting gender analysis...")
    plot_gender_analysis(df)
    
    print("Plotting citation analysis...")
    plot_citation_analysis(df)
    
    print("Plotting CCDF analysis...")
    plot_ccdf_analysis(df)

if __name__ == "__main__":
    main()