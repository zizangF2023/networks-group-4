# Academic Network Analysis

A Python-based tool for analyzing academic collaboration networks and their impact on research metrics. This tool processes coauthorship data to generate visualizations and statistics about research collaboration patterns, gender distribution, and citation impacts.

## Features

- Collaboration impact analysis across different network sizes
- Gender-based collaboration and citation analysis
- Network centrality correlation with academic impact metrics (h-index and citations)
- Complementary Cumulative Distribution Function (CCDF) analysis with power-law fitting
- Visualization of collaboration patterns and their effects on academic performance

## Requirements

```
pandas
networkx
matplotlib
seaborn
numpy
```

## Data Requirements

The tool expects three CSV files in a `dataset` directory:
- `coauthorship.csv`: Contains coauthorship network data with columns 'Author ID' and 'Co-author ID'
- `authorsFeatures.csv`: Contains author metrics with columns including 'Author ID', 'Citation Count', and 'h-index'
- `authorsFields.csv`: Contains author demographic data including 'Author ID' and 'Gender'

## Usage

```python
from academic_network import main

# Run all analyses
main()
```

## Analysis Components

1. **Impact Analysis**: Visualizes how the number of collaborators affects citation counts and h-index
2. **Gender Analysis**: 
   - Shows gender distribution across different collaboration levels
   - Compares citation patterns between genders at various collaboration levels
3. **Citation Analysis**: Examines correlations between network centrality and academic impact metrics
4. **CCDF Analysis**: Performs power-law fitting analysis on degree centrality, h-index, and citation counts

## Output

The script generates several visualizations:
- Bar/line plots showing collaboration impact on citations and h-index
- Gender distribution and citation boxplots
- Scatter plots with regression lines for centrality vs. impact metrics
- CCDF plots with power-law fitting for network properties