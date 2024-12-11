import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns
from scholarly import scholarly
from mpl_toolkits.basemap import Basemap


'''
--------------------------------------------------------------
Initial Setup
--------------------------------------------------------------
'''
random.seed(18755)


'''
--------------------------------------------------------------
Load Dataset
--------------------------------------------------------------
'''
# Load co-authorship data
coauthorship_df = pd.read_csv('data/coauthorship.csv')
complete_graph = nx.Graph()
edges = list(zip(coauthorship_df['Author ID'], coauthorship_df['Co-author ID']))
complete_graph.add_edges_from(edges)

# Load authors' fields of interest data
authors_fields_df = pd.read_csv('data/authorsFields.csv')
# Get rid of "" in the fields of interest entries
authors_fields_df['Field of Interest'] = authors_fields_df['Field of Interest'].str.strip('"')
# Convert to dictionary with Author ID as key and Field of Interest as value
authors_fields = dict(zip(authors_fields_df['Author ID'], authors_fields_df['Field of Interest']))

# Load authors' features data
authors_features_df = pd.read_csv('data/authorsFeatures.csv')
# Rename columns to desired key names
authors_features_df = authors_features_df.rename(columns={
        "Institute ID": "instituteID",
        "Citation Count": "citationCount",
        "h-index": "hIndex",
        "Gender": "gender",
        "Country": "country"
    })
# Convert to dictionary with Author ID as key and a dict with features as value
# E.g. To get the citation count of author QcRldecAAAAJ by accessing authors_features['QcRldecAAAAJ']['citationCount']
authors_features = authors_features_df.set_index('Author ID').T.to_dict()


'''
--------------------------------------------------------------
Basic Network Analysis
--------------------------------------------------------------
'''
print("\n-------------Stats of the Overall Network-------------")
# Calculate number of nodes
num_nodes = complete_graph.number_of_nodes()
print("Number of nodes in the network:", num_nodes)

# Calculate number of edges
num_edges = complete_graph.number_of_edges()
print("Number of edges in the network:", num_edges)

# Calculate the number of components in the network
num_components = nx.number_connected_components(complete_graph)
print(f"Number of Components in the network: {num_components}")

# Results:
# Number of nodes in the network: 134115
# Number of edges in the network: 153683
# Number of Components in the network: 15125

print("\n-------------Stats of the Giant Component-------------")
# Compute the size of the giant component
largest_cc = max(nx.connected_components(complete_graph), key=len)
G = complete_graph.subgraph(largest_cc)
size_giant_component = G.number_of_nodes()
print(f"Size of the giant component: {size_giant_component}")

# Calculate stats of the giant component
average_degree = sum(dict(G.degree()).values()) / len(G)
print(f"Average Degree of the giant component: {average_degree}")

# average_path_length = nx.average_shortest_path_length(G)
# print(f"Average Path Length of the giant component: {average_path_length}")

# diameter = nx.diameter(G)
# print(f"Diameter of the giant component: {diameter}")

average_clustering_coefficient = nx.average_clustering(G)
print(f"Average Clustering Coefficient of the giant component: {average_clustering_coefficient}")

global_clustering_coefficient = nx.transitivity(G)
print(f"Global Clustering Coefficient of the giant component: {global_clustering_coefficient}")

# Results:
# Size of the giant component: 89474
# Average Degree of the giant component: 2.7207233386235106
# Average Clustering Coefficient of the giant component: 0.16028181036918698
# Global Clustering Coefficient of the giant component: 0.17048518997610226

print("\n-------------Plot Degree Distribution-------------")
# Get the top 5 nodes with the highest degree
top_5_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]
print("Top 5 nodes with the highest degree:", top_5_nodes)

# Plot the degree distribution as a histogram
degrees = [d for _, d in G.degree()]
plt.figure(figsize=(8, 6))
plt.hist(degrees, bins=range(min(degrees), max(degrees)+2), edgecolor='black', align='left')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig('graphs/degree_distribution_hist.png', format='png')
plt.close()

# Plot the degree distribution using CCDF
degree_counts = np.bincount(degrees)
k_values = np.arange(len(degree_counts))
ccdf = np.cumsum(degree_counts[::-1])[::-1] / sum(degree_counts)
plt.figure(figsize=(8, 6))
plt.loglog(k_values, ccdf, 'bo')
plt.xlabel('Degree k')
plt.ylabel('P(degree > k)')
plt.title('CCDF in Log-Log Scale')
plt.grid(True)

# Save the graph
plt.savefig('graphs/degree_distribution_CCDF.png', format='png')
plt.close()


'''
--------------------------------------------------------------
Construct Sub-graph of Fields of Interest
--------------------------------------------------------------
'''
def construct_field_subgraph(coauthorship_graph, authors_fields_dict, weight_threshold=30):
    author_field_graph = nx.Graph()

    # Loop through each edge in the co-authorship graph
    for author1, author2 in coauthorship_graph.edges:
        # Get the fields of interest for each author
        field1 = authors_fields_dict.get(author1)
        field2 = authors_fields_dict.get(author2)

        # If either author has no field information, skip this edge
        if not isinstance(field1, str) or not isinstance(field2, str):
            continue

        # Sort field names to ensure consistent ordering in the undirected graph
        field_pair = tuple(sorted([field1, field2]))

        # Only add edges between different fields
        if field_pair[0] != field_pair[1]:
            # If the edge already exists, increment the weight; otherwise, set it to 1
            if author_field_graph.has_edge(*field_pair):
                author_field_graph[field_pair[0]][field_pair[1]]['weight'] += 1
            else:
                author_field_graph.add_edge(field_pair[0], field_pair[1], weight=1)

    # Remove edges with weight lower than weight_threshold
    edges_to_remove = [(u, v) for u, v, data in author_field_graph.edges(data=True) if data['weight'] < weight_threshold]
    author_field_graph.remove_edges_from(edges_to_remove)

    # Remove nodes that have no edges
    isolated_nodes = [node for node, degree in author_field_graph.degree() if degree == 0]
    author_field_graph.remove_nodes_from(isolated_nodes)

    return author_field_graph

print("\n-------------Construct the Fields of Interest Subgraph-------------")
field_graph = construct_field_subgraph(G, authors_fields, weight_threshold = 80)

# Print out the first 5 pairs of fields of interest that have the highest connection
sorted_field_graph_edges = sorted(field_graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
print("Top 5 pairs of fields that have the highest academic connection:")
for u, v, data in sorted_field_graph_edges[:5]:
    print(f"Field1: {u}, Field2: {v}, Weight: {data['weight']}")

print("\n-------------Draw the Fields of Interest Subgraph-------------")
# Nodes and edges with weight < 100 are removed to make the graph cleaner
plt.figure(figsize=(19, 19))

elarge = [(u, v) for (u, v, d) in field_graph.edges(data=True) if d["weight"] > 800]
esmedium = [(u, v) for (u, v, d) in field_graph.edges(data=True) if 300 < d["weight"] <= 800]
esmall = [(u, v) for (u, v, d) in field_graph.edges(data=True) if d["weight"] <= 300]

pos = nx.circular_layout(field_graph, scale=0.7)

# nodes
nx.draw_networkx_nodes(field_graph, pos, node_size=1500)

# edges
nx.draw_networkx_edges(field_graph, pos, edgelist=elarge, width=3, alpha=0.5, edge_color='red')
nx.draw_networkx_edges(field_graph, pos, edgelist=esmedium, width=2, alpha=0.4, edge_color='green')
nx.draw_networkx_edges(field_graph, pos, edgelist=esmall, width=1, alpha=0.2, edge_color="black")

# node labels
nx.draw_networkx_labels(field_graph, pos, font_family="sans-serif", font_weight="bold", font_size=10)

plt.savefig('graphs/field_graph.png', format='png', bbox_inches='tight')
plt.close()


'''
--------------------------------------------------------------
Analyze Sub-graph of Fields of Interest
--------------------------------------------------------------
'''
print("\n-------------Analysis of Fields of Interest Subgraph-------------")
# Degree Centrality (= degree of a node / total number of nodes)
degree_centrality = nx.degree_centrality(field_graph)
# Betweenness Centrality (it measures how often a node lies on the shortest path between two other nodes)
betweenness_centrality = nx.betweenness_centrality(field_graph, weight='weight')
# Closeness Centrality (it measures how close a node is to all other nodes in a network)
closeness_centrality = nx.closeness_centrality(field_graph)
# Eigenvector Centrality (it measures a node’s influence in a network by considering the centrality of its neighbors)
eigenvector_centrality = nx.eigenvector_centrality(field_graph, weight='weight')

# Sorting the fields by centrality
degree_sorted = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
betweenness_sorted = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
closeness_sorted = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
eigenvector_sorted = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)

print("Fields by Degree Centrality (Top 10):")
print(degree_sorted[:10])

print("\nFields by Betweenness Centrality (Top 10):")
print(betweenness_sorted[:10])

print("\nFields by Closeness Centrality (Top 10):")
print(closeness_sorted[:10])

print("\nFields by Eigenvector Centrality (Top 10):")
print(eigenvector_sorted[:10])

print("\n-------------Plot the Analysis of Fields of Interest Subgraph-------------")
# Select only the top N fields to make the plot readable
top_n = 10
plt.figure(figsize=(16, 10))

# Extract top fields and centrality scores for each measure
fields = [x[0] for x in degree_sorted[:top_n]]

degree_dict = dict(degree_sorted)
betweenness_dict = dict(betweenness_sorted)
closeness_dict = dict(closeness_sorted)
eigenvector_dict = dict(eigenvector_sorted)

degree_values = [degree_dict[field] for field in fields]
betweenness_values = [betweenness_dict[field] for field in fields]
closeness_values = [closeness_dict[field] for field in fields]
eigenvector_values = [eigenvector_dict[field] for field in fields]

# Generate rankings for each field in each centrality measure
degree_ranks = {field: rank + 1 for rank, (field, _) in enumerate(degree_sorted)}
betweenness_ranks = {field: rank + 1 for rank, (field, _) in enumerate(betweenness_sorted)}
closeness_ranks = {field: rank + 1 for rank, (field, _) in enumerate(closeness_sorted)}
eigenvector_ranks = {field: rank + 1 for rank, (field, _) in enumerate(eigenvector_sorted)}

# Set up the bar width and positions for each centrality type in the plot
bar_width = 0.2
indices = np.arange(len(fields))

# Plot each centrality as a bar plot
bars_degree = plt.bar(indices - bar_width*1.5, degree_values, width=bar_width, label='Degree Centrality', color='skyblue')
bars_closeness = plt.bar(indices - bar_width/2, closeness_values, width=bar_width, label='Closeness Centrality', color='lightgreen')
bars_eigenvector = plt.bar(indices + bar_width/2, eigenvector_values, width=bar_width, label='Weighted Eigenvector Centrality', color='purple')
bars_betweenness = plt.bar(indices + bar_width*1.5, betweenness_values, width=bar_width, label='Weighted Betweenness Centrality', color='salmon')


# Add labels showing the rank on top of each bar
for i, field in enumerate(fields):
    plt.text(bars_degree[i].get_x() + bars_degree[i].get_width() / 2, degree_values[i] + 0.01,
             f'#{degree_ranks[field]}', ha='center', va='bottom', color='skyblue', fontweight='bold', fontsize=8)
    plt.text(bars_closeness[i].get_x() + bars_closeness[i].get_width() / 2, closeness_values[i] + 0.01,
             f'#{closeness_ranks[field]}', ha='center', va='bottom', color='lightgreen', fontweight='bold', fontsize=8)
    plt.text(bars_eigenvector[i].get_x() + bars_eigenvector[i].get_width() / 2, eigenvector_values[i] + 0.01,
             f'#{eigenvector_ranks[field]}', ha='center', va='bottom', color='purple', fontweight='bold', fontsize=8)
    plt.text(bars_betweenness[i].get_x() + bars_betweenness[i].get_width() / 2, betweenness_values[i] + 0.01,
             f'#{betweenness_ranks[field]}', ha='center', va='bottom', color='salmon', fontweight='bold', fontsize=8)

# Add field labels to the x-axis
plt.xticks(indices, fields, rotation=45, ha='right')

# Add labels, legend, and title
plt.ylabel('Centrality Measure Value')
plt.title('Comparison of Centrality Measures for Top Fields')
plt.legend()

# Save the plot
plt.tight_layout()
plt.savefig('graphs/field_graph_measures.png', format='png', bbox_inches='tight')
plt.close()


'''
--------------------------------------------------------------
Construct Sub-graph of Countries
--------------------------------------------------------------
'''
def construct_country_subgraph(coauthorship_graph, authors_features_dict, weight_threshold=30):
    author_country_graph = nx.Graph()

    # Loop through each edge in the co-authorship graph
    for author1, author2 in coauthorship_graph.edges:
        # Get the country of each author
        author1_features = authors_features_dict.get(author1)
        author2_features = authors_features_dict.get(author2)

        country1 = None if author1_features is None else author1_features.get('country')
        country2 = None if author2_features is None else author2_features.get('country')

        # If either author has no country information, skip this edge
        if not isinstance(country1, str) or not isinstance(country2, str):
            continue

        # Sort field names to ensure consistent ordering in the undirected graph
        country_pair = tuple(sorted([country1, country2]))

        # Only add edges between different countries
        if country_pair[0] != country_pair[1]:
            # If the edge already exists, increment the weight; otherwise, set it to 1
            if author_country_graph.has_edge(*country_pair):
                author_country_graph[country_pair[0]][country_pair[1]]['weight'] += 1
            else:
                author_country_graph.add_edge(country_pair[0], country_pair[1], weight=1)

    # Remove edges with weight lower than weight_threshold
    edges_to_remove = [(u, v) for u, v, data in author_country_graph.edges(data=True) if data['weight'] < weight_threshold]
    author_country_graph.remove_edges_from(edges_to_remove)

    # Remove nodes that have no edges
    isolated_nodes = [node for node, degree in author_country_graph.degree() if degree == 0]
    author_country_graph.remove_nodes_from(isolated_nodes)

    return author_country_graph

print("\n-------------Construct the Country Subgraph-------------")
country_graph = construct_country_subgraph(G, authors_features, weight_threshold = 80)

# Print out the first 5 pairs of countries that have the highest connection
sorted_country_graph_edges = sorted(country_graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
print("Top 5 pairs of countries that have the highest academic connection:")
for u, v, data in sorted_country_graph_edges:
    print(f"Country1: {u}, Country2: {v}, Weight: {data['weight']}")

# Get the top 5 pairs of countries by collaboration weight
top_5_pairs = sorted_country_graph_edges[:5]

# Extract the country pairs and their weights for plotting
country_pairs = ["US-UK", "US-Germany", "Mexico-Spain", "US-India", "US-China"]
weights = [data['weight'] for _, _, data in top_5_pairs]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.bar(country_pairs, weights, width=0.4, color='skyblue', edgecolor='black')
plt.ylabel("Co-Authorship Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('graphs/country_top_pairs.png', format='png', bbox_inches="tight")
plt.close()

print("\n-------------Plot the Country Subgraph-------------")
country_geo_data = {
    'HU': {'latitude': 47.1625, 'longitude': 19.5033},  # Hungary
    'DK': {'latitude': 56.2639, 'longitude': 9.5018},   # Denmark
    'ES': {'latitude': 40.4637, 'longitude': -3.7492},  # Spain
    'FR': {'latitude': 46.6034, 'longitude': 1.8883},   # France
    'PL': {'latitude': 51.9194, 'longitude': 19.1451},  # Poland
    'ID': {'latitude': -0.7893, 'longitude': 113.9213}, # Indonesia
    'JP': {'latitude': 36.2048, 'longitude': 138.2529}, # Japan
    'NL': {'latitude': 52.1326, 'longitude': 5.2913},   # Netherlands
    'IL': {'latitude': 31.0461, 'longitude': 34.8516},  # Israel
    'US': {'latitude': 37.0902, 'longitude': -95.7129}, # United States
    'RO': {'latitude': 45.9432, 'longitude': 24.9668},  # Romania
    'MX': {'latitude': 23.6345, 'longitude': -102.5528},# Mexico
    'IR': {'latitude': 32.4279, 'longitude': 53.6880},  # Iran
    'TW': {'latitude': 23.6978, 'longitude': 120.9605}, # Taiwan
    'BD': {'latitude': 23.6850, 'longitude': 90.3563},  # Bangladesh
    'BE': {'latitude': 50.5039, 'longitude': 4.4699},   # Belgium
    'AR': {'latitude': -38.4161, 'longitude': -63.6167},# Argentina
    'AU': {'latitude': -25.2744, 'longitude': 133.7751},# Australia
    'VN': {'latitude': 14.0583, 'longitude': 108.2772}, # Vietnam
    'GR': {'latitude': 39.0742, 'longitude': 21.8243},  # Greece
    'NG': {'latitude': 9.0820, 'longitude': 8.6753},    # Nigeria
    'CO': {'latitude': 4.5709, 'longitude': -74.2973},  # Colombia
    'PT': {'latitude': 39.3999, 'longitude': -8.2245},  # Portugal
    'GB': {'latitude': 55.3781, 'longitude': -3.4360},  # United Kingdom
    'IT': {'latitude': 41.8719, 'longitude': 12.5674},  # Italy
    'KR': {'latitude': 35.9078, 'longitude': 127.7669}, # South Korea
    'AT': {'latitude': 47.5162, 'longitude': 14.5501},  # Austria
    'UA': {'latitude': 48.3794, 'longitude': 31.1656},  # Ukraine
    'CN': {'latitude': 35.8617, 'longitude': 104.1954}, # China
    'RU': {'latitude': 61.5240, 'longitude': 105.3188}, # Russia
    'IE': {'latitude': 53.4129, 'longitude': -8.2439},  # Ireland
    'FI': {'latitude': 61.9241, 'longitude': 25.7482},  # Finland
    'DE': {'latitude': 51.1657, 'longitude': 10.4515},  # Germany
    'BR': {'latitude': -14.2350, 'longitude': -51.9253},# Brazil
    'EG': {'latitude': 26.8206, 'longitude': 30.8025},  # Egypt
    'ZA': {'latitude': -30.5595, 'longitude': 22.9375}, # South Africa
    'PK': {'latitude': 30.3753, 'longitude': 69.3451},  # Pakistan
    'CL': {'latitude': -35.6751, 'longitude': -71.5430},# Chile
    'CZ': {'latitude': 49.8175, 'longitude': 15.4730},  # Czech Republic
    'CA': {'latitude': 56.1304, 'longitude': -106.3468},# Canada
    'PH': {'latitude': 12.8797, 'longitude': 121.7740}, # Philippines
    'SE': {'latitude': 60.1282, 'longitude': 18.6435},  # Sweden
    'TR': {'latitude': 38.9637, 'longitude': 35.2433},  # Turkey
    'IN': {'latitude': 20.5937, 'longitude': 78.9629},  # India
}

plt.figure(figsize=(15, 10))
m = Basemap(projection='merc',
            llcrnrlat=-60, urcrnrlat=85,
            llcrnrlon=-180, urcrnrlon=180,
            resolution='l')
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgray', lake_color='lightblue')

# Get positions based on latitude and longitude from country_geo_data
pos = {country: m(data['longitude'], data['latitude']) for country, data in country_geo_data.items()}

# Draw nodes
node_sizes = [200 + 30 * country_graph.degree(n) for n in country_graph.nodes]  # Adjust node size based on degree
nx.draw_networkx_nodes(country_graph, pos, node_size=node_sizes, node_color="skyblue", edgecolors="black")

# Draw edges with width proportional to weight
elarge = [(u, v) for (u, v, d) in country_graph.edges(data=True) if d["weight"] > 300]
esmedium = [(u, v) for (u, v, d) in country_graph.edges(data=True) if 150 < d["weight"] <= 300]
esmall = [(u, v) for (u, v, d) in country_graph.edges(data=True) if d["weight"] <= 150]

nx.draw_networkx_edges(country_graph, pos, edgelist=elarge, width=2.5, alpha=0.6, edge_color='red')
nx.draw_networkx_edges(country_graph, pos, edgelist=esmedium, width=1.5, alpha=0.4, edge_color='green')
nx.draw_networkx_edges(country_graph, pos, edgelist=esmall, width=1, alpha=0.3, edge_color="black")

# Draw node labels
nx.draw_networkx_labels(country_graph, pos, font_size=9, font_family="sans-serif", font_weight="bold", font_color="darkblue")

# Title and display
plt.savefig('graphs/country_geolocation_graph_basemap.png', format='png', bbox_inches="tight")
plt.close()


'''
--------------------------------------------------------------
Analyze Sub-graph of Countries
--------------------------------------------------------------
'''
print("\n-------------Analysis of Country Subgraph-------------")
# Degree Centrality (= degree of a node / total number of nodes)
degree_centrality = nx.degree_centrality(country_graph)
# Betweenness Centrality (it measures how often a node lies on the shortest path between two other nodes)
betweenness_centrality = nx.betweenness_centrality(country_graph, weight='weight')
# Closeness Centrality (it measures how close a node is to all other nodes in a network)
closeness_centrality = nx.closeness_centrality(country_graph)
# Eigenvector Centrality (it measures a node’s influence in a network by considering the centrality of its neighbors)
eigenvector_centrality = nx.eigenvector_centrality(country_graph, weight='weight')

# Sorting the fields by centrality
degree_sorted = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
betweenness_sorted = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
closeness_sorted = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
eigenvector_sorted = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)

print("Countries by Degree Centrality (Top 10):")
print(degree_sorted[:10])

print("\nCountries by Betweenness Centrality (Top 10):")
print(betweenness_sorted[:10])

print("\nCountries by Closeness Centrality (Top 10):")
print(closeness_sorted[:10])

print("\nCountries by Eigenvector Centrality (Top 10):")
print(eigenvector_sorted[:10])

print("\n-------------Plot the Analysis of Country Subgraph-------------")
# Select only the top N countries to make the plot readable
top_n = 10
plt.figure(figsize=(14, 8))

# Extract top countries and centrality scores for each measure
countries = [x[0] for x in degree_sorted[:top_n]]

degree_dict = dict(degree_sorted)
betweenness_dict = dict(betweenness_sorted)
closeness_dict = dict(closeness_sorted)
eigenvector_dict = dict(eigenvector_sorted)

degree_values = [degree_dict[country] for country in countries]
betweenness_values = [betweenness_dict[country] for country in countries]
closeness_values = [closeness_dict[country] for country in countries]
eigenvector_values = [eigenvector_dict[country] for country in countries]

# Generate rankings for each country in each centrality measure
degree_ranks = {country: rank + 1 for rank, (country, _) in enumerate(degree_sorted)}
betweenness_ranks = {country: rank + 1 for rank, (country, _) in enumerate(betweenness_sorted)}
closeness_ranks = {country: rank + 1 for rank, (country, _) in enumerate(closeness_sorted)}
eigenvector_ranks = {country: rank + 1 for rank, (country, _) in enumerate(eigenvector_sorted)}

# Set up the bar width and positions for each centrality type in the plot
bar_width = 0.2
indices = np.arange(len(countries))

# Plot each centrality as a bar plot
bars_degree = plt.bar(indices - bar_width*1.5, degree_values, width=bar_width, label='Degree Centrality', color='skyblue')
bars_closeness = plt.bar(indices - bar_width/2, closeness_values, width=bar_width, label='Closeness Centrality', color='lightgreen')
bars_eigenvector = plt.bar(indices + bar_width/2, eigenvector_values, width=bar_width, label='Weighted Eigenvector Centrality', color='purple')
bars_betweenness = plt.bar(indices + bar_width*1.5, betweenness_values, width=bar_width, label='Weighted Betweenness Centrality', color='salmon')


# Add labels showing the rank on top of each bar
for i, country in enumerate(countries):
    plt.text(bars_degree[i].get_x() + bars_degree[i].get_width() / 2, degree_values[i] + 0.01,
             f'#{degree_ranks[country]}', ha='center', va='bottom', color='skyblue', fontweight='bold', fontsize=8)
    plt.text(bars_closeness[i].get_x() + bars_closeness[i].get_width() / 2, closeness_values[i] + 0.01,
             f'#{closeness_ranks[country]}', ha='center', va='bottom', color='lightgreen', fontweight='bold', fontsize=8)
    plt.text(bars_eigenvector[i].get_x() + bars_eigenvector[i].get_width() / 2, eigenvector_values[i] + 0.01,
             f'#{eigenvector_ranks[country]}', ha='center', va='bottom', color='purple', fontweight='bold', fontsize=8)
    plt.text(bars_betweenness[i].get_x() + bars_betweenness[i].get_width() / 2, betweenness_values[i] + 0.01,
             f'#{betweenness_ranks[country]}', ha='center', va='bottom', color='salmon', fontweight='bold', fontsize=8)

# Add country labels to the x-axis
plt.xticks(indices, countries, rotation=45, ha='right')

# Add labels, legend, and title
plt.ylabel('Centrality Measure Value')
plt.title('Comparison of Centrality Measures for Top Countries')
plt.legend()

# Save the plot
plt.tight_layout()
plt.savefig('graphs/country_graph_measures.png', format='png', bbox_inches='tight')
plt.close()


'''
--------------------------------------------------------------
Community Detection
--------------------------------------------------------------
'''
print("\n-------------Community Detection-------------")
communities = nx.community.louvain_communities(G)

# Initialize lists to store community attributes
community_data = []

# Calculate attributes for each community
for community in communities:
    community_authors = [authors_features[author_id] for author_id in community if author_id in authors_features]

    # Gather metrics for the community
    community_size = len(community_authors)
    mean_citation_count = np.mean([author['citationCount'] for author in community_authors])
    mean_h_index = np.mean([author['hIndex'] for author in community_authors])
    female_ratio = sum(1 for author in community_authors if author['gender'] == 'female') / community_size
    us_ratio = sum(1 for author in community_authors if author['country'] == 'US') / community_size

    # Store all attributes with size-based ranking
    community_data.append({
        "Community Size (Logarithm)": community_size,
        "Mean Citation Count (Logarithm)": mean_citation_count,
        "Mean h Index (Logarithm)": mean_h_index,
        "Female Ratio": female_ratio,
        "US ratio": us_ratio
    })

print(f"number of communities: {len(community_data)}")

# Convert list of dictionaries to a DataFrame
df = pd.DataFrame(community_data)

# Log-transform the columns to handle skewness and scale the data
df_log = df.copy()
df_log["Community Size (Logarithm)"] = np.log1p(df["Community Size (Logarithm)"])
df_log["Mean Citation Count (Logarithm)"] = np.log1p(df["Mean Citation Count (Logarithm)"])
df_log["Mean h Index (Logarithm)"] = np.log1p(df["Mean h Index (Logarithm)"])
# for col in df.columns:
#     df_log[col] = np.log1p(df[col])

# Plot pairwise relationships using Seaborn
sns.pairplot(df_log, kind="reg", diag_kind='hist')
plt.savefig('graphs/community_attributes.png', format='png')
plt.close()


'''
--------------------------------------------------------------
Individual Community Analysis
--------------------------------------------------------------
'''
print("\n-------------Randomly Select a Community for Analysis-------------")
# Randomly select one community
random_community = random.choice(communities)

# Subgraph containing only the nodes and edges of the selected community
community_graph = complete_graph.subgraph(random_community)

# Draw the community graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(community_graph, seed=42)  # Use spring layout for visualization
nx.draw(
    community_graph,
    pos,
    with_labels=False,
    node_size=50,
    node_color="skyblue",
    edge_color="gray",
    alpha=0.7
)

plt.title("Graph of Randomly Selected Community", fontsize=16)
plt.savefig("graphs/random_community_graph.png", format="png")
plt.close()


# Filter authors in the community who have feature data
community_authors = [author_id for author_id in random_community if author_id in authors_features]

# Basic stats of the community
print("\n-------------Analysis of a Randomly Selected Community-------------")
community_size = len(community_authors)
print(f"Size of the community: {community_size}")

# Extract citation counts, h-indices, genders, and countries
citation_counts = [authors_features[author_id]['citationCount'] for author_id in community_authors]
h_indices = [authors_features[author_id]['hIndex'] for author_id in community_authors]
genders = [authors_features[author_id]['gender'] for author_id in community_authors]
countries = [authors_features[author_id]['country'] for author_id in community_authors]
fields_of_interest = [authors_fields.get(author_id, "Unknown") for author_id in community_authors]

# Analyze citation counts
mean_citations = np.mean(citation_counts) if citation_counts else 0
median_citations = np.median(citation_counts) if citation_counts else 0
max_citations = max(citation_counts) if citation_counts else 0
print(f"Mean Citation Count: {mean_citations}")
print(f"Median Citation Count: {median_citations}")
print(f"Max Citation Count: {max_citations}")

# Analyze h-index
mean_h_index = np.mean(h_indices) if h_indices else 0
median_h_index = np.median(h_indices) if h_indices else 0
max_h_index = max(h_indices) if h_indices else 0
print(f"Mean h-index: {mean_h_index}")
print(f"Median h-index: {median_h_index}")
print(f"Max h-index: {max_h_index}")

# Analyze gender distribution
female_count = genders.count('female')
male_count = genders.count('male')
print(f"Number of females: {female_count}")
print(f"Number of males: {male_count}")
print(f"Female ratio: {female_count / community_size if community_size > 0 else 0}")

# Analyze country distribution
country_distribution = pd.Series(countries).value_counts()[:23]
print("Country distribution:")
print(country_distribution)

# Analyze fields of interest
fields_distribution = pd.Series(fields_of_interest).value_counts()[:13]
print("Fields of Interest distribution:")
print(fields_distribution)

# Calculate mean citation count and h-index across all authors
all_citation_counts = [authors_features[author_id]['citationCount'] for author_id in authors_features if authors_features[author_id]['citationCount'] is not None]
all_h_indices = [authors_features[author_id]['hIndex'] for author_id in authors_features if authors_features[author_id]['hIndex'] is not None]

global_mean_citation_count = np.mean(all_citation_counts) if all_citation_counts else 0
global_mean_h_index = np.mean(all_h_indices) if all_h_indices else 0
print(f"Global Mean Citation Count: {global_mean_citation_count}")
print(f"Global Mean h-index: {global_mean_h_index}")

# Visualize citation count distribution in the community with global and community mean lines
plt.figure(figsize=(10, 6))
sns.histplot(citation_counts, bins=20, color='blue')

# Add dotted lines for the global and community means
plt.axvline(global_mean_citation_count, color='red', linestyle='--', linewidth=3, label=f'Global Mean: {global_mean_citation_count:.0f}')
plt.axvline(mean_citations, color='orange', linestyle='--', linewidth=3, label=f'Community Mean: {mean_citations:.0f}')

# Add titles and labels
plt.title("Citation Count Distribution in the Community")
plt.xlabel("Citation Count")
plt.ylabel("Frequency")
plt.legend()

# Save the plot
plt.savefig("graphs/random_community_citation_distribution.png")
plt.close()

# Visualize h-index distribution in the community with global and community mean lines
plt.figure(figsize=(10, 6))
sns.histplot(h_indices, bins=20, color='green')

# Add dotted lines for the global and community means
plt.axvline(global_mean_h_index, color='red', linestyle='--', linewidth=3, label=f'Global Mean: {global_mean_h_index:.0f}')
plt.axvline(mean_h_index, color='orange', linestyle='--', linewidth=3, label=f'Community Mean: {mean_h_index:.0f}')

# Add titles and labels
plt.title("h-index Distribution in the Community")
plt.xlabel("h-index")
plt.ylabel("Frequency")
plt.legend()

# Save the plot
plt.savefig("graphs/random_community_h_index_distribution.png")
plt.close()

# Visualize country distribution
plt.figure(figsize=(10, 6))
country_distribution.plot(kind='bar', color='orange')
plt.title("Country Distribution in the Community")
plt.ylabel("Number of Authors")
plt.savefig("graphs/random_community_country_distribution.png")
plt.close()

# Visualize fields of interest distribution
plt.figure(figsize=(14, 11))
fields_distribution.plot(kind='bar', color='purple')
plt.title("Fields of Interest Distribution in the Community")
plt.ylabel("Number of Authors")
plt.xticks(rotation=15, ha='right')
plt.savefig("graphs/random_community_fields_distribution.png")
plt.close()
