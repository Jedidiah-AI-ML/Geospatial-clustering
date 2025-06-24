## Geospatial Clustering of Delivery Locations
This project performs geospatial analysis and clustering of delivery locations across India. Using real-world distance calculations and interactive visualizations, it helps identify key delivery zones and optimize logistics strategies.

## Features
- Geodesic Distance Calculation between pickup and delivery points

- Interactive Map of delivery locations using Plotly

- K-Means Clustering to segment delivery locations into geographic zones

- Cluster Visualization with centroids for strategic insights

- Optimized Zoning for logistics simplification

## Tech Stack
Python

Pandas, NumPy

scikit-learn

Plotly

Geopy

## Setup
Install dependencies:

bash
Copy
Edit
pip install pandas numpy scikit-learn geopy plotly
Load dataset (update path if needed):

python
Copy
Edit
data = pd.read_csv("path_to_your_data/train.csv")
Run the script to visualize and cluster delivery locations.

## Sample Output
Interactive map of delivery points

Clustered zones with centroids

Labeled delivery zones like "Central Delivery Zone" and "Southern Delivery Zone"
