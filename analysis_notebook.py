import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import FileLink, display

# Cell 2: Define the Model Training Function
def train_model(input_path):
    data = pd.read_csv(input_path)  # Reads the dataset from the specified path
    features = ['org_count', 'gpe_count', 'person_count', 'article_length', 'sentiment_score']
    X = data[features]  # Feature matrix
    y = data['engagement_metric']  # Target variable for engagement metric (adjust based on your dataset)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the RandomForestRegressor model
    model = RandomForestRegressor()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the target on the test data
    y_pred = model.predict(X_test)

    # Print out the performance metrics (Mean Absolute Error)
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    return model, data  # Return the trained model and the data

# Cell 3: Define the Plotting Function
def plot_visualizations(data):
    # Plot for Entity Frequency
    entity_counts = data[['org_count', 'gpe_count', 'person_count']].sum()
    entity_counts.plot(kind='bar', title='Entity Frequency')  # Create a bar plot
    plt.savefig("outputs/visualizations/entity_frequency.png")  # Save plot as image
    plt.show()  # Display the plot

    # Scatter plot for Article Length vs Engagement Metric
    sns.scatterplot(data=data, x='article_length', y='engagement_metric')
    plt.title('Article Length vs Engagement')  # Add title to the scatter plot
    plt.savefig("outputs/visualizations/scatter_article_length.png")  # Save plot as image
    plt.show()  # Display the plot

    # Heatmap for Correlation Matrix
    correlation = data.corr()  # Calculate the correlation between numeric variables
    sns.heatmap(correlation, annot=True, cmap='coolwarm')  # Create heatmap with annotations
    plt.savefig("outputs/visualizations/heatmap_correlation.png")  # Save plot as image
    plt.show()  # Display the plot

# Cell 4: Create the Directory for Saving Plots
import os
os.makedirs('outputs/visualizations', exist_ok=True)

# Cell 5: Train Model and Plot Visualizations
model, data = train_model('./data/news.csv')  # Use relative path to your CSV file
plot_visualizations(data)  # Generate and save visualizations

# Cell 6: Create Download Links for Output Files (Optional)
# Provide download links for the generated visualizations
display(FileLink(r'outputs/visualizations/entity_frequency.png'))  # Link to entity frequency plot
display(FileLink(r'outputs/visualizations/scatter_article_length.png'))  # Link to scatter plot
display(FileLink(r'outputs/visualizations/heatmap_correlation.png'))  # Link to heatmap
