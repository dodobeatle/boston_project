
import matplotlib.pyplot as plt
import pandas as pd
from seaborn import heatmap

def plot_correlation_matrix(preprocessed_data: pd.DataFrame):
    # Calculate Pearson correlation matrix
    correlation_matrix = preprocessed_data.corr(method='pearson')
    
    # Create a figure for the correlation matrix
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Pearson Correlation Matrix')
    plt.tight_layout()
    
    # Log the correlation matrix figure to MLflow
    fig_path = "./images/"+"correlation_matrix.png"
    plt.savefig(fig_path)
    plt.close()
    return fig_path

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Precios reales')
    plt.ylabel('Predicciones')
    plt.title('Valores reales vs Predicciones')
    plt.tight_layout()
    fig_name = "./images/predictions_plot.png"
    plt.savefig(fig_name)
    plt.close()
    return fig_name