import matplotlib.pyplot as plt

def plot_true_vs_pred_random_forest(y_true, y_pred_flat, y_pred_agg):
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label='True', marker='o')
    plt.plot(y_pred_flat, label='RF Flattened', marker='x')
    plt.plot(y_pred_agg, label='RF Aggregated', marker='s')
    plt.title("True vs Predicted Comparison")
    plt.xlabel("Sample Index")
    plt.ylabel("Springback")
    plt.legend()
    plt.show()

def plot_feature_importance_random_forest(feat_imp_df, title="Feature Importance", top_n=20):
    plt.figure(figsize=(12,6))
    plt.barh(feat_imp_df['feature'][:top_n][::-1], feat_imp_df['importance'][:top_n][::-1])
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.show()