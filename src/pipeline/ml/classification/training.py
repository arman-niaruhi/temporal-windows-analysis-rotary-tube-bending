
import numpy as np
import torch
import json

from torch.utils.data import DataLoader
from src.pipeline.ml.classification.utils.model import LSTMSequenceClassifier
from src.pipeline.ml.classification.utils.classification_utils import ClassifierPreprocessor
from src.pipeline.ml.classification.utils.dataset_utils import SegmentDataset3DSequenceWithMask
from src.pipeline.preprocessing.loader import DataLoader as DataLoaderETL
from src.pipeline.ml.classification.utils.feature_analysis_utils import dropout_importance, feature_ablation_importance, gradient_importance_sequence,integrated_gradients_importance,occlusion_importance,permutation_importance_sequence, plot_feature_importance, compare_methods


def training_pipeline(model_path_root: str,
        database_path: str,
        annotation_json_path: str,
        experiment_ids_path: str,
        machine_part: str,
        eliminated_columns: list,
        label: str,
        pipeline_config):
    loader = DataLoaderETL(database_path)
    dataframes = loader.load_all_data_from_sqlite()
    classifier_preprocessor = ClassifierPreprocessor(sensors_df=dataframes[machine_part],
                                                    annotation_json=annotation_json_path)

    with open(experiment_ids_path, "r") as f:
        experiment_groups = json.load(f)

    sensors_df, annotation_dict = classifier_preprocessor.read_data()
    sensors_df = classifier_preprocessor.delete_columns(eliminated_columns=eliminated_columns)
    if label == "All":
        sensors_df = classifier_preprocessor.assign_labels()
    else:
        sensors_df = classifier_preprocessor.assign_one_label(target_label=label) # Clamping, De-Clamping, Mandrel Extraction
    sensors_df = classifier_preprocessor.normalize_and_encode_labels()


    train_df, val_df, test_df, experiment_ids = classifier_preprocessor.split_experiments(
        experiment_groups=experiment_groups
    )

    # Use SegmentDataset3DSequenceWithMask for per-timestep prediction
    train_dataset, val_dataset, test_dataset = classifier_preprocessor.create_datasets(
        train_df, val_df, test_df, SegmentDataset3DSequenceWithMask
    )

    batch_size = pipeline_config.get("dataloader_config").get("batch_size", 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    idx_to_label = {v: k for k, v in train_dataset.label_to_idx.items()}



    feature_cols = classifier_preprocessor.get_feature_cols()
    input_size = len(feature_cols)
    hidden_size = pipeline_config.get("model_config").get("hidden_size", 64)
    num_layers = pipeline_config.get("model_config").get("num_layers", 2)
    num_classes = len(train_dataset.unique_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSequenceClassifier(input_size, hidden_size, num_layers, num_classes, bidirectional=True).to(device)

    idx_to_label = {v: k for k, v in train_dataset.label_to_idx.items()}

    model_path = f"{model_path_root}/{machine_part}/{label}"
    training_flag = pipeline_config.get("training_config").get("training", False)
    if training_flag:
        model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=pipeline_config.get("training_config").get("num_epochs", 300),
            learning_rate=pipeline_config.get("training_config").get("learning_rate", 1e-5),
            patience=pipeline_config.get("training_config").get("patience", 3),
            device=device,
            idx_to_label=idx_to_label,
            model_path=model_path,
            run_name=label,
            experiment_name=machine_part
        )
    else:
        state_dict = torch.load(
            f"{model_path}/Activity_Detector.pth",
            map_location=device
        )
        model.load_state_dict(state_dict)
        model.eval()
    return model, sensors_df, test_loader, device, feature_cols
    


# Assuming model, test_loader, device, and idx_to_label are already defined

def get_predictions_and_data(model, data_loader, device):
    """
    Extract predictions and data from the test loader.
    Returns only valid (non-padded) timesteps.
    """
    model.eval()
    
    all_features = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for X_batch, y_batch, mask_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Get predictions
            outputs = model(X_batch)
            predictions = outputs.argmax(dim=-1)
            
            # Extract only valid timesteps (where mask == 1)
            for i in range(X_batch.shape[0]):
                valid_mask = mask_batch[i] == 1
                valid_indices = torch.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    all_features.append(X_batch[i, valid_indices, :].cpu().numpy())
                    all_labels.append(y_batch[i, valid_indices].cpu().numpy())
                    all_predictions.append(predictions[i, valid_indices].cpu().numpy())
    
    # Concatenate all samples
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)
    
    return X, y, y_pred


def analyze_features(model, sensors_df, test_loader, device):
    to_remove = ["Experiment_ID", "Label", "Label_encoded"]
    column_names = [col for col in sensors_df.columns if col not in to_remove]

    # Get feature names from the dataset
    feature_names = column_names

    # Dictionary to store all importance scores
    all_importances = {}
    perm_importances, perm_std = permutation_importance_sequence(
        model, test_loader, device, n_repeats=10
    )
    all_importances['Permutation'] = perm_importances
    plot_feature_importance(perm_importances, perm_std, feature_names, "Permutation")

    grad_importances = gradient_importance_sequence(
        model, test_loader, device, n_batches=10
    )
    all_importances['Gradient'] = grad_importances
    plot_feature_importance(grad_importances, None, feature_names, "Gradient")

    intgrad_importances = integrated_gradients_importance(
        model, test_loader, device, n_samples=50, steps=30
    )
    all_importances['Integrated Gradients'] = intgrad_importances
    plot_feature_importance(intgrad_importances, None, feature_names, "IntegratedGradients")

    occlusion_importances = occlusion_importance(
        model, test_loader, device, occlusion_value=0.0
    )
    all_importances['Occlusion'] = occlusion_importances
    plot_feature_importance(occlusion_importances, None, feature_names, "Occlusion")

    ablation_importances = feature_ablation_importance(
        model, test_loader, device
    )
    all_importances['Ablation'] = ablation_importances
    plot_feature_importance(ablation_importances, None, feature_names, "Ablation")

    dropout_importances = dropout_importance(
        model, test_loader, device, n_repeats=20, dropout_rate=0.5
    )
    all_importances['Dropout'] = dropout_importances
    plot_feature_importance(dropout_importances, None, feature_names, "Dropout")

    compare_methods(all_importances, feature_names)

    # Save all results
    np.savez(
        "feature_importance_all_methods.npz",
        permutation_importance=perm_importances,
        permutation_std=perm_std,
        gradient_importance=grad_importances,
        integrated_gradients_importance=intgrad_importances,
        occlusion_importance=occlusion_importances,
        ablation_importance=ablation_importances,
        dropout_importance=dropout_importances,
        feature_names=feature_names
    )

    # Calculate average rank for each feature
    feature_ranks = np.zeros((len(feature_names), len(all_importances)))
    for i, (method_name, importances) in enumerate(all_importances.items()):
        # Rank features (higher importance = lower rank number)
        ranks = len(importances) - np.argsort(np.argsort(importances))
        feature_ranks[:, i] = ranks

