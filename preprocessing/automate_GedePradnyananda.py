import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os
import mlflow

def preprocess_diabetes_dataset(input_path: str, output_dir: str, target_column: str = 'Diabetes_binary') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess diabetes dataset with SMOTE resampling and train-test split.
    Saves intermediate and final datasets to CSV files.
    """
    # Load dataset
    df = pd.read_csv(input_path)

    # Pilih kolom yang diperlukan
    selected_columns = ['Age', 'Sex', 'GenHlth', 'MentHlth', 'PhysHlth',
                        'BMI', 'HvyAlcoholConsump', 'HighChol', 'PhysActivity', target_column]
    df = df[selected_columns]

    # Pisahkan fitur dan target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Resample dengan SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Buat DataFrame hasil resample
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled

    # Buat folder output
    os.makedirs(output_dir, exist_ok=True)

    # Simpan dataset setelah SMOTE
    cleaned_path = os.path.join(output_dir, "dataset_smote.csv")
    df_resampled.to_csv(cleaned_path, index=False)

    # Split train/test
    train_df, test_df = train_test_split(df_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    train_path = os.path.join(output_dir, "train_data.csv")
    test_path = os.path.join(output_dir, "test_data.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_df, test_df, cleaned_path, train_path, test_path

if __name__ == "__main__":
    input_file = "preprocessing/diabet_preprocessing.csv"
    output_dir = "preprocessing"

    mlflow.set_tracking_uri("file:./mlruns")

    with mlflow.start_run(run_name="Preprocessing_Diabetes_SMOTE"):
        train_df, test_df, cleaned_path, train_path, test_path = preprocess_diabetes_dataset(input_file, output_dir)

        mlflow.log_param("input_file", input_file)
        mlflow.log_param("output_dir", output_dir)
        mlflow.log_metric("rows_train", train_df.shape[0])
        mlflow.log_metric("rows_test", test_df.shape[0])

        mlflow.log_artifact(cleaned_path)
        mlflow.log_artifact(train_path)
        mlflow.log_artifact(test_path)
