import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(waiting_times_file, weather_file):
    """
    Charge les deux fichiers CSV et les merge sur DATETIME
    """
    # Chargement des données principales
    df_main = pd.read_csv(waiting_times_file)
    df_weather = pd.read_csv(weather_file)
    
    print("=== DONNÉES CHARGÉES ===")
    print(f"Données principales: {df_main.shape}")
    print(f"Données météo: {df_weather.shape}")
    
    # Conversion des datetime
    df_main['DATETIME'] = pd.to_datetime(df_main['DATETIME'])
    df_weather['DATETIME'] = pd.to_datetime(df_weather['DATETIME'])
    
    # Fusion des données
    # On va chercher la météo la plus proche pour chaque observation
    df_main['datetime_rounded'] = df_main['DATETIME'].dt.round('H')  # Arrondir à l'heure
    df_weather['datetime_rounded'] = df_weather['DATETIME'].dt.round('H')
    
    # Merge sur l'heure arrondie
    df_merged = df_main.merge(df_weather, on='datetime_rounded', how='left', suffixes=('', '_weather'))
    
    print(f"Données après fusion: {df_merged.shape}")
    print(f"Valeurs manquantes météo après fusion: {df_merged['temp'].isnull().sum()}")
    
    return df_merged

def explore_data(df):
    """
    Exploration complète des données
    """
    print("\n=== EXPLORATION DES DONNÉES ===")
    print(f"Forme: {df.shape}")
    
    # Variable cible
    target = 'WAIT_TIME_IN_2H'
    print(f"\nVariable cible ({target}):")
    print(f"- Min: {df[target].min()}")
    print(f"- Max: {df[target].max()}")
    print(f"- Moyenne: {df[target].mean():.2f}")
    print(f"- Médiane: {df[target].median():.2f}")
    print(f"- Valeurs manquantes: {df[target].isnull().sum()}")
    
    # Attractions
    print(f"\nAttractions uniques: {df['ENTITY_DESCRIPTION_SHORT'].unique()}")
    print(df['ENTITY_DESCRIPTION_SHORT'].value_counts())
    
    # Valeurs manquantes par colonne
    print("\nValeurs manquantes par colonne:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    for col, count in missing.items():
        percentage = (count / len(df)) * 100
        print(f"- {col}: {count} ({percentage:.1f}%)")
    
    return df

def create_temporal_features(df):
    """
    Crée les features temporelles à partir de DATETIME
    """
    df = df.copy()
    
    # Features de base
    df['year'] = df['DATETIME'].dt.year
    df['month'] = df['DATETIME'].dt.month
    df['day'] = df['DATETIME'].dt.day
    df['hour'] = df['DATETIME'].dt.hour
    df['minute'] = df['DATETIME'].dt.minute
    df['dayofweek'] = df['DATETIME'].dt.dayofweek  # 0=Lundi
    df['dayofyear'] = df['DATETIME'].dt.dayofyear
    df['week'] = df['DATETIME'].dt.isocalendar().week
    
    # Features dérivées
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['quarter'] = df['DATETIME'].dt.quarter
    
    # Saisons
    season_map = {12: 'winter', 1: 'winter', 2: 'winter',
                  3: 'spring', 4: 'spring', 5: 'spring',
                  6: 'summer', 7: 'summer', 8: 'summer',
                  9: 'autumn', 10: 'autumn', 11: 'autumn'}
    df['season'] = df['month'].map(season_map)
    
    # Périodes de la journée
    def get_time_period(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    df['time_period'] = df['hour'].apply(get_time_period)
    
    # Heures de pointe parc (approximatif)
    df['is_peak_hour'] = ((df['hour'] >= 10) & (df['hour'] <= 16)).astype(int)
    
    # Features cycliques pour capturer la nature cyclique du temps
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    print("✓ Features temporelles créées")
    return df

def create_weather_features(df):
    """
    Crée les features météorologiques
    """
    df = df.copy()
    
    # Features météo de base déjà présentes
    weather_cols = ['temp', 'dew_point', 'feels_like', 'pressure', 'humidity', 
                   'wind_speed', 'rain_1h', 'snow_1h', 'clouds_all']
    
    # Gestion des valeurs manquantes météo
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Features dérivées météo
    if all(col in df.columns for col in ['temp', 'humidity']):
        # Index de confort (approximation simple)
        df['comfort_index'] = df['temp'] * (1 - df['humidity'] / 100)
    
    # Catégorisation météo
    if 'temp' in df.columns:
        df['temp_category'] = pd.cut(df['temp'], 
                                   bins=[-np.inf, 10, 20, 25, np.inf],
                                   labels=['cold', 'cool', 'pleasant', 'warm'])
    
    if 'rain_1h' in df.columns:
        df['is_raining'] = (df['rain_1h'] > 0).astype(int)
        df['rain_category'] = pd.cut(df['rain_1h'].fillna(0), 
                                   bins=[-0.1, 0, 0.5, 2, np.inf],
                                   labels=['no_rain', 'light', 'moderate', 'heavy'])
    
    if 'snow_1h' in df.columns:
        df['is_snowing'] = (df['snow_1h'] > 0).astype(int)
    
    if 'clouds_all' in df.columns:
        df['cloud_category'] = pd.cut(df['clouds_all'], 
                                    bins=[-1, 25, 50, 75, 101],
                                    labels=['clear', 'partly_cloudy', 'mostly_cloudy', 'overcast'])
    
    # Météo "bonne" pour un parc d'attraction
    conditions = []
    if 'temp' in df.columns:
        conditions.append((df['temp'] >= 15) & (df['temp'] <= 28))
    if 'rain_1h' in df.columns:
        conditions.append(df['rain_1h'].fillna(0) <= 0.5)
    if 'wind_speed' in df.columns:
        conditions.append(df['wind_speed'] <= 10)
    
    if conditions:
        df['good_weather'] = np.all(conditions, axis=0).astype(int)
    
    print("✓ Features météo créées")
    return df

def create_attraction_features(df):
    """
    Crée les features liées aux attractions
    """
    df = df.copy()
    
    # Encoding de l'attraction
    le_attraction = LabelEncoder()
    df['attraction_encoded'] = le_attraction.fit_transform(df['ENTITY_DESCRIPTION_SHORT'])
    
    # Popularité de l'attraction (fréquence d'apparition)
    attraction_counts = df['ENTITY_DESCRIPTION_SHORT'].value_counts()
    df['attraction_popularity'] = df['ENTITY_DESCRIPTION_SHORT'].map(attraction_counts)
    
    # Normalisation de la capacité
    if 'ADJUST_CAPACITY' in df.columns:
        scaler = StandardScaler()
        df['capacity_normalized'] = scaler.fit_transform(df[['ADJUST_CAPACITY']])
        
        # Catégorisation de la capacité
        df['capacity_category'] = pd.qcut(df['ADJUST_CAPACITY'], 
                                        q=3, labels=['low', 'medium', 'high'])
    
    # Features sur le downtime
    if 'DOWNTIME' in df.columns:
        df['has_downtime'] = (df['DOWNTIME'] > 0).astype(int)
        df['downtime_category'] = pd.cut(df['DOWNTIME'], 
                                       bins=[-1, 0, 30, 60, np.inf],
                                       labels=['no_downtime', 'short', 'medium', 'long'])
    
    print("✓ Features attractions créées")
    return df

def create_wait_time_features(df):
    """
    Crée les features basées sur les temps d'attente et événements
    """
    df = df.copy()
    
    # Gestion des valeurs manquantes pour les temps vers événements
    event_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']
    
    for col in event_cols:
        if col in df.columns:
            # Remplacer null par une valeur élevée (pas d'événement proche)
            df[col] = df[col].fillna(999)
    
    # Features dérivées
    if all(col in df.columns for col in event_cols):
        df['min_time_to_event'] = df[event_cols].min(axis=1)
        df['avg_time_to_event'] = df[event_cols].mean(axis=1)
        
        # Proximité d'événements
        df['near_parade_1'] = (df['TIME_TO_PARADE_1'] <= 60).astype(int)  # dans l'heure
        df['near_parade_2'] = (df['TIME_TO_PARADE_2'] <= 60).astype(int)
        df['near_night_show'] = (df['TIME_TO_NIGHT_SHOW'] <= 60).astype(int)
        df['near_any_event'] = ((df['TIME_TO_PARADE_1'] <= 60) | 
                               (df['TIME_TO_PARADE_2'] <= 60) | 
                               (df['TIME_TO_NIGHT_SHOW'] <= 60)).astype(int)
        
        # Très proche d'un événement
        df['very_near_event'] = (df['min_time_to_event'] <= 30).astype(int)
    
    # Features sur le temps d'attente actuel
    if 'CURRENT_WAIT_TIME' in df.columns:
        df['current_wait_category'] = pd.cut(df['CURRENT_WAIT_TIME'], 
                                           bins=[-1, 0, 15, 30, 60, np.inf],
                                           labels=['no_wait', 'short', 'medium', 'long', 'very_long'])
        
        # Ratio temps actuel vs capacité
        if 'ADJUST_CAPACITY' in df.columns:
            df['wait_capacity_ratio'] = df['CURRENT_WAIT_TIME'] / (df['ADJUST_CAPACITY'] + 1)
    
    print("✓ Features temps d'attente créées")
    return df

def create_lag_features(df):
    """
    Crée des features de lag (valeurs précédentes) pour chaque attraction
    """
    df = df.copy()
    df = df.sort_values(['ENTITY_DESCRIPTION_SHORT', 'DATETIME'])
    
    # Lags pour chaque attraction séparément
    lag_features = ['CURRENT_WAIT_TIME', 'ADJUST_CAPACITY']
    
    for feature in lag_features:
        if feature in df.columns:
            for lag in [1, 2, 3]:  # 1, 2, 3 observations précédentes
                df[f'{feature}_lag_{lag}'] = df.groupby('ENTITY_DESCRIPTION_SHORT')[feature].shift(lag)
    
    # Moyennes mobiles
    for feature in lag_features:
        if feature in df.columns:
            df[f'{feature}_rolling_mean_3'] = df.groupby('ENTITY_DESCRIPTION_SHORT')[feature].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
            df[f'{feature}_rolling_std_3'] = df.groupby('ENTITY_DESCRIPTION_SHORT')[feature].rolling(window=3, min_periods=1).std().reset_index(0, drop=True)
    
    print("✓ Features de lag créées")
    return df

def create_interaction_features(df):
    """
    Crée des features d'interaction
    """
    df = df.copy()
    
    # Interactions importantes
    if 'good_weather' in df.columns and 'is_weekend' in df.columns:
        df['good_weather_weekend'] = df['good_weather'] * df['is_weekend']
    
    if 'is_peak_hour' in df.columns and 'capacity_normalized' in df.columns:
        df['peak_capacity_interaction'] = df['is_peak_hour'] * df['capacity_normalized']
    
    if 'temp' in df.columns and 'is_weekend' in df.columns:
        df['temp_weekend_interaction'] = df['temp'] * df['is_weekend']
    
    if 'near_any_event' in df.columns and 'CURRENT_WAIT_TIME' in df.columns:
        df['event_current_wait_interaction'] = df['near_any_event'] * df['CURRENT_WAIT_TIME']
    
    print("✓ Features d'interaction créées")
    return df

def encode_categorical_features(df):
    """
    Encode les features catégorielles
    """
    df = df.copy()
    
    # One-hot encoding pour les variables catégorielles
    categorical_cols = ['season', 'time_period', 'temp_category', 'rain_category', 
                       'cloud_category', 'capacity_category', 'downtime_category', 
                       'current_wait_category']
    
    # Filtrer les colonnes qui existent réellement
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    if categorical_cols:
        df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        print(f"✓ One-hot encoding appliqué à {len(categorical_cols)} colonnes")
        return df_encoded
    
    return df

def handle_missing_values(df, target_col='WAIT_TIME_IN_2H'):
    """
    Gère les valeurs manquantes finales
    """
    df = df.copy()
    
    # Supprimer les lignes où la target est manquante
    initial_rows = len(df)
    df = df.dropna(subset=[target_col])
    dropped_rows = initial_rows - len(df)
    print(f"✓ Supprimé {dropped_rows} lignes avec target manquante")
    
    # Pour les autres colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(target_col)
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Pour les colonnes catégorielles restantes
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)
    
    print(f"✓ Valeurs manquantes traitées")
    return df

def full_preprocessing_pipeline(waiting_times_file, weather_file):
    """
    Pipeline complet de preprocessing
    """
    print("=== DÉBUT DU PREPROCESSING ===")
    
    # 1. Chargement et fusion
    df = load_data(waiting_times_file, weather_file)
    
    # 2. Exploration
    df = explore_data(df)
    
    # 3. Features temporelles
    df = create_temporal_features(df)
    
    # 4. Features météo
    df = create_weather_features(df)
    
    # 5. Features attractions
    df = create_attraction_features(df)
    
    # 6. Features temps d'attente
    df = create_wait_time_features(df)
    
    # 7. Features de lag
    df = create_lag_features(df)
    
    # 8. Features d'interaction
    df = create_interaction_features(df)
    
    # 9. Encoding catégoriel
    df = encode_categorical_features(df)
    
    # 10. Gestion valeurs manquantes finales
    df = handle_missing_values(df)
    
    print(f"\n=== PREPROCESSING TERMINÉ ===")
    print(f"Forme finale: {df.shape}")
    print(f"Features créées: {df.shape[1]} colonnes")
    
    # Colonnes finales
    print("\nTypes de features créées:")
    feature_types = {
        'temporal': [col for col in df.columns if any(x in col.lower() for x in ['year', 'month', 'day', 'hour', 'week', 'season', 'time', 'sin', 'cos'])],
        'weather': [col for col in df.columns if any(x in col.lower() for x in ['temp', 'rain', 'snow', 'wind', 'cloud', 'pressure', 'humidity', 'weather'])],
        'attraction': [col for col in df.columns if any(x in col.lower() for x in ['attraction', 'capacity', 'downtime'])],
        'wait_time': [col for col in df.columns if any(x in col.lower() for x in ['wait', 'parade', 'show', 'event'])],
        'lag': [col for col in df.columns if 'lag' in col.lower() or 'rolling' in col.lower()],
        'interaction': [col for col in df.columns if 'interaction' in col.lower()],
        'original': ['DATETIME', 'ENTITY_DESCRIPTION_SHORT', 'WAIT_TIME_IN_2H']
    }
    
    for feature_type, cols in feature_types.items():
        print(f"- {feature_type}: {len(cols)} features")
    
    return df

def prepare_for_modeling(df, target_col='WAIT_TIME_IN_2H', test_size=0.2):
    """
    Prépare les données pour la modélisation
    """
    # Colonnes à exclure des features
    exclude_cols = ['DATETIME', 'ENTITY_DESCRIPTION_SHORT', target_col, 'datetime_rounded', 'DATETIME_weather']
    
    # Features finales
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"\n=== PRÉPARATION POUR MODÉLISATION ===")
    print(f"Nombre de features: {X.shape[1]}")
    print(f"Nombre d'observations: {X.shape[0]}")
    print(f"Target stats - Min: {y.min()}, Max: {y.max()}, Mean: {y.mean():.2f}")
    
    # Split temporel (respecter l'ordre chronologique)
    split_date = df['DATETIME'].quantile(1 - test_size)
    train_mask = df['DATETIME'] < split_date
    
    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]
    
    print(f"Train set: {X_train.shape[0]} observations")
    print(f"Test set: {X_test.shape[0]} observations")
    print(f"Split date: {split_date}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def save_processed_data(df, filename='processed_data.csv'):
    """
    Sauvegarde les données préprocessées
    """
    df.to_csv(filename, index=False)
    print(f"✓ Données sauvegardées dans {filename}")

# Exemple d'utilisation complète
if __name__ == "__main__":
    # Lancement du preprocessing complet
    df_processed = full_preprocessing_pipeline('/Users/pierrebressollette/Desktop/Hackathon perso/GitHubFolder/waiting_times_train.csv', '/Users/pierrebressollette/Desktop/Hackathon perso/GitHubFolder/weather_data.csv')
    
    # Préparation pour la modélisation
    X_train, X_test, y_train, y_test, features = prepare_for_modeling(df_processed)
    
    # Sauvegarde optionnelle
    # save_processed_data(df_processed)
    
    print("\n=== DONNÉES PRÊTES POUR LA MODÉLISATION ===")
    print("Variables disponibles:")
    print("- df_processed: DataFrame complet avec toutes les features")
    print("- X_train, X_test: Features d'entraînement et de test")
    print("- y_train, y_test: Target d'entraînement et de test")
    print("- features: Liste des noms de features")
    
    print(f"\nPremières features: {features[:10]}")
    print("...")
    print(f"Dernières features: {features[-10:]}")

# Fonctions utilitaires pour l'analyse
def analyze_feature_importance_preparation(df, target_col='WAIT_TIME_IN_2H'):
    """
    Prépare une analyse rapide des corrélations pour guider la sélection de features
    """
    # Sélection des colonnes numériques seulement
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_df = df[numeric_cols]
    
    # Corrélations avec la target
    correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
    
    print("\n=== TOP 20 FEATURES PAR CORRÉLATION ===")
    print(correlations.head(20))
    
    return correlations

def get_feature_summary(df):
    """
    Résumé des features créées
    """
    print("\n=== RÉSUMÉ DES FEATURES ===")
    print(f"Total features: {df.shape[1]}")
    
    # Compte par type
    feature_counts = {
        'Temporal': len([col for col in df.columns if any(x in col.lower() for x in ['year', 'month', 'day', 'hour', 'week', 'season', 'time', 'sin', 'cos'])]),
        'Weather': len([col for col in df.columns if any(x in col.lower() for x in ['temp', 'rain', 'snow', 'wind', 'cloud', 'pressure', 'humidity', 'weather'])]),
        'Attraction': len([col for col in df.columns if any(x in col.lower() for x in ['attraction', 'capacity', 'downtime'])]),
        'Wait/Events': len([col for col in df.columns if any(x in col.lower() for x in ['wait', 'parade', 'show', 'event'])]),
        'Lag': len([col for col in df.columns if 'lag' in col.lower() or 'rolling' in col.lower()]),
        'Interaction': len([col for col in df.columns if 'interaction' in col.lower()]),
        'Encoded': len([col for col in df.columns if any(x in col for x in ['_season_', '_time_period_', '_category_'])])
    }
    
    for feature_type, count in feature_counts.items():
        print(f"- {feature_type}: {count}")
    
    return feature_counts