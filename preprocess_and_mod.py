import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Essayer d'importer XGBoost, sinon utiliser GradientBoosting
try:
    from xgboost import XGBRegressor
    USE_XGBOOST = True
    print("✅ XGBoost disponible")
except ImportError:
    USE_XGBOOST = False
    print("⚠️ XGBoost non disponible, utilisation de GradientBoosting à la place")

# Configuration pour optimiser les performances
REDUCE_MEMORY = True  # Réduire l'utilisation mémoire
SAMPLE_SIZE = None    # Mettre un nombre (ex: 10000) pour échantillonner les données d'entraînement
USE_SIMPLE_MODEL = False  # True pour utiliser un seul modèle (très rapide)

print("\n🚀 Pipeline Optimisée pour Mac")
print("="*60)

# 1. CHARGEMENT DES DONNÉES
print("\n1. Chargement des données...")
train_df = pd.read_csv('/Users/pierrebressollette/Desktop/Hackathon perso/GitHubFolder/waiting_times_train.csv')
test_df = pd.read_csv('/Users/pierrebressollette/Desktop/Hackathon perso/GitHubFolder/waiting_times_X_test_val.csv')
weather_df = pd.read_csv('/Users/pierrebressollette/Desktop/Hackathon perso/GitHubFolder/weather_data.csv')

print(f"   - Train: {train_df.shape}")
print(f"   - Test: {test_df.shape}")
print(f"   - Weather: {weather_df.shape}")

# Optimisation mémoire
if REDUCE_MEMORY:
    print("\n   Optimisation de la mémoire...")
    # Convertir les float64 en float32
    float_cols = train_df.select_dtypes(include=['float64']).columns
    train_df[float_cols] = train_df[float_cols].astype('float32')
    test_df[float_cols.intersection(test_df.columns)] = test_df[float_cols.intersection(test_df.columns)].astype('float32')
    weather_df[weather_df.select_dtypes(include=['float64']).columns] = weather_df[weather_df.select_dtypes(include=['float64']).columns].astype('float32')

# Échantillonnage si demandé (pour tests rapides)
if SAMPLE_SIZE and len(train_df) > SAMPLE_SIZE:
    print(f"\n   Échantillonnage à {SAMPLE_SIZE} lignes pour accélérer...")
    train_df = train_df.sample(n=SAMPLE_SIZE, random_state=42)

# 2. PRÉPARATION DES DONNÉES
print("\n2. Préparation des données...")

# Convertir les dates
train_df['DATETIME'] = pd.to_datetime(train_df['DATETIME'])
test_df['DATETIME'] = pd.to_datetime(test_df['DATETIME'])
weather_df['DATETIME'] = pd.to_datetime(weather_df['DATETIME'])

# Fusionner avec les données météo (version optimisée)
print("   - Fusion avec les données météo...")

# Réduire weather_df aux colonnes essentielles seulement
essential_weather_cols = ['DATETIME', 'temp', 'humidity', 'rain_1h', 'clouds_all']
weather_df_reduced = weather_df[essential_weather_cols].copy()

# Arrondir les dates à l'heure pour faciliter la fusion
train_df['DATETIME_HOUR'] = train_df['DATETIME'].dt.floor('H')
test_df['DATETIME_HOUR'] = test_df['DATETIME'].dt.floor('H')
weather_df_reduced['DATETIME_HOUR'] = weather_df_reduced['DATETIME'].dt.floor('H')

# Moyenner les données météo par heure (plus rapide)
weather_hourly = weather_df_reduced.groupby('DATETIME_HOUR').mean().reset_index()

# Fusion simple et rapide
train_df = train_df.merge(weather_hourly, on='DATETIME_HOUR', how='left')
test_df = test_df.merge(weather_hourly, on='DATETIME_HOUR', how='left')

# Garder la colonne DATETIME originale pour les features et le résultat final
# (on avait juste besoin de DATETIME_HOUR pour la fusion)
# Ne pas supprimer DATETIME_HOUR car on en a besoin pour create_features_light

# 3. FEATURE ENGINEERING LÉGER
print("\n3. Feature Engineering (version optimisée)...")

def create_features_light(df):
    """Version allégée du feature engineering - seulement les features essentielles"""
    df = df.copy()
    
    # Features temporelles de base
    # Utiliser DATETIME si elle existe, sinon DATETIME_HOUR
    datetime_col = 'DATETIME' if 'DATETIME' in df.columns else 'DATETIME_HOUR'
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['month'] = df[datetime_col].dt.month
    
    # Indicateurs simples mais efficaces
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    df['is_peak_hour'] = df['hour'].isin([11, 12, 13, 14, 15, 16, 17]).astype('int8')
    
    # Features cycliques (seulement pour l'heure - le plus important)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype('float32')
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype('float32')
    
    # Une seule interaction clé
    df['capacity_x_wait'] = (df['ADJUST_CAPACITY'] * df['CURRENT_WAIT_TIME']).astype('float32')
    
    # Feature météo simple si disponible
    if 'rain_1h' in df.columns:
        df['is_raining'] = (df['rain_1h'] > 0).astype('int8')
    
    return df

# Appliquer le feature engineering
train_df = create_features_light(train_df)
test_df = create_features_light(test_df)

# Statistiques par attraction (version simplifiée)
print("   - Calcul des statistiques par attraction...")
attraction_stats = train_df.groupby('ENTITY_DESCRIPTION_SHORT')['WAIT_TIME_IN_2H'].agg(['mean', 'median']).round(1)
attraction_stats.columns = ['wait_2h_mean', 'wait_2h_median']

train_df = train_df.merge(attraction_stats, on='ENTITY_DESCRIPTION_SHORT', how='left')
test_df = test_df.merge(attraction_stats, on='ENTITY_DESCRIPTION_SHORT', how='left')

# 4. PRÉPARATION DES FEATURES FINALES
print("\n4. Sélection des features...")

# Encoder l'attraction
le = LabelEncoder()
train_df['entity_encoded'] = le.fit_transform(train_df['ENTITY_DESCRIPTION_SHORT']).astype('int16')
test_df['entity_encoded'] = le.transform(test_df['ENTITY_DESCRIPTION_SHORT']).astype('int16')

# Features essentielles seulement
feature_cols = [
    'ADJUST_CAPACITY', 'CURRENT_WAIT_TIME', 'DOWNTIME',
    'entity_encoded',
    'hour', 'day_of_week', 'month',
    'is_weekend', 'is_peak_hour',
    'hour_sin', 'hour_cos',
    'capacity_x_wait',
    'wait_2h_mean', 'wait_2h_median'
]

# Ajouter les features de parade si disponibles
parade_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']
for col in parade_cols:
    if col in train_df.columns:
        # Remplacer les NaN par -1 (indicateur d'absence)
        train_df[col] = train_df[col].fillna(-1).astype('float32')
        test_df[col] = test_df[col].fillna(-1).astype('float32')
        feature_cols.append(col)

# Ajouter les features météo si disponibles
weather_cols = ['temp', 'humidity', 'rain_1h', 'clouds_all', 'is_raining']
for col in weather_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(train_df[col].median()).astype('float32')
        test_df[col] = test_df[col].fillna(train_df[col].median()).astype('float32')
        feature_cols.append(col)

# Préparer X et y
X_train = train_df[feature_cols].fillna(0)
y_train = train_df['WAIT_TIME_IN_2H']
X_test = test_df[feature_cols].fillna(0)

print(f"   Nombre de features: {len(feature_cols)}")
print(f"   Shape X_train: {X_train.shape}")
print(f"   Shape X_test: {X_test.shape}")

# 5. ENTRAÎNEMENT DU MODÈLE
print("\n5. Entraînement du modèle...")

if USE_SIMPLE_MODEL:
    # Version ultra-rapide : un seul modèle
    print("   Mode rapide : Un seul modèle")
    
    if USE_XGBOOST:
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    
    # Validation rapide
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    model.fit(X_tr, y_tr)
    
    # Évaluation
    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"   RMSE validation: {rmse:.2f}")
    
    # Réentraînement sur toutes les données
    print("   Réentraînement sur toutes les données...")
    model.fit(X_train, y_train)
    
    # Prédictions finales
    y_pred = model.predict(X_test)
    
else:
    # Version équilibrée : 2 modèles complémentaires
    if USE_XGBOOST:
        print("   Mode équilibré : XGBoost + Random Forest")
    else:
        print("   Mode équilibré : GradientBoosting + Random Forest")
    
    # Split pour validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    # Modèle 1: XGBoost ou GradientBoosting
    if USE_XGBOOST:
        print("   - Entraînement XGBoost...")
        gb_model = XGBRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    else:
        print("   - Entraînement GradientBoosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    
    gb_model.fit(X_tr, y_tr)
    y_pred_gb = gb_model.predict(X_val)
    rmse_gb = np.sqrt(mean_squared_error(y_val, y_pred_gb))
    print(f"     RMSE: {rmse_gb:.2f}")
    
    # Modèle 2: Random Forest (version légère)
    print("   - Entraînement Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,  # Réduit de 200
        max_depth=15,      # Limité
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',  # Réduit le nombre de features considérées
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_val)
    rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
    print(f"     RMSE: {rmse_rf:.2f}")
    
    # Ensemble simple (moyenne pondérée)
    print("\n   - Création de l'ensemble...")
    weight_gb = 0.7  # Boosting généralement plus performant
    weight_rf = 0.3
    
    y_pred_ensemble = weight_gb * y_pred_gb + weight_rf * y_pred_rf
    rmse_ensemble = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))
    print(f"     RMSE Ensemble: {rmse_ensemble:.2f}")
    
    # Réentraînement sur toutes les données
    print("\n   - Réentraînement final...")
    gb_model.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    # Prédictions finales
    y_pred = weight_gb * gb_model.predict(X_test) + weight_rf * rf.predict(X_test)

# 6. GÉNÉRATION DES RÉSULTATS
print("\n6. Génération des prédictions...")

# Post-processing : s'assurer que les prédictions sont positives
y_pred = np.maximum(y_pred, 0)

# Vérifier que nous avons bien la colonne DATETIME
if 'DATETIME' not in test_df.columns:
    print("   ⚠️ Colonne DATETIME manquante, utilisation de DATETIME_HOUR")
    datetime_for_output = test_df['DATETIME_HOUR']
else:
    datetime_for_output = test_df['DATETIME']

# Créer le DataFrame de résultats
results_df = pd.DataFrame({
    'DATETIME': datetime_for_output,
    'ENTITY_DESCRIPTION_SHORT': test_df['ENTITY_DESCRIPTION_SHORT'],
    'y_pred': np.round(y_pred, 2),
    'KEY': 'Validation'
})

# Sauvegarder
output_file = '/Users/pierrebressollette/Desktop/Hackathon perso/GitHubFolder/predictions_wait_time.csv'
results_df.to_csv(output_file, index=False)

# 7. RÉSUMÉ
print("\n" + "="*60)
print("✅ PIPELINE TERMINÉE AVEC SUCCÈS!")
print("="*60)
print(f"\n📊 Statistiques des prédictions:")
print(f"   - Nombre de prédictions: {len(y_pred)}")
print(f"   - Min: {y_pred.min():.1f} minutes")
print(f"   - Max: {y_pred.max():.1f} minutes")
print(f"   - Moyenne: {y_pred.mean():.1f} minutes")
print(f"   - Médiane: {np.median(y_pred):.1f} minutes")
print(f"\n💾 Fichier sauvegardé: {output_file}")

# Feature importance (optionnel)
if not USE_SIMPLE_MODEL and hasattr(gb_model, 'feature_importances_'):
    print(f"\n🔍 Top 10 features importantes:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10).to_string(index=False))

print("\n🚀 Optimisations appliquées:")
print("   ✓ Réduction mémoire (float32)")
print("   ✓ Features essentielles uniquement")
print("   ✓ Données météo agrégées par heure")
print("   ✓ Modèles optimisés pour la vitesse")
if USE_SIMPLE_MODEL:
    print("   ✓ Mode ultra-rapide activé")
if USE_XGBOOST:
    print("   ✓ XGBoost utilisé")
else:
    print("   ✓ GradientBoosting utilisé (fallback)")