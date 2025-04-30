#!/usr/bin/env python3
"""
Player Ranking System for Rhythm Games
-------------------------------------
A comprehensive machine learning ranking system that combines multiple approaches
to rank players based on their performance data.

Features:
- Case-insensitive player name handling
- Multiple ranking algorithms (Basic ML, Ensemble, TrueSkill)
- Feature importance analysis
- Customizable weighting
- Time-based filtering
- Visualizations

This all-in-one file contains all necessary functionality.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
import datetime
import json
from scipy.stats import rankdata
import os
from datetime import datetime, timedelta

# ============================================================================
# Player Name Normalizer
# ============================================================================

class PlayerNameNormalizer:
    """
    Utility class to handle player name case sensitivity issues.
    This class ensures that player names in the provided list are matched 
    correctly with their counterparts in the database, even if the case differs.
    """
    
    def __init__(self, db_connection):
        """Initialize with a database connection"""
        self.conn = db_connection
        self.db_player_names = self._get_db_player_names()
        self.name_mapping = {}
    
    def _get_db_player_names(self):
        """Get all player names from the database"""
        query = "SELECT username FROM gamers"
        df = pd.read_sql_query(query, self.conn)
        return df['username'].tolist()
    
    def create_name_mapping(self, player_list):
        """
        Create a mapping between input player names and database names.
        This handles case differences and similar variations.
        
        Args:
            player_list: List of player names to map
            
        Returns:
            dict: Mapping from input names to database names
        """
        self.name_mapping = {}
        
        for player in player_list:
            # Try exact match first
            if player in self.db_player_names:
                self.name_mapping[player] = player
                continue
            
            # Try case-insensitive match
            db_match = next((db_name for db_name in self.db_player_names 
                           if db_name.lower() == player.lower()), None)
            
            if db_match:
                self.name_mapping[player] = db_match
            else:
                # Handle whitespace differences (like " peter" vs "peter")
                db_match = next((db_name for db_name in self.db_player_names 
                               if db_name.lower().strip() == player.lower().strip()), None)
                
                if db_match:
                    self.name_mapping[player] = db_match
                else:
                    # Keep original if no match found
                    self.name_mapping[player] = player
                    print(f"Warning: No case-insensitive match found for player '{player}'")
        
        return self.name_mapping
    
    def get_normalized_name(self, player_name):
        """Get the normalized database name for a player"""
        if player_name in self.name_mapping:
            return self.name_mapping[player_name]
        
        # If not in mapping, try to find it now
        if player_name in self.db_player_names:
            return player_name
            
        # Try case-insensitive match
        db_match = next((db_name for db_name in self.db_player_names 
                       if db_name.lower() == player_name.lower()), None)
        
        if db_match:
            return db_match
            
        # Handle whitespace differences
        db_match = next((db_name for db_name in self.db_player_names 
                       if db_name.lower().strip() == player_name.lower().strip()), None)
        
        if db_match:
            return db_match
            
        # Return original if no match
        return player_name
    
    def normalize_player_list(self, player_list):
        """
        Apply normalization to an entire player list
        
        Args:
            player_list: List of player names to normalize
            
        Returns:
            list: List with normalized player names
        """
        # Create mapping if not already done
        if not self.name_mapping:
            self.create_name_mapping(player_list)
        
        # Apply mapping to list
        normalized_list = [self.get_normalized_name(player) for player in player_list]
        
        return normalized_list
    
    def find_player_by_name(self, name_fragment, limit=5):
        """
        Find players whose names contain the given fragment
        Useful for autocomplete or correction suggestions
        
        Args:
            name_fragment: Partial player name to search for
            limit: Maximum number of results to return
            
        Returns:
            list: List of matching player names
        """
        query = """
        SELECT username
        FROM gamers
        WHERE LOWER(username) LIKE LOWER(?)
        LIMIT ?
        """
        
        df = pd.read_sql_query(query, self.conn, params=(f'%{name_fragment}%', limit))
        return df['username'].tolist()
    
    def print_potential_mismatches(self, player_list):
        """
        Print information about potential player name mismatches
        
        Args:
            player_list: List of player names to check
        """
        if not self.name_mapping:
            self.create_name_mapping(player_list)
        
        print("\nPotential player name mismatches:")
        print("----------------------------------")
        
        for input_name, db_name in self.name_mapping.items():
            if input_name != db_name and db_name in self.db_player_names:
                print(f"Input: '{input_name}' -> Database: '{db_name}'")
        
        not_found = [name for name, mapped in self.name_mapping.items() 
                    if mapped not in self.db_player_names]
        
        if not_found:
            print("\nPlayers not found in database:")
            for name in not_found:
                print(f"- '{name}'")
                # Suggest potential matches
                suggestions = self.find_player_by_name(name.strip())
                if suggestions:
                    print(f"  Did you mean: {', '.join(suggestions)}?")


# ============================================================================
# Basic ML Ranking System
# ============================================================================

class PlayerRankingML:
    def __init__(self, db_connection, cutoff_date):
        self.conn = db_connection
        self.cutoff_date = cutoff_date
        self.feature_importance = None
        self.model = None
        
    def get_player_scores(self, player_name):
        """Fetch all scores for a given player up to the cutoff date"""
        # Use LOWER() to ensure case-insensitive matching
        query = """
        SELECT 
            s.score, s.cleared, s.full_combo, s.misses, s.max_combo, 
            s.perfect1, s.perfect2, s.early, s.late, s.steps,
            c.difficulty, c.difficulty_display, c.meter, c.pass_count, c.play_count,
            sg.title, sg.bpm
        FROM scores s
        JOIN gamers g ON s.gamer_id = g._id
        JOIN charts c ON s.song_chart_id = c._id
        JOIN songs sg ON c.song_id = sg._id
        WHERE LOWER(g.username) = LOWER(?) AND s.created_at <= ?
        ORDER BY s.created_at DESC
        """
        df = pd.read_sql_query(query, self.conn, params=(player_name, self.cutoff_date))
        return df
    
    def calculate_performance_metrics(self, player_name):
        """Calculate performance metrics for a player"""
        scores_df = self.get_player_scores(player_name)
        
        if scores_df.empty:
            # Return default values if no scores found
            return {
                'player': player_name,
                'avg_score': 0,
                'max_score': 0,
                'clear_rate': 0,
                'fc_rate': 0,
                'avg_accuracy': 0,
                'difficulty_performance': 0,
                'consistency': 0,
                'versatility': 0,
                'total_scores': 0
            }
        
        # Filter out any problematic rows
        scores_df = scores_df.dropna(subset=['score', 'difficulty'])
        
        if scores_df.empty:
            return {
                'player': player_name,
                'avg_score': 0,
                'max_score': 0,
                'clear_rate': 0,
                'fc_rate': 0,
                'avg_accuracy': 0,
                'difficulty_performance': 0,
                'consistency': 0,
                'versatility': 0,
                'total_scores': 0
            }
        
        # Calculate basic metrics
        total_scores = len(scores_df)
        
        # Calculate score metrics
        # Normalize scores by steps to make them comparable - handle division by zero
        scores_df['normalized_score'] = scores_df.apply(
            lambda row: row['score'] / row['steps'] if row['steps'] > 0 else 0, 
            axis=1
        )
        avg_norm_score = scores_df['normalized_score'].mean()
        max_norm_score = scores_df['normalized_score'].max()
        
        # Calculate accuracy - handle division by zero
        scores_df['accuracy'] = scores_df.apply(
            lambda row: ((row['perfect1'] + row['perfect2']) / row['steps']) * 100 if row['steps'] > 0 else 0,
            axis=1
        )
        avg_accuracy = scores_df['accuracy'].mean()
        
        # Calculate clear and full combo rates
        clear_rate = scores_df['cleared'].mean() * 100
        fc_rate = scores_df['full_combo'].mean() * 100
        
        # Calculate difficulty performance (how well they do on harder charts)
        # Weighted average score based on difficulty
        difficulty_sum = np.sum(scores_df['difficulty'])
        if difficulty_sum > 0:
            difficulty_performance = np.sum(scores_df['normalized_score'] * scores_df['difficulty']) / difficulty_sum
        else:
            difficulty_performance = 0
        
        # Calculate consistency (lower std dev = more consistent)
        if len(scores_df) >= 3:
            score_std = scores_df['normalized_score'].std()
            consistency = 1 / (score_std + 0.001) * 10 if not np.isnan(score_std) else 0
        else:
            consistency = 0
        
        # Calculate versatility (variety of song difficulties played)
        versatility = len(scores_df['difficulty'].unique()) / 20 * 100  # Assuming max 20 difficulties
        
        return {
            'player': player_name,
            'avg_score': avg_norm_score if not np.isnan(avg_norm_score) else 0,
            'max_score': max_norm_score if not np.isnan(max_norm_score) else 0,
            'clear_rate': clear_rate if not np.isnan(clear_rate) else 0,
            'fc_rate': fc_rate if not np.isnan(fc_rate) else 0,
            'avg_accuracy': avg_accuracy if not np.isnan(avg_accuracy) else 0,
            'difficulty_performance': difficulty_performance if not np.isnan(difficulty_performance) else 0,
            'consistency': consistency if not np.isnan(consistency) else 0,
            'versatility': versatility if not np.isnan(versatility) else 0,
            'total_scores': total_scores
        }
    
    def get_all_player_metrics(self, player_list):
        """Get metrics for all players"""
        all_metrics = []
        for player in player_list:
            metrics = self.calculate_performance_metrics(player)
            all_metrics.append(metrics)
        
        return pd.DataFrame(all_metrics)
    
    def train_ranking_model(self, player_metrics_df, known_ranks=None):
        """Train a model to predict player ranks"""
        # If we have known ranks (e.g., from previous tournaments)
        if known_ranks is not None:
            # Use known_ranks as target
            X = player_metrics_df.drop(['player'], axis=1)
            y = known_ranks
        else:
            # Create a simple scoring formula as our target
            player_metrics_df['target_score'] = (
                player_metrics_df['avg_score'] * 0.3 +
                player_metrics_df['max_score'] * 0.2 +
                player_metrics_df['clear_rate'] * 0.1 +
                player_metrics_df['fc_rate'] * 0.1 +
                player_metrics_df['avg_accuracy'] * 0.1 +
                player_metrics_df['difficulty_performance'] * 0.1 +
                player_metrics_df['consistency'] * 0.05 +
                player_metrics_df['versatility'] * 0.05
            )
            # Scale metrics for better performance
            scaler = StandardScaler()
            X = scaler.fit_transform(player_metrics_df.drop(['player', 'target_score'], axis=1))
            y = player_metrics_df['target_score']
        
        # Train a RandomForest regression model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Store feature importances
        feature_names = player_metrics_df.drop(['player'], axis=1).columns
        if 'target_score' in feature_names:
            feature_names = feature_names.drop('target_score')
            
        self.feature_importance = dict(zip(feature_names, model.feature_importances_))
        self.model = model
        
        return model
    
    def predict_player_ranks(self, player_metrics_df):
        """Predict ranks for all players"""
        if self.model is None:
            self.train_ranking_model(player_metrics_df)
        
        X = player_metrics_df.drop(['player'], axis=1)
        if 'target_score' in X.columns:
            X = X.drop(['target_score'], axis=1)
            
        # Predict scores
        pred_scores = self.model.predict(X)
        
        # Create ranking from scores (higher score = better rank)
        player_metrics_df['predicted_score'] = pred_scores
        
        # Sort by predicted score and assign ranks
        ranked_players = player_metrics_df.sort_values('predicted_score', ascending=False)
        ranked_players['rank'] = range(1, len(ranked_players) + 1)
        
        return ranked_players[['player', 'predicted_score', 'rank']]
    
    def plot_feature_importance(self):
        """Plot feature importance from the model"""
        if self.feature_importance is None:
            return
        
        # Sort feature importance
        sorted_features = dict(sorted(self.feature_importance.items(), 
                                     key=lambda item: item[1], 
                                     reverse=True))
        
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_features.keys(), sorted_features.values())
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Importance in Player Ranking')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    def evaluate_ranking(self, predicted_ranks, actual_ranks=None):
        """Evaluate the ranking against actual ranks if available"""
        if actual_ranks is None:
            # Use original player list order as "actual" ranks for demonstration
            actual_ranks = pd.DataFrame({
                'player': predicted_ranks['player'],
                'actual_rank': range(1, len(predicted_ranks) + 1)
            })
        
        # Merge predicted and actual ranks
        merged_ranks = pd.merge(predicted_ranks, actual_ranks, on='player')
        
        # Calculate rank differences
        merged_ranks['rank_diff'] = merged_ranks['rank'] - merged_ranks['actual_rank']
        
        # Calculate evaluation metrics
        mae = np.mean(np.abs(merged_ranks['rank_diff']))
        rmse = np.sqrt(np.mean(merged_ranks['rank_diff'] ** 2))
        
        # Plot rank differences
        plt.figure(figsize=(14, 6))
        plt.bar(merged_ranks['player'], merged_ranks['rank_diff'], color='purple')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xticks(rotation=90)
        plt.xlabel('Players')
        plt.ylabel('Rank Difference (Predicted - Actual)')
        plt.title(f'Ranking Deviation (MAE: {mae:.2f}, RMSE: {rmse:.2f})')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('rank_deviation.png')
        plt.close()
        
        return {'mae': mae, 'rmse': rmse, 'rank_diffs': merged_ranks}
    
    def save_ranking_results(self, ranked_players, filename='player_rankings.csv'):
        """Save the ranking results to a CSV file"""
        ranked_players.to_csv(filename, index=False)
        print(f"Rankings saved to {filename}")


# ============================================================================
# Ensemble Ranking System
# ============================================================================

class EnsembleRankingSystem:
    def __init__(self, db_connection, cutoff_date="2024-10-17"):
        self.conn = db_connection
        self.cutoff_date = cutoff_date
        self.models = {}
        self.player_features = None
        self.rankings = pd.DataFrame()
        
    def extract_player_features(self, player_name):
        """Extract comprehensive features for a player"""
        # Base query to get scores - using LOWER() for case-insensitive matching
        query = """
        SELECT 
            s._id as score_id, s.score, s.cleared, s.full_combo, s.misses, s.max_combo, 
            s.perfect1, s.perfect2, s.early, s.late, s.green, s.yellow, s.red,
            s.grade, s.steps, s.created_at, s.updated_at,
            c._id as chart_id, c.difficulty, c.difficulty_display, c.meter, 
            c.pass_count, c.play_count, c.steps_author,
            sg._id as song_id, sg.title, sg.artist, sg.bpm
        FROM scores s
        JOIN gamers g ON s.gamer_id = g._id
        JOIN charts c ON s.song_chart_id = c._id
        JOIN songs sg ON c.song_id = sg._id
        WHERE LOWER(g.username) = LOWER(?) AND s.created_at <= ?
        ORDER BY s.created_at DESC
        """
        
        df = pd.read_sql_query(query, self.conn, params=(player_name, self.cutoff_date))
        
        if df.empty:
            # Return default features for players with no data
            return self._get_default_features(player_name)
        
        # Convert dates to datetime objects
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # === BASIC METRICS ===
        # Score performance
        avg_score = df['score'].mean()
        max_score = df['score'].max()
        
        # Normalize scores by steps - handle division by zero
        df['score_per_step'] = df.apply(lambda row: row['score'] / row['steps'] if row['steps'] > 0 else 0, axis=1)
        avg_score_per_step = df['score_per_step'].mean()
        
        # Clear rates
        clear_rate = df['cleared'].mean()
        fc_rate = df['full_combo'].mean()
        
        # Accuracy - handle division by zero
        df['timing_accuracy'] = df.apply(
            lambda row: (row['perfect1'] + row['perfect2']) / row['steps'] if row['steps'] > 0 else 0, 
            axis=1
        )
        avg_accuracy = df['timing_accuracy'].mean()
        
        # === DIFFICULTY METRICS ===
        # How player performs on different difficulties
        diff_performance = {}
        for diff in df['difficulty_display'].unique():
            diff_df = df[df['difficulty_display'] == diff]
            if not diff_df.empty:
                diff_performance[diff] = diff_df['score_per_step'].mean()
        
        # Calculate weighted difficulty score (higher difficulty charts weighted more)
        # Handle division by zero
        weighted_diff_score = 0
        if np.sum(df['difficulty']) > 0:
            weighted_diff_score = np.sum(df['score_per_step'] * df['difficulty']) / np.sum(df['difficulty'])
        
        # === TIMING METRICS ===
        # Timing deviation (early vs late)
        timing_balance = 0
        if df['early'].sum() + df['late'].sum() > 0:
            timing_balance = (df['early'].sum() - df['late'].sum()) / (df['early'].sum() + df['late'].sum())
        
        # === CONSISTENCY METRICS ===
        # Score consistency (lower std = more consistent) - handle divide by zero
        score_std = df['score_per_step'].std()
        score_consistency = 1 / (score_std + 0.001) * 10 if not np.isnan(score_std) else 0
        
        # Consistency over time (recent performance vs overall)
        recent_df = df.sort_values('created_at', ascending=False).head(min(10, len(df)))
        recent_avg_score = recent_df['score_per_step'].mean()
        
        # Handle NaN values in score trend calculation
        if np.isnan(recent_avg_score) or np.isnan(avg_score_per_step):
            score_trend = 0
        else:
            score_trend = recent_avg_score - avg_score_per_step
        
        # === VERSATILITY METRICS ===
        # Different types of charts played
        unique_charts = len(df['chart_id'].unique())
        unique_songs = len(df['song_id'].unique())
        unique_difficulties = len(df['difficulty'].unique())
        
        # === PROGRESSION METRICS ===
        # Improvement over time - with proper NaN handling
        improvement = 0
        df = df.sort_values('created_at')
        if len(df) >= 3:
            # Use first half vs second half scores to check improvement
            mid_point = len(df) // 2
            first_half = df.iloc[:mid_point]
            second_half = df.iloc[mid_point:]
            
            first_half_mean = first_half['score_per_step'].mean()
            second_half_mean = second_half['score_per_step'].mean()
            
            # Check for NaN values before subtraction
            if not np.isnan(first_half_mean) and not np.isnan(second_half_mean):
                improvement = second_half_mean - first_half_mean
            else:
                improvement = 0
            
        # === COMBINE ALL FEATURES ===
        features = {
            'player': player_name,
            'avg_score': avg_score if not np.isnan(avg_score) else 0,
            'max_score': max_score if not np.isnan(max_score) else 0,
            'avg_score_per_step': avg_score_per_step if not np.isnan(avg_score_per_step) else 0,
            'clear_rate': clear_rate if not np.isnan(clear_rate) else 0,
            'fc_rate': fc_rate if not np.isnan(fc_rate) else 0,
            'avg_accuracy': avg_accuracy if not np.isnan(avg_accuracy) else 0,
            'weighted_diff_score': weighted_diff_score if not np.isnan(weighted_diff_score) else 0,
            'timing_balance': timing_balance if not np.isnan(timing_balance) else 0,
            'score_consistency': score_consistency if not np.isnan(score_consistency) else 0,
            'score_trend': score_trend if not np.isnan(score_trend) else 0,
            'unique_charts': unique_charts,
            'unique_songs': unique_songs,
            'unique_difficulties': unique_difficulties,
            'improvement': improvement if not np.isnan(improvement) else 0,
            'total_plays': len(df)
        }
        
        # Add difficulty-specific performance
        for diff, score in diff_performance.items():
            safe_diff_name = f"diff_{diff.replace(' ', '_')}"
            features[safe_diff_name] = score if not np.isnan(score) else 0
        
        return features
    
    def _get_default_features(self, player_name):
        """Return default features for players with no data"""
        return {
            'player': player_name,
            'avg_score': 0,
            'max_score': 0,
            'avg_score_per_step': 0,
            'clear_rate': 0,
            'fc_rate': 0,
            'avg_accuracy': 0,
            'weighted_diff_score': 0,
            'timing_balance': 0,
            'score_consistency': 0,
            'score_trend': 0,
            'unique_charts': 0,
            'unique_songs': 0,
            'unique_difficulties': 0,
            'improvement': 0,
            'total_plays': 0
        }
    
    def extract_all_player_features(self, player_list):
        """Extract features for all players"""
        all_features = []
        for player in player_list:
            print(f"Extracting features for {player}...")
            features = self.extract_player_features(player)
            all_features.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Fill missing values and replace infinity
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        self.player_features = df
        return df
    
    def get_chart_difficulty_features(self):
        """Get additional features about chart difficulties"""
        query = """
        SELECT 
            c._id, c.difficulty, c.difficulty_display, c.meter, 
            c.pass_count, c.play_count, c.steps_author,
            json_extract(c.graph, '$') as graph_data
        FROM charts c
        """
        
        charts_df = pd.read_sql_query(query, self.conn)
        
        # Process graph data
        charts_df['graph_complexity'] = charts_df['graph_data'].apply(
            lambda x: np.std(json.loads(x)) if x else 0
        )
        
        # Calculate difficulty ratios
        charts_df['pass_ratio'] = charts_df['pass_count'] / (charts_df['play_count'] + 1)
        
        return charts_df
    
    def train_model_score_prediction(self):
        """Train a model to predict player scores"""
        if self.player_features is None:
            raise ValueError("Player features not extracted yet")
        
        X = self.player_features.drop(['player'], axis=1)
        
        # Create a target based on weighted metrics
        y = (
            X['avg_score_per_step'] * 0.3 + 
            X['fc_rate'] * 0.2 + 
            X['avg_accuracy'] * 0.2 + 
            X['weighted_diff_score'] * 0.2 + 
            X['score_consistency'] * 0.1
        )
        
        # Replace any NaN or infinite values before scaling
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Store model
        self.models['score_prediction'] = {
            'model': model,
            'scaler': scaler,
            'feature_names': X.columns
        }
        
        # Calculate feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        
        return model, feature_importance
    
    def train_model_consistency(self):
        """Train a model focusing on player consistency"""
        if self.player_features is None:
            raise ValueError("Player features not extracted yet")
        
        X = self.player_features.drop(['player'], axis=1)
        
        # Target is consistency-focused
        y = (
            X['score_consistency'] * 0.4 + 
            X['avg_accuracy'] * 0.3 + 
            X['fc_rate'] * 0.3
        )
        
        # Replace any NaN or infinite values before scaling
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Store model
        self.models['consistency'] = {
            'model': model,
            'scaler': scaler,
            'feature_names': X.columns
        }
        
        return model
    
    def train_model_difficulty_handling(self):
        """Train a model focusing on how players handle difficult charts"""
        if self.player_features is None:
            raise ValueError("Player features not extracted yet")
        
        X = self.player_features.drop(['player'], axis=1)
        
        # Target is difficulty-focused
        y = (
            X['weighted_diff_score'] * 0.5 + 
            X['unique_difficulties'] * 0.3 + 
            X['improvement'] * 0.2
        )
        
        # Replace any NaN or infinite values before scaling
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
        
        # Store model
        self.models['difficulty_handling'] = {
            'model': model,
            'scaler': scaler,
            'feature_names': X.columns
        }
        
        return model
    
    def train_all_models(self):
        """Train all ranking models"""
        self.train_model_score_prediction()
        self.train_model_consistency()
        self.train_model_difficulty_handling()
    
    def predict_rankings(self):
        """Predict rankings using all models"""
        if not self.models:
            self.train_all_models()
        
        predictions = {}
        
        for model_name, model_data in self.models.items():
            # Get the model, scaler and feature names
            model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            
            # Prepare data
            X = self.player_features[feature_names]
            X_scaled = scaler.transform(X)
            
            # Predict
            pred_scores = model.predict(X_scaled)
            
            # Convert to ranks (higher score = better rank = lower number)
            ranks = rankdata(-pred_scores)
            
            # Store predictions
            predictions[model_name] = {
                'scores': pred_scores,
                'ranks': ranks
            }
        
        # Ensemble the rankings (average ranks across models)
        all_ranks = np.column_stack([predictions[model]['ranks'] for model in self.models])
        ensemble_ranks = np.mean(all_ranks, axis=1)
        
        # Create final ranking
        final_ranking = pd.DataFrame({
            'player': self.player_features['player'],
            'ensemble_score': -ensemble_ranks,  # Convert back to score (higher is better)
        })
        
        # Sort and add rank column
        final_ranking = final_ranking.sort_values('ensemble_score', ascending=False)
        final_ranking['rank'] = range(1, len(final_ranking) + 1)
        
        # Store rankings
        self.rankings = final_ranking
        
        return final_ranking
    
    def compare_to_original_order(self, original_player_list):
        """Compare ranking to original order"""
        if self.rankings.empty:
            raise ValueError("No rankings available. Run predict_rankings() first.")
        
        # Create mapping from original list to assumed ranks (1-indexed)
        original_ranks = pd.DataFrame({
            'player': original_player_list,
            'original_rank': range(1, len(original_player_list) + 1)
        })
        
        # Merge with predicted rankings
        comparison = pd.merge(self.rankings, original_ranks, on='player')
        
        # Calculate rank deviation
        comparison['rank_diff'] = comparison['rank'] - comparison['original_rank']
        
        # Calculate metrics
        mae = np.mean(np.abs(comparison['rank_diff']))
        rmse = np.sqrt(np.mean(comparison['rank_diff'] ** 2))
        
        # Plot comparison
        plt.figure(figsize=(14, 6))
        plt.bar(comparison['player'], comparison['rank_diff'], color='purple')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xticks(rotation=90)
        plt.xlabel('Players')
        plt.ylabel('Rank Difference (Predicted - Original)')
        plt.title(f'Ensemble Ranking Deviation (MAE: {mae:.2f}, RMSE: {rmse:.2f})')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('ensemble_ranking_deviation.png')
        plt.close()
        
        return comparison, {'mae': mae, 'rmse': rmse}
        
    def visualize_feature_importance(self):
        """Visualize feature importance from score prediction model"""
        if 'score_prediction' not in self.models:
            raise ValueError("Score prediction model not trained yet")
        
        model = self.models['score_prediction']['model']
        feature_names = self.models['score_prediction']['feature_names']
        
        # Get feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance for Player Ranking')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance_ensemble.png')
        plt.close()
        
    def save_rankings(self, filename='ensemble_rankings.csv'):
        """Save the ensemble rankings to a CSV file"""
        if self.rankings.empty:
            raise ValueError("No rankings available. Run predict_rankings() first.")
        
        self.rankings.to_csv(filename, index=False)
        print(f"Rankings saved to {filename}")


# ============================================================================
# TrueSkill Ranking System
# ============================================================================

class TrueSkillRanking:
    def __init__(self, db_connection, cutoff_date="2024-10-17"):
        self.conn = db_connection
        self.cutoff_date = cutoff_date
        self.player_skills = {}
        self.player_uncertainties = {}
        
    def initialize_skills(self, player_list):
        """Initialize skills for all players"""
        for player in player_list:
            self.player_skills[player] = 1500  # Initial skill level (like Elo)
            self.player_uncertainties[player] = 300  # Initial uncertainty
            
    def get_player_matchups(self):
        """Generate player matchups from score data"""
        # This query compares scores on the same charts
        query = """
        SELECT 
            g1.username as player1,
            g2.username as player2,
            s1.score as score1,
            s2.score as score2,
            c.difficulty as difficulty,
            c.difficulty_display as difficulty_type
        FROM scores s1
        JOIN scores s2 ON s1.song_chart_id = s2.song_chart_id AND s1._id != s2._id
        JOIN gamers g1 ON s1.gamer_id = g1._id
        JOIN gamers g2 ON s2.gamer_id = g2._id
        JOIN charts c ON s1.song_chart_id = c._id
        WHERE s1.created_at <= ? AND s2.created_at <= ?
        LIMIT 10000
        """
        
        matchups_df = pd.read_sql_query(query, self.conn, params=(self.cutoff_date, self.cutoff_date))
        return matchups_df
    
    def update_skills_from_matchups(self, matchups_df, k_factor=32):
        """Update player skills based on matchups"""
        for _, matchup in matchups_df.iterrows():
            player1 = matchup['player1']
            player2 = matchup['player2']
            
            # Skip if either player is not in our list
            if player1 not in self.player_skills or player2 not in self.player_skills:
                continue
                
            score1 = matchup['score1']
            score2 = matchup['score2']
            
            # Determine outcome (1 if player1 wins, 0.5 if tie, 0 if player2 wins)
            if score1 > score2:
                outcome = 1
            elif score1 < score2:
                outcome = 0
            else:
                outcome = 0.5
                
            # Scale k-factor by difficulty
            difficulty = matchup['difficulty']
            adjusted_k = k_factor * (difficulty / 10)  # Scale by difficulty
            
            # Get current skills
            skill1 = self.player_skills[player1]
            skill2 = self.player_skills[player2]
            
            # Calculate expected outcome (ELO formula)
            expected1 = 1 / (1 + 10 ** ((skill2 - skill1) / 400))
            expected2 = 1 - expected1
            
            # Update skills
            self.player_skills[player1] += adjusted_k * (outcome - expected1)
            self.player_skills[player2] += adjusted_k * ((1 - outcome) - expected2)
            
            # Update uncertainties (decrease with more matchups)
            self.player_uncertainties[player1] *= 0.99
            self.player_uncertainties[player2] *= 0.99
    
    def generate_rankings(self):
        """Generate rankings from player skills"""
        # Combine skills and uncertainties
        ranking_data = []
        for player, skill in self.player_skills.items():
            uncertainty = self.player_uncertainties[player]
            # Adjust skill by uncertainty (conservative estimate)
            adjusted_skill = skill - (uncertainty * 0.5)
            ranking_data.append({
                'player': player,
                'skill': skill,
                'uncertainty': uncertainty,
                'adjusted_skill': adjusted_skill
            })
        
        # Convert to DataFrame and sort
        rankings_df = pd.DataFrame(ranking_data)
        rankings_df = rankings_df.sort_values('adjusted_skill', ascending=False)
        rankings_df['rank'] = range(1, len(rankings_df) + 1)
        
        return rankings_df
    
    def run_ranking(self, player_list, iterations=3):
        """Run the TrueSkill-inspired ranking algorithm"""
        self.initialize_skills(player_list)
        
        # Get matchups
        print("Generating player matchups...")
        matchups = self.get_player_matchups()
        
        # Run multiple iterations
        for i in range(iterations):
            print(f"Running iteration {i+1}/{iterations}...")
            self.update_skills_from_matchups(matchups)
            
        # Generate final rankings
        return self.generate_rankings()


# ============================================================================
# Main Ranking Functions
# ============================================================================

def normalize_player_names_for_ranking(player_list, conn):
    """Normalize player names to match database entries"""
    print("\nNormalizing player names to match database entries...")
    
    # Create normalizer
    normalizer = PlayerNameNormalizer(conn)
    
    # Create mapping
    name_mapping = normalizer.create_name_mapping(player_list)
    
    # Print potential mismatches
    normalizer.print_potential_mismatches(player_list)
    
    # Normalize the list
    normalized_players = normalizer.normalize_player_list(player_list)
    
    # Show before/after for players with changes
    changes = [(old, new) for old, new in zip(player_list, normalized_players) if old != new]
    if changes:
        print("\nPlayer name normalizations applied:")
        for old, new in changes:
            print(f"  '{old}' -> '{new}'")
    else:
        print("No player name normalizations needed.")
    
    return normalized_players

def rank_players_with_ml(player_list, cutoff_date="2024-10-17", db_path="new_data.db"):
    """
    Run multiple ranking approaches and aggregate results
    
    Args:
        player_list: List of player names to rank
        cutoff_date: Only use scores up to this date
        db_path: Path to SQLite database
        
    Returns:
        DataFrame with final rankings
    """
    print(f"Ranking {len(player_list)} players with data up to {cutoff_date}...")
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        print(f"Successfully connected to database: {db_path}")
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None
    
    # Normalize player names
    normalizer = PlayerNameNormalizer(conn)
    name_mapping = normalizer.create_name_mapping(player_list)
    normalized_players = normalizer.normalize_player_list(player_list)
    
    # Print normalization information
    changes = [(old, new) for old, new in zip(player_list, normalized_players) if old != new]
    if changes:
        print("\nPlayer name normalizations applied:")
        for old, new in changes:
            print(f"  '{old}' -> '{new}'")
    
    # Check for players not found in database
    not_found = [name for name in normalized_players 
                if name not in normalizer.db_player_names]
    
    if not_found:
        print("\nWarning: The following players were not found in the database:")
        for name in not_found:
            print(f"- '{name}'")
            # Suggest potential matches
            suggestions = normalizer.find_player_by_name(name.strip())
            if suggestions:
                print(f"  Did you mean: {', '.join(suggestions)}?")
    
    # Run ensemble ranking
    print("\n=== ENSEMBLE RANKING ===")
    ensemble_ranker = EnsembleRankingSystem(conn, cutoff_date)
    ensemble_ranker.extract_all_player_features(normalized_players)
    ensemble_ranker.train_all_models()
    ensemble_rankings = ensemble_ranker.predict_rankings()
    
    # Rename the rank column to avoid naming conflicts in merge
    ensemble_rankings = ensemble_rankings.rename(columns={'rank': 'rank_ensemble'})
    
    # Run TrueSkill ranking
    print("\n=== TRUESKILL RANKING ===")
    trueskill_ranker = TrueSkillRanking(conn, cutoff_date)
    trueskill_rankings = trueskill_ranker.run_ranking(normalized_players)
    
    # Rename the rank column to avoid naming conflicts in merge
    trueskill_rankings = trueskill_rankings.rename(columns={'rank': 'rank_trueskill'})
    
    # Run basic ML ranking
    print("\n=== BASIC ML RANKING ===")
    basic_ranker = PlayerRankingML(conn, cutoff_date)
    player_metrics = basic_ranker.get_all_player_metrics(normalized_players)
    basic_ranker.train_ranking_model(player_metrics)
    basic_rankings = basic_ranker.predict_player_ranks(player_metrics)
    
    # Rename the rank column to avoid naming conflicts in merge
    basic_rankings = basic_rankings.rename(columns={'rank': 'rank_basic'})
    
    # Aggregate rankings
    print("\n=== AGGREGATING RANKINGS ===")
    all_rankings = pd.DataFrame({'player': normalized_players})
    
    # Add each ranking - using direct column names to avoid suffix issues
    all_rankings = pd.merge(
        all_rankings, 
        ensemble_rankings[['player', 'rank_ensemble']], 
        on='player', 
        how='left'
    )
    
    all_rankings = pd.merge(
        all_rankings, 
        trueskill_rankings[['player', 'rank_trueskill']], 
        on='player', 
        how='left'
    )
    
    all_rankings = pd.merge(
        all_rankings, 
        basic_rankings[['player', 'rank_basic']], 
        on='player', 
        how='left'
    )
    
    # Calculate mean rank with explicit column references
    rank_columns = ['rank_ensemble', 'rank_trueskill', 'rank_basic']
    # Check which columns actually exist in the dataframe
    existing_rank_columns = [col for col in rank_columns if col in all_rankings.columns]
    
    # Only calculate mean if we have at least one ranking
    if existing_rank_columns:
        all_rankings['mean_rank'] = all_rankings[existing_rank_columns].mean(axis=1)
    else:
        # Fallback if no ranking columns exist
        print("Warning: No ranking columns found for mean calculation")
        all_rankings['mean_rank'] = range(1, len(all_rankings) + 1)
    
    # Sort by mean rank
    final_rankings = all_rankings.sort_values('mean_rank')
    final_rankings['final_rank'] = range(1, len(final_rankings) + 1)
    
    # Map back to original names for output if needed
    reverse_mapping = {v: k for k, v in name_mapping.items()}
    final_rankings['original_name'] = final_rankings['player'].map(
        lambda x: reverse_mapping.get(x, x)
    )
    
    # Save final rankings
    final_rankings.to_csv('final_ml_rankings.csv', index=False)
    
    # Display top 10 players
    print("\nTop 10 Players (Final Ranking):")
    print(final_rankings.head(10)[['original_name', 'player', 'final_rank', 'mean_rank']])
    
    # Plot ranking comparison
    try:
        import matplotlib.pyplot as plt
        
        # Get top 20 players for visualization
        top_players = final_rankings.head(20)
        
        plt.figure(figsize=(12, 8))
        
        # Create bar positions
        bar_width = 0.2
        r1 = np.arange(len(top_players))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Check which columns exist for plotting
        can_plot = True
        if 'rank_ensemble' not in top_players.columns:
            print("Warning: ensemble ranking results missing")
            can_plot = False
        if 'rank_trueskill' not in top_players.columns:
            print("Warning: trueskill ranking results missing")
            can_plot = False
        if 'rank_basic' not in top_players.columns:
            print("Warning: basic ML ranking results missing")
            can_plot = False
            
        if can_plot:
            # Create bars
            plt.bar(r1, top_players['rank_ensemble'], width=bar_width, label='Ensemble Ranking')
            plt.bar(r2, top_players['rank_trueskill'], width=bar_width, label='TrueSkill Ranking')
            plt.bar(r3, top_players['rank_basic'], width=bar_width, label='Basic ML Ranking')
            
            # Add labels and title
            plt.xlabel('Players')
            plt.ylabel('Rank')
            plt.title('Comparison of Ranking Methods (Top 20 Players)')
            plt.xticks([r + bar_width for r in range(len(top_players))], top_players['player'], rotation=90)
            
            # Add legend
            plt.legend()
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('ranking_comparison.png')
            plt.close()
            print("Ranking comparison plot saved as 'ranking_comparison.png'")
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Close database connection
    conn.close()
    
    return final_rankings

def compare_rankings(players, ranking_list):
    """Compare different ranking approaches"""
    # Create DataFrame for comparison
    comparison = pd.DataFrame({'player': players})
    
    # Add each ranking
    for ranks, name in ranking_list:
        # Standardize column names
        if ranks is None:
            continue
            
        if 'rank' in ranks.columns:
            rank_col = ranks[['player', 'rank']].copy()
        else:
            # Find rank-like column
            rank_cols = [col for col in ranks.columns if 'rank' in col.lower()]
            if rank_cols:
                rank_col = ranks[['player', rank_cols[0]]].copy()
                rank_col.columns = ['player', 'rank']
            else:
                continue
                
        # Rename rank column with method name
        rank_col.columns = ['player', f'{name}_rank']
        
        # Merge with comparison
        comparison = pd.merge(comparison, rank_col, on='player', how='left')
    
    # Calculate agreement between methods
    rank_cols = [col for col in comparison.columns if col.endswith('_rank')]
    
    # Only proceed if we have multiple ranking methods
    if len(rank_cols) > 1:
        # Calculate correlation matrix
        rank_corr = comparison[rank_cols].corr(method='spearman')
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(rank_corr, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(rank_cols)):
            for j in range(len(rank_cols)):
                plt.text(j, i, f'{rank_corr.iloc[i, j]:.2f}', 
                         ha='center', va='center', color='white', fontweight='bold')
        
        plt.colorbar(label='Spearman Correlation')
        plt.xticks(range(len(rank_cols)), rank_cols, rotation=45)
        plt.yticks(range(len(rank_cols)), rank_cols)
        plt.title('Correlation Between Different Ranking Methods')
        plt.tight_layout()
        plt.savefig('ranking_correlation.png')
        plt.close()
        
        print("Ranking correlation plot saved as 'ranking_correlation.png'")
    
    # Display top 10 players according to each method
    print("\nTop 10 players by different methods:")
    for col in rank_cols:
        method = col.replace('_rank', '')
        top10 = comparison.sort_values(col).head(10)['player'].reset_index(drop=True)
        print(f"\n{method} ranking top 10:")
        for i, player in enumerate(top10):
            print(f"{i+1}. {player}")
    
    return comparison

# Function to customize the ranking system
def customize_ranking_system(player_list, cutoff_date="2024-10-17", db_path="new_data.db"):
    """Create a custom weighting scheme for player ranking"""
    try:
        conn = sqlite3.connect(db_path)
        print(f"Successfully connected to database: {db_path}")
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None
    
    print("\n=== CUSTOMIZED RANKING SYSTEM ===")
    
    # Normalize player names
    normalized_players = normalize_player_names_for_ranking(player_list, conn)
    
    # Get player metrics
    ranker = PlayerRankingML(conn, cutoff_date)
    player_metrics = ranker.get_all_player_metrics(normalized_players)
    
    # Create a custom weighting scheme
    # This gives more weight to full combos and accuracy
    player_metrics['custom_score'] = (
        player_metrics['avg_score'] * 0.2 +
        player_metrics['max_score'] * 0.1 +
        player_metrics['clear_rate'] * 0.1 +
        player_metrics['fc_rate'] * 0.25 +  # Increased weight for full combos
        player_metrics['avg_accuracy'] * 0.25 +  # Increased weight for accuracy
        player_metrics['difficulty_performance'] * 0.1
    )
    
    # Sort by custom score
    custom_ranking = player_metrics.sort_values('custom_score', ascending=False)
    custom_ranking['rank'] = range(1, len(custom_ranking) + 1)
    
    # Show top 10 players
    print("\nTop 10 players (custom weighting):")
    print(custom_ranking[['player', 'rank', 'custom_score']].head(10))
    
    # Create a visualization comparing original vs custom ranking
    top_20_players = custom_ranking.head(20)
    
    plt.figure(figsize=(12, 8))
    plt.bar(top_20_players['player'], top_20_players['custom_score'], color='purple')
    plt.xticks(rotation=90)
    plt.title('Top 20 Players by Custom Score')
    plt.tight_layout()
    plt.savefig('custom_ranking.png')
    plt.close()
    
    print("Custom ranking visualization saved as 'custom_ranking.png'")
    
    # Close connection
    conn.close()
    
    return custom_ranking

# Function to focus on specific time periods
def time_based_ranking(player_list, months=3, db_path="new_data.db"):
    """Rank players based on recent performances only"""
    # Calculate cutoff date (e.g., last 3 months)
    cutoff = datetime.now() - timedelta(days=30 * months)
    cutoff_date = cutoff.strftime("%Y-%m-%d")
    
    print(f"\n=== RANKING BASED ON LAST {months} MONTHS ===")
    print(f"Using data from {cutoff_date} to present")
    
    # Use the main ranking function with the adjusted cutoff date
    return rank_players_with_ml(player_list, cutoff_date=cutoff_date, db_path=db_path)


# ============================================================================
# Usage Examples
# ============================================================================

def run_example():
    """Run an example of the player ranking system"""
    # Example player list with case variations
    players = [
        "JimboSMX", "HINTZ", "spencer", "swagman", "paranoiaboi",
        "masongos", "Grady", "jjk.", "chezmix", "inglomi",
        "jellyslosh", "wdrm", "eesa", "senpi", "janus5k",
        "tayman", "emcat", "pilot", "mxl100", "snowstorm",
        "datcoreedoe", "big matt", "werdwerdus", "cathadan",
        "shinoBEE", " peter"  # Note the space and mixed case
    ]
    
    # Run the ranking with name normalization
    rankings = rank_players_with_ml(players)
    
    # You can also run time-based ranking
    # recent_rankings = time_based_ranking(players, months=3)
    
    # Or create a custom weighting
    # custom_rankings = customize_ranking_system(players)
    
    return rankings

# Function that runs when the script is executed directly
def main():
    """Main function that runs when script is executed directly"""
    print("=" * 80)
    print("PLAYER RANKING SYSTEM FOR RHYTHM GAMES")
    print("=" * 80)
    print("\nThis system ranks players based on their performance data using multiple ML approaches.")
    print("It automatically handles case differences in player names and provides comprehensive")
    print("visualizations of the ranking results.")
    
    # Ask if user wants to use the example player list or provide their own
    choice = input("\nDo you want to use the example player list? (y/n): ").strip().lower()
    
    if choice.startswith('n'):
        # Get player list from user
        print("\nEnter player names (one per line). Type 'done' when finished:")
        players = []
        while True:
            player = input().strip()
            if player.lower() == 'done':
                break
            if player:  # Skip empty lines
                players.append(player)
    else:
        # Use example player list
        players = [
            "JimboSMX",
            "hintz",
            "spencer",
            "swagman",
            "paranoiaboi",
            "masongos",
            "grady",
            "jjk.",
            "chezmix",
            "inglomi",
            "jellyslosh",
            "wdrm",
            "eesa",
            "senpi",
            "Janus5k",
            "tayman",
            "emcat",
            "pilot",
            "mxl100",
            "snowstorm",
            "datcoreedoe",
            "big matt",
            "werdwerdus",
            "cathadan",
            "shinobee",
            "mesyr",
            "zom585",
            "ctemi",
            "zephyrnobar",
            "sydia",
            "kren",
            "xosen",
            "sweetjohnnycage",
            "cheesecake",
            "ditz",
            "enderstevemc",
            "jiajiabinks",
            "meeko",
            "momosapien",
            "auby",
            "arual",
            "dogfetus",
            "noir",
            "dali",
            " peter",
            "jokko",
            "butterbutt",
            "jewel",
            "beatao",
            "maverick",
        ]

        print(f"\nUsing example list with {len(players)} players.")
    
    # Ask for cutoff date
    use_default_date = input("\nUse default cutoff date (2024-10-17)? (y/n): ").strip().lower()
    if use_default_date.startswith('n'):
        print("Enter cutoff date (YYYY-MM-DD):")
        cutoff_date = input().strip()
    else:
        cutoff_date = "2024-10-17"
    
    # Ask for database path
    use_default_db = input("\nUse default database path (new_data.db)? (y/n): ").strip().lower()
    if use_default_db.startswith('n'):
        print("Enter database path:")
        db_path = input().strip()
    else:
        db_path = "new_data.db"
    
    # Run the ranking
    print("\nRunning player ranking system...")
    rankings = rank_players_with_ml(players, cutoff_date=cutoff_date, db_path=db_path)
    
    if rankings is not None:
        print("\nRanking complete! Results saved to 'final_ml_rankings.csv'")
        print("Visualizations have been generated as PNG files.")
    else:
        print("\nRanking failed. Please check your database path and try again.")
    
    return rankings

# Entry point for the script
if __name__ == "__main__":
    main()
