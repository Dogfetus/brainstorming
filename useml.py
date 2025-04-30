from ml import rank_players_with_ml

# Your list of players (case doesn't matter)
players = [
    "JimboSMX", "hintz", "spencer", "swagman", "paranoiaboi",
    "masongos", "grady", "jjk.", "chezmix", "inglomi",
    "jellyslosh", "wdrm", "eesa", "senpi", "Janus5k",
    # Add the rest of your players here
]

# Run the ranking
rankings = rank_players_with_ml(players)

# Print the top 10 players
print("\nTop 10 Players:")
top_10 = rankings[['player', 'final_rank']].head(10)
for _, row in top_10.iterrows():
    print(f"{row['final_rank']}. {row['player']}")
