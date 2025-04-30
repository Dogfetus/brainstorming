import sqlite3
import os
import requests
import time

def download_cover_thumbnails():
    conn = sqlite3.connect("new_data.db")
    cursor = conn.cursor()
    query = """
        SELECT title, cover_thumb
        FROM songs
        WHERE cover_thumb IS NOT NULL;
    """
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()

    base_url = "https://data.stepmaniax.com/"
    output_dir = "cover_thumbnails"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for title, cover_thumb in results:
        if cover_thumb:
            url = base_url + cover_thumb
            file_extension = os.path.splitext(cover_thumb)[1]
            output_file = os.path.join(output_dir, f"{title}{file_extension}")

            response = requests.get(url)
            if response.status_code == 200:
                with open(output_file, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded: {output_file}")
            else:
                print(f"Failed to download: {url}")

            # Add a slight delay of 0.5 seconds between each download
            time.sleep(1)

# Execute the function to download cover thumbnails
download_cover_thumbnails()
