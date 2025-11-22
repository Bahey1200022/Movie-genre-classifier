
import asyncio
import aiohttp
import os
from tmdb import route
import csv
from tqdm import tqdm
API_KEY = "79f1fab39c2786b7ef9932b7b325832c"
SAVE_POSTER_DIR = "posters/"
CSV_FILE = "movies.csv"
os.makedirs(SAVE_POSTER_DIR, exist_ok=True)

async def download_file(session, url, path):
    async with session.get(url) as resp:
        if resp.status == 200:
            with open(path, "wb") as f:
                f.write(await resp.read())
            return True
        return False

async def main():
    base = route.Base()
    base.key = API_KEY

    movie_route = route.Movie()
    movie_route.base = base
    failed_downloads = 0
    succesful_downloads = 0
    with open("logs.txt", "w") as logger:

    # Example: download 50 movies
        for i in range(1,500):
            movies_page = await movie_route.popular(page=i)
            movies = movies_page["results"]
            csv_rows = []

            async with aiohttp.ClientSession() as session:
                
                    for m in tqdm(movies):
                        movie_id = m["id"]
                        title = m["title"]
                        poster_path = m["poster_path"]
                        # print(f'Title: {title}')
                        # print(f'Path: {poster_path}')

                        # Get full details (genres)
                        details = await movie_route.details(movie_id)
                        genres = [g["name"] for g in details["genres"]]
                        if (len(genres) == 0):
                            continue
                        # Download poster
                        if poster_path:
                            img_filename = f'{movie_id}.jpg'
                            url = "https://image.tmdb.org/t/p/original" + poster_path
                            img_path = os.path.join(SAVE_POSTER_DIR, img_filename)
                            ok = await download_file(session, url, img_path)
                            if not ok:
                                logger.write(f"Movie {title} poster could not be downloaded\n")
                                failed_downloads += 1
                                continue
                            succesful_downloads +=1
                                
                        # Prepare CSV row
                        csv_rows.append([img_filename, genres,title])
                    logger.write(f"Downloaded {succesful_downloads} movies succesfully")
                    

                    with open("output.csv", "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(csv_rows)

    

    await base.session.close()

asyncio.run(main())
