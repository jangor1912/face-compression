from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from common import utils
from config.config import Config
from video.deconstructor.deconstructor import Deconstructor
from video.downloader.downloader import Downloader


class VideoManager(object):
    def __init__(self, directory, max_results=50):
        config = Config().CONF
        self.develper_key = config['youtube']['developer_key']
        self.youtube_api_service_name = 'youtube'
        self.youtube_api_version = 'v3'
        self.baseUrl = 'https://www.youtube.com/watch?v='
        self.max_results = max_results
        self.downloaded_urls_csv = config['path']['downloaded_urls']

        self.downloader = Downloader(directory)
        self.deconstructor = Deconstructor()

    def youtube_search(self, query, video_duration="short", **kwargs):
        youtube = build(self.youtube_api_service_name,
                        self.youtube_api_version,
                        developerKey=self.develper_key)

        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=self.max_results,
            type="video",
            videoDuration=video_duration,
            **kwargs
        ).execute()

        results = search_response.get('items', [])
        results = [self.baseUrl + result['id']['videoId'] for result in results
                   if result['id']['kind'] == 'youtube#video']
        return results

    def start_processing(self, queries):
        downloaded_urls = set(utils.load_csv_rows(self.downloaded_urls_csv, 'url'))
        for query in queries:
            urls = self.youtube_search(query)
            for url in urls:
                if url in downloaded_urls:
                    print(f"Url {url} has been already processed")
                    continue
                # append new url to csv and set
                utils.append_csv_row(self.downloaded_urls_csv, ["url"], {"url": url})
                downloaded_urls.add(url)

                try:
                    # callback_method = self.deconstructor.after_download
                    print("Started downloading from url={}".format(url))
                    self.downloader.download_video(url)
                except Exception as url_exception:
                    print("Url = {} failed to process".format(url))
                    print("\tException message = {}".format(str(url_exception)))


if __name__ == '__main__':
    queries = ["interview", "standup", "vlog", "speech", "music video", "reaction", "talks", "face expression", "news",
               "daily vlog", "acting", "challenge", "tik tok", "workout", "super model"]
    manager = VideoManager("/media/jan/Elements SE/Magisterka/youtube_dataset", max_results=100)
    try:
        manager.start_processing(queries)
    except HttpError as e:
        print('An HTTP error %d occurred:\n%s' % (e.resp.status, e.content))
