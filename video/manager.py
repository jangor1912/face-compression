from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import utils
from config.config import get_config
from video.deconstructor.deconstructor import Deconstructor
from video.downloader.downloader import Downloader


class VideoManager(object):
    def __init__(self, max_results=50):
        config = get_config()
        self.develper_key = config['youtube']['developer_key']
        self.youtube_api_service_name = 'youtube'
        self.youtube_api_version = 'v3'
        self.baseUrl = 'https://www.youtube.com/watch?v='
        self.max_results = max_results
        self.downloaded_urls_csv = config['csv']['downloaded_urls']

        self.downloader = Downloader()
        self.deconstructor = Deconstructor()

    def youtube_search(self, query):
        youtube = build(self.youtube_api_service_name,
                        self.youtube_api_version,
                        developerKey=self.develper_key)

        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=self.max_results
        ).execute()

        results = search_response.get('items', [])
        results = [self.baseUrl + result['id']['videoId'] for result in results
                   if result['id']['kind'] == 'youtube#video']
        return results

    def start_processing(self, queries):
        for query in queries:
            urls = self.youtube_search(query)
            for url in urls:
                # check if url hasn't been downloaded yet
                urls = utils.load_csv_rows(self.downloaded_urls_csv, 'url')
                if url in urls:
                    continue
                urls = []

                # append new url to csv
                with open(self.downloaded_urls_csv, 'a') as csv_file:
                    csv_file.write(url)

                try:
                    callback_method = self.deconstructor.after_download
                    print("Started downloading from url={}".format(url))
                    self.downloader.download_video(url, callback_method)
                except Exception as url_exception:
                    print("Url = {} failed to process".format(url))
                    print("\tException message = {}".format(str(url_exception)))


if __name__ == '__main__':
    manager = VideoManager(max_results=1)
    try:
        manager.start_processing(['Chorwacja i Wlochy 2017 jan gorazda'])
    except HttpError as e:
        print('An HTTP error %d occurred:\n%s' % (e.resp.status, e.content))

