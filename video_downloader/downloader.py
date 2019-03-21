import sys

from pytube import YouTube


class Downloader(object):
    def __init__(self, res='1080p', vcodec=None, acodec=None):
        self.res = res
        self.vcodec = vcodec
        self.acodec = acodec
        self.mime_type = 'video/mp4'

    def download_video(self, url, callback_method=None):
        filter_kwargs = {"mime_type": self.mime_type}
        if self.acodec:
            filter_kwargs["audio_codec"] = self.acodec
        if self.vcodec:
            filter_kwargs["video_codec"] = self.vcodec

        yt = YouTube(url)
        yt.register_on_complete_callback(callback_method)
        video = yt.streams.filter(**filter_kwargs).first()
        if video is None:
            raise RuntimeError("Video with such parameters do not exists")
        video.download()


if __name__ == "__main__":
    url = sys.argv[1]
    downloader = Downloader()
    downloader.download_video(url)
