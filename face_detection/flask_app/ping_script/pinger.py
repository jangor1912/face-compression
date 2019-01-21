import sys
from time import time
from uuid import uuid4

import requests


def main():
    id = uuid4()
    image = sys.argv[1]
    ip = sys.argv[2]
    request_number = int(sys.argv[4])
    url = "http://{}/detect_face".format(ip)

    sum = 0
    for i in range(request_number):
        start = time()
        files = {'image': open(image, 'rb')}
        response = requests.post(url, files=files)
        end = time()
        response_time = end-start
        print("My id={}, request_number={}, response_time={}, response={}".format(id, i, response_time, response))
        sum += response_time
    average_time = sum / request_number
    print("My id={}, Average time of {} request is {}".format(id,
                                                              request_number,
                                                              average_time))


if __name__ == "__main__":
    main()
