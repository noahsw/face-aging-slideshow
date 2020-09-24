import requests, json, os, json_tricks, sys, getopt

def download_media_item(media_item, photos_path):
    width = media_item['mediaMetadata']['width']
    height = media_item['mediaMetadata']['height']

    baseUrl = media_item['baseUrl'] + "=w" + width + "-h" + height

    timestamp = media_item['mediaMetadata']['creationTime']

    path = photos_path + "/" + timestamp + "-" + media_item['id'] + "-" + media_item['filename']

    if os.path.isfile(path):
        return

    if "-COLLAGE" in media_item['filename']:
        return

    r = requests.get(baseUrl, stream = True)
    with open(path, 'wb') as f:
        for chunk in r:
            f.write(chunk)

    data = {}
    data['googleMetadata'] = media_item
    data['timestamp'] = timestamp
    json_path = path + ".json"
    with open(json_path, 'w') as outfile:
        outfile.write(json_tricks.dumps(data))

    print("Saved " + path)


def main():
    key = ""
    authorization = ""

    try:
        opts, args = getopt.getopt(sys.argv[1:] , "", ["key=","authorization="])
    except getopt.GetoptError:
        print("download-photos.py --key=<key> --authorizaton=<authorization>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("download-photos.py --key=<key> --authorizaton=<authorization>")
            sys.exit()
        elif opt in ("--key"):
            key = arg
        elif opt in ("--authorization"):
            authorization = arg

    headers = {
        'authority': 'content-photoslibrary.googleapis.com',
        'x-goog-encode-response-if-executable': 'base64',
        'x-origin': 'https://explorer.apis.google.com',
        'x-clientdetails': 'appVersion=5.0%20(Macintosh%3B%20Intel%20Mac%20OS%20X%2010_15_6)%20AppleWebKit%2F537.36%20(KHTML%2C%20like%20Gecko)%20Chrome%2F85.0.4183.83%20Safari%2F537.36&platform=MacIntel&userAgent=Mozilla%2F5.0%20(Macintosh%3B%20Intel%20Mac%20OS%20X%2010_15_6)%20AppleWebKit%2F537.36%20(KHTML%2C%20like%20Gecko)%20Chrome%2F85.0.4183.83%20Safari%2F537.36',
        'authorization': 'Bearer ' + authorization,
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest',
        'x-javascript-user-agent': 'apix/3.0.0 google-api-javascript-client/1.1.0',
        'x-referer': 'https://explorer.apis.google.com',
        'accept': '*/*',
##        'x-client-data': 'CIu2yQEIorbJAQjBtskBCKmdygEI4KHKAQiXrMoBCJm1ygEI9cfKAQjnyMoBCOnIygEIq8nKAQiW1soBCLvXygEI9NfKAQ==',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
##        'referer': 'https://content-photoslibrary.googleapis.com/static/proxy.html?usegapi=1&jsh=m%3B%2F_%2Fscs%2Fapps-static%2F_%2Fjs%2Fk%3Doz.gapi.en.myOGgYJo9ys.O%2Fam%3DwQE%2Fd%3D1%2Fct%3Dzgms%2Frs%3DAGLTcCMR2Cg_3Iqxcgmos-E9G6cjWQG_Kw%2Fm%3D__features__',
        'accept-language': 'en-US,en;q=0.9',
    }

    params = (
        ('key', key),
    )

    response = requests.get('https://content-photoslibrary.googleapis.com/v1/albums', headers=headers, params=params)

    albums = response.json()['albums']

    i = 0
    for album in albums:
        i += 1
        print(str(i) + ". " + album['title'])

    print("")
    print("Which album?")
    album_index = int(input())

    album = albums[album_index - 1]
    album_id = album['id']
    album_name = album['title']

    ### MAKE FOLDER OF ALBUM_ID
    photos_path = "photos/" + album_name + " - " + album_id
    if not os.path.exists(photos_path):
        os.makedirs(photos_path)


    ### ITERATE THROUGH FILES
    next_page_token = ""
    while True:
        data = '{"albumId":"' + album_id + '", "pageToken":"' + next_page_token + '", "pageSize":"100"}'
        response = requests.post('https://content-photoslibrary.googleapis.com/v1/mediaItems:search', headers=headers, params=params, data=data)
        print(response.content)

        json = response.json()
        media_items = json['mediaItems']
        for media_item in media_items:
            if media_item['mimeType'] == "image/jpeg" or media_item['mimeType'] == "image/heif":
                if int(media_item['mediaMetadata']['width']) >= 1080:
                    download_media_item(media_item, photos_path)

        if "nextPageToken" in json:
            next_page_token = json['nextPageToken']
        else:
            break


if __name__ == "__main__":
    main()