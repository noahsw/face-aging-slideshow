# face-aging-slideshow
Create a chronological slideshow of a specific face in a collection of photos.

This is inspired by projects like https://mymodernmet.com/time-lapse-video-children-growing-up-frans-hofmeester/. I wanted to do the same for my kids, based on everyday photos of them.


## Install
```
brew install libsvm
brew install imagemagick # for converting HEIC to JPG
pipenv shell
```


## Configure
1. Download Google Photos to `photos`



## Usage
```
python3 download-photos.py --key=KEY --authorization=AUTHORIZATION
```