# Fashion Detection and Recommendation project

## 1.Install requirement:
```
pip install -r requirements.txt
```
To use selenium, you have to install geckodriver first
## 2. Download pre-trained weights
Download `yolov3-df2_15000.weights` from https://drive.google.com/drive/folders/1b7laIv9-oeh59XSV6aOO50eMKbTGsPoP and put into static/models/ folder

## 3. Run Server:
```
python server.py
```

## 4. Public server through ngrok
In order to use google images search, you must public your image links.
Public local server through ngrok, and use public server 
Go to http://ngrok-random-string-server:5000 to test
Upload your images and get the results
