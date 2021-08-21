from flask import Flask, request
import news_detection as news
app = Flask(__name__)

@app.route('/detectHyperpartisan',methods=['GET'])
def hyperpartisan ():
    response = news.detect(request.data)
    return response