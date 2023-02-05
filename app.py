from flask import Flask, render_template, request, redirect, url_for
import os
import openai

app = Flask(__name__)
openai.api_key = os.environ.get('OPENAI_KEY')

response_log = []
input_log = []

def getResponse(curIn):
    completion = openai.Completion.create(engine="text-davinci-003",
                                        prompt=curIn,
                                        temperature=0.2,
                                        n=1,
                                        max_tokens=100)
    output = completion.choices[0].text
    return str(output)

@app.route('/home')
@app.route('/')
def home():
    return render_template('home.html')

@app.route("/result", methods=["POST"])
def result():
    input_value = request.form["input_value"]
    response = getResponse(input_value)
    response_log.insert(0, (input_value, response))
    return render_template("home.html", chat_response=response, responses = response_log)

@app.route('/education')
def education():
    return render_template('education.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host='0.0.0.0', port=port)