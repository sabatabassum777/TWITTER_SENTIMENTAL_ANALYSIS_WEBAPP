from flask import Flask, render_template , request
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from ml_model import vect, tfidf

app=Flask(__name__)
#desirialization

model=pickle.load(open('model.pkl',"rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict' , methods=['POST','GET'])
def predict():
    comment=[x for x in request.form.values()]
    print(comment)

    x=vect.transform(comment)
    x_tfidf=tfidf.transform(x)

    output=model.predict(x_tfidf)
    print(output)

    x_prob=model.predict_proba(x_tfidf)
    print(x_prob)
    x_prob='{0:.{1}f}'.format(x_prob[0][1],2)
    print(x_prob)
    if output[0]==0:
        return render_template('index_result2.html',tweet='{}'.format(comment),pred='Negative Tweet',prob='{}'.format(x_prob))
    elif output[0]==1:
        return render_template('index_result1.html',tweet='{}'.format(comment), pred='Neutral Tweet',prob='{}'.format(x_prob))
    elif  output[0]==2:
        return render_template('index_result.html',tweet='{}'.format(comment), pred='Positive Tweet', prob='{}'.format(x_prob))
    else:
        return render_template('index.html', pred='An Error!! had occured in Prediction')
if __name__ =='__main__':
    app.run(debug=True)