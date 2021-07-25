import numpy as np
import pickle

from flask import Flask, redirect, url_for,jsonify, request,render_template
app = Flask(__name__)
model = pickle.load(open('./model/model.pkl', 'rb'))

@app.route('/')
def index():
   return render_template("index.html")

@app.route('/ff',methods = ['POST', 'GET'])
def ff():
   return render_template("form.html")

@app.route('/predict',methods = ['POST'])
def predict():
   values=[]
 
   variance=request.form['variance']
   values.append(variance)
   
   asymmetry=request.form['asymmetry']
   values.append(asymmetry)

   kurtosis=request.form['kurtosis']
   values.append(kurtosis)

   Imageentropy=request.form['Imageentropy']
   values.append(Imageentropy)
   
   final_values=[np.array(values)]
   
   
   prediction=model.predict(final_values)
   
   
   result=prediction
   
   
   if result==0:
       return render_template('result.html',variance=variance,asymmetry=asymmetry,kurtosis=kurtosis,Imageentropy=Imageentropy,rrr=0)
   else:
       return render_template('result.html',variance=variance,asymmetry=asymmetry,kurtosis=kurtosis,Imageentropy=Imageentropy,rrr=1)
     


if __name__ == '__main__':
   app.run(debug=True,use_reloader=False)
