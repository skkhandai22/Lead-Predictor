import numpy as np
import pandas as pd
from flask import Flask , request ,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
  return render_template('Lead-Predictor.html')
@app.route('/predict',methods=['POST'])
def predict():
    to_predict= np.array(to_predict_list).reshape(1,13)
    loaded_model=pickle.load(open('model.pkl','rb'))
    result=loaded_model.predict(to_predict)
    return result
    
@app.route('/predict_api',methods=['POST'])
def result():
  if request.method=='POST':
    to_predict_list=request.form.to_dict()
    to_predict_list=list(to_predict_list.values())
    to_predict_list=list(map(int,to_predict_list))
    result=[]
    result=predict(to_predict_list)
    if result[0]==0 and result[1]==0:
      prediction="These Wont be converted to neither MQL nor SQL"
    elif result[0]==1 and result[1]==0:
      prediction="They would be converted to MQL but not to SQL"
    elif result[0]==1 and result[1]==1:
      prediction="They would be converted to MQl as well as SQL"
    else:
      prediction="Oops Some problem occurred"
  
  
if __name__ == "__main__":
    app.run(debug=True)
    
