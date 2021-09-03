import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image

pickle_in=open("model.pkl","rb")
rf=pickle.load(pickle_in)

def predict_lead(companyaccount,leadSource,accountType,status,mqlMed,industry,lineOfService,leadChannel,marketingSubj,marketingCampaign,mqlSubj,mqlServ,mqlChannel):
    prediction=rf.predict([company/account,leadSource,accountType,status,mqlMed,industry,lineOfService,leadChannel,marketingSubj,marketingCampaign,mqlSubj,mqlServ,mqlChannel])
    print(prediction)
    return prediction
def main():
    st.title("Lead Predictor")
    html_temp="""
    <div style ="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Lead Predictor</h2>
    </div>
    """

    st.markdown(html_temp,unsafe_allow_html=True)
    
    companyaccount=st.text_input("company/account","Type Here")
    leadSource=st.text_input("Lead Source","Type Here")
    accountType=st.text_input('Select Account Type',('New Account','Existing Account'))
    status=st.text_input('Select Lead Status',('Working','New Lead','Not Interested Qualified','Send to Marketing','Unreachable','Do Not Call(DNC)'))
    mqlMed=st.text_input("MQL Medium","Type Here")
    industry=st.text_input("Industry","Type Here")
    lineOfService=st.text_input("Line Of Service","Type Here")
    leadChannel=st.text_input("Lead Channel","Type Here")
    marketingSubj=st.text_input("Marketing Subject","Type Here")
    marketingCampaign=st.text_input("Marketing Campaign Name","Type Here")
    mqlSubj=st.text_input("MQL Subject","Type Here")
    mqlServ=st.text_input("MQL Service","Type Here")
    mqlChannel=st.text_input("MQL Channel","Type Here")

    result=""
    if st.button("Predict"):
        result=predict_lead(companyaccount,leadSource,accountType,status,mqlMed,industry,lineOfService,leadChannel,marketingSubj,marketingCampaign,mqlSubj,mqlServ,mqlChannel)
        st.success("The Output is {}",format.result)


if __name__=='__main__':
    main()
