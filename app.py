# from importlib.resources import open_binary, read_binary
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib

# Import distribution param
dis = pd.read_csv('distribution.csv',index_col=0)
dis = dis.sort_values(by='features')


# Import model
import xgboost
filename = 'xgboost.json'
print('version---',xgboost.__version__)
model = xgboost.XGBClassifier()
model.load_model(filename)


# Sidebar
st.sidebar.title('Biomarkers')    
lab_ls = []
for lab in dis['features']:
    lab_v = st.sidebar.number_input(label='{}'.format(lab), value=0.0, format='%f')
    lab_ls.append(lab_v)


# Scale
dis['lab'] = lab_ls
dis['stand'] = (dis['lab'] - dis['means'])/dis['stds']

# Main page
# st.write(dis)
# st.text(dis['stand'].to_list())
lab_arr = np.array([dis['stand'].to_list()])
# st.text(lab_arr)# [1,5]
## prediction
# model.predict(lab_arr)
model.predict_proba(lab_arr)
p = model.predict_proba(lab_arr)[:,1]
# print(dis['lab'].to_list())
   
if(dis['lab'].to_list()==[0.0, 0.0, 0.0, 0.0, 0.0]):
    cls = 'N/A'
elif p>0.9394049:
    # roc threshold
    cls = 'HFrEF'    
else:
    cls = 'HFpEF'

st.markdown('# Prediction') 
c1, c2, c3 = st.columns(3)
with c2:
    st.markdown('## {}'.format(cls))
# with c3:
#     st.text('Probability: {}'.format(p[0]))
#     st.text('Threshold: '+ '0.9394049')


# st.markdown('- Classification: {}     \n- Probability: {}    \n - Threshold: 0.9394049'.format("HFpEF", p[0]))
# st.markdown('***')


# shap == 0.41.0
import shap
explainer = shap.TreeExplainer(model)
df = pd.DataFrame(lab_arr)
df.columns = dis['features']
shap_values = explainer(df)

# waterfall
st.set_option('deprecation.showPyplotGlobalUse', False)
fig = shap.plots.waterfall(shap_values[0], )

st.markdown('# Explainer:')
st.pyplot(fig)

st.markdown('# Info:')
st.markdown('- Classification: {}     \n- Probability: {}    \n - Threshold: 0.9394049'.format(cls, p[0]))

