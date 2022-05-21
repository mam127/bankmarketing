# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hydralit as hy
import hydralit_components as hc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Function for making spaces
def space(n,element): # n: number of lines
    for i in range(n):
        element.write("")

# html command for a red line
red_line="""<hr style="height:4px;border:none;color:#DC143C;background-color:#C00000;"/>"""

# html commnad for a grey line
grey_line="""<hr style="height:1px;border:none;no shade;"/>"""

# Setting page layout
st.set_page_config(layout='wide',initial_sidebar_state='collapsed',)

# Loading Dataset
df = pd.read_csv("https://raw.githubusercontent.com/mam127/Project-370/main/bank.csv")

# Sidebar Formatting
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 550px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 550px;
        margin-left: -550px;
    }
     
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .css-4yfn50 {
    background-color: rgb(240, 242, 246);
    background-attachment: fixed;
    flex-shrink: 0;
    height: 100vh;
    overflow: auto;
    padding: 0rem 1rem;
    position: relative;
    transition: margin-left 300ms ease 0s, box-shadow 300ms ease 0s;
    width: 21rem;
    z-index: 100;
    margin-left: 0px;
}    
    """,
    unsafe_allow_html=True,
)

# Filter Word Formatting
st.markdown(
    """
    <style>
    strong {
    font-weight: 1000;
}    
    """,
    unsafe_allow_html=True,
)
st.sidebar.header("**Filters**")

# Filters Formatting
st.markdown(
    """
    <style>
  .css-qrbaxs {
    font-size: 18px;
    color: rgb(49, 51, 63);
    margin-bottom: 7px;
    height: auto;
    min-height: 1.5rem;
    vertical-align: middle;
    display: flex;
    flex-direction: row;
    -webkit-box-align: center;
    align-items: center;
}   
    """,
    unsafe_allow_html=True,
)

# age filter
age = st.sidebar.slider('Age', min(df['age']), max(df['age']),(min(df['age']), max(df['age'])))
space(1,st.sidebar)

# education filter
education_list = ['primary','secondary','tertiary','unknown']
education = st.sidebar.multiselect('Education',education_list,education_list)
space(1,st.sidebar)

# contact filter
contact_list = ['cellular', 'telephone','unknown']
contact = st.sidebar.multiselect('Contact',contact_list,contact_list)
space(1,st.sidebar)

# month filter
month_list = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
month = st.sidebar.multiselect('Month',month_list,month_list)
space(1,st.sidebar)

# campaign filter
campaign = st.sidebar.slider('Number of Contacts', min(df['campaign']), max(df['campaign']),(min(df['campaign']), max(df['campaign'])))
space(1,st.sidebar)

# applying filters
df = df[ (df['age']>=age[0]) & (df['age']<=age[1]) & (df.education.isin(education))  & (df.contact.isin(contact))  & (df.month.isin(month)) &
(df['campaign']>=campaign[0]) & (df['campaign']<=campaign[1])]

# specify the primary menu definition
menu_data = [
    {'id':"Overview",'icon': "far fa-copy", 'label':"Overview",'ttip':"Overview"},
    {'id':"Dashboard",'icon': "far fa-chart-bar", 'label':"Dashboard",'ttip':"Dashboard"},
    {'id':"Model",'icon': "fa fa-laptop",'label':"ML Model",'ttip':"Model"},  
    {'id':"Meta Data",'icon': "fa fa-database", 'label':"Meta Data",'ttip':"Meta Data"}]
over_theme = {'txc_inactive': 'white','menu_background':'#C00000'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='sticky', #jumpy or not-jumpy, but sticky or pinned
)

# Home Page
if menu_id=='Home':
    col1,col2,col3 = st.columns([1.5,2,1])
    col2.title('Bank Marketing Campaign')
    # MSBA Logo
    html_string = '''<!DOCTYPE html>
<html>
<body>
 <a href="https://www.aub.edu.lb/osb/MSBA/Pages/default.aspx">
  <img src="https://github.com/mam127/Project-370/blob/main/Logo%20MSBA.png?raw=true.png" width=200" height="200" />
 </a>
</body>
</html>'''
    col1,col2,col3 = st.columns([2.5,2,1])
    space(1,col2)
    col2.markdown(html_string, unsafe_allow_html=True)
    col1,col2,col3 = st.columns([1.1,2,1.1])
    space(1,col1)
    col1.subheader("Done by [Mahdi Mohammad](https://www.linkedin.com/in/mahdi-mohammad-7b5034201/?originalSubdomain=lb)")
    space(1,col3)
    col3.subheader("Professor [Wissam Sammouri](https://www.aub.edu.lb/Pages/profile.aspx?memberid=ws42)")



# Overview Page
if menu_id=='Overview':
    st.subheader("Description")
    st.write('''
    This is the classic marketing bank dataset uploaded originally in the UCI Machine Learning Repository. The dataset gives information about 
    a marketing campaign of a financial institution. It is required to analyze this data in order to find the best ways to look for future strategies in order 
    to improve future marketing campaigns for the bank.''')
    st.write(grey_line, unsafe_allow_html=True)
    st.subheader("Content")
    st.write('''
    * The dataset contains **''' + str(df.shape[0]) + " rows** and **" + str(df.shape[1]) + " columns**." + 
    '''
    * The features of this dataset are: **''' + str(df.columns.tolist()) + "**")
    st.write(grey_line, unsafe_allow_html=True)
    st.subheader("Numerical Features Statistics")
    st.write(df.describe())
    st.write(grey_line, unsafe_allow_html=True)
    st.subheader("Categorical Features Statistics")
    # List of categorical features
    cat_features = ["job", "marital", "education", "default", "housing", "loan", "contact","month", "poutcome", "deposit"]
    # Modes of categorical features
    modes_list = []
    for f in cat_features:
        modes_list.append(df[f].mode()[0])
    modes_df = pd.DataFrame({"Categorical Variable":cat_features, "Mode":modes_list})
    st.write(modes_df)

# Dashboard Page
if menu_id=='Dashboard': 
    st.markdown(
    """
    <style>
  .css-1rh8hwn {
    font-size: 16px;
    color: rgb(49, 51, 63);
    height: auto;
    min-height: 1.5rem;
    vertical-align: middle;
    display: flex;
    flex-direction: row;
    -webkit-box-align: center;
    align-items: center;
    margin-bottom: 0px;
}   
    """,
    unsafe_allow_html=True,
)

    col1,col2,col3,col4,col5,col6 = st.columns([0.5,1,1,1,1,1])
    col2.metric('Average Balance $', float("{:.2f}".format(df["balance"].mean())))
    col4.metric('Avg. Last Contact Duration (Sec)', float("{:.2f}".format(df["duration"].mean())))
    col6.metric('Deposit Subscriptions', df[df['deposit']=='yes']['deposit'].count())
    
    space(2,st)

    # Pie Chart
    col1,col2,col3=st.columns([1,0.5,1])
    fig = px.pie(df, names='deposit',hole=0.5,color='deposit', color_discrete_map={'yes':'#C00000','no':'#D4D2D2'}, title='Subscriptions to a Term Deposit')
    col1.plotly_chart(fig, use_container_width=True, sharing="streamlit")
    col1.write('''**What is the percentage of clients who have subscribed for a term deposit?**''')
    col1.write('**' + str(round(df[df['deposit']=='yes']['deposit'].count()/df.shape[0] *100,1)) + "%** of clients have subscribed for a term deposit.")

    # Bar Chart 1: Deposit Subscription per Education Level
    df_bar=df[df['deposit']=='yes']
    df_bar=df_bar.groupby(['education'],as_index=False).count()
    fig = go.Figure([go.Bar(x=df_bar['deposit'],y=df_bar["education"],orientation='h',marker_color='#C00000')])
    fig.update_layout(title={'text': "Deposit Subscription per Education Level"})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    col3.plotly_chart(fig, use_container_width=True, sharing="streamlit")
    col3.write('''**Which level of education has the highest number of term deposit subscriptions?**''')
    level = df_bar['education'][df_bar.deposit == df_bar.deposit.max()].tolist()
    col3.write('The **' + level[0] + "** level has the highest number of term deposit subscriptions.")

    space(5,col1)
    space(3,col3)

    # Bar Chart 2: Deposit Subscription per Contact Type
    df_bar=df[df['deposit']=='yes']
    df_bar=df_bar.groupby(['contact'],as_index=False).count()
    fig = go.Figure([go.Bar(x=df_bar['deposit'],y=df_bar["contact"],orientation='h',marker_color='#C00000')])
    fig.update_layout(title={'text': "Deposit Subscription per Contact Type"})
    fig.update_layout( paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    col1.plotly_chart(fig, use_container_width=True, sharing="streamlit")
    col1.write('''**Which contact communication type has the highest number of term deposit subscriptions?**''')
    con = df_bar['contact'][df_bar.deposit == df_bar.deposit.max()].tolist()
    col1.write('The **' + con[0] + "** communication gives the highest number of term deposit subscriptions.")

    # Bar Chart 3: Deposit Subscription per Marital Status
    df_bar=df[df['deposit']=='yes']
    df_bar=df_bar.groupby(['marital'],as_index=False).count()
    fig = go.Figure([go.Bar(x=df_bar['deposit'],y=df_bar["marital"],orientation='h',marker_color='#C00000')])
    fig.update_layout(title={'text': "Deposit Subscription per Marital Status"})
    fig.update_layout( paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    col3.plotly_chart(fig, use_container_width=True, sharing="streamlit")
    col3.write('''**How term deposit subscriptions are distributed among different marital status?**''')
    status = df_bar['marital'][df_bar.deposit == df_bar.deposit.max()].tolist()
    col3.write('Clients who are **' + status[0] + "** shows the highest number of term deposit subscriptions.")

    space(3,col1)
    space(4,col3)

    # Bar Chart 4: Deposit Subscription per Job Type
    df_bar=df[df['deposit']=='yes']
    df_bar=df_bar.groupby(['job'],as_index=False).count()
    fig = go.Figure([go.Bar(x=df_bar['deposit'],y=df_bar["job"],orientation='h',marker_color='#C00000')])
    fig.update_layout(title={'text': "Deposit Subscription per Job Type"})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    col1.plotly_chart(fig, use_container_width=True, sharing="streamlit")
    col1.write('''**Which job type shows the highest number of term deposit subscriptions?**''')
    jtype = df_bar['job'][df_bar.deposit == df_bar.deposit.max()].tolist()
    col1.write('The **' + jtype[0] + "** job status shows the highest number of term deposit subscriptions.")

    # Line Chart: 
    df_line=df[df['deposit']=='yes']
    df_line['month']=pd.to_datetime(df_line['month'], format='%b').dt.month
    df_line=df_line.groupby(['month'],as_index=False).count()
    fig = go.Figure([go.Scatter(x=df_line['month'], y=df_line['deposit'],line=dict(color="#C00000"))])
    fig.update_layout(title={'text': "Deposit Subscription Over Months"})
    fig.update_xaxes(dtick="M1",tickformat="%b",ticklabelmode="period")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    col3.plotly_chart(fig, use_container_width=True, sharing="streamlit")
    col3.write('''**How do term deposit subscriptions vary over months?**''')
    mhigh = df_line['month'][df_line.deposit == df_line.deposit.max()].tolist()
    mlow = df_line['month'][df_line.deposit == df_line.deposit.min()].tolist()
    mdic={1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
    col3.write('**' + mdic[mhigh[0]] + "** shows the highest number of term deposit subscriptions whereas **" + mdic[mlow[0]] + '** shows the lowset number.')

    space(4,col1)
    space(4,col3)

    # Correlation Plot
    df1 = df[["age", "balance", "duration", "campaign"]]
    cr = df1.corr(method='pearson')
    fig = go.Figure(go.Heatmap(x=cr.columns, y=cr.columns, z=cr.values.tolist(), colorscale='OrRd', zmin=-1, zmax=1))
    fig.update_layout(title={'text': "Correlation Plot"})
    col1.plotly_chart(fig, use_container_width=True, sharing="streamlit")
    col1.write('''**What does this plot show?**''')
    col1.write('This plot aims to discover the linear realationship between the numerical features in the dataset.')

    
    # Boxplot 1: Last Contact Duration (in seconds) vs. Deposit Subscription Boxplot
    fig = go.Figure()
    fig.add_trace(go.Box(y=df["duration"],x=df['deposit'],marker_color='#C00000'))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(title={'text': "Last Contact Duration (seconds) vs. Deposit Subscription Boxplot"})
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    col3.plotly_chart(fig, use_container_width=True, sharing="streamlit")


    space(4,col1)
    space(10,col3)

    # Boxplot 2: Number of Contacts Performed vs. Deposit Subscription Boxplot
    fig = go.Figure()
    fig.add_trace(go.Box(y=df["campaign"],x=df['deposit'],marker_color='#C00000'))
    fig.update_layout(title={'text': "Number of Contacts Performed vs. Deposit Subscription Boxplot"})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    col1.plotly_chart(fig, use_container_width=True, sharing="streamlit")

    
    # Boxplot 3: Age vs. Deposit Subscription Boxplot
    fig = go.Figure()
    fig.add_trace(go.Box(y=df["age"],x=df['deposit'],marker_color='#C00000'))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(title={'text': "Age vs. Deposit Subscription Boxplot"})
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    col3.plotly_chart(fig, use_container_width=True, sharing="streamlit")

    space(170,col2)
    html='''<center><p style="border:3px; border-style:solid; border-color:#C00000; padding: 2em;
    "><b>There is a possible association between the features whose boxplots are not on the same level.</p>'''
    col2.write(html,unsafe_allow_html=True)


# Models Page
if menu_id=='Model':
    # transforms yes and no to 0,1
    def bool_to_dummy(row, column):
        return 1 if row[column] == 'yes' else 0

    # getting mean of column instead of a value
    def thre_to_mean(row, column, threshold, df):
        if row[column] <= threshold:
            return row[column]
        else:
            mean = df[df[column] <= threshold][column].mean()
            return mean
    
    def clean(df):
        cleaned_df = df.copy()

        # apply bool to dummy function
        bool_columns = ['default','deposit','loan', 'housing']
        for col in bool_columns:
            cleaned_df[col + '_bool'] = df.apply(lambda row: bool_to_dummy(row, col),axis=1)

        # drop old columns
        cleaned_df = cleaned_df.drop(columns = bool_columns)

        # transform categorical features to dummies
        cat_features = ['job', 'contact', 'poutcome','month','education','marital']
        for col in  cat_features:
            cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',drop_first=True, dummy_na=False)], axis=1)

        # drop unwanted columns
        cleaned_df = cleaned_df.drop(columns = ['pdays'])
    
        # impute values and drop old columns
        cleaned_df['campaign_new'] = df.apply(lambda row: thre_to_mean(row, 'campaign', 34, cleaned_df),axis=1)
        cleaned_df['previous_new'] = df.apply(lambda row: thre_to_mean(row, 'previous', 34, cleaned_df),axis=1)
        cleaned_df = cleaned_df.drop(columns = ['campaign', 'previous'])
        
        return cleaned_df

    
    # perform cleaning
    cleaned_df = clean(df)

    X = cleaned_df.drop(columns = 'deposit_bool')
    y = cleaned_df[['deposit_bool']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

    # build XGBoost model
    xgb = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7)
    xgb.fit(X_train,y_train.squeeze().values)

    # calculate with top 15 features
    y_test_preds = xgb.predict(X_test)

    # top features from xgb model
    headers = ["name", "score"]
    v = sorted(zip(X_train.columns, xgb.feature_importances_), key=lambda x: x[1] * -1)
    xgb_feature_importances = pd.DataFrame(v, columns = headers)

    # plot feature importances
    st.header("XGBoost Predictive Model")
    st.write(red_line, unsafe_allow_html=True)

    st.subheader("Accuracy")
    st.write("The XGBoost model was able to predict the success and fail of the marketing campaign with an accuarcy of **" + 
    str(round(accuracy_score(y_test, y_test_preds)*100,2)) +" %**.")
    st.write(grey_line, unsafe_allow_html=True)

    st.subheader("Top Features")
    x_pos = np.arange(0, len(xgb_feature_importances))
    fig = go.Figure([go.Bar(x=x_pos,y=xgb_feature_importances['score'],orientation='v',marker_color='#C00000')])

    fig.update_layout(xaxis = dict(tickmode = 'array',tickvals = x_pos,ticktext = xgb_feature_importances['name']))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, sharing="streamlit")
   
    st.write('''The top features affecting the bank marketing campaign are:''')
    st.write('''
    * **poutcome:** the outcome of the previous marketing campaigns (success or failure)
    * **duration:** the last contact duration (in seconds)
    * **month:** the last contact month of year
    * **housing:** whether the client has a housing loan or not
    * **loan:** whether the client has a personal loan or not''')
    st.write(grey_line, unsafe_allow_html=True)

    st.subheader("Conclusion")
    st.write('''The bank should highly take into consideration the outcome of previous marketing campaigns, the last contact duration,
     the last contact month of year, and the loans of the clients as theese features appear to highly affect the marketing campaigns designed by the bank.''')

# Meta Data Page
if menu_id=='Meta Data':
    # Table of data
    st.subheader("Table of Data")    
    df = pd.read_csv("https://raw.githubusercontent.com/mam127/Project-370/main/bank.csv")
    st.write(df)
    st.write(grey_line, unsafe_allow_html=True)

    # Description of Variables
    st.subheader("Description of Features")
    col1,col2,col3,col4,col5,col6= st.columns([1,1,1,1,1,1])
    if col1.button("Age"):
        col1.write("Age of the client")

    if col2.button("Job"):
        col2.write("Type of client's job")

    if col3.button("Marital"):
        col3.write("Marital status: divorced, married, single, or unknown")

    if col4.button("Education"):
        col4.write("Level of education: primary, secondary, tertiary, or unknown")

    if col5.button("Default"):
        col5.write("Has credit in default?")

    if col6.button("Housing"):
        col6.write("Has housing loan?")

    if col1.button("Loan"):
        col1.write("Has personal loan?")

    if col2.button("Balance"):
        col1.write("Balance of each client")

    if col3.button("Contact"):
        col3.write("Contact communication type: cellular, telephone, or unknown")

    if col4.button("Month"):
        col4.write("Last month of year contacted")

    if col5.button("Day"):
        col5.write("Last day of week contacted")

    if col6.button("Duration"):
        col6.write("Last contact duration in seconds")

    if col1.button("Campaign"):
        col1.write('''Number of contacts performed''')

    if col2.button("pdays"):
        col2.write("Number of days that passed after which the client was last contacted")

    if col3.button("previous"):
        col3.write("Number of contacts performed before this campaign")

    if col4.button("poutcome"):
        col4.write("Outcome of previous campaigns")

    if col5.button("Deposit"):
        col5.write("Has the client subscribed for a term deposit?")


    

