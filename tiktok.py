from flask import Flask, redirect, url_for, render_template, request
import joblib
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import io
import base64
import dataclean
import dataload
app = Flask(__name__)

#---------------------------home------------------------------#
@app.route('/')  
def home():  
    return render_template("welcome.html")
#---------------------------home------------------------------#
#
#
#---------------------------5 sub pages------------------------------#
@app.route('/description')   
def description():  
    return render_template('description.html')

@app.route('/explore')   
def explore():  
    #df = pd.read_csv('trending_sub.csv').head()
    return render_template('exploration.html')#,  tables=[pd.DataFrame.to_html()])

@app.route('/visual')   
def visual():  
    #df = pd.read_csv('trending_sub.csv').head()
    return render_template('visualization.html')#,  tables=[df.to_html()]

@app.route('/extract')   
def extract():  
    #implement ML to select features
    '''from xgboost import XGBRegressor
    from xgboost import plot_importance
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import SelectFromModel
    from sklearn.metrics import r2_score
    
    df = dataclean.clean()
    x = df[['shareCount','playCount','commentCount','musicMetaMusicOriginal','videoMetaHeight','videoMetaWidth','videoMetaDuration','text_len','authorMetaVerified']]
    y = df['diggCount']
    x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.7, random_state=30)

    #XGBoost(default)
    xgb = XGBRegressor(random_state=30)
    xgb.fit(x_train, y_train)

    #feature importance
    feature_names = list(x_train.columns)
    feature_importances = list(xgb.feature_importances_)
    dataf = pd.DataFrame({'feature_names':feature_names, 'feature_importances':feature_importances})
    dataf.sort_values('feature_importances', ascending=False, inplace=True)
    dataf.reset_index(drop=True, inplace=True)
    #print(dataf)

    #show feature importance
    plot1 = plot_importance(xgb)
    img = io.BytesIO()
    fig = plot1.get_figure()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # select features using threshold
    thresholds = sorted(xgb.feature_importances_)
    lst = []
    for thresh in thresholds:
        selection = SelectFromModel(xgb, threshold=thresh, prefit=True)
        select_x_train = selection.transform(x_train)
        # train model
        selection_model = XGBRegressor(random_state=30)
        selection_model.fit(select_x_train, y_train)
        # eval model
        select_x_test = selection.transform(x_test)
        y_pred = selection_model.predict(select_x_test)
        dic = {}
        dic['Thresh'] = thresh
        dic['n'] = select_x_train.shape[1]
        dic['R2 Score'] = r2_score(y_test, y_pred)
        lst.append(dic)
        #print("Thresh=%.3f, n=%d, R2 Score: %.5f" % (thresh, select_x_train.shape[1], r2_score(y_test, y_pred)))
    dff = pd.DataFrame(lst)'''
    #return render_template('load.html', plot_url = plot_url, tables = [dff.to_html()])
    return render_template('extract-img.html')

@app.route('/pred', methods=['POST', 'GET'])   
def pred():  
    return render_template('prediction.html')#, tables=[df.to_html()]
#---------------------------5 sub pages------------------------------#
#
#
#---------------------------exploration------------------------------#
@app.route('/explore/load')   
def loading(): 
    dataload.load('trending.csv')
    return render_template('load.html', content='Load data successfully!')

@app.route('/explore/clean')   
def clean(): 
    #df = dataclean.clean()
    return render_template('data-clean.html') # tables=[dataclean.clean().head().to_html()]) 

@app.route('/explore/clean/load')   
def clean_data_load():  
    dataload.load('trending_sub.csv')
    return render_template('load.html', content='Load cleaned data successfully!')

@app.route('/explore/clean/structure')   
def structure():  
    df = pd.read_csv('trending_sub.csv')
    return render_template('datastructure.html', content=df.shape, describe=f'It has {df.shape[0]} rows and {df.shape[1]} columns.')

@app.route('/explore/clean/info')   
def info():  
    ##df = pd.read_csv('trending_sub.csv')
    ##buffer = io.StringIO()
    ##df.info(buf=buffer)
    ##re = buffer.getvalue()
    ##dff = pd.DataFrame(re.split("\n"), columns=['info'])
    return render_template('shape-img.html')# tables=[dff.to_html()])


#---------------------------exploration------------------------------#
#
#
#---------------------------visualization------------------------------#
@app.route('/visual/visual1')   
def visual1():  
    '''df = dataclean.clean()  #suppose to get data from Firebase directly
    img = io.BytesIO()
    import missingno as msno
    plot = msno.matrix(df, labels=True)
    fig = plot.get_figure()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()'''
    ##return render_template('img.html', plot_url = plot_url)
    return render_template('img.html')


@app.route('/visual/visual2')   
def visual2(): 
    #df = dataclean.clean()  #suppose to get data from Firebase directly
    #img = io.BytesIO()
    #y = [1,2,3,4,5]
    #x = [0,2,1,3,4]
    #plt.plot(x,y)
    #plt.savefig(img, format='png')
    #img.seek(0)
    #plot_url = base64.b64encode(img.getvalue()).decode()
    #return render_template('img.html', plot_url=plot_url)
    return render_template('img2.html')
    

@app.route('/visual/visual3')   
def visual3():
    return render_template('img3.html')
#---------------------------visualization------------------------------#
#
#
#---------------------------feature extraction------------------------------#
@app.route('/extract/load')   
def feature_load():  
    #dataload.load()
    return render_template('load.html', content='Load features successfully!')
#---------------------------feature extraction------------------------------#
#
#
#---------------------------prediction------------------------------#
#@app.route('/pred/pred1/load')   
#def result_load():
#    #dataload.load()
#    return render_template('load.html', content='Store results successfully!')

@app.route('/pred/pred1', methods = ['POST','GET'])   
def pred1():
    shareCount = request.form.get('shareCount')
    playCount = request.form.get('playCount')
    commentCount = request.form.get('commentCount')
    videoMetaDuration = request.form.get('videoMetaDuration')
    text_len = request.form.get('text_len')
    dic = {}
    dic['shareCount'] = int(shareCount)
    dic['playCount'] = int(playCount)
    dic['commentCount'] = int(commentCount)
    dic['videoMetaDuration'] = int(videoMetaDuration)
    dic['text_len'] = int(text_len)
    x = pd.DataFrame(dic, index = [0])
    model = joblib.load('xgb.model')
    y = model.predict(x)[0]
    dataload.load(y, shareCount=shareCount, playCount=playCount, commentCount=commentCount, videoMetaDuration=videoMetaDuration, text_len=text_len)
    return render_template('pred1.html', content=f'digCount prediction: {y}, successfully stored in Firebase.')

#---------------------------prediction------------------------------#
#
#
if __name__ =="__main__":
    app.run() 