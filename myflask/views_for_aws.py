from flask import render_template
from flask import request
from myflask import app
from myflask import models
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import os
from urllib.parse import urlparse
from skimage import data, segmentation, color, io
#from skimage.future import graph
#import cyvlfeat



# Python code to connect to Postgres
# You may need to modify this based on your OS, 
# as detailed in the postgres dev setup materials.
user = 'postgres' #add your Postgres username here      
host = 'localhost'
dbname = 'harrods_db'
password = 'abcd1234'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user, host = host, password = password)


# Index/Cover page
# Save the uploaded image to static/uploads
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/db')
def test_item():
    sql_query = """                                                                       
                SELECT * FROM harrodsdressheels_table;          
                """
    df = pd.read_sql_query(sql_query,con)

    test_items = ""
    for i in range(0,5):
        test_items += df.iloc[i]['murl']
        test_items += "<br>"
    return test_items


@app.route('/match', methods=['GET', 'POST'])
def match_item():
    if request.method == 'GET':
        return flask.redirect(url_for('index'))

    file = request.files['query']

# add root folder path
 ROOT_FOLDER = os.path.realpath(__file__).split('views')[0]

# Read data base
    sql_query = """                                                                       
                SELECT * FROM harrodsdressheels_table;          
                """
    df = pd.read_sql_query(sql_query,con)   


    uploads_root = os.path.join(ROOT_FOLDER,'static/uploads')
    save_path = os.path.join(uploads_root, file.filename)
    file.save(save_path)
#    url_path = os.path.join('/static/uploads', file.filename)
    url_path = save_path

# call models.py, to give the top 3 matching items
    

    im = io.imread(save_path)
    color_similarity,main_color = models.compute_dress_similarity(save_path)



# compare the main color
    df1 = df[['pid','mid','murl']] 
    df1['distance'] = (df['c0']-main_color[0])**2+(df['c1']-main_color[1])**2+(df['c2']-main_color[2])**2+(df['c3']-main_color[3])**2+(df['c4']-main_color[4])**2+(df['c5']-main_color[5])**2+(df['c6']-main_color[6])**2+(df['c7']-main_color[7])**2+(df['c8']-main_color[8])**2
    
    df1 = df1.sort_values(['distance'],ascending = True)

    color_similarity = df1.index[0:3].tolist()
 #   print(main_color)

# 1_30 modification

    name_list = []
    name_list.append(str(df.loc[color_similarity[0],'mname']))
    name_list.append(str(df.loc[color_similarity[1],'mname']))
    name_list.append(str(df.loc[color_similarity[2],'mname']))


    match_image_list = []
    prefix_img = ROOT_FOLDER+'static/data/Harrods'
    tmp = str(df.loc[color_similarity[0],'mid'])+'_1.jpg'

    print("The image source folder is")
    print(tmp)

    match_image_list.append(os.path.join(prefix_img, tmp))
    tmp = str(df.loc[color_similarity[1],'mid'])+'_1.jpg'
    match_image_list.append(os.path.join(prefix_img, tmp))

    tmp = str(df.loc[color_similarity[2],'mid'])+'_1.jpg'
    match_image_list.append(os.path.join(prefix_img', tmp))



    prod_image_list = []
    tmp = str(df.loc[color_similarity[0],'pid'])+'_1.jpg'
    prod_image_list.append(os.path.join(prefix_img, tmp))
    tmp = str(df.loc[color_similarity[1],'pid'])+'_1.jpg'
    prod_image_list.append(os.path.join(prefix_img, tmp))
    tmp = str(df.loc[color_similarity[2],'pid'])+'_1.jpg'
    prod_image_list.append(os.path.join(prefix_img, tmp))

    shoe_url = []
    shoe_url.append(df.loc[color_similarity[0],'murl'])
    shoe_url.append(df.loc[color_similarity[1],'murl'])
    shoe_url.append(df.loc[color_similarity[2],'murl'])
# end of 1_30 modification

    return render_template(
        'match.html',
        filepath=url_path,
        matches_prod_image=prod_image_list,
	matches_name = name_list,
        matches_image = match_image_list,
	match_url = shoe_url
    )



# give the similarity of the dress


