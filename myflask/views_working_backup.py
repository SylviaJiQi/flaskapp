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
from skimage.future import graph


# Python code to connect to Postgres
# You may need to modify this based on your OS, 
# as detailed in the postgres dev setup materials.
user = 'postgres' #add your Postgres username here      
host = 'localhost'
dbname = 'harrods_db'
password = 'Science.2017'
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
                SELECT * FROM harrodsdressheels_table WHERE mprice < 200;          
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
    uploads_root = os.path.join('/home/miracle/flask/myflask/static', 'uploads')
    save_path = os.path.join(uploads_root, file.filename)
    file.save(save_path)
    url_path = os.path.join('/static/uploads', file.filename)

    sql_query = """                                                                       
                SELECT mid, mname, pid, r, g, b FROM harrodsdressheels_table;          
                """
    df = pd.read_sql_query(sql_query,con)

    name_list = ''
    match_image_list = ''
    url_list = ''
    prod_image_list = ''

    img = models.load_image(save_path)	

    im_rgb = models.compute_rgb(img)

    df['color_dist'] = (df['r']-im_rgb[0])**2+(df['g']-im_rgb[1])**2+(df['b']-im_rgb[2])**2
    min_id = df['color_dist'].idxmin()



    name_list += df.iloc[min_id]['mname']

        
    tmp = str(df.iloc[min_id]['mid'])+'_1.jpg'

    match_image_list += os.path.join('/static/data/Harrods', tmp)
 
    tmp = str(df.iloc[min_id]['pid'])+'_1.jpg'
    prod_image_list = os.path.join('/static/data/Harrods', tmp)

#    shoe_url = 


    return render_template(
        'match.html',
        filepath=url_path,
        matches_prod_image=prod_image_list,
	matches_name = name_list,
        matches_image = match_image_list,
    )



# give the similarity of the dress


