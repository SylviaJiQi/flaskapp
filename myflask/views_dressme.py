import base64
import os
import re
import uuid

import flask
from flask import (
    redirect, render_template, request, url_for,
    )

from myflask import config
#from findmydress.db import models
#from findmydress.db.models import ItemImage, write_s3_image, ImageDerivative
#from findmydress.web import match
#from findmydress.web.app import app
#from myflask.app import app
from myflask import app
from myflask import match


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/match/<short_code>', methods=['GET'])

def match_saved_item(short_code):
    #find item url for short_code and render template with url
    session = models.Session()
    match = (session.query(models.ImageMatchRequest)
                        .filter(models.ImageMatchRequest.short_code == short_code)
                        .one_or_none())

    if not match:
        return "Not Found", 404

    return render_template(
        'match.html',
        filepath=match.get_s3_image_url(),
        matches = []
        )
#    return render_template(
#        'match.html',
#
#        matches = []
#        )


@app.route('/match', methods=['GET', 'POST'])
def match_item():
    if request.method == 'GET':
        return flask.redirect(url_for('index'))

    file = request.files['query']
    uploads_root = os.path.join('/home/miracle/flask/myflask/static', 'uploads')
    save_path = os.path.join(uploads_root, file.filename)
#    url_path = os.path.join('/static/uploads', file.filename)
    url_path = save_path
    file.save(save_path)

    probability_list = match.find_similar_items(
        match.load_model(),
        match.extract_image_features(save_path),
        )

    session = models.Session()
    dress_ids = [r['dress_id'] for r in probability_list if r['dress_id']]
    dress_items = session.query(models.Item).filter(models.Item.id.in_(dress_ids))
    dress_item_map = {d.id: d for d in dress_items}

    dress_images = (session.query(models.ItemImage)
                    .filter(models.ItemImage.item_id.in_(dress_ids))
                    .filter(models.ItemImage.position == 0))
    dress_image_map = {d.item_id: d for d in dress_images}

    for prob_result in probability_list:
        # get an item from dress_index for prob_result
        item_id = prob_result['dress_id']
        prob_result['item'] = dress_item_map.get(item_id)
        prob_result['image'] = dress_image_map.get(item_id)

    probability_list.sort(key=lambda x: x['probability'], reverse=True)

    return render_template(
        'match.html',
        filepath=url_path,
        matches=probability_list,
    )


@app.route('/items/<int:item_id>', methods=['GET'])
def item_detail(item_id):
    session = models.Session()
    item = session.query(models.Item).get(item_id)
    if not item:
        return "Not Found", 404

    return render_template(
        'item-detail.html',
        item=item,
        )


@app.route('/images/')
def images():
    session = models.Session()
    images = session.query(models.ItemImage)[:30]

    result_obj = {
        'labels': config.STATIC_LABELS,
        'imageURLs': [flask.url_for('image_file', image_id=i.id) for i in images],
        'annotationURLs': [flask.url_for('image_annotation_file', image_id=i.id) for i in images],
        }
    return flask.jsonify(result_obj)


@app.route('/images/<int:image_id>')
def image_detail(image_id):
    session = models.Session()
    image = models.ItemImage.query.get(image_id)
    return "TODO: fill in image detail"


@app.route('/images/<int:image_id>/file')
def image_file(image_id):
    session = models.Session()
    image = session.query(models.ItemImage).get(image_id)

    aws_session = aws.get_aws_session()
    s3 = aws_session.client('s3')
    return flask.redirect(image.get_s3_image_url(s3))


@app.route('/images/<int:image_id>/annotation', methods=['GET', 'POST'])
def image_annotation_file(image_id):
    session = models.Session()
    annotation_image = (session.query(ImageDerivative)
                        .filter(ImageDerivative.original_image == image_id)
                        .filter(ImageDerivative.type == 'annotation')
                        .one_or_none())

    if request.method == 'GET':
        if not annotation_image:
            return "Not Found", 404

        aws_session = aws.get_aws_session()
        s3 = aws_session.client('s3')
        return flask.redirect(annotation_image.get_s3_image_url(s3))

    elif request.method == 'POST':
        mimetype, data = parse_data_url(flask.request.form['imgDataURL'])
        # TODO: validate that this is the same size as the original
        aws_session = aws.get_aws_session()
        s3 = aws_session.resource('s3')

        image_id = uuid.uuid4().hex
        path = 'files/images/annotations/{}'.format(image_id)
        url = write_s3_image(data, mimetype, path)

        if not annotation_image:
            annotation_image = ImageDerivative(
                original_image=image_id,
                image_s3_url=url,
                type='annotation',
            )
        else:
            annotation_image.image_s3_url = url
        session.add(annotation_image)
        session.commit()

        return url


DATA_URL_PATTERN = re.compile(r'data:(?P<mimetype>.+);(?P<encoding>base64),(?P<encoded_data>.+)$')
def parse_data_url(url):
    match = DATA_URL_PATTERN.match(url)
    if not match:
        raise ValueError("Couldn't parse data URL")

    mimetype = match.group('mimetype')
    data = base64.b64decode(match.group('encoded_data'))
    return mimetype, data
