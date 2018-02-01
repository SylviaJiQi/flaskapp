#! /usr/bin/env python
from findmydress import config
from findmydress.web.app import app
from findmydress.web import views


if __name__ == '__main__':
    app.run(
        host=config.FLASK_BIND_HOST,
        port=config.FLASK_BIND_PORT,
        debug=config.FLASK_DEBUG,
        )
