import json
import logging
import os
import sys
import traceback

import json_logging
from app.exceptions import BackendError
from dotenv import load_dotenv
from flask import Flask, current_app, jsonify
from flask_cors import CORS
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect
from nte_aisdk.errors import AuthError, SDKError
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)

load_dotenv()

def setup_logging(flask_app: Flask):
    logging.basicConfig()
    logging.getLogger().setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    logging.getLogger().handlers = [logging.StreamHandler(stream=sys.stdout)]
    json_logging.init_flask(enable_json=True)
    json_logging.init_request_instrument(flask_app)


def init_flask(app):
    app.config.from_object(
        f'config.{os.environ.get("CX_POD_ENV", "Local").capitalize()}'
    )

    with app.app_context():
        # init ai client including access control before registering blueprint
        from app.clients import ai

        ai.init_app(app)

        # register blueprint
        from app.routes import bp

        app.register_blueprint(bp)
    return app


def extensions(app):
    # logging
    setup_logging(app)

    # security
    # uncomment to enable CSRF protection
    # csrf = CSRFProtect()
    # csrf.init_app(app)

    CORS(
        app,
        origins=app.config["ALLOW_ORIGINS"].split(","),
        supports_credentials=True
    )

    Talisman(app, force_https=False)

    @app.after_request
    def add_cache_header(response):
        response.headers[
            "Cache-Control"
        ] = "no-cache, no-store, max-age=0, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["Surrogate-Control"] = "public, max-age=0"
        return response

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        logger.error(e)
        response = e.get_response()
        response.data = json.dumps({
            "code": e.code,
            "name": e.name,
            "description": e.description,
        })
        response.content_type = "application/json"
        return response

    @app.errorhandler(SDKError)
    def handle_sdk_exception(error):
        logger.error(error)
        response = {
            "message": str(error.message),
        }
        status_code = error.status_code if getattr(error, "status_code", None) else 500
        return response, status_code

    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        logger.error(error)
        return jsonify({
            "message": str(error),
            "errors": error.errors(include_url=False, include_context=False),
        }), 400

    @app.errorhandler(AuthError)
    def handle_auth_error(error):
        logger.error(error)
        return jsonify({
            "message": error.message,
        }), 403

    @app.errorhandler(BackendError)
    def handle_backend_error(error):
        logger.error(error)
        return jsonify({
            "message": error.message,
        }), error.status_code

    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.error(e)
        traceback.print_exc()
        return "Server Internal Error", 500

    return app


def clients(app):
    from app.clients.ai import ai

    ai.init_app(app)
    return app

def create_app():
    app = Flask(__name__)
    app = init_flask(app)
    app = extensions(app)
    return app


app = create_app()
