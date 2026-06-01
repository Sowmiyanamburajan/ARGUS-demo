#!/usr/bin/env bash
export FLASK_APP=app.py
export FLASK_ENV=development
python -m pip install -r requirements.txt
python app.py
