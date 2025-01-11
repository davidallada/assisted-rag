#!/bin/bash

cd /app

django-admin startproject backend

cd backend

python manage.py startapp assistedrag

python manage.py migrate
