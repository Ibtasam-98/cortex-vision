# cortex_vision (ML Backend)

This repository contains the Python-based machine learning backend for the Cortex Vision mobile application. This backend is responsible for receiving eye images, processing them using machine learning algorithms, and predicting the likelihood of cataracts. It exposes a Flask API for communication with the mobile application.

## Overview

The Cortex Vision mobile application (available in a separate repository) allows users to upload images of their eyes. This backend processes these images to detect features relevant to cataract diagnosis and utilizes trained machine learning models to generate a prediction. The communication between the mobile app and this backend is done via a RESTful Flask API.

## Repository Structure
