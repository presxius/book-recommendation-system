from app import app, initialize_system

if __name__ == "__main__":
    # This file is just for local development
    # Railway uses gunicorn directly
    initialize_system()
    app.run(host='0.0.0.0', port=5000, debug=False)