from app import app, initialize_system

if __name__ == "__main__":
    # Initialize system and run app
    if initialize_system():
        print("System initialized successfully!")
    else:
        print("System initialization failed, running in limited mode")
    app.run()