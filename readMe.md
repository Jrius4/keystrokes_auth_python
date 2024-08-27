Flask app with keystroke-based continuous authentication. Users will enter their username, and the app will use various neural network models (CNN, LSTM, GAN, MLP) for authentication. The project will use MySQL for data storage, and keystroke data and features will be stored as CSV files, with paths to these files saved in the database. The trained model paths will also be stored in the database.

        start app
            flask db init
            flask db migrate
            flask db upgrade
