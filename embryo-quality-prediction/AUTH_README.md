# Authentication System for Embryo Quality Prediction

This document explains how to set up and use the authentication system for the Embryo Quality Prediction application.

## Overview

The authentication system provides:
- User registration with email and password
- User login with email and password
- Session management for authenticated users
- Access control to protect application routes

## Prerequisites

- MySQL Server 5.7+ installed and running
- Python 3.7+ with required packages (see requirements.txt)
- bcrypt package for password hashing

## Setup Instructions

1. Install required Python packages:

```bash
pip install -r requirements.txt
```

2. Set up the authentication database:

```bash
# Run the authentication database setup script
python setup_auth_db.py --host localhost --user suba --password Suba@123 --database embryo_predictions
```

This script will:
- Create the users table in the database if it doesn't exist
- Set up stored procedures for user authentication
- Configure the database connection

## Database Schema

The authentication system uses the following table:

### users
- `id`: Auto-increment primary key
- `email`: User's email address (unique)
- `password`: Hashed password (using bcrypt)
- `created_at`: When the user account was created
- `last_login`: When the user last logged in

## Usage

### Starting the Application

```bash
# Start the Flask web application
python app/app.py
```

### User Registration

1. Access the registration page at http://localhost:5000/register
2. Enter your email and password
3. Click "Sign Up" to create an account

### User Login

1. Access the login page at http://localhost:5000/login
2. Enter your email and password
3. Click "Sign In" to log in to the application

### User Logout

1. Click the "Sign Out" button in the top right corner of the application

## Security Features

- Passwords are hashed using bcrypt before storage
- Session data is stored securely
- All application routes are protected with the login_required decorator
- Input validation for registration and login forms

## Customization

You can customize the authentication system by:
- Editing the templates in app/templates/login.html and register.html
- Modifying the authentication routes in app/app.py
- Adding additional user profile fields to the users table

## Troubleshooting

If you encounter issues with the authentication system:

1. Check that MySQL is running and accessible
2. Verify that the database credentials are correct
3. Ensure that all required packages are installed
4. Check the application logs for error messages

## Database Connection

The authentication system uses the following default database connection parameters:
- Host: localhost
- User: suba
- Password: Suba@123
- Database: embryo_predictions

To customize these settings, modify the parameters when running the setup script or edit the connection parameters in src/auth_utils.py. 