#!/usr/bin/env python
"""
Setup Authentication Database Script

This script sets up the authentication database for the Embryo Quality Prediction application.
It creates the necessary tables for user authentication.
"""

import os
import sys
import argparse

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from src.auth_utils import setup_auth_database

def main():
    """Setup authentication database tables and stored procedures."""
    
    parser = argparse.ArgumentParser(description='Setup Authentication Database')
    parser.add_argument('--host', default='localhost', help='MySQL host')
    parser.add_argument('--user', default='suba', help='MySQL username')
    parser.add_argument('--password', default='Suba@123', help='MySQL password')
    parser.add_argument('--database', default='embryo_predictions', help='MySQL database name')
    
    args = parser.parse_args()
    
    print("Setting up authentication database...")
    success = setup_auth_database(
        host=args.host,
        user=args.user,
        password=args.password,
        database=args.database
    )
    
    if success:
        print("Authentication database setup completed successfully!")
    else:
        print("Authentication database setup failed. Check logs for details.")
        sys.exit(1)
        
    print(f"\nYou can now start the application with:")
    print(f"cd {SCRIPT_DIR}")
    print(f"python app/app.py")
    
    print("\nTo sign in to the application, first create a user account.")
    print("Open http://localhost:5000/register in your browser.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 