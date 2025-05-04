CREATE DATABASE boston_project;
CREATE USER "dodobeatle@gmail.com" WITH ENCRYPTED PASSWORD 'airbyte';
GRANT ALL PRIVILEGES ON DATABASE mlops TO "dodobeatle@gmail.com";
GRANT ALL ON SCHEMA public TO "dodobeatle@gmail.com";
GRANT USAGE ON SCHEMA public TO "dodobeatle@gmail.com";
ALTER DATABASE boston_project OWNER TO "dodobeatle@gmail.com";