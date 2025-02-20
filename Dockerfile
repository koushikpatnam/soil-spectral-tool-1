# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install flask pandas numpy scikit-learn tensorflow keras matplotlib seaborn reportlab plotly python-ternary





# Expose port 8080
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
