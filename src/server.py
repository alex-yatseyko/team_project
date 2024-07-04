import pickle
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import pandas as pd
import urllib.parse as urlparse

# Load the model
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'r') as file:
                self.wfile.write(file.read().encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = urlparse.parse_qs(post_data.decode())

            # Extract input features
            try:
                features = {
                    'person_age': float(data.get('person_age', [0])[0]),
                    'person_income': float(data.get('person_income', [0])[0]),
                    'person_emp_length': float(data.get('person_emp_length', [0])[0]),
                    'loan_amnt': float(data.get('loan_amnt', [0])[0]),
                    'loan_int_rate': float(data.get('loan_int_rate', [0])[0])
                }
            except ValueError as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Invalid input parameters: {str(e)}".encode())
                return

            # Convert features to DataFrame
            features_df = pd.DataFrame([features])

            # Make a prediction
            try:
                prediction = model.predict(features_df)[0]
            except ValueError as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Invalid input parameters: {str(e)}".encode())
                return

            # Send the response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'loan_status': int(prediction)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}')
    httpd.serve_forever()

if __name__ == "__main__":
    run()
