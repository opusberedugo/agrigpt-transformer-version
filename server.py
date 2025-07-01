from flask import Flask, request, jsonify
from model import DeepRoot

app = Flask(__name__)

# Initialize DeepRoot with error handling
try:
    deeproot = DeepRoot()
    print("DeepRoot model initialized successfully")
except Exception as e:
    print(f"Error initializing DeepRoot: {e}")
    deeproot = None

@app.route('/introduce', methods=['GET'])
def introduce():
    try:
        if deeproot is None:
            return jsonify({"error": "Model not initialized"}), 500
        
        introduction = deeproot.introduce()
        return jsonify({"introduction": introduction})
    except Exception as e:
        print(f"Error in introduce endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/chat', methods=['POST', 'GET'])
def chat():
    try:
        if deeproot is None:
            return jsonify({"error": "Model not initialized"}), 500
        
        # Handle both GET and POST requests properly
        if request.method == 'POST':
            # For POST requests, try to get message from JSON body first, then query params
            data = request.get_json(silent=True)
            if data and 'message' in data:
                message = data['message']
            else:
                message = request.args.get('message')
        else:  # GET request
            message = request.args.get('message')
        
        print(f"Received message: {message}")
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        if not message.strip():  # Check for empty/whitespace-only messages
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # Generate response with error handling
        response = deeproot.generate_response(message)
        return jsonify({"message": response})
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Failed to generate response"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)